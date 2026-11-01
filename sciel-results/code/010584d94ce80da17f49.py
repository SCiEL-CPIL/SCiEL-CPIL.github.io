"""Combined approach: algebraic structured search for tensor decomposition."""
import time
import random
import numpy as np
from scipy import linalg
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def compute_R(U, V, W):
        """Compute the reconstructed tensor."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return R

    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        R = compute_R(U, V, W)
        return np.sum((R - target) ** 2)

    def discrete_search(rank, max_iters=50000, T_init=1.0, T_final=0.01):
        """
        Perform a discrete local search (Simulated Annealing) on the half-integer grid.
        """
        # Initialize with sparse random half-integers
        # We want to encourage sparsity, so start with mostly zeros
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Randomly set some entries to non-zero half-integers
        # Density ~ 1/10
        for c in range(rank):
            for i in range(nm):
                if random.random() < 0.1:
                    U[i, c] = random.choice(vals)
            for i in range(mp):
                if random.random() < 0.1:
                    V[i, c] = random.choice(vals)
            for i in range(pn):
                if random.random() < 0.1:
                    W[i, c] = random.choice(vals)
        
        current_error = tensor_error(U, V, W)
        best_error = current_error
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        T = T_init
        cooling_rate = (T_final / T_init) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Pick a random move
            mat_type = random.randint(0, 2) # 0: U, 1: V, 2: W
            if mat_type == 0:
                row = random.randint(0, nm - 1)
                col = random.randint(0, rank - 1)
                old_val = U[row, col]
                new_val = random.choice(vals)
                if new_val == old_val: continue
                U[row, col] = new_val
                new_error = tensor_error(U, V, W)
                # Revert if not accepted
                if new_error > current_error:
                    delta = new_error - current_error
                    if random.random() > np.exp(-delta / T):
                        U[row, col] = old_val
                        continue
            elif mat_type == 1:
                row = random.randint(0, mp - 1)
                col = random.randint(0, rank - 1)
                old_val = V[row, col]
                new_val = random.choice(vals)
                if new_val == old_val: continue
                V[row, col] = new_val
                new_error = tensor_error(U, V, W)
                if new_error > current_error:
                    delta = new_error - current_error
                    if random.random() > np.exp(-delta / T):
                        V[row, col] = old_val
                        continue
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, rank - 1)
                old_val = W[row, col]
                new_val = random.choice(vals)
                if new_val == old_val: continue
                W[row, col] = new_val
                new_error = tensor_error(U, V, W)
                if new_error > current_error:
                    delta = new_error - current_error
                    if random.random() > np.exp(-delta / T):
                        W[row, col] = old_val
                        continue
            
            current_error = new_error
            if current_error < best_error:
                best_error = current_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if best_error < 1e-9:
                    return best_U, best_V, best_W
            
            T *= cooling_rate
        
        return best_U, best_V, best_W

    best_result = None
    best_rank = 42
    
    # Try ranks from lower bound (21) to schoolbook (42)
    # We prioritize lower ranks
    for rank in range(21, 43):
        if time.time() > deadline - 20:
            break
            
        # Try a few random restarts for each rank
        # Time budget per rank: 5 seconds
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts and time.time() < deadline - 25:
            U, V, W = discrete_search(rank, max_iters=10000)
            if verify(U, V, W):
                best_result = (U, V, W)
                best_rank = rank
                break
            attempts += 1
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
