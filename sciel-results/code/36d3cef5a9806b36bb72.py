import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    start_time = time.time()
    time_limit = 145.0
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = get_schoolbook()
    
    # Precompute all (i,j,k) indices for fast error computation
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def compute_error(U, V, W, r):
        """Compute sum of squared errors for all (i,j,k) triples."""
        err = 0.0
        for i, j, k in triples:
            s = 0.0
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            for c in range(r):
                s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            err += (s - 1.0) ** 2
        return err
    
    def is_valid(U, V, W, r):
        """Check if decomposition is exact."""
        for i, j, k in triples:
            s = 0.0
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            for c in range(r):
                s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            if abs(s - 1.0) > 1e-8:
                return False
        return True
    
    def check_half_integer(U, V, W):
        """Check all entries are integer or half-integer."""
        for mat in (U, V, W):
            for row in mat:
                for val in row:
                    if abs(2.0*val - round(2.0*val)) > 1e-8:
                        return False
        return True
    
    # Try to find lower rank using structured search
    # Strategy: Start from schoolbook, try to eliminate columns by finding
    # alternative decompositions that cover multiple (i,j,k) constraints
    
    # For each rank from 59 down to 40, try to find a valid decomposition
    # Use local search with restarts
    
    for target_rank in range(59, 39, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        found = False
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Initialize: start from schoolbook and randomly perturb
            # Keep most schoolbook structure but modify some entries
            U = np.zeros((nm, target_rank), dtype=float)
            V = np.zeros((mp, target_rank), dtype=float)
            W = np.zeros((pn, target_rank), dtype=float)
            
            # Fill with schoolbook-like pattern for first min(target_rank, 60) columns
            # Then add random perturbation
            num_cols = min(target_rank, n*m*p)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < target_rank:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = 1.0
                            col += 1
            
            # Add random perturbation to all entries (small chance)
            if target_rank < n*m*p:
                # We have fewer columns than schoolbook, so we need to be clever
                # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
                candidates = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                U = np.random.choice(candidates, size=(nm, target_rank))
                V = np.random.choice(candidates, size=(mp, target_rank))
                W = np.random.choice(candidates, size=(pn, target_rank))
            
            # Local search with simulated annealing
            current_err = compute_error(U, V, W, target_rank)
            best_err = current_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            temperature = 1.0
            cooling_rate = 0.9995
            min_temp = 0.01
            
            steps_per_restart = 2000
            for step in range(steps_per_restart):
                if time.time() - start_time > time_limit - 10:
                    break
                if temperature < min_temp:
                    break
                
                # Pick a random variable to modify
                var = random.randint(0, 2)
                if var == 0:
                    idx = random.randint(0, nm-1)
                    c = random.randint(0, target_rank-1)
                    old_val = U[idx, c]
                    new_val = random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                    if new_val == old_val:
                        continue
                    U[idx, c] = new_val
                    new_err = compute_error(U, V, W, target_rank)
                    delta = new_err - current_err
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            if best_err < 1e-8:
                                if is_valid(best_U, best_V, best_W, target_rank) and check_half_integer(best_U, best_V, best_W):
                                    found = True
                                    break
                    else:
                        U[idx, c] = old_val
                elif var == 1:
                    idx = random.randint(0, mp-1)
                    c = random.randint(0, target_rank-1)
                    old_val = V[idx, c]
                    new_val = random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                    if new_val == old_val:
                        continue
                    V[idx, c] = new_val
                    new_err = compute_error(U, V, W, target_rank)
                    delta = new_err - current_err
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            if best_err < 1e-8:
                                if is_valid(best_U, best_V, best_W, target_rank) and check_half_integer(best_U, best_V, best_W):
                                    found = True
                                    break
                    else:
                        V[idx, c] = old_val
                else:
                    idx = random.randint(0, pn-1)
                    c = random.randint(0, target_rank-1)
                    old_val = W[idx, c]
                    new_val = random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                    if new_val == old_val:
                        continue
                    W[idx, c] = new_val
                    new_err = compute_error(U, V, W, target_rank)
                    delta = new_err - current_err
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            if best_err < 1e-8:
                                if is_valid(best_U, best_V, best_W, target_rank) and check_half_integer(best_U, best_V, best_W):
                                    found = True
                                    break
                    else:
                        W[idx, c] = old_val
                
                temperature *= cooling_rate
            
            if found:
                best_result = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
