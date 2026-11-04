"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    return {
        "U": [[int(x) if abs(x - round(x)) < 1e-9 else (int(2*x)/2) for x in row] for row in U.tolist()],
        "V": [[int(x) if abs(x - round(x)) < 1e-9 else (int(2*x)/2) for x in row] for row in V.tolist()],
        "W": [[int(x) if abs(x - round(x)) < 1e-9 else (int(2*x)/2) for x in row] for row in W.tolist()],
    }

def _verify(U, V, W, n, m, p, tol=1e-8):
    # Compute C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and compare to identity-like tensor: C[i,k] = delta_{i,k} for i<k<n? 
    # Actually matrix product: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = delta (product tensor)
    # Our decomposition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # The product tensor has T[i,j,k] = 1 always? No.
    # The matrix multiplication tensor is: for fixed i,j,k, the coefficient is 1.
    # So sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should equal 1 for all i,j,k.
    # Let me verify: in schoolbook, for each (i,j,k) there is exactly one column c where
    # U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1, so the sum is 1. Yes.
    rank = U.shape[1]
    # Build tensors
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.max(np.abs(T - 1.0)) < tol

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # Start with schoolbook
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = _to_lists(U0, V0, W0)
    
    # Try to find lower rank decompositions using randomized search
    # We search for rank R where sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k
    # This is a system of n*m*p = 36 polynomial equations.
    
    # Use a structured search: try ranks from 32 down to some minimum
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with schoolbook pattern plus random perturbation
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # First, copy the schoolbook pattern for the first target_rank columns
            if target_rank <= n * m * p:
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
            
            # Add random small perturbations to encourage exploration
            if target_rank < n * m * p:
                # For the remaining structure, try different patterns
                # Add random half-integer entries
                for r in range(target_rank):
                    for idx in range(n * m):
                        if U[idx, r] == 0:
                            if np.random.random() < 0.3:
                                U[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    for idx in range(m * p):
                        if V[idx, r] == 0:
                            if np.random.random() < 0.3:
                                V[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    for idx in range(p * n):
                        if W[idx, r] == 0:
                            if np.random.random() < 0.3:
                                W[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # Try to refine using iterative adjustment
            # Simple local search: adjust entries to reduce error
            for iteration in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Compute error
                T = np.zeros((n, m, p))
                for r in range(target_rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                
                err = T - 1.0
                max_err = np.max(np.abs(err))
                
                if max_err < 1e-8:
                    # Success!
                    best_rank = target_rank
                    best_result = _to_lists(U, V, W)
                    return best_result
                
                # Find the worst (i,j,k) and try to adjust
                worst_idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
                wi, wj, wk = worst_idx
                werr = err[wi, wj, wk]
                
                # Try adjusting entries in U[wi*m+wj, :], V[wj*p+wk, :], W[wk*n+wi, :]
                # Pick a random column and adjust
                r = np.random.randint(target_rank)
                sign = 1.0 if werr > 0 else -1.0
                # Adjust U
                U[wi * m + wj, r] += sign * 0.1 * np.random.random()
                # Snap to half-integers
                U[wi * m + wj, r] = np.round(U[wi * m + wj, r] * 2) / 2
                
                # Occasionally try other columns
                if np.random.random() < 0.5:
                    r2 = np.random.randint(target_rank)
                    V[wj * p + wk, r2] += sign * 0.1 * np.random.random()
                    V[wj * p + wk, r2] = np.round(V[wj * p + wk, r2] * 2) / 2
                if np.random.random() < 0.5:
                    r3 = np.random.randint(target_rank)
                    W[wk * n + wi, r3] += sign * 0.1 * np.random.random()
                    W[wk * n + wi, r3] = np.round(W[wk * n + wi, r3] * 2) / 2
            
            # After local search, check if we improved
            T = np.zeros((n, m, p))
            for r in range(target_rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            
            if np.max(np.abs(T - 1.0)) < 1e-8:
                best_rank = target_rank
                best_result = _to_lists(U, V, W)
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
