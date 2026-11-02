"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the multiplication tensor T with shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; zero otherwise."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Reconstruct tensor: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T_recon - T))
    return max_err < tol

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_random_search(n, m, p, T, target_rank, time_limit, rng):
    """Attempt to find a rank-target_rank decomposition via randomized optimization."""
    nm, mp, pn = n * m, m * p, p * n
    start = time.time()
    
    # Initialize with random factors
    U = rng.standard_normal((nm, target_rank))
    V = rng.standard_normal((mp, target_rank))
    W = rng.standard_normal((pn, target_rank))
    
    # Normalize columns to avoid scale issues
    for c in range(target_rank):
        norm_U = np.linalg.norm(U[:, c])
        norm_V = np.linalg.norm(V[:, c])
        norm_W = np.linalg.norm(W[:, c])
        if norm_U > 1e-12:
            U[:, c] /= norm_U
        if norm_V > 1e-12:
            V[:, c] /= norm_V
        if norm_W > 1e-12:
            W[:, c] /= norm_W
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Gradient-free local search: try small perturbations
    lr = 0.1
    for iteration in range(500):
        if time.time() - start > time_limit:
            break
        
        # Compute current error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-9:
            break
        
        # Compute gradients
        # dL/dU[a,r] = sum_{b,c} (T_recon[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        diff = T_recon - T
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
        
        # Take a step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Re-normalize
        for c in range(target_rank):
            norm_U = np.linalg.norm(U[:, c])
            norm_V = np.linalg.norm(V[:, c])
            norm_W = np.linalg.norm(W[:, c])
            if norm_U > 1e-12:
                U[:, c] /= norm_U
            if norm_V > 1e-12:
                V[:, c] /= norm_V
            if norm_W > 1e-12:
                W[:, c] /= norm_W
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build the target tensor
    T = _build_tensor(n, m, p)
    
    # Schoolbook baseline
    schoolbook = _schoolbook(n, m, p)
    best_rank = n * m * p  # 96
    best_result = schoolbook
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But for matrix multiplication, the true rank is often higher.
    # Known: 2x2x2 has rank 7 (vs schoolbook 8)
    # Let's try ranks from 95 down, but focus on promising ranges
    
    rng = np.random.default_rng(42)
    
    # Try a few candidate ranks
    # For 3x4x8, we expect rank to be somewhat below 96
    # Try ranks: 90, 85, 80, 75, 70, 65, 60, 55, 50, 48
    candidate_ranks = [90, 85, 80, 75, 70, 65, 60, 55, 50, 48, 45, 40]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Allocate time per rank attempt
        time_per_rank = min(15, (deadline - time.time()) / max(1, len(candidate_ranks)))
        
        result = _try_random_search(n, m, p, T, target_rank, time_per_rank, rng)
        
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Convert to lists with integer/half-integer entries if possible
            # Check if entries are close to integers or half-integers
            U_list = [[round(val) if abs(val - round(val)) < 1e-8 else 
                       round(val * 2) / 2 if abs(val * 2 - round(val * 2)) < 1e-8 else val
                       for val in row] for row in U_arr]
            V_list = [[round(val) if abs(val - round(val)) < 1e-8 else 
                       round(val * 2) / 2 if abs(val * 2 - round(val * 2)) < 1e-8 else val
                       for val in row] for row in V_arr]
            W_list = [[round(val) if abs(val - round(val)) < 1e-8 else 
                       round(val * 2) / 2 if abs(val * 2 - round(val * 2)) < 1e-8 else val
                       for val in row] for row in W_arr]
            
            # Verify the rounded version
            if _verify_decomposition(U_list, V_list, W_list, T, tol=1e-7):
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                print(f"Found rank {target_rank} decomposition!")
            else:
                # Use the unrounded version if it's valid
                if _verify_decomposition(U_arr, V_arr, W_arr, T, tol=1e-9):
                    # Convert to float lists
                    U_list = [[float(val) for val in row] for row in U_arr]
                    V_list = [[float(val) for val in row] for row in V_arr]
                    W_list = [[float(val) for val in row] for row in W_arr]
                    best_rank = target_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    print(f"Found rank {target_rank} decomposition (float entries)!")
    
    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
