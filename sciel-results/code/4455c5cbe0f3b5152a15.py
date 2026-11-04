"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication."""
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    rank = U.shape[1]
    # Efficient check: for each i, check that the implied multiplication is correct
    for i in range(n):
        # For fixed i, we need: sum_j A[i,j]*B[j,k] = C[i,k]
        # In tensor form: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j,k
        # Let's verify by checking the bilinear form
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # For each i, check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # Reshape and use einsum-like operations
    for i in range(n):
        # U_i: shape (m, rank) = U[i*m:(i+1)*m, :]
        U_i = U[i*m:(i+1)*m, :]
        # W_i: shape (p, rank) = W[i::n, :]  (since W[k*n+i, r])
        W_i = W[i::n, :]  # shape (p, rank)
        # For each j: sum_r U_i[j,r] * V[j*p+k,r] * W_i[k,r] = 1 for all k
        # Let M_j = V[j*p:(j+1)*p, :]  shape (p, rank)
        # Then we need: sum_r U_i[j,r] * M_j[k,r] * W_i[k,r] = 1 for all k
        # This is: (U_i[j,:] * M_j) dot W_i element-wise, summed... 
        # Actually: for each k: sum_r U_i[j,r] * M_j[k,r] * W_i[k,r] = 1
        # = sum_r (U_i[j,r] * M_j[k,r]) * W_i[k,r]
        # Let's compute: for each j, compute matrix D_j where D_j[k,r] = U_i[j,r]*M_j[k,r]
        # Then for each k: sum_r D_j[k,r]*W_i[k,r] = 1
        for j in range(m):
            M_j = V[j*p:(j+1)*p, :]  # (p, rank)
            # D_j[k,r] = U_i[j,r] * M_j[k,r]
            D_j = M_j * U_i[j, :]  # (p, rank), broadcasting U_i[j,:] over rows
            # Check: for each k, sum_r D_j[k,r] * W_i[k,r] = 1
            checks = D_j * W_i  # (p, rank)
            row_sums = checks.sum(axis=1)  # (p,)
            if not np.allclose(row_sums, np.ones(p), atol=tol):
                return False
    return True

def _alternate_solve(U, V, W, n, m, p, max_iter=200, tol=1e-10):
    """Alternating least squares: fix two, solve for third."""
    rank = U.shape[1]
    
    for it in range(max_iter):
        # Solve for V: for each (j,k), find v = V[j*p+k,:] 
        # such that for all i: sum_r U[i*m+j,r] * v[r] * W[k*n+i,r] = 1
        # Linear system: A * v = 1, where A[i,r] = U[i*m+j,r] * W[k*n+i,r]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = v
                except:
                    return None, None, None
        
        # Solve for W: for each (k,i), find w = W[k*n+i,:]
        # such that for all j: sum_r U[i*m+j,r] * V[j*p+k,r] * w[r] = 1
        # Linear system: B * w = 1, where B[j,r] = U[i*m+j,r] * V[j*p+k,r]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                B = np.zeros((m, rank))
                for j in range(m):
                    B[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = w
                except:
                    return None, None, None
        
        # Solve for U: for each (i,j), find u = U[i*m+j,:]
        # such that for all k: sum_r u[r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # Linear system: C * u = 1, where C[k,r] = V[j*p+k,r] * W[k*n+i,r]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                C = np.zeros((p, rank))
                for k in range(p):
                    C[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u, _, _, _ = np.linalg.lstsq(C, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = u
                except:
                    return None, None, None
        
        V = V_new
        W = W_new
        U = U_new
    
    # Final verification
    if _verify_fast(U, V, W, n, m, p, tol=1e-7):
        return U, V, W
    return None, None, None

def _try_structured_decomposition(n, m, p, target_rank, seed=0):
    """Try to find a decomposition with target_rank using structured initialization."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n*m, target_rank)
    V = rng.randn(m*p, target_rank)
    W = rng.randn(p*n, target_rank)
    
    # Normalize
    U /= np.sqrt(target_rank)
    V /= np.sqrt(target_rank)
    W /= np.sqrt(target_rank)
    
    U, V, W = _alternate_solve(U, V, W, n, m, p, max_iter=300, tol=1e-10)
    
    if U is not None:
        return U, V, W
    return None, None, None

def _try_block_structure(n, m, p, target_rank, seed=0):
    """Try a block-structured initialization based on 3x3 blocks."""
    rng = np.random.RandomState(seed)
    
    # For 3x3x8, we can think of it as 8 blocks of 3x3
    # Try to use a 3x3 decomposition pattern
    # Known: 3x3x3 can be done in rank 23 (Strassen-like)
    # For 3x3x8, lower bound is max(9, 24, 24) = 24
    
    # Try rank 24 first with structured init
    if target_rank < 24:
        return None, None, None
    
    # Initialize using a pattern
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Use random initialization as fallback
    U = rng.randn(n*m, target_rank)
    V = rng.randn(m*p, target_rank)
    W = rng.randn(p*n, target_rank)
    U /= np.sqrt(target_rank)
    V /= np.sqrt(target_rank)
    W /= np.sqrt(target_rank)
    
    U, V, W = _alternate_solve(U, V, W, n, m, p, max_iter=500, tol=1e-10)
    
    if U is not None:
        return U, V, W
    return None, None, None

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound upward
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            seed = 42 + target_rank * 1000 + attempt * 7
            
            U, V, W = _try_structured_decomposition(n, m, p, target_rank, seed=seed)
            
            if U is not None and V is not None and W is not None:
                # Verify once more with tight tolerance
                if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower? No, we're going up.
            # Actually we want the minimum, so if we found one at target_rank, that's our answer
            # unless we can find a lower one. But we started from lower_bound, so target_rank is minimal.
            break
    
    return best_result
# EVOLVE-BLOCK-END
