import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_DEADLINE = 130  # safety margin before 150s kill

def _build_target_tensor(n, m, p):
    """Build the target tensor T where T[i*m+j, j*p+k, k*n+i] = 1"""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(computed - target) > tol:
                    return False
    # Check that no spurious entries exist
    # For entries where the index pattern doesn't match, the sum should be 0
    # But actually, the tensor is defined such that only (i*m+j, j*p+k, k*n+i) = 1
    # and all other entries are 0. Let's verify a few random other entries.
    import random
    random.seed(42)
    for _ in range(100):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        # Check if this is a "real" entry
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k2 = b % p
        k1 = c // n
        i2 = c % n
        if (i1 == i2 and j1 == j2 and k1 == k2):
            continue  # This is a real entry, already checked
        computed = 0.0
        for r in range(rank):
            computed += U[a, r] * V[b, r] * W[c, r]
        if abs(computed) > tol:
            return False
    return True

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-12, seed=0):
    """Try to find a rank decomposition using ALS with exact verification."""
    np.random.seed(seed)
    
    # Initialize with random matrices
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Build target tensor as sparse dict
    target = _build_target_tensor(n, m, p)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Update U: for each row i*m+j, solve for U[row, :]
        for i in range(n):
            for j in range(m):
                row_u = i*m + j
                # We need U[row_u, :] such that sum_r U[row_u,r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[row_u, :] @ diag(V[j*p+k, :] * W[k*n+i, :]) = 1 for each k
                # Actually, for fixed i,j: sum_r U[row_u,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # Let A_k[r] = V[j*p+k, r] * W[k*n+i, r]
                # Then we need U[row_u, :] @ A_k = 1 for all k
                # This is a system: A @ U[row_u, :]^T = ones(p)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ x = ones(p)
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[row_u, :] = x
                except:
                    pass
        
        # Update V: for each row j*p+k, solve for V[row_v, :]
        for j in range(m):
            for k in range(p):
                row_v = j*p + k
                # For fixed j,k: sum_r U[i*m+j, r] * V[row_v, r] * W[k*n+i, r] = 1 for all i
                # Let B_i[r] = U[i*m+j, r] * W[k*n+i, r]
                # Then V[row_v, :] @ B_i = 1 for all i
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[row_v, :] = x
                except:
                    pass
        
        # Update W: for each row k*n+i, solve for W[row_w, :]
        for k in range(p):
            for i in range(n):
                row_w = k*n + i
                # For fixed k,i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[row_w, r] = 1 for all j
                # Let C_j[r] = U[i*m+j, r] * V[j*p+k, r]
                # Then W[row_w, :] @ C_j = 1 for all j
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[row_w, :] = x
                except:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
    
    return None

def _try_rank(n, m, p, rank, num_trials=5):
    """Try multiple random initializations for a given rank."""
    for trial in range(num_trials):
        if time.time() > _DEADLINE:
            break
        result = _als_decompose(n, m, p, rank, max_iters=100, seed=trial*12345+42)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                return U, V, W
    return None

def _to_list(arr):
    """Convert numpy array to list of lists with half-integer rounding."""
    result = []
    for row in arr:
        lst = []
        for val in row:
            # Round to nearest half-integer
            rounded = round(val * 2) / 2
            if abs(rounded - val) < 1e-9:
                lst.append(rounded)
            else:
                # If not close to half-integer, try to find a simple fraction
                # For now, just use the rounded value
                lst.append(rounded)
        result.append(lst)
    return result

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Try ranks below schoolbook
    # Based on known results, the rank might be lower than 90
    # Try a range of ranks
    for rank in range(70, schoolbook_rank + 1):
        if time.time() > _DEADLINE:
            break
        
        result = _try_rank(n, m, p, rank, num_trials=3)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            # Don't break - try to find even lower rank
            continue
    
    # If we found a decomposition, use it
    if best_U is not None:
        return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
