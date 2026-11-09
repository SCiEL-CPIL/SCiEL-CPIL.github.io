"""
Matrix multiplication 2x3x10 - rank-minimal tensor decomposition search.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 10]
_START_TIME = time.time()
DEADLINE = _START_TIME + 145  # 150s wall clock, keep 5s margin

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W, best_rank = _schoolbook(n, m, p)
    
    # Try iterative alternating solve for lower ranks
    # Work downwards from schoolbook-1 to find any valid decomposition
    min_rank = max(n*m, m*p, p*n)  # 30
    
    # Try ranks from just below schoolbook down
    # Limit search to avoid timeout; we have 145s total.
    # Schoolbook is 60. Let's try a few ranks below.
    # If we don't find a better one quickly, we return schoolbook.
    search_limit = 10 # Try at most 10 ranks
    for rank in range(rank_schoolbook - 1, max(min_rank, rank_schoolbook - 1 - search_limit), -1):
        if time.time() > DEADLINE - 10: # Leave 10s for final return/verification
            break
        
        U, V, W = _alternating_solve(n, m, p, rank)
        if U is not None:
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
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
    return U, V, W, rank

def _round_to_half_int(x):
    return np.round(x * 2) / 2.0

def _verify_exact_half_int(n, m, p, U, V, W):
    # Convert to 2x integers to check exactness
    U_int = (2 * U).astype(np.int64)
    V_int = (2 * V).astype(np.int64)
    W_int = (2 * W).astype(np.int64)
    
    # Check if rounding was exact (i.e., original was close enough to half-int)
    if not np.allclose(U, U_int / 2.0, atol=1e-9): return False
    if not np.allclose(V, V_int / 2.0, atol=1e-9): return False
    if not np.allclose(W, W_int / 2.0, atol=1e-9): return False
    
    rank = U_int.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :])
                if s != 8:
                    return False
    return True

def _alternating_solve(n, m, p, rank, max_iters=500, seed=42):
    """
    Alternating solve with rounding and exact verification.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5: # Leave margin for verification and return
            return None, None, None
        
        # Solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w_row, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = w_row
                except:
                    return None, None, None
        W = W_new
        
        # Solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u_row, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = u_row
                except:
                    return None, None, None
        U = U_new
        
        # Solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v_row, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = v_row
                except:
                    return None, None, None
        V = V_new
        
        # Check convergence: round and verify exactly
        if it % 10 == 0:
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            if _verify_exact_half_int(n, m, p, U_r, V_r, W_r):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Final check
    U_r = _round_to_half_int(U)
    V_r = _round_to_half_int(V)
    W_r = _round_to_half_int(W)
    if _verify_exact_half_int(n, m, p, U_r, V_r, W_r):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None, None, None

def _verify_decomposition_np(n, m, p, U, V, W, tol=1e-8):
    """Verify decomposition using numpy for speed."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
