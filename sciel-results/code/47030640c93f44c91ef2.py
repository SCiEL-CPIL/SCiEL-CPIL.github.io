"""Matrix multiplication 3x4x5 rank-minimal decomposition search."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N

# Deadline guard: evaluator kills after 150s; keep 10s safety margin.
DEADLINE = 140.0

def _time_left():
    return DEADLINE - time.time()

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] = 1 if all dims valid (trivially all 1).
    Actually T is a 3-way tensor of size n*m*p where T[i,j,k]=1 for all i,j,k in ranges.
    But the decomposition constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    # The target is just 1 for every (i,j,k)
    return None  # We just need sum_c U_ic * V_jc * W_kc = 1 for all (i,j,k)

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _als_step(U, V, W, n, m, p, rank):
    """
    Perform one step of Alternating Least Squares.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Returns updated U, V, W.
    """
    # Reshape for easier indexing
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # 1. Solve for W given U, V
    # For each (i, k), solve A * x = 1 where A[j, c] = U[i, j, c] * V[j, k, c]
    W_new_r = np.zeros((p, n, rank))
    for i in range(n):
        for k in range(p):
            # A is (m, rank)
            # A[j, c] = U_r[i, j, c] * V_r[j, k, c]
            A = U_r[i, :, :] * V_r[:, k, :] # Broadcasting: (m, rank) * (m, rank) -> (m, rank)
            # Solve A * x = 1
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W_new_r[k, i, :] = x
            except:
                W_new_r[k, i, :] = W_r[k, i, :]
    W_new = W_new_r.reshape(p * n, rank)
    
    # 2. Solve for V given U, W_new
    # For each (j, k), solve A * x = 1 where A[i, c] = U[i, j, c] * W[k, i, c]
    V_new_r = np.zeros((m, p, rank))
    for j in range(m):
        for k in range(p):
            # A is (n, rank)
            # A[i, c] = U_r[i, j, c] * W_new_r[k, i, c]
            A = U_r[:, j, :] * W_new_r[k, :, :] # Broadcasting: (n, rank) * (n, rank) -> (n, rank)
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V_new_r[j, k, :] = x
            except:
                V_new_r[j, k, :] = V_r[j, k, :]
    V_new = V_new_r.reshape(m * p, rank)
    
    # 3. Solve for U given V_new, W_new
    # For each (i, j), solve A * x = 1 where A[k, c] = V[j, k, c] * W[k, i, c]
    U_new_r = np.zeros((n, m, rank))
    for i in range(n):
        for j in range(m):
            # A is (p, rank)
            # A[k, c] = V_new_r[j, k, c] * W_new_r[k, i, c]
            A = V_new_r[j, :, :] * W_new_r[:, i, :] # Broadcasting: (p, rank) * (p, rank) -> (p, rank)
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new_r[i, j, :] = x
            except:
                U_new_r[i, j, :] = U_r[i, j, :]
    U_new = U_new_r.reshape(n * m, rank)
    
    return U_new, V_new, W_new

def _try_rank(rank, n, m, p, deadline):
    """
    Try to find a decomposition of given rank using random initialization + ALS.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Number of attempts based on time left
    time_per_attempt = 0.5 # seconds
    n_tries = max(5, int(_time_left() / time_per_attempt))
    
    for attempt in range(n_tries):
        if _time_left() < 2:
            break
        
        # Random initialization
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # ALS iterations
        for iteration in range(100):
            if _time_left() < 2:
                break
            
            U_old = U.copy()
            V_old = V.copy()
            W_old = W.copy()
            
            U, V, W = _als_step(U, V, W, n, m, p, rank)
            
            # Check convergence
            diff = np.linalg.norm(U - U_old) + np.linalg.norm(V - V_old) + np.linalg.norm(W - W_old)
            if diff < 1e-6:
                break
        
        # Snap to half-integers?
        # The problem statement says "integer/half-integer entries".
        # If the solution is close to such values, snapping might help verification.
        # Let's try snapping and verifying.
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        # Also verify without snapping (in case the true solution is not half-integer, 
        # though the prompt implies it is)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _try_structural_rank(rank, n, m, p):
    """
    Try structural decompositions.
    """
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from schoolbook rank and try to improve
    # Schoolbook rank = n*m*p = 60
    best_result = None
    best_rank = n * m * p
    
    # Generate schoolbook solution as fallback
    U_fb = [[0] * best_rank for _ in range(n * m)]
    V_fb = [[0] * best_rank for _ in range(m * p)]
    W_fb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    
    # We might not find the absolute minimum, but let's try up to 50 or so
    # to ensure we find something better than 60 if possible.
    max_search_rank = 55
    
    for rank in range(lower_bound, min(best_rank, max_search_rank) + 1):
        if _time_left() < 5:
            break
        
        result = _try_rank(rank, n, m, p, DEADLINE)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found the minimum rank in this range, stop
    
    return best_result
