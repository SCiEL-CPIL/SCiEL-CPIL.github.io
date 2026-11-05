"""Matrix multiplication 3x4x5 rank-minimal decomposition search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N

# Deadline guard: evaluator kills after 150s; keep 10s safety margin.
DEADLINE = 140.0

def _time_left():
    return DEADLINE - time.time()

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
    W_new_r = np.zeros((p, n, rank))
    for i in range(n):
        for k in range(p):
            A = U_r[i, :, :] * V_r[:, k, :]
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W_new_r[k, i, :] = x
            except:
                W_new_r[k, i, :] = W_r[k, i, :]
    W_new = W_new_r.reshape(p * n, rank)
    
    # 2. Solve for V given U, W_new
    V_new_r = np.zeros((m, p, rank))
    for j in range(m):
        for k in range(p):
            A = U_r[:, j, :] * W_new_r[k, :, :]
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V_new_r[j, k, :] = x
            except:
                V_new_r[j, k, :] = V_r[j, k, :]
    V_new = V_new_r.reshape(m * p, rank)
    
    # 3. Solve for U given V_new, W_new
    U_new_r = np.zeros((n, m, rank))
    for i in range(n):
        for j in range(m):
            A = V_new_r[j, :, :] * W_new_r[:, i, :]
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new_r[i, j, :] = x
            except:
                U_new_r[i, j, :] = U_r[i, j, :]
    U_new = U_new_r.reshape(n * m, rank)
    
    return U_new, V_new, W_new

def _try_rank(rank, n, m, p, deadline, max_tries=50):
    """
    Try to find a decomposition of given rank using random initialization + ALS.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    time_per_attempt = 0.5
    n_tries = min(max_tries, max(5, int(_time_left() / time_per_attempt)))
    
    for attempt in range(n_tries):
        if _time_left() < 2:
            break
        
        # Use different random seeds for diversity
        np.random.seed(attempt * 12345 + rank)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Normalize to avoid numerical issues
        U = U / np.linalg.norm(U) * np.sqrt(nm)
        V = V / np.linalg.norm(V) * np.sqrt(mp)
        W = W / np.linalg.norm(W) * np.sqrt(pn)
        
        for iteration in range(150):
            if _time_left() < 2:
                break
            
            U_old = U.copy()
            V_old = V.copy()
            W_old = W.copy()
            
            U, V, W = _als_step(U, V, W, n, m, p, rank)
            
            diff = np.linalg.norm(U - U_old) + np.linalg.norm(V - V_old) + np.linalg.norm(W - W_old)
            if diff < 1e-8:
                break
        
        # Try rounding to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        # Try rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
        # Check if the unrounded solution is close enough
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _block_decomposition(n, m, p):
    """
    Build the schoolbook decomposition with rank n*m*p.
    """
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, n*m*p

def _try_block_rank(rank, n, m, p):
    """
    Try to find a decomposition using block structure.
    This is a heuristic that tries to exploit the block structure of the matrices.
    """
    # For 3x4x5, we can try various block partitions
    # The idea is that some blocks might have lower rank than their full size
    # But for exact multiplication, this is unlikely to help much
    
    # Try a simple heuristic: if rank is close to n*m*p, use schoolbook
    # Otherwise, try to find a sparse solution
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from schoolbook rank and try to improve
    # Schoolbook rank = n*m*p = 60
    best_result = None
    best_rank = n * m * p
    
    # Generate schoolbook solution as fallback
    U_fb, V_fb, W_fb, _ = _block_decomposition(n, m, p)
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try ranks from lower bound upward
    # The lower bound for matrix multiplication is max(n*m, m*p, p*n)
    # For 3x4x5: max(12, 20, 15) = 20
    # But the actual lower bound is higher; let's start from a reasonable value
    
    # Known results suggest that for small matrices, the rank is close to n*m*p
    # Let's try from 50 down to see if we can find something better than 60
    
    # First, try to find any solution better than schoolbook
    # Start from rank 55 and go down
    
    for rank in range(55, 15, -1):
        if _time_left() < 10:
            break
        
        result = _try_rank(rank, n, m, p, DEADLINE, max_tries=20)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            # Don't break; try to find even lower rank
            # But if we found a good one, we can stop after a few more tries
            if rank <= 40:
                break
    
    # If we didn't find anything better, try a few more ranks with more attempts
    if best_rank == n * m * p:
        for rank in range(55, 45, -1):
            if _time_left() < 5:
                break
            result = _try_rank(rank, n, m, p, DEADLINE, max_tries=30)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    return best_result
