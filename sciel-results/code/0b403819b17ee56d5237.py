"""Optimized program for 2x3 * 3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _to_half_int(x, tol=1e-9):
    """Round to half-integer (k/2) if close enough, else return None."""
    ix = int(round(2 * x))
    if abs(2 * x - ix) < tol:
        return ix
    return None

def _check_solution(U, V, W):
    """Check if U, V, W (as lists of lists) give a valid decomposition."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    s += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True

def _residual(U, V, W):
    """Compute the residual sum of squares."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                res += (s - 1.0) ** 2
    return res

def _snap_to_half_int(U, V, W, tol=1e-6):
    """Try to snap entries to half-integers and check validity."""
    U_snap = []
    for row in U:
        new_row = []
        for x in row:
            h = _to_half_int(x, tol)
            if h is not None:
                new_row.append(h // 2 if h % 2 == 0 else h / 2.0)
            else:
                return None, None, None
        U_snap.append(new_row)
    
    V_snap = []
    for row in V:
        new_row = []
        for x in row:
            h = _to_half_int(x, tol)
            if h is not None:
                new_row.append(h // 2 if h % 2 == 0 else h / 2.0)
            else:
                return None, None, None
        V_snap.append(new_row)
    
    W_snap = []
    for row in W:
        new_row = []
        for x in row:
            h = _to_half_int(x, tol)
            if h is not None:
                new_row.append(h // 2 if h % 2 == 0 else h / 2.0)
            else:
                return None, None, None
        W_snap.append(new_row)
    
    if _check_solution(U_snap, V_snap, W_snap):
        return U_snap, V_snap, W_snap
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Start with Schoolbook (Rank 42)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def optimize(U, V, W, max_iters=500, tol=1e-10):
        """Run coordinate descent to minimize residual."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        rank = len(U[0])
        prev_res = float('inf')
        stuck_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        M[k, :] = v_row * w_row
                    t = np.ones(p)
                    try:
                        u_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        U[i * m + j, :] = u_sol
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        u_row = U[i * m + j, :]
                        w_row = W[k * n + i, :]
                        M[i, :] = u_row * w_row
                    t = np.ones(n)
                    try:
                        v_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        V[j * p + k, :] = v_sol
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        M[j, :] = u_row * v_row
                    t = np.ones(m)
                    try:
                        w_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        W[k * n + i, :] = w_sol
                    except:
                        pass
                        
            res = _residual(U, V, W)
            if abs(prev_res - res) < tol:
                stuck_count += 1
                if stuck_count > 10:
                    break
            else:
                stuck_count = 0
            prev_res = res
            
        # Final polish: try to snap to half-integers
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        U_snap, V_snap, W_snap = _snap_to_half_int(U_list, V_list, W_list, tol=1e-5)
        if U_snap is not None:
            return U_snap, V_snap, W_snap, res
        return None, None, None, res

    # Try ranks from 41 down to 35 with multiple restarts and different initializations
    rng = np.random.default_rng(12345)
    
    for rank in range(41, 34, -1):
        if time.time() > deadline:
            break
            
        # Try multiple initialization strategies
        num_restarts = 3
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            if r == 0:
                # Random initialization with scaling
                U = rng.random((n * m, rank))
                V = rng.random((m * p, rank))
                W = rng.random((p * n, rank))
                scale = 2.0
                U *= scale
                V *= scale
                W *= scale
            elif r == 1:
                # Sparse random initialization
                U = (rng.random((n * m, rank)) > 0.5).astype(float) * 2.0
                V = (rng.random((m * p, rank)) > 0.5).astype(float) * 2.0
                W = (rng.random((p * n, rank)) > 0.5).astype(float) * 2.0
            else:
                # Near-schoolbook: start with schoolbook structure but with random signs
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for c in range(min(rank, 42)):
                    i = (c // 21) % 2
                    j = (c // 7) % 3
                    k = c % 7
                    U[i * m + j, c] = 1.0 if rng.random() > 0.5 else -1.0
                    V[j * p + k, c] = 1.0 if rng.random() > 0.5 else -1.0
                    W[k * n + i, c] = 1.0 if rng.random() > 0.5 else -1.0
            
            U_opt, V_opt, W_opt, res = optimize(U, V, W, max_iters=300)
            
            if U_opt is not None:
                return {"U": U_opt, "V": V_opt, "W": W_opt}
                
    # If nothing found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
