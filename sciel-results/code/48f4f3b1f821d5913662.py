import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS
SCHOOLBOOK_RANK = n * m * p

def _schoolbook():
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

def _verify(U, V, W):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = len(W[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_mat = U[:, c].reshape((n, m))
        v_mat = V[:, c].reshape((m, p))
        w_mat = W[:, c].reshape((p, n))
        
        # Vectorized update for T
        # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # For each i, j:
        #   T[i,j,:] += u_mat[i,j] * (v_mat[j,:] * w_mat[:,i])
        for i in range(n):
            for j in range(m):
                u_val = u_mat[i, j]
                if u_val != 0:
                    w_vec = w_mat[:, i]
                    T[i, j, :] += u_val * (v_mat[j, :] * w_vec)
                    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _try_rank(r, time_limit=10, seed=None):
    """
    Try to find a decomposition of rank r using alternating least squares with structured initialization.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    max_attempts = 100
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize U, V, W
        # We use a mix of random and structured initializations
        # Structured: based on subsets of indices
        if attempt < 10:
            # Try to initialize close to schoolbook but with merging
            # This is hard to do directly, so we just use random but with specific scaling
            U = rng.standard_normal((n * m, r))
            V = rng.standard_normal((m * p, r))
            W = rng.standard_normal((p * n, r))
        else:
            # Pure random
            U = rng.standard_normal((n * m, r))
            V = rng.standard_normal((m * p, r))
            W = rng.standard_normal((p * n, r))
            
        # Normalize to prevent overflow/underflow
        for c in range(r):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            total_norm = norm_u * norm_v * norm_w
            if total_norm > 1e-9:
                scale = (1.0 / total_norm) ** (1.0/3.0)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
            else:
                # If norm is 0, re-initialize
                U[:, c] = rng.standard_normal(n * m)
                V[:, c] = rng.standard_normal(m * p)
                W[:, c] = rng.standard_normal(p * n)
        
        # Alternating Least Squares
        for it in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # 1. Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        v_row = V[j*p + np.arange(p), :]
                        w_col = W[np.arange(p)*n + i, :]
                        A[k, :] = v_row[k, :] * w_col[k, :]
                    
                    b = np.ones(p)
                    U_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m + j, :] = U_vec
            
            # 2. Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        u_row = U[i*m + j, :]
                        w_row = W[k*n + i, :]
                        A[i, :] = u_row * w_row
                    b = np.ones(n)
                    V_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p + k, :] = V_vec
            
            # 3. Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        u_row = U[i*m + j, :]
                        v_row = V[j*p + k, :]
                        A[j, :] = u_row * v_row
                    b = np.ones(m)
                    W_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n + i, :] = W_vec
            
            # Check residual
            T = np.zeros((n, m, p))
            for c in range(r):
                u_mat = U[:, c].reshape((n, m))
                v_mat = V[:, c].reshape((m, p))
                w_mat = W[:, c].reshape((p, n))
                for i in range(n):
                    for j in range(m):
                        u_val = u_mat[i, j]
                        if u_val != 0:
                            w_vec = w_mat[:, i]
                            T[i, j, :] += u_val * (v_mat[j, :] * w_vec)
            
            residual = np.sum((T - 1.0)**2)
            if residual < 1e-8:
                # Try to snap to integers/half-integers
                # We can try snapping to integers first, then half-integers, then quarters?
                # The problem allows half-integers.
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
                # Also try pure integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    return None

def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s margin for 150s limit
    
    # The theoretical lower bound for rank of 3x4x5 is not trivially known to be < 60.
    # Strassen's bound is 2^(log2(n)+log2(m)+log2(p)) which is not integer.
    # For 2x2x2 it's 7.
    # For 3x3x3 it's 23 (Landsberg).
    # For 3x4x5, we just try to find something < 60.
    
    # We will search for ranks from 59 downwards.
    # We allocate time budget carefully.
    
    # First, try a few high ranks quickly to see if we can get a quick win.
    # Then try lower ranks.
    
    ranks_to_try = list(range(59, 40, -1)) # 59 down to 41
    
    # If we have time, try lower
    if time.time() - start_time < 60:
        ranks_to_try += list(range(40, 30, -1))
        
    for r in ranks_to_try:
        if time.time() - start_time > DEADLINE:
            break
        
        # Time budget per rank
        # Lower ranks are harder, but let's just give a fixed budget for now
        # or proportional to (60-r)?
        # Let's give 2 seconds per rank on average.
        time_budget = 2.0
        
        # Try with a few different seeds
        # The _try_rank function has internal randomization via seed parameter if we pass it, 
        # but currently it uses np.random default. Let's make it accept seed or just rely on global state.
        # Actually, _try_rank uses np.random.randn which uses global state.
        # Let's just call it.
        
        result = _try_rank(r, time_limit=time_budget)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start}s")
    print(f"Rank: {len(res['U'][0])}")
    if not _verify(res['U'], res['V'], res['W']):
        print("Verification Failed!")
    else:
        print("Verification Passed")
