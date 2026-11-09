"""Rank-minimal tensor decomposition for 3x4 x 4x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _check_decomp(U, V, W, tol=1e-9):
    """Check if U*V*W (outer product sum) equals all-ones tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Compute tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # This is a contraction over c: einsum('ijc,jkc,kic->ijk')
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=tol)

def _try_rank_r_search(r, max_iters=100000, seed=42):
    """Try to find a rank-r decomposition using structured initialization + local search."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Candidate values for half-integers
    vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    for attempt in range(10):
        # Initialize with random half-integer matrices
        U = np.random.choice(vals, size=(n*m, r)).tolist()
        V = np.random.choice(vals, size=(m*p, r)).tolist()
        W = np.random.choice(vals, size=(p*n, r)).tolist()
        
        Ur = np.array(U, dtype=float).reshape(n, m, r)
        Vr = np.array(V, dtype=float).reshape(m, p, r)
        Wr = np.array(W, dtype=float).reshape(p, n, r)
        
        def compute_loss():
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            return np.sum((T - 1.0)**2)
        
        loss = compute_loss()
        if loss < best_loss:
            best_loss = loss
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        
        # Local search
        improved = True
        iters = 0
        # We want to spend about 5 seconds per attempt if 10 attempts
        # 100000 iters / 10 attempts = 10000 iters per attempt
        # Each iter does 100 random moves? Too slow.
        # Let's do 1 move per iter, but more iters.
        while improved and iters < 5000:
            improved = False
            iters += 1
            # Pick a random variable to change
            mat_idx = rng.choice([0, 1, 2])
            if mat_idx == 0:
                ri = rng.randint(0, n*m - 1)
                ci = rng.randint(0, r - 1)
                old_val = U[ri][ci]
                # Try neighbors
                for new_val in [old_val - 0.5, old_val + 0.5]:
                    if new_val < -4.0 or new_val > 4.0: continue
                    U[ri][ci] = new_val
                    Ur[ri // m][ri % m][ci] = new_val
                    new_loss = compute_loss()
                    if new_loss < loss - 1e-12:
                        loss = new_loss
                        improved = True
                        if loss < best_loss:
                            best_loss = loss
                            best_U = [row[:] for row in U]
                            best_V = [row[:] for row in V]
                            best_W = [row[:] for row in W]
                        break
                    else:
                        U[ri][ci] = old_val
                        Ur[ri // m][ri % m][ci] = old_val
            elif mat_idx == 1:
                ri = rng.randint(0, m*p - 1)
                ci = rng.randint(0, r - 1)
                old_val = V[ri][ci]
                for new_val in [old_val - 0.5, old_val + 0.5]:
                    if new_val < -4.0 or new_val > 4.0: continue
                    V[ri][ci] = new_val
                    Vr[ri // p][ri % p][ci] = new_val
                    new_loss = compute_loss()
                    if new_loss < loss - 1e-12:
                        loss = new_loss
                        improved = True
                        if loss < best_loss:
                            best_loss = loss
                            best_U = [row[:] for row in U]
                            best_V = [row[:] for row in V]
                            best_W = [row[:] for row in W]
                        break
                    else:
                        V[ri][ci] = old_val
                        Vr[ri // p][ri % p][ci] = old_val
            else:
                ri = rng.randint(0, p*n - 1)
                ci = rng.randint(0, r - 1)
                old_val = W[ri][ci]
                for new_val in [old_val - 0.5, old_val + 0.5]:
                    if new_val < -4.0 or new_val > 4.0: continue
                    W[ri][ci] = new_val
                    Wr[ri // n][ri % n][ci] = new_val
                    new_loss = compute_loss()
                    if new_loss < loss - 1e-12:
                        loss = new_loss
                        improved = True
                        if loss < best_loss:
                            best_loss = loss
                            best_U = [row[:] for row in U]
                            best_V = [row[:] for row in V]
                            best_W = [row[:] for row in W]
                        break
                    else:
                        W[ri][ci] = old_val
                        Wr[ri // n][ri % n][ci] = old_val
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W
    
    if best_loss < 1e-6:
        # Try rounding if it's close
        Ur = np.round(np.array(best_U) * 2) / 2.0
        Vr = np.round(np.array(best_V) * 2) / 2.0
        Wr = np.round(np.array(best_W) * 2) / 2.0
        if _check_decomp(Ur.tolist(), Vr.tolist(), Wr.tolist()):
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
        if _check_decomp(best_U, best_V, best_W):
            return best_U, best_V, best_W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Try ranks in increasing order of difficulty (lower rank is harder)
    # Rank 36 is the target.
    ranks_to_try = [36, 38, 40, 42, 44, 46]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        try:
            result = _try_rank_r_search(r, max_iters=50000, seed=42+r)
            if result is not None:
                U, V, W = result
                if _check_decomp(U, V, W):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Fallback to schoolbook
    return _schoolbook()
