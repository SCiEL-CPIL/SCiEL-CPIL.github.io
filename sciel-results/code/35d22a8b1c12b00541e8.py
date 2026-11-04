"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _snap_numpy(arr):
    return np.round(2 * arr) / 2.0

def _verify_numpy(Ua, Va, Wa, n, m, p, atol=1e-9):
    if Ua.shape != (n*m, Ua.shape[1]) or Va.shape != (m*p, Va.shape[1]) or Wa.shape != (p*n, Wa.shape[1]):
        return False
    if Ua.shape[1] != Va.shape[1] or Ua.shape[1] != Wa.shape[1]:
        return False
    rank = Ua.shape[1]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=atol)

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    return _verify_numpy(Ua, Va, Wa, n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _alt_solve(U, V, W, n, m, p, max_iter=50, tol=1e-12, polish=False):
    """Alternating linear solve to polish a near-solution."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    rank = Ua.shape[1]
    
    actual_tol = tol if not polish else 1e-14
    actual_max_iter = max_iter if not polish else min(max_iter * 3, 200)
    
    for it in range(actual_max_iter):
        if _remaining() < 1.0:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                V_j = Va[j*p:(j+1)*p, :]
                W_i = Wa[i::n, :]
                M = V_j * W_i
                try:
                    Ua[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                U_j = Ua[j::m, :]
                W_k = Wa[k::p, :]
                M = U_j * W_k
                try:
                    Va[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_i = Ua[i*m:(i+1)*m, :]
                V_k = Va[k::p, :]
                M = U_i * V_k
                try:
                    Wa[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < actual_tol:
            break
    
    return Ua, Va, Wa

def _init_random(n, m, p, rank, rng):
    """Random initialization."""
    Ua = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    Va = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    Wa = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    return Ua, Va, Wa

def _init_structured(n, m, p, rank, rng):
    """Structured initialization based on partial schoolbook structure."""
    Ua = np.zeros((n*m, rank))
    Va = np.zeros((m*p, rank))
    Wa = np.zeros((p*n, rank))
    
    # Try to place some 1s in structured positions
    for c in range(rank):
        # Randomly choose which (i,j), (j,k), (k,i) to activate
        i = rng.randint(0, n-1)
        j = rng.randint(0, m-1)
        k = rng.randint(0, p-1)
        
        # With some probability, make it a clean 1
        if rng.random() < 0.3:
            Ua[i*m+j, c] = 1.0
            Va[j*p+k, c] = 1.0
            Wa[k*n+i, c] = 1.0
        else:
            # Random values
            Ua[i*m+j, c] = rng.uniform(-1, 1)
            Va[j*p+k, c] = rng.uniform(-1, 1)
            Wa[k*n+i, c] = rng.uniform(-1, 1)
    
    return Ua, Va, Wa

def _init_perturbed_schoolbook(n, m, p, rank, rng):
    """Perturbed schoolbook-like initialization."""
    Ua = np.zeros((n*m, rank))
    Va = np.zeros((m*p, rank))
    Wa = np.zeros((p*n, rank))
    
    # Start with some schoolbook terms
    used_cols = set()
    for c in range(min(rank, n*m*p)):
        i = c // (m*p)
        rem = c % (m*p)
        j = rem // p
        k = rem % p
        
        if c < rank:
            # Add some randomness
            noise = 0.1 if rng.random() < 0.2 else 0.0
            Ua[i*m+j, c] = 1.0 + noise * rng.uniform(-1, 1)
            Va[j*p+k, c] = 1.0 + noise * rng.uniform(-1, 1)
            Wa[k*n+i, c] = 1.0 + noise * rng.uniform(-1, 1)
            used_cols.add(c)
    
    # Fill remaining columns with random values
    for c in range(rank):
        if c not in used_cols:
            Ua[:, c] = [rng.uniform(-0.5, 0.5) for _ in range(n*m)]
            Va[:, c] = [rng.uniform(-0.5, 0.5) for _ in range(m*p)]
            Wa[:, c] = [rng.uniform(-0.5, 0.5) for _ in range(p*n)]
    
    return Ua, Va, Wa

def _try_rank(n, m, p, rank, max_seeds=20, time_per_seed=2.0):
    """Try to find a valid decomposition of given rank."""
    if _remaining() < 3.0:
        return None
    
    best_result = None
    best_error = np.inf
    
    for seed in range(max_seeds):
        if _remaining() < 3.0:
            break
        
        rng = random.Random(seed * 7919 + rank * 104729)
        
        # Choose initialization strategy
        init_type = seed % 3
        if init_type == 0:
            Ua, Va, Wa = _init_random(n, m, p, rank, rng)
        elif init_type == 1:
            Ua, Va, Wa = _init_structured(n, m, p, rank, rng)
        else:
            Ua, Va, Wa = _init_perturbed_schoolbook(n, m, p, rank, rng)
        
        # Run alternating solve
        Ua, Va, Wa = _alt_solve(Ua, Va, Wa, n, m, p, max_iter=30, tol=1e-10)
        
        # Check error
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_result = (Ua.copy(), Va.copy(), Wa.copy())
        
        if error < 1e-8:
            # Try to snap to half-integers
            Ua_s = _snap_numpy(Ua)
            Va_s = _snap_numpy(Va)
            Wa_s = _snap_numpy(Wa)
            
            if _verify_numpy(Ua_s, Va_s, Wa_s, n, m, p, atol=1e-9):
                return Ua_s, Va_s, Wa_s
            
            # Try without snapping
            if _verify_numpy(Ua, Va, Wa, n, m, p, atol=1e-9):
                return Ua, Va, Wa
    
    # If we got close but not exact, try a polish pass on the best result
    if best_result is not None and best_error < 1e-6 and _remaining() > 5.0:
        Ua, Va, Wa = best_result
        Ua, Va, Wa = _alt_solve(Ua, Va, Wa, n, m, p, max_iter=100, tol=1e-14, polish=True)
        
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < 1e-8:
            Ua_s = _snap_numpy(Ua)
            Va_s = _snap_numpy(Va)
            Wa_s = _snap_numpy(Wa)
            
            if _verify_numpy(Ua_s, Va_s, Wa_s, n, m, p, atol=1e-9):
                return Ua_s, Va_s, Wa_s
            
            if _verify_numpy(Ua, Va, Wa, n, m, p, atol=1e-9):
                return Ua, Va, Wa
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    min_rank = max(n*m, m*p, n*p)
    
    # For 3x5x6, min_rank = 30
    # Try ranks from min_rank upward with smart time budgeting
    # Focus on a few key ranks rather than exhaustive search
    
    # Strategy: try ranks 30-50 with more seeds, then 50-90 with fewer
    ranks_to_try = list(range(min_rank, min(min_rank + 21, best_rank + 1)))
    
    # Add some strategic ranks
    for r in [40, 50, 60, 70, 80]:
        if min_rank <= r < best_rank and r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try = sorted(set(ranks_to_try))
    
    for rank in ranks_to_try:
        if _remaining() < 10.0:
            break
        
        # Allocate more time to lower ranks
        if rank < 40:
            max_seeds = 20
        elif rank < 60:
            max_seeds = 15
        else:
            max_seeds = 10
        
        result = _try_rank(n, m, p, rank, max_seeds=max_seeds)
        if result is not None:
            Ua, Va, Wa = result
            if _verify_numpy(Ua, Va, Wa, n, m, p):
                best = {"U": Ua.tolist(), "V": Va.tolist(), "W": Wa.tolist()}
                best_rank = rank
                break
    
    # Final validation
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
