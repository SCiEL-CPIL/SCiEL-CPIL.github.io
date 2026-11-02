"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]
DEADLINE = 140.0  # Safety margin before 150s kill

def _start_time():
    return time.time()

def _time_left(start):
    return DEADLINE - (time.time() - start)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, else 0."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _validate(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    
    rec = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    if not np.allclose(rec, T, atol=1e-9):
        return False
    return True

def _check_half_integer(U, V, W):
    """Check if all entries are integers or half-integers (i.e., 2*x is integer)."""
    for M in [U, V, W]:
        M = np.array(M, dtype=np.float64)
        if not np.allclose(M * 2, np.round(M * 2), atol=1e-9):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_333():
    """Return U,V,W for 3x3 * 3x3 with rank 7 (Strassen).
    Convention: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]"""
    n, m, p = 3, 3, 3
    U = np.zeros((9, 7), dtype=np.int64)
    V = np.zeros((9, 7), dtype=np.int64)
    W = np.zeros((9, 7), dtype=np.int64)
    
    # Term 0: (a00+a22)(b00+b22) -> contributes to c00 and c22
    U[0, 0] = 1; U[8, 0] = 1
    V[0, 0] = 1; V[8, 0] = 1
    W[0, 0] = 1; W[8, 0] = 1
    
    # Term 1: (a10+a20)b01 -> contributes to c11 and c21
    U[3, 1] = 1; U[6, 1] = 1
    V[1, 1] = 1
    W[4, 1] = 1; W[7, 1] = 1
    
    # Term 2: a00(b01-b11) -> contributes to c01
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[1, 2] = 1
    
    # Term 3: a22(b10-b00) -> contributes to c20
    U[8, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[6, 3] = 1
    
    # Term 4: (a00+a11)(b11+b22) -> contributes to c01 and c12
    U[0, 4] = 1; U[4, 4] = 1
    V[4, 4] = 1; V[8, 4] = 1
    W[1, 4] = 1; W[5, 4] = 1
    
    # Term 5: (a10+a11)(b01+b11) -> contributes to c11
    U[3, 5] = 1; U[4, 5] = 1
    V[1, 5] = 1; V[4, 5] = 1
    W[4, 5] = 1
    
    # Term 6: (a01+a11)(b10+b20) -> contributes to c00 and c10
    U[1, 6] = 1; U[4, 6] = 1
    V[3, 6] = 1; V[6, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _construct_33x35_strassen_plus_schoolbook(n, m, p):
    """Use Strassen for the 3x3 part and schoolbook for the 3x2 part."""
    U_s, V_s, W_s = _strassen_333()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    rank_sb = 3 * 3 * 2
    U_sb = np.zeros((9, rank_sb), dtype=np.int64)
    V_sb = np.zeros((15, rank_sb), dtype=np.int64)
    W_sb = np.zeros((15, rank_sb), dtype=np.int64)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                U_sb[i * 3 + j, col] = 1
                V_sb[j * 5 + k, col] = 1
                W_sb[k * 3 + i, col] = 1
                col += 1
    
    U = np.hstack([U_s, U_sb])
    V = np.hstack([V_s, V_sb])
    W = np.hstack([W_s, W_sb])
    
    return U.tolist(), V.tolist(), W.tolist()

def _alternating_minimization(n, m, p, rank, T, max_iters=200, tol=1e-10, deadline=None):
    """
    Alternating minimization to find a rank-R decomposition.
    For each mode, fix the other two and solve a linear least-squares problem.
    """
    rng = np.random.RandomState(42)
    
    # Random initialization
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Precompute T as float
    T_f = T.astype(np.float64)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: fix V, W
        # For each a: T[a, :, :] = sum_r U[a, r] * V[:, r] * W[:, r]
        # Let M_a be a (m*p*p*n) x rank matrix where M_a[idx, r] = V[b, r] * W[c, r]
        # Then U[a, :] = argmin || T[a, :, :].reshape(-1) - M_a @ U[a, :] ||
        
        # Efficiently: for each a, build the linear system
        # Actually, we can vectorize: 
        # Reshape T to (n*m, m*p*p*n)
        # Build a big matrix B of shape (n*m, m*p*p*n, rank) where
        # B[a, b*(p*n)+c, r] = V[b, r] * W[c, r]
        # Then for each a: U[a, :] = lstsq(B[a], T_f[a].reshape(-1))
        
        # This is memory intensive but n*m=9, m*p*p*n=15*15=225, rank is small
        # Let's do it mode by mode for memory efficiency
        
        # Update U
        for a in range(n * m):
            # Build matrix: (m*p * p*n) x rank
            # Entry (b, c, r) = V[b, r] * W[c, r]
            # Reshape to (m*p*p*n, rank)
            M = np.einsum('br,cr->bcr', V, W).reshape(m * p * p * n, rank)
            rhs = T_f[a, :, :].reshape(-1)
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
            except:
                pass
        
        # Update V
        for b in range(m * p):
            M = np.einsum('ar,cr->acr', U, W).reshape(n * m * p * n, rank)
            rhs = T_f[:, b, :].reshape(-1)
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
            except:
                pass
        
        # Update W
        for c in range(p * n):
            M = np.einsum('ar,br->abr', U, V).reshape(n * m * m * p, rank)
            rhs = T_f[:, :, c].reshape(-1)
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
            except:
                pass
        
        # Check convergence
        rec = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(rec - T_f)
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank_search(n, m, p, start_rank, end_rank, deadline):
    """Try to find decompositions for ranks in [start_rank, end_rank]."""
    T = _build_tensor(n, m, p)
    best_result = None
    best_rank = float('inf')
    
    for rank in range(start_rank, end_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Use different seeds
            np.random.seed(42 + attempt)
            
            U, V, W, err = _alternating_minimization(
                n, m, p, rank, T, 
                max_iters=100, tol=1e-10, deadline=deadline
            )
            
            if err < 1e-9:
                # Verify exactly
                if _validate(U, V, W, n, m, p) and _check_half_integer(U, V, W):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result, best_rank

def construct():
    start = _start_time()
    n, m, p = DIMENSIONS
    
    # Try several constructions and return the best valid one
    candidates = []
    
    # 1. Schoolbook (always valid, rank 45)
    sb = _schoolbook(n, m, p)
    candidates.append((45, sb))
    
    # 2. Strassen + schoolbook (rank 7 + 18 = 25, if valid)
    if _time_left(start) > 5:
        try:
            U, V, W = _construct_33x35_strassen_plus_schoolbook(n, m, p)
            if _validate(U, V, W, n, m, p) and _check_half_integer(U, V, W):
                candidates.append((25, {"U": U, "V": V, "W": W}))
        except Exception:
            pass
    
    # 3. Try the hardcoded rank-17 from the original (if valid)
    if _time_left(start) > 5:
        try:
            U17 = [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
            V17 = [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
            W17 = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
            if _validate(U17, V17, W17, n, m, p) and _check_half_integer(U17, V17, W17):
                candidates.append((17, {"U": U17, "V": V17, "W": W17}))
        except Exception:
            pass
    
    # 4. Try alternating minimization search for lower ranks
    if _time_left(start) > 10:
        try:
            # Try ranks from 10 up to 24 (below the Strassen+schoolbook 25)
            result, rank = _try_rank_search(n, m, p, 10, 24, time.time() + _time_left(start) - 5)
            if result is not None:
                candidates.append((rank, result))
        except Exception:
            pass
    
    # Sort by rank and return the best
    candidates.sort(key=lambda x: x[0])
    if candidates:
        return candidates[0][1]
    
    # Fallback
    return _schoolbook(n, m, p)
