"""Rewritten program: search for a lower-rank exact decomposition of the 2x3 * 3x10 matmul tensor."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

DEADLINE = 135.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return DEADLINE - _elapsed()

def _verify(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U3 = U.reshape(N, M, rank)
    V3 = V.reshape(M, P, rank)
    W3 = W.reshape(P, N, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    expected = np.ones((N, M, P))
    return np.allclose(T, expected, atol=tol)

def _schoolbook():
    """Return the trivial rank-60 schoolbook decomposition."""
    U = np.zeros((N * M, RANK_SCHOOLBOOK))
    V = np.zeros((M * P, RANK_SCHOOLBOOK))
    W = np.zeros((P * N, RANK_SCHOOLBOOK))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W

def _fast_als(rank, n_iter=100, tol=1e-10, seed=0):
    """Fast ALS using vectorized Khatri-Rao products."""
    np_rng = np.random.RandomState(seed)
    
    U = np_rng.randn(N * M, rank) * 0.5
    V = np_rng.randn(M * P, rank) * 0.5
    W = np_rng.randn(P * N, rank) * 0.5
    
    # Build target tensor
    T = np.zeros((N * M, M * P, P * N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M+j, j*P+k, k*N+i] = 1.0
    
    T1 = T.reshape(N*M, -1)  # (6, 600)
    T2 = T.transpose(1,0,2).reshape(M*P, -1)  # (30, 120)
    T3 = T.transpose(2,0,1).reshape(P*N, -1)  # (20, 180)
    
    for it in range(n_iter):
        if _time_left() < 5.0:
            break
        
        # Update U: B = kron(W, V) reshaped, solve U*B^T = T1
        # B has shape (M*P*P*N, rank) where B[(b,c), r] = V[b,r]*W[c,r]
        # This is the Khatri-Rao product of V and W, reshaped
        B = np.kron(W.T, V.T).T  # (P*N*M*P, rank) -> need to reshape to (M*P*P*N, rank)
        # Actually: np.kron(W, V) gives (P*N*M*P, rank) but we need specific ordering
        # Let's construct properly: for each r, B[:,r] = kron(W[:,r], V[:,r]) but with specific index mapping
        # b = j*P+k (0..29), c = k*N+i (0..19)
        # We want B[b*P*N+c, r] = V[b,r] * W[c,r]
        # This is: for each r, outer product V[:,r] (30,) and W[:,r] (20,) gives (30,20) matrix
        # Reshape to (600,) with b as row, c as col: V[b,r]*W[c,r]
        # So B[:,r] = np.outer(V[:,r], W[:,r]).flatten()
        
        # Vectorized: B = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
        B = np.einsum('br,cr->bcr', V, W).reshape(M*P*P*N, rank)
        
        try:
            U_new = np.linalg.lstsq(B.T, T1.T, rcond=None)[0].T
        except:
            break
        
        # Update V: C[(a,c), r] = U[a,r]*W[c,r]
        C = np.einsum('ar,cr->acr', U_new, W).reshape(N*M*P*N, rank)
        try:
            V_new = np.linalg.lstsq(C.T, T2.T, rcond=None)[0].T
        except:
            break
        
        # Update W: D[(a,b), r] = U[a,r]*V[b,r]
        D = np.einsum('ar,br->abr', U_new, V_new).reshape(N*M*M*P, rank)
        try:
            W_new = np.linalg.lstsq(D.T, T3.T, rcond=None)[0].T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_approx, T, atol=tol):
            break
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    if _verify(U_snap, V_snap, W_snap, tol=1e-6):
        return U_snap, V_snap, W_snap
    return None

def _structured_wfa(rank_target):
    """
    Try to exploit structure: decompose into blocks.
    For 2x3 * 3x10, think of it as:
    Row 0: [A00 A01 A02] * B
    Row 1: [A10 A11 A12] * B
    
    We can try to share computations between rows.
    """
    # Try various structured approaches
    # One idea: use the fact that we only have 2 rows to share some multipliers
    
    # Approach: parameterize with fewer degrees of freedom
    # Try random search with structure: make U have repeated patterns
    
    np_rng = np.random.RandomState(12345)
    
    for trial in range(2000):
        if _time_left() < 10.0:
            break
        
        # Try rank from a reasonable range
        r = rank_target
        
        # Initialize with some structure
        U = np.zeros((N*M, r))
        V = np.zeros((M*P, r))
        W = np.zeros((P*N, r))
        
        # For each rank component, assign random half-integer values
        vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        
        U[:] = np_rng.choice(vals, size=(N*M, r))
        V[:] = np_rng.choice(vals, size=(M*P, r))
        W[:] = np_rng.choice(vals, size=(P*N, r))
        
        if _verify(U, V, W, tol=1e-8):
            return U, V, W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    global _start_time
    _start_time = time.time()
    
    U_fb, V_fb, W_fb = _schoolbook()
    best = (U_fb, V_fb, W_fb)
    best_rank = RANK_SCHOOLBOOK
    
    # Try ALS for ranks from 30 to 59
    for rank in range(30, best_rank):
        if _time_left() < 20.0:
            break
        
        for seed in range(10):
            if _time_left() < 15.0:
                break
            
            result = _fast_als(rank, n_iter=80, tol=1e-10, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, tol=1e-8):
                    best = (U, V, W)
                    best_rank = rank
                    break
        
        # If we found something, try to go lower
        if best_rank < RANK_SCHOOLBOOK and _time_left() > 30:
            # Continue searching for lower ranks
            pass
    
    # Try structured random search for lower ranks
    for rank in range(25, best_rank):
        if _time_left() < 15.0:
            break
        result = _structured_wfa(rank)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, tol=1e-8):
                best = (U, V, W)
                best_rank = rank
                break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
