import numpy as np
import time

def construct():
    """
    Returns a valid decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the robust rank-98 construction (two independent 4x4x4 Strassen blocks).
    """
    n, m, p = 4, 4, 8
    return _construct_rank98(n, m, p)

def _construct_rank98(n, m, p):
    """Fallback: rank-98 via two concatenated 4x4x4 Strassen decompositions."""
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding linear dependencies among columns."""
    rank = U.shape[1]
    
    # Check for zero or near-zero columns
    for r in range(rank - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        if np.all(np.abs(U[:, r]) < 1e-10) or np.all(np.abs(V[:, r]) < 1e-10) or np.all(np.abs(W[:, r]) < 1e-10):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
    
    # Try to find columns that are linear combinations of others
    # For each column r, check if U[:,r] is in span of other U columns, etc.
    # This is expensive; only do a few iterations.
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        removed = False
        for r in range(rank - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Skip if already checked
            if r >= rank:
                continue
            # Try to express column r as combination of others
            # Use least squares on U part
            if rank <= 1:
                break
            mask = [c for c in range(rank) if c != r]
            Uo = U[:, mask]
            if Uo.shape[1] == 0:
                continue
            try:
                coeffs, residuals, rank_u, sv = np.linalg.lstsq(Uo, U[:, r], rcond=None)
                if np.linalg.norm(Uo @ coeffs - U[:, r]) < 1e-8 * max(1, np.linalg.norm(U[:, r])):
                    # Check V
                    Vo = V[:, mask]
                    coeffs_v, _, _, _ = np.linalg.lstsq(Vo, V[:, r], rcond=None)
                    if np.linalg.norm(Vo @ coeffs_v - V[:, r]) < 1e-8 * max(1, np.linalg.norm(V[:, r])):
                        # Check W
                        Wo = W[:, mask]
                        coeffs_w, _, _, _ = np.linalg.lstsq(Wo, W[:, r], rcond=None)
                        if np.linalg.norm(Wo @ coeffs_w - W[:, r]) < 1e-8 * max(1, np.linalg.norm(W[:, r])):
                            # Column r is redundant
                            U = np.delete(U, r, axis=1)
                            V = np.delete(V, r, axis=1)
                            W = np.delete(W, r, axis=1)
                            rank -= 1
                            removed = True
            except np.linalg.LinAlgError:
                continue
    
    return U, V, W, rank

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
