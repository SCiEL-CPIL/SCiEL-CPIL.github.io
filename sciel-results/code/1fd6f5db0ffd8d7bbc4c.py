import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by removing linearly dependent columns.
    Uses SVD to detect numerical rank and greedy removal.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Build the tensor for verification
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def _is_valid(U_c, V_c, W_c, tol=1e-9):
        """Check if the decomposition is still valid."""
        if time.time() > deadline:
            return False
        T_dec = np.einsum('ar,br,cr->abc', U_c, V_c, W_c)
        return np.allclose(T_dec, T_target, atol=tol)
    
    def _try_remove_column(idx, U_c, V_c, W_c):
        """Try removing column idx and check validity."""
        U_new = np.delete(U_c, idx, axis=1)
        V_new = np.delete(V_c, idx, axis=1)
        W_new = np.delete(W_c, idx, axis=1)
        if _is_valid(U_new, V_new, W_new):
            return U_new, V_new, W_new
        return None
    
    # Greedy column removal: repeatedly try to remove each column
    current_rank = best_rank
    U_c, V_c, W_c = best_U, best_V, best_W
    removed = set()
    
    for col in range(best_rank):
        if time.time() > deadline:
            break
        if col in removed:
            continue
        result = _try_remove_column(col, U_c, V_c, W_c)
        if result is not None:
            U_c, V_c, W_c = result
            removed.add(col)
            current_rank -= 1
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
    
    # If we reduced rank, update best
    if best_rank < U.shape[1]:
        return best_U, best_V, best_W
    
    return U, V, W

def _build_rank_39():
    """
    Build the known rank 39 decomposition.
    5 Strassen blocks (2x2x2) + 1 Schoolbook block (2x2x1).
    """
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1

        # m1 = (a00+a11)(b00+b11)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (a10+a11)b00
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a00(b01-b11)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = a11(b10-b00)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a00+a01)b11
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (a10-a00)(b00+b01)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (a01-a11)(b10+b11)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # Last column k=10: Schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1

    U[0, r] += 1
    V[v_e2, r] += 1
    W[w_c02, r] += 1
    r += 1

    U[1, r] += 1
    V[v_g2, r] += 1
    W[w_c02, r] += 1
    r += 1

    U[2, r] += 1
    V[v_e2, r] += 1
    W[w_c12, r] += 1
    r += 1

    U[3, r] += 1
    V[v_g2, r] += 1
    W[w_c12, r] += 1
    r += 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Build the known rank 39 solution
    U, V, W = _build_rank_39()
    
    # 2. Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 44
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # 3. Try to reduce rank by removing linearly dependent columns
    if time.time() < deadline - 5:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p, deadline)
        # Re-verify after reduction
        if not _verify(U, V, W, n, m, p):
            # Rebuild if verification fails
            U, V, W = _build_rank_39()
    
    # Return the best verified solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
