import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build rank-44 solution
    U44, V44, W44 = build_rank44(n, m, p)
    
    # Try to find rank-43 by removing one column and re-solving
    # We try all 44 columns, but with a time limit
    # We also try to find rank-42 by removing two columns if time permits
    
    # First, try rank 43
    for col in range(44):
        if time.time() > deadline - 10:
            break
        result = try_reduce_to_rank(U44, V44, W44, [col], n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If rank 43 failed, try rank 42 (remove 2 columns)
    # We only try a few pairs to save time
    import itertools
    pairs = list(itertools.combinations(range(44), 2))
    # Shuffle to avoid bias
    np.random.seed(42)
    np.random.shuffle(pairs)
    
    for pair in pairs[:20]: # Try first 20 pairs
        if time.time() > deadline - 5:
            break
        result = try_reduce_to_rank(U44, V44, W44, list(pair), n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Return rank-44
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}

def build_rank44(n, m, p):
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # 7 terms for 2x2x2 Strassen
        # M1 = (A11+A22)(B11+B22)
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        # M2 = (A21+A22)B11
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1

        # M3 = A11(B21-B22)
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        # M4 = A22(B11-B12)
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1

        # M5 = (A11+A12)B22
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        # M6 = (A21-A11)(B11+B12)
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        # M7 = (A12-A22)(B21+B22)
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        # 4 terms for 1x2x2 schoolbook
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    add_122(2, 0, 0)
    add_122(2, 2, 0)
    add_122(2, 0, 2)
    add_122(2, 2, 2)

    return U, V, W

def check(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1) > 1e-6:
                    return False
    return True

def try_reduce_to_rank(U, V, W, remove_cols, n, m, p, deadline):
    R = U.shape[1] - len(remove_cols)
    if R <= 0:
        return None
        
    U_new = np.delete(U, remove_cols, axis=1).copy()
    V_new = np.delete(V, remove_cols, axis=1).copy()
    W_new = np.delete(W, remove_cols, axis=1).copy()

    # Precompute index arrays for speed
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    triples = np.array(triples)

    def fast_check(U, V, W):
        for idx in range(len(triples)):
            a, b, c = triples[idx]
            val = np.dot(U[a], V[b] * W[c])
            if abs(val - 1) > 1e-6:
                return False
        return True

    # ALS iterations
    for it in range(50):
        if time.time() > deadline - 2:
            break

        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V_new[j * p + k, :] * W_new[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U_new[i * m + j, :] * W_new[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    pass

    # Round to half-integers
    U_new = np.round(U_new * 2) / 2
    V_new = np.round(V_new * 2) / 2
    W_new = np.round(W_new * 2) / 2

    if fast_check(U_new, V_new, W_new):
        return U_new, V_new, W_new

    # Local search: try flipping individual half-integer values
    for ls_iter in range(200):
        if time.time() > deadline - 1:
            break
        
        violated = set()
        for idx in range(len(triples)):
            a, b, c = triples[idx]
            val = np.dot(U_new[a], V_new[b] * W_new[c])
            if abs(val - 1) > 1e-6:
                violated.add(a)
                violated.add(b)
                violated.add(c)
        
        if not violated:
            return U_new, V_new, W_new
        
        improved = False
        for a in violated:
            if a < n * m:
                for r in range(R):
                    old = U_new[a, r]
                    for delta in [0.5, -0.5]:
                        U_new[a, r] = old + delta
                        if fast_check(U_new, V_new, W_new):
                            return U_new, V_new, W_new
                        U_new[a, r] = old
            elif a < n * m + m * p:
                v_idx = a - n * m
                for r in range(R):
                    old = V_new[v_idx, r]
                    for delta in [0.5, -0.5]:
                        V_new[v_idx, r] = old + delta
                        if fast_check(U_new, V_new, W_new):
                            return U_new, V_new, W_new
                        V_new[v_idx, r] = old
            else:
                w_idx = a - n * m - m * p
                for r in range(R):
                    old = W_new[w_idx, r]
                    for delta in [0.5, -0.5]:
                        W_new[w_idx, r] = old + delta
                        if fast_check(U_new, V_new, W_new):
                            return U_new, V_new, W_new
                        W_new[w_idx, r] = old
    
    return None
