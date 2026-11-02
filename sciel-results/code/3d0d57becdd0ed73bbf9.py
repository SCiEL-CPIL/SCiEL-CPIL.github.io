import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Tries multiple blocking strategies with Strassen and selects the lowest rank.
    """
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def build_decomposition_from_terms(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-10):
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err

    def add_strassen(terms, i_start, j_start, k_start):
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook_term(terms, i, j, k):
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))

    def try_decomposition(j_blocks, k_blocks):
        terms = []
        for (j0, j1) in j_blocks:
            for (k0, k1) in k_blocks:
                if j1 - j0 == 2 and k1 - k0 == 2:
                    add_strassen(terms, 0, j0, k0)
        covered = set()
        for (j0, j1) in j_blocks:
            for (k0, k1) in k_blocks:
                if j1 - j0 == 2 and k1 - k0 == 2:
                    for j in range(j0, j1):
                        for k in range(k0, k1):
                            covered.add((j, k))
        for j in range(m):
            for k in range(p):
                if (j, k) not in covered:
                    for i in range(n):
                        add_schoolbook_term(terms, i, j, k)
        U, V, W = build_decomposition_from_terms(terms)
        valid, err = verify_decomposition(U, V, W)
        if valid:
            return U, V, W, len(terms)
        return None, None, None, float('inf')
    
    best_U, best_V, best_W, best_R = None, None, None, float('inf')
    j_options = [(0,2), (1,3), (2,4), (3,5)]
    k_options = [(0,2), (1,3), (2,4), (3,5)]
    
    for j0 in j_options:
        for j1 in j_options:
            if j0[0] >= j1[0] or j0[1] <= j1[0]:
                continue
            for k0 in k_options:
                for k1 in k_options:
                    if k0[0] >= k1[0] or k0[1] <= k1[0]:
                        continue
                    U, V, W, R = try_decomposition([j0, j1], [k0, k1])
                    if R < best_R:
                        best_R, best_U, best_V, best_W = R, U, V, W
                    if time.time() > deadline - 5:
                        break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    for j0 in j_options:
        for k0 in k_options:
            if time.time() > deadline - 5:
                break
            U, V, W, R = try_decomposition([j0], [k0])
            if R < best_R:
                best_R, best_U, best_V, best_W = R, U, V, W
    
    if best_U is None:
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
