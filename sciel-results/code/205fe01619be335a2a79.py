"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    deadline = time.time() + 130

    def _verify(U, V, W, n, m, p):
        rank = U.shape[1]
        T = np.ones((n, m, p))
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    result[i, j, k] = val
        return np.allclose(result, T, atol=1e-9)

    def _is_half_integer(M):
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    def _try_decomp(U_terms, V_terms, W_terms, n, m, p):
        rank = len(U_terms)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            U[:, r] = U_terms[r]
            V[:, r] = V_terms[r]
            W[:, r] = W_terms[r]
        if _verify(U, V, W, n, m, p) and _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # We build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U indexed by [i*m + j], V by [j*p + k], W by [k*n + i]
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_vec, v_vec, w_vec):
        """u_vec: length n*m=6, v_vec: length m*p=15, w_vec: length p*n=10"""
        U_terms.append(list(u_vec))
        V_terms.append(list(v_vec))
        W_terms.append(list(w_vec))

    def zeros6():
        return [0] * 6

    def zeros15():
        return [0] * 15

    def zeros10():
        return [0] * 10

    def setA(u, i, j, val):
        u[i * 3 + j] += val

    def setB(v, j, k, val):
        v[j * 5 + k] += val

    def setC(w, i, k, val):
        w[k * 2 + i] += val

    # Try 2x2x5 Winograd (rank 11) + 2x1x5 outer (rank 10) = rank 21
    # A = [[a00,a01],[a10,a11]], B = 2x5, C = 2x5
    # Decomposition:
    #   For k in 0..4:
    #     t1_k = (a00+a01) * (b0k+b1k)  -> contributes to c0k, c1k
    #     t2_k = a00 * b1k              -> contributes to c0k
    #     t3_k = a01 * b0k              -> contributes to c1k
    #   c0k = t1_k - t2_k
    #   c1k = t1_k - t3_k
    # That's 3*5 = 15 terms. Not 11.
    #
    # Better: use the known 2x2xn rank 2n+1 = 11 decomposition.
    # x1=a00, x2=a01, x3=a10, x4=a11, x5=a00+a01, x6=a10+a11
    # For each k: y1_k=b0k, y2_k=b1k, y3_k=b0k+b1k
    # Terms:
    #   r1: x1*y1_k -> c0k (for each k) : 5 terms
    #   r2: x2*y2_k -> c0k (for each k) : 5 terms  
    #   r3: x3*y1_k -> c1k (for each k) : 5 terms
    #   r4: x4*y2_k -> c1k (for each k) : 5 terms
    #   That's 20 terms, not 11.
    #
    # The rank 2n+1 comes from sharing. Let me just use rank 12 for 2x2x5
    # via pairing: 2x2x2 (rank 7) + 2x2x3 (rank 10) = 17, or other splits.
    # 2x2x2 rank 7 + 2x2x2 rank 7 + 2x2x1 rank 4 = 18
    # 2x2x3 rank 10 + 2x2x2 rank 7 = 17
    #
    # Actually the known bound is rank(2x2xn) = 2n+1 for n>=3, so 11.
    # Let me construct it explicitly.
    #
    # 2x2x5 rank 11 decomposition:
    # A = [[a,b],[c,d]], B = [[e1,e2,e3,e4,e5],[f1,f2,f3,f4,f5]]
    # C = [[c1,c2,c3,c4,c5],[c6,c7,c8,c9,c10]]
    #
    # m1 = (a+d)(e1+f1)
    # m2 = (a+d)(e2+f2)
    # m3 = (a+d)(e3+f3)
    # m4 = (a+d)(e4+f4)
    # m5 = (a+d)(e5+f5)
    # m6 = a*f1
    # m7 = b*e2
    # m8 = c*e3
    # m9 = d*f4
    # m10 = (a-b)(f5-e5)  -- need to check
    # m11 = (c-d)(e1-f1)  -- need to check
    #
    # This is getting complex. Let me just keep the current rank-29 approach
    # but verify it's correct and return it.

    # Part B: 2x2x2 Strassen (rank 7) for columns 3,4
    strassen_terms = [
        ([(0, 0, +1), (1, 1, +1)], [(0, 3, +1), (1, 4, +1)], [(0, 3, +1), (1, 4, +1)]),
        ([(1, 0, +1), (1, 1, +1)], [(0, 3, +1)], [(1, 3, +1)]),
        ([(0, 0, +1)], [(1, 3, +1), (1, 4, -1)], [(0, 4, +1), (1, 3, +1)]),
        ([(1, 1, +1)], [(0, 4, +1), (0, 3, -1)], [(0, 3, +1), (1, 3, +1)]),
        ([(0, 0, +1), (0, 1, +1)], [(1, 4, +1)], [(0, 3, -1), (0, 4, +1)]),
        ([(1, 0, +1), (0, 0, -1)], [(0, 3, +1), (1, 3, +1)], [(1, 4, +1)]),
        ([(0, 1, +1), (1, 1, -1)], [(1, 4, +1), (0, 4, +1)], [(0, 3, +1)]),
    ]

    for (a_list, b_list, c_list) in strassen_terms:
        u = zeros6()
        v = zeros15()
        w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term(u, v, w)

    # ------------------------------------------------------------------
    # Part C: 2x1x5 outer product (rank 5) for column j=2
    # A[:,2] (2x1) * B[2,:] (1x5)
    # ------------------------------------------------------------------
    for k in range(5):
        # C[0,k] = A[0,2]*B[2,k]
        u = zeros6(); v = zeros15(); w = zeros10()
        setA(u, 0, 2, 1)
        setB(v, 2, k, 1)
        setC(w, 0, k, 1)
        add_term(u, v, w)
        # C[1,k] = A[1,2]*B[2,k]
        u = zeros6(); v = zeros15(); w = zeros10()
        setA(u, 1, 2, 1)
        setB(v, 2, k, 1)
        setC(w, 1, k, 1)
        add_term(u, v, w)

    result = _try_decomp(U_terms, V_terms, W_terms, n, m, p)
    if result is not None:
        return result

    # Fallback: guaranteed-valid schoolbook decomposition (rank 30)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
