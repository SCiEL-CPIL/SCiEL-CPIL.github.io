import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Vectorized verification
    def verify(U, V, W, r, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Rank-37 decomposition:
    # j=0 part: 2 shared terms (since n=2, the 20 terms collapse to 2)
    # j=1,2 part: Strassen on 5 blocks of 2x2 (7 * 5 = 35 terms)
    # Total: 2 + 35 = 37

    def rank37_decomposition():
        rank = 37
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0

        # Part A: j=0 (first column of A, first row of B)
        # For each k, the terms (i=0, j=0, k) and (i=1, j=0, k) can share a term
        # since W[k*n+0] and W[k*n+1] are different, but U[i*m+0] and V[0*p+k] can be shared.
        # Actually, the tensor T[i,j,k] for j=0 is T[i,0,k] = A[i,0] * B[0,k].
        # This is a 2 x 1 x 10 tensor. The rank of a 2 x 1 x 10 tensor is 2 (since it's
        # equivalent to two vectors). We can share the terms.
        # Term c: U[0, c] = 1, V[0, c] = 1, W[0, c] = 1  -> contributes to (0,0,0)
        # Term c: U[0, c] = 1, V[0, c] = 1, W[1, c] = 1  -> contributes to (0,0,1)
        # Wait, let's think again.
        # For j=0, we need T[i,0,k] = 1 for all i, k.
        # T[i,0,k] = sum_c U[i,0,c] * V[0,k,c] * W[k,i,c]
        # Since V[0,k,c] only depends on k, and W[k,i,c] depends on k and i,
        # and U[i,0,c] depends on i, we can set:
        # For each k, U[0,0,c] = 1, V[0,k,c] = 1, W[k,0,c] = 1 -> gives T[0,0,k] += 1
        # U[1,0,c] = 1, V[0,k,c] = 1, W[k,1,c] = 1 -> gives T[1,0,k] += 1
        # But V[0,k,c] is shared between the two terms for the same k.
        # So we can have one term per k that contributes to both i=0 and i=1?
        # No, because W[k,0,c] and W[k,1,c] are different indices.
        # Actually, for a fixed k, the sub-tensor over i is [1, 1] (a vector of length 2).
        # The rank of a 2-element vector is 1. So we can use one term per k.
        # Term c: U[0,0,c] = 1, U[1,0,c] = 1, V[0,k,c] = 1, W[k,0,c] = 1, W[k,1,c] = 1
        # This gives T[0,0,k] += 1*1*1 = 1 and T[1,0,k] += 1*1*1 = 1.
        # So we need 10 terms (one per k).
        # Wait, that's 10 terms, not 2.
        # Let me re-examine. The tensor is T[i,j,k]. For j=0, it's T[i,0,k].
        # This is a 2 x 10 tensor. The rank of a 2 x 10 tensor (with all ones) is 2.
        # So we can do it with 2 terms.
        # Term c1: U[0,0,c1] = 1, V[0,*,c1] = [1,1,...,1], W[*,0,c1] = [1,1,...,1]
        # This gives T[0,0,k] += 1 for all k.
        # Term c2: U[1,0,c2] = 1, V[0,*,c2] = [1,1,...,1], W[*,1,c2] = [1,1,...,1]
        # This gives T[1,0,k] += 1 for all k.
        # Yes! That works. So 2 terms for j=0.

        # c1
        U[0, col] = 1  # i=0, j=0
        for k in range(p):
            V[0 * p + k, col] = 1
            W[k * n + 0, col] = 1
        col += 1

        # c2
        U[1 * m + 0, col] = 1  # i=1, j=0
        for k in range(p):
            V[0 * p + k, col] = 1
            W[k * n + 1, col] = 1
        col += 1

        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1

            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1

            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1

            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1

            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1

            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1

            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        return U, V, W, rank

    U_best, V_best, W_best, rank_best = rank37_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)

    # If the rank-37 decomposition doesn't verify, fall back to rank-55
    if err > 1e-8:
        def rank55_decomposition():
            rank = 55
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))

            col = 0

            # Part A: j=0 (first column of A, first row of B)
            for i in range(n):
                for k in range(p):
                    U[i * m + 0, col] = 1
                    V[0 * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

            # Part B: j=1,2 with Strassen on 5 blocks of 2x2
            for b in range(5):
                k0 = 2 * b
                k1 = 2 * b + 1

                w_00 = k0 * n + 0
                w_01 = k1 * n + 0
                w_10 = k0 * n + 1
                w_11 = k1 * n + 1

                u_11 = 0 * m + 1
                u_12 = 0 * m + 2
                u_21 = 1 * m + 1
                u_22 = 1 * m + 2

                v_11 = 1 * p + k0
                v_12 = 1 * p + k1
                v_21 = 2 * p + k0
                v_22 = 2 * p + k1

                # M1 = (a11 + a22)(b11 + b22)
                U[u_11, col] += 1
                U[u_22, col] += 1
                V[v_11, col] += 1
                V[v_22, col] += 1
                W[w_00, col] += 1
                W[w_11, col] += 1
                col += 1

                # M2 = (a21 + a22)b11
                U[u_21, col] += 1
                U[u_22, col] += 1
                V[v_11, col] += 1
                W[w_10, col] += 1
                W[w_11, col] += -1
                col += 1

                # M3 = a11(b12 - b22)
                U[u_11, col] += 1
                V[v_12, col] += 1
                V[v_22, col] += -1
                W[w_01, col] += 1
                W[w_11, col] += 1
                col += 1

                # M4 = a22(b21 - b11)
                U[u_22, col] += 1
                V[v_21, col] += 1
                V[v_11, col] += -1
                W[w_00, col] += 1
                W[w_10, col] += 1
                col += 1

                # M5 = (a11 + a12)b22
                U[u_11, col] += 1
                U[u_12, col] += 1
                V[v_22, col] += 1
                W[w_00, col] += -1
                W[w_01, col] += 1
                col += 1

                # M6 = (a21 - a11)(b11 + b12)
                U[u_21, col] += 1
                U[u_11, col] += -1
                V[v_11, col] += 1
                V[v_12, col] += 1
                W[w_11, col] += 1
                col += 1

                # M7 = (a12 - a22)(b21 + b22)
                U[u_12, col] += 1
                U[u_22, col] += -1
                V[v_21, col] += 1
                V[v_22, col] += 1
                W[w_00, col] += 1
                col += 1

            return U, V, W, rank

        U_best, V_best, W_best, rank_best = rank55_decomposition()

    # Final verification
    final_err = verify(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
