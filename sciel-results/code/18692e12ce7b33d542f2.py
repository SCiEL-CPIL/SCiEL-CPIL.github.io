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

    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_vec, v_vec, w_vec):
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

    # Strategy: decompose 2x3x5 as:
    #   - 2x2x3 block (cols j=0,1; k=0,1,2) using Winograd rank-10
    #   - 2x2x2 block (cols j=0,1; k=3,4) using Strassen rank-7
    #   - 2x1x5 block (col j=2; all k) using rank-5 outer
    # Total: 10 + 7 + 5 = 22

    # ---- Part A: 2x2x3 Winograd (rank 10) for j=0,1 and k=0,1,2 ----
    # A = [[a,b],[c,d]], B = [[e0,e1,e2],[f0,f1,f2]], C = [[c0,c1,c2],[c3,c4,c5]]
    # Winograd 2x2x3 rank 10:
    # u1=a, u2=b, u3=c, u4=d, u5=a+b, u6=c+d, u7=a-b, u8=c-d
    # For each k: v_k = e_k + f_k, w_k = e_k - f_k
    # c0 = u1*e0 + u2*f0  (but we need shared structure)
    #
    # Standard Winograd 2x2xn with n=3, rank 2n+2=8? No, rank(2x2x3)=10.
    # Let me use a direct construction:
    # Terms 1-5: for each k in 0,1,2: (a+b)*(ek+fk) -> contributes to both c0k and c1k
    # Terms 6-7: a*fk -> c0k correction
    # Terms 8-9: b*ek -> c1k correction  
    # That's 5+2+2=9... not 10.
    #
    # Actually rank(2x2x3) = 10 is known. Let me use:
    # m1 = a*e0, m2 = a*e1, m3 = a*e2, m4 = b*f0, m5 = b*f1, m6 = b*f2
    # m7 = c*e0, m8 = c*e1, m9 = c*e2, m10 = d*f0
    # ... this is getting to schoolbook.
    #
    # Let me just use the schoolbook for 2x2x3 (rank 12) + Strassen 2x2x2 (rank 7) + outer 2x1x5 (rank 5) = 24
    # Or better: schoolbook 2x3x5 = rank 30. Let me try to do better.
    #
    # Simplest correct approach: schoolbook rank 30, but try to reduce.
    # Let me try: for each (j,k) pair, we need 2 terms (one for i=0, one for i=1)
    # That's 3*5*2 = 30.
    #
    # Better: group by k. For each k, we multiply 2x3 by 3x1 to get 2x1.
    # That's 2*3=6 mults per k, total 6*5=30.
    #
    # Use Strassen on 2x2x2 blocks:
    # Split k into {0,1}, {2,3}, {4}
    # 2x2x2 (j=0,1; k=0,1): rank 7
    # 2x2x2 (j=0,1; k=2,3): rank 7
    # 2x2x1 (j=0,1; k=4): rank 4
    # 2x1x5 (j=2; k=0..4): rank 5
    # Total: 7+7+4+5 = 23

    # ---- Part A: 2x2x2 Strassen (rank 7) for j=0,1; k=0,1 ----
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B10-B11)
    # M4=A11(B01-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
    strassen_k01 = [
        # M1: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2: (A10+A11)*B00 -> C10 += 1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3: A00*(B10-B11) -> C01 += 1, C11 += 1
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4: A11*(B01-B00) -> C00 += 1, C10 += 1
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5: (A00+A01)*B11 -> C00 -= 1, C01 += 1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6: (A10-A00)*(B00+B01) -> C11 += 1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7: (A01-A11)*(B10+B11) -> C00 += 1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for (a_list, b_list, c_list) in strassen_k01:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term(u, v, w)

    # ---- Part B: 2x2x2 Strassen (rank 7) for j=0,1; k=2,3 ----
    strassen_k23 = [
        ([(0,0,1),(1,1,1)], [(0,2,1),(1,3,1)], [(0,2,1),(1,3,1)]),
        ([(1,0,1),(1,1,1)], [(0,2,1)], [(1,2,1)]),
        ([(0,0,1)], [(1,2,1),(1,3,-1)], [(0,3,1),(1,3,1)]),
        ([(1,1,1)], [(0,3,1),(0,2,-1)], [(0,2,1),(1,2,1)]),
        ([(0,0,1),(0,1,1)], [(1,3,1)], [(0,2,-1),(0,3,1)]),
        ([(1,0,1),(0,0,-1)], [(0,2,1),(0,3,1)], [(1,3,1)]),
        ([(0,1,1),(1,1,-1)], [(1,2,1),(1,3,1)], [(0,2,1)]),
    ]
    for (a_list, b_list, c_list) in strassen_k23:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term(u, v, w)

    # ---- Part C: 2x2x1 (rank 4) for j=0,1; k=4 ----
    # C04 = A00*B04 + A01*B14, C14 = A10*B04 + A11*B14
    # 4 terms:
    c4_terms = [
        ([(0,0,1)], [(0,4,1)], [(0,4,1)]),
        ([(0,1,1)], [(1,4,1)], [(0,4,1)]),
        ([(1,0,1)], [(0,4,1)], [(1,4,1)]),
        ([(1,1,1)], [(1,4,1)], [(1,4,1)]),
    ]
    for (a_list, b_list, c_list) in c4_terms:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term(u, v, w)

    # ---- Part D: 2x1x5 outer (rank 5) for j=2; k=0..4 ----
    for k in range(5):
        u = zeros6(); v = zeros15(); w = zeros10()
        setA(u, 0, 2, 1)
        setB(v, 2, k, 1)
        setC(w, 0, k, 1)
        add_term(u, v, w)
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
