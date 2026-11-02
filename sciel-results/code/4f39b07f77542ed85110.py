"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t_start = time.time()
    deadline = t_start + 110

    def build_from_terms(terms):
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for r, (u_d, v_d, w_d) in enumerate(terms):
            for (i, j), c in u_d.items():
                U[i * m + j][r] = float(c)
            for (j, k), c in v_d.items():
                V[j * p + k][r] = float(c)
            for (i, k), c in w_d.items():
                W[k * n + i][r] = float(c)
        return U, V, W

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        np.random.seed(12345)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        C_calc = np.zeros((n, p))
        R = U_arr.shape[1]
        for r in range(R):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            for i in range(n):
                for k in range(p):
                    C_calc[i, k] += u.dot(v) * w[k, i]
        return np.max(np.abs(C_calc - C_exp)) < 1e-9

    # Strassen rank-7 kernel for 2x2 blocks, with index offsets
    def strassen_terms(i0, j0, k0):
        i1, j1, k1 = i0+1, j0+1, k0+1
        return [
            ( {(i0,j0):1,(i1,j1):1}, {(j0,k0):1,(j1,k1):1}, {(i0,k0):1,(i1,k1):1} ),
            ( {(i1,j0):1,(i1,j1):1}, {(j0,k0):1}, {(i1,k0):1,(i1,k1):-1} ),
            ( {(i0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(i0,k1):1,(i1,k1):1} ),
            ( {(i1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(i0,k0):1,(i1,k0):1} ),
            ( {(i0,j0):1,(i0,j1):1}, {(j1,k1):1}, {(i0,k0):-1,(i0,k1):1} ),
            ( {(i1,j0):1,(i0,j0):-1}, {(j0,k0):1,(j0,k1):1}, {(i1,k1):1} ),
            ( {(i0,j1):1,(i1,j1):-1}, {(j1,k0):1,(j1,k1):1}, {(i0,k0):1} ),
        ]

    def schoolbook_2x2x3(i0, j0, k0):
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    ii, jj, kk = i0+i, j0+j, k0+k
                    terms.append(
                        ( {(ii,jj):1}, {(jj,kk):1}, {(ii,kk):1} )
                    )
        return terms

    best_rank = None
    best_result = None

    # Strategy A: four independent 2x2x3 subproblems, Strassen for first two cols + schoolbook for third
    terms_A = []
    for i0 in (0, 2):
        for j0 in (0, 2):
            for k0 in (0, 3):
                if k0 < 4:
                    terms_A.extend(strassen_terms(i0, j0, k0))
                else:
                    terms_A.extend(schoolbook_2x2x3(i0, j0, k0))
    rank_A = len(terms_A)
    U_A, V_A, W_A = build_from_terms(terms_A)
    if verify(U_A, V_A, W_A, n, m, p):
        best_rank = rank_A
        best_result = {"U": U_A, "V": V_A, "W": W_A}

    # Strategy B: treat as 4x4*4x4 (Strassen rank 28) + 4x4*4x2 (schoolbook rank 32)
    terms_B = []
    for i0 in (0, 2):
        for j0 in (0, 2):
            for k0 in (0, 2):
                terms_B.extend(strassen_terms(i0, j0, k0))
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                terms_B.append( ( {(i,j):1}, {(j,k):1}, {(i,k):1} ) )
    rank_B = len(terms_B)
    U_B, V_B, W_B = build_from_terms(terms_B)
    if verify(U_B, V_B, W_B, n, m, p):
        if best_rank is None or rank_B < best_rank:
            best_rank = rank_B
            best_result = {"U": U_B, "V": V_B, "W": W_B}

    # Strategy C: full 4x4x6 Strassen-style: 2x2 blocks with 3-col B
    # For each 2x2 A-block and 2x2 B-block (cols 0-1), use Strassen
    # For cols 2-3 and 4-5, use schoolbook within 2x2x2
    terms_C = []
    for i0 in (0, 2):
        for j0 in (0, 2):
            # cols 0,1 via Strassen on 2x2x2
            terms_C.extend(strassen_terms(i0, j0, 0))
            # cols 2,3 via schoolbook 2x2x2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ii, jj, kk = i0+i, j0+j, 2+k
                        terms_C.append( ( {(ii,jj):1}, {(jj,kk):1}, {(ii,kk):1} ) )
            # cols 4,5 via schoolbook 2x2x2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ii, jj, kk = i0+i, j0+j, 4+k
                        terms_C.append( ( {(ii,jj):1}, {(jj,kk):1}, {(ii,kk):1} ) )
    rank_C = len(terms_C)
    U_C, V_C, W_C = build_from_terms(terms_C)
    if verify(U_C, V_C, W_C, n, m, p):
        if best_rank is None or rank_C < best_rank:
            best_rank = rank_C
            best_result = {"U": U_C, "V": V_C, "W": W_C}

    # Strategy D: full schoolbook fallback
    terms_D = []
    for i in range(4):
        for j in range(4):
            for k in range(6):
                terms_D.append( ( {(i,j):1}, {(j,k):1}, {(i,k):1} ) )
    U_D, V_D, W_D = build_from_terms(terms_D)
    if best_rank is None:
        best_result = {"U": U_D, "V": V_D, "W": W_D}
        best_rank = len(terms_D)

    # Convert to lists of numbers (int or half-integer)
    if best_result is not None:
        U_out = best_result["U"]
        V_out = best_result["V"]
        W_out = best_result["W"]
        def clean(mat):
            out = []
            for row in mat:
                out.append([int(2*v)/2 if abs(v*2-round(v*2))>1e-9 else int(round(v)) for v in row])
            return out
        best_result = {"U": clean(U_out), "V": clean(V_out), "W": clean(W_out)}
        return best_result
    else:
        # ultimate fallback
        U = [[0]*96 for _ in range(16)]
        V = [[0]*96 for _ in range(24)]
        W = [[0]*96 for _ in range(24)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
