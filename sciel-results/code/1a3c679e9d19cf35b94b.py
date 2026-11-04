"""Strassen-based decomposition for 3x5 by 5x7 matrix multiplication."""
import time
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = _START_TIME + (_DEADLINE - (time.time() - _START_TIME))
    if time.time() > deadline:
        deadline = _START_TIME + 5.0
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook fallback: rank = n*m*p = 105
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Pad to 4x8 * 8x8, apply Strassen on 2x2 blocks of (2x4) and (4x4).
    # This gives 7 products, each of rank 4 (intermediate dim), total rank 28.
    # But we need to verify correctness.
    
    # A_padded: 4x8, A_p[i,j] = A[i,j] if i<3,j<5 else 0
    # B_padded: 8x8, B_p[i,j] = B[i,j] if i<5,j<7 else 0
    
    # Strassen partition:
    # A00=A_p[0:2,0:4], A01=A_p[0:2,4:8], A10=A_p[2:4,0:4], A11=A_p[2:4,4:8]
    # B00=B_p[0:4,0:4], B01=B_p[0:4,4:8], B10=B_p[4:8,0:4], B11=B_p[4:8,4:8]
    
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11),
    # M4=A11(B10-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01),
    # M7=(A01-A11)(B10+B11)
    # C[0:2,0:4]=M1+M4-M5+M7, C[0:2,4:8]=M3+M5,
    # C[2:4,0:4]=M2+M4, C[2:4,4:8]=M1-M2+M3+M7
    
    # We only need C_orig[i,k] for i in 0..2, k in 0..6.
    
    # Build decomposition using contributions dict.
    from fractions import Fraction as F
    
    contributions = {}  # (u_idx, v_idx, w_idx) -> coeff
    
    def add_contrib(u_idx, v_idx, w_idx, coeff):
        if coeff == 0:
            return
        key = (u_idx, v_idx, w_idx)
        contributions[key] = contributions.get(key, F(0)) + coeff
    
    # A_side(t, a, l): linear combo of A_p entries for row a, intermediate l
    # B_side(t, l, b): linear combo of B_p entries for intermediate l, col b
    # c_positions(t, a, b): list of (ci, ck, sign) for C_padded positions
    
    def A_side(t, a, l):
        d = {}
        if t == 1:
            d[(a, l)] = d.get((a, l), F(0)) + F(1)
            d[(a+2, l+4)] = d.get((a+2, l+4), F(0)) + F(1)
        elif t == 2:
            d[(a+2, l)] = d.get((a+2, l), F(0)) + F(1)
            d[(a+2, l+4)] = d.get((a+2, l+4), F(0)) + F(1)
        elif t == 3:
            d[(a, l)] = d.get((a, l), F(0)) + F(1)
        elif t == 4:
            d[(a+2, l+4)] = d.get((a+2, l+4), F(0)) + F(1)
        elif t == 5:
            d[(a, l)] = d.get((a, l), F(0)) + F(1)
            d[(a, l+4)] = d.get((a, l+4), F(0)) + F(1)
        elif t == 6:
            d[(a+2, l)] = d.get((a+2, l), F(0)) + F(1)
            d[(a, l)] = d.get((a, l), F(0)) - F(1)
        elif t == 7:
            d[(a, l+4)] = d.get((a, l+4), F(0)) + F(1)
            d[(a+2, l+4)] = d.get((a+2, l+4), F(0)) - F(1)
        return d
    
    def B_side(t, l, b):
        d = {}
        if t == 1:
            d[(l, b)] = d.get((l, b), F(0)) + F(1)
            d[(l+4, b+4)] = d.get((l+4, b+4), F(0)) + F(1)
        elif t == 2:
            d[(l, b)] = d.get((l, b), F(0)) + F(1)
        elif t == 3:
            d[(l, b+4)] = d.get((l, b+4), F(0)) + F(1)
            d[(l+4, b+4)] = d.get((l+4, b+4), F(0)) - F(1)
        elif t == 4:
            d[(l+4, b)] = d.get((l+4, b), F(0)) + F(1)
            d[(l, b)] = d.get((l, b), F(0)) - F(1)
        elif t == 5:
            d[(l+4, b+4)] = d.get((l+4, b+4), F(0)) + F(1)
        elif t == 6:
            d[(l, b)] = d.get((l, b), F(0)) + F(1)
            d[(l, b+4)] = d.get((l, b+4), F(0)) + F(1)
        elif t == 7:
            d[(l+4, b)] = d.get((l+4, b), F(0)) + F(1)
            d[(l+4, b+4)] = d.get((l+4, b+4), F(0)) + F(1)
        return d
    
    def c_positions(t, a, b):
        res = []
        if t == 1:
            res.append((a, b, F(1)))
            res.append((a+2, b+4, F(1)))
        elif t == 2:
            res.append((a+2, b, F(1)))
            res.append((a+2, b+4, F(-1)))
        elif t == 3:
            res.append((a, b+4, F(1)))
            res.append((a+2, b+4, F(1)))
        elif t == 4:
            res.append((a, b, F(1)))
            res.append((a+2, b, F(1)))
        elif t == 5:
            res.append((a, b, F(-1)))
            res.append((a, b+4, F(1)))
        elif t == 6:
            res.append((a+2, b, F(1)))
        elif t == 7:
            res.append((a, b, F(1)))
            res.append((a+2, b+4, F(1)))
        return res
    
    for t in range(1, 8):
        for a in range(2):
            for l in range(4):
                for b in range(4):
                    A_coeffs = A_side(t, a, l)
                    B_coeffs = B_side(t, l, b)
                    for (ai, aj), ac in A_coeffs.items():
                        for (bj, bk), bc in B_coeffs.items():
                            if ai >= 3 or aj >= 5 or bj >= 5 or bk >= 7:
                                continue
                            for (ci, ck, sign) in c_positions(t, a, b):
                                if ci >= 3 or ck >= 7:
                                    continue
                                u_idx = ai * m + aj
                                v_idx = bj * p + bk
                                w_idx = ck * n + ci
                                add_contrib(u_idx, v_idx, w_idx, ac * bc * sign)
    
    rank = len(contributions)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_idx, v_idx, w_idx) in enumerate(sorted(contributions.keys())):
        coeff = contributions[(u_idx, v_idx, w_idx)]
        U[u_idx][col] = float(coeff)
        V[v_idx][col] = float(coeff)
        W[w_idx][col] = float(coeff)
    
    # Verify with numpy
    import numpy as np
    np.random.seed(42)
    A_test = np.random.randint(0, 10, size=(n, m)).astype(float)
    B_test = np.random.randint(0, 10, size=(m, p)).astype(float)
    C_test = A_test @ B_test
    
    C_comp = np.zeros((n, p))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for t in range(rank):
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += U_arr[i*m+j, t] * V_arr[j*p+k, t]
                C_comp[i, k] += s * W_arr[k*n+i, t]
    
    if not np.allclose(C_comp, C_test, atol=1e-9):
        # Strassen decomposition is wrong, fall back to schoolbook
        return schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
