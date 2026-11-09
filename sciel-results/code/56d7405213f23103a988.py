import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

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
        return {"U": U, "V": V, "W": W}, rank

    def target_set():
        t = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t.add((i * m + j, j * p + k, k * n + i))
        return t

    TGT = target_set()

    def verify_exact_int(U, V, W, rank):
        """Verify using integers (entries multiplied by 2)."""
        # Check if entries are half-integers (integers after *2)
        # We assume the input is already scaled or we scale here.
        # For speed, we assume the caller passes integers representing 2*x.
        
        # Build tensor
        T = {}
        for r in range(rank):
            for a in range(n * m):
                ua = U[a][r]
                if ua == 0:
                    continue
                for b in range(m * p):
                    vb = V[b][r]
                    if vb == 0:
                        continue
                    for c in range(p * n):
                        wc = W[c][r]
                        if wc == 0:
                            continue
                        key = (a, b, c)
                        T[key] = T.get(key, 0) + ua * vb * wc
        
        # Check target entries (should be 1, so 8 in scaled integer if we scaled by 2? No, 1*1*1=1)
        # If we scale U,V,W by 2, the product is scaled by 8.
        # So target is 8.
        for key in TGT:
            if T.get(key, 0) != 8:
                return False
        for key, val in T.items():
            if key not in TGT and val != 0:
                return False
        return True

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        STRASSEN = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,-1),(1,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1)]),
        ]

        BLOCKS = [
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 2, 2, 0),
            (0, 2, 2, 0, 0, 0),
            (0, 2, 2, 2, 2, 0),
        ]

        col = 0
        for (ar_off, ac_off, br_off, bc_off, cr_off, cc_off) in BLOCKS:
            for (u_c, v_c, w_c) in STRASSEN:
                for (ar, ac, coeff) in u_c:
                    U[ar_off + ar][ac_off + ac] = coeff
                for (br, bc, coeff) in v_c:
                    V[br_off + br][bc_off + bc] = coeff
                for (cr, cc, coeff) in w_c:
                    W[cr_off + cr][cc_off + cc] = coeff
                col += 1
        
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()

    if time.time() < deadline - 10:
        strassen, r28 = build_strassen_28()
        # Scale by 2 for integer verification
        U_int = [[2*x for x in row] for row in strassen["U"]]
        V_int = [[2*x for x in row] for row in strassen["V"]]
        W_int = [[2*x for x in row] for row in strassen["W"]]
        
        if verify_exact_int(U_int, V_int, W_int, r28):
            best_result = strassen
            best_rank = r28
        else:
            # Fallback to schoolbook if verification fails
            pass

    return {
        "U": best_result["U"],
        "V": best_result["V"],
        "W": best_result["W"]
    }
