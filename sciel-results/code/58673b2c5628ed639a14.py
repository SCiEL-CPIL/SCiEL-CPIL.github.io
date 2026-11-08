import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin vs 150s kill

    # ---------- helpers ----------
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

    def verify_exact(U, V, W, rank):
        """Exact verification using Fraction; entries must be half-integers."""
        for M in (U, V, W):
            for row in M:
                for x in row:
                    fx = Fraction(x) if not isinstance(x, Fraction) else x
                    if (fx * 2).denominator != 1:
                        return False
        T = {}
        for r in range(rank):
            for a in range(n * m):
                ua = U[a][r]
                if ua == 0:
                    continue
                ua_f = Fraction(ua) if not isinstance(ua, Fraction) else ua
                for b in range(m * p):
                    vb = V[b][r]
                    if vb == 0:
                        continue
                    vb_f = Fraction(vb) if not isinstance(vb, Fraction) else vb
                    for c in range(p * n):
                        wc = W[c][r]
                        if wc == 0:
                            continue
                        wc_f = Fraction(wc) if not isinstance(wc, Fraction) else wc
                        key = (a, b, c)
                        T[key] = T.get(key, Fraction(0)) + ua_f * vb_f * wc_f
        for key in TGT:
            if T.get(key, Fraction(0)) != Fraction(1):
                return False
        for key, val in T.items():
            if key not in TGT and val != 0:
                return False
        return True

    # ---------- Strassen 2x2x2 core ----------
    STRASSEN = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,-1),(1,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1)]),
    ]

    # Rectangular 2x4 by 4x4: split into 2 blocks of 2x2 by 2x2
    # Block 0: A[:, 0:2] * B[0:2, :] -> C[:, 0:2]
    # Block 1: A[:, 2:4] * B[2:4, :] -> C[:, 2:4]
    BLOCKS = [
        (0, 0, 0, 0, 0, 0),  # A11*B11 -> C11
        (0, 2, 2, 0, 2, 0),  # A12*B21 -> C12
    ]

    def build_strassen_28():
        rank = 14
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for (ar_off, ac_off, br_off, bc_off, cr_off, cc_off) in BLOCKS:
            for (u_c, v_c, w_c) in STRASSEN:
                for (ar, ac, coeff) in u_c:
                    U[ar_off + ar, ac_off + ac] = coeff
                for (br, bc, coeff) in v_c:
                    V[br_off + br, bc_off + bc] = coeff
                for (cr, cc, coeff) in w_c:
                    W[cr_off + cr, cc_off + cc] = coeff
                col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # ---------- Main search ----------
    best_result, best_rank = schoolbook()

    # Try Strassen rank 14 (2 blocks x 7)
    if time.time() < deadline - 20:
        strassen, r14 = build_strassen_28()
        if verify_exact(strassen["U"], strassen["V"], strassen["W"], r14):
            best_result = strassen
            best_rank = r14
            print(f"Strassen rank {r14} verified exactly")
        else:
            print("Strassen rank 14 failed exact verification")

    # If rank 14 works, try to reduce further via linear dependency search
    if best_rank > 10 and time.time() < deadline - 15:
        strassen, r14 = build_strassen_28()
        U_arr = np.array(strassen["U"], dtype=float)
        V_arr = np.array(strassen["V"], dtype=float)
        W_arr = np.array(strassen["W"], dtype=float)

        # Try to find linear dependencies among columns using SVD
        # For each subset of columns, check if removing one column can be
        # compensated by adjusting another
        for r_remove in range(r14):
            if time.time() > deadline - 10:
                break
            U_reduced = np.delete(U_arr, r_remove, axis=1)
            V_reduced = np.delete(V_arr, r_remove, axis=1)
            W_reduced = np.delete(W_arr, r_remove, axis=1)
            
            # Check if the remaining decomposition still works
            # (it won't in general, but verify to be safe)
            U_l = U_reduced.tolist()
            V_l = V_reduced.tolist()
            W_l = W_reduced.tolist()
            if verify_exact(U_l, V_l, W_l, r14 - 1):
                best_result = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = r14 - 1
                print(f"Reduced to rank {best_rank}")
                break

    # Try rank 12 via structured search
    if best_rank > 12 and time.time() < deadline - 5:
        import random
        random.seed(42)
        for attempt in range(100):
            if time.time() > deadline - 3:
                break
            rank = 12
            U = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(p*n)]
            if verify_exact(U, V, W, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break

    # Ensure return format matches interface
    result = {
        "U": best_result["U"],
        "V": best_result["V"],
        "W": best_result["W"]
    }
    return result
# EVOLVE-BLOCK-END
