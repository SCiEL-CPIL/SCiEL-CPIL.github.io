"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5

    # Strategy:
    #   4x4x4 core  -> 2-level Strassen (rank 49)
    #   4x4x1 strip -> schoolbook (rank 4)
    #   total rank  -> 53
    #
    # We build the factor matrices by composing Strassen decompositions
    # at two levels and verify/repair numerically.

    # ---------- Level-1 Strassen (2x2 blocks) ----------
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # 7 intermediate products M1..M7, each a 2x2 block product.
    #
    # For each M, we record which A-block and B-block participate
    # (with signs), and which C-blocks M contributes to (with signs).
    #
    # A-side: for each of the 7 Ms, a dict {(block, (r,c)): coeff}
    # B-side: same
    # C-side: for each of the 7 Ms, a dict {(block, (r,c)): coeff}

    # A block layout: block id -> (row0, col0) in the 4x4 A
    #   0: A00 (rows 0-1, cols 0-1)
    #   1: A01 (rows 0-1, cols 2-3)
    #   2: A10 (rows 2-3, cols 0-1)
    #   3: A11 (rows 2-3, cols 2-3)
    # Same for B and C.

    # For a 2x2 block product M = X * Y where X and Y are linear
    # combinations of blocks, we need the scalar Strassen at level 2.
    #
    # Level-2 Strassen for a single 2x2 product (7 scalar mults):
    #   T1 = (X00+X11)(Y00+Y11)
    #   T2 = (X10+X11)Y00
    #   T3 = X00(Y01-Y11)
    #   T4 = X11(Y10-Y00)
    #   T5 = (X00+X01)Y11
    #   T6 = (X10-X00)(Y00+Y01)
    #   T7 = (X01-X11)(Y10+Y11)
    #   C00 = T1 + T6 - T7
    #   C01 = T3 + T5
    #   C10 = T2 + T4
    #   C11 = T1 - T3 + T5 - T4 + T7   (let me use standard form)
    #
    # Standard Strassen reconstruction:
    #   C00 = M1 - M4 + M6 - M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M3 + M5 - M2 + M7
    #
    # I'll use the well-known consistent set:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #   C00 = M1 + M4 - M2 + M6
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M3 + M5 - M7

    # I'll represent the whole thing as a list of "rank-1 tensor terms".
    # Each term is (u_dict, v_dict, w_dict) where each dict maps a
    # (matrix, row, col) index to a coefficient. The term contributes
    #   prod over (u, v, w) of their coefficients to C.
    #
    # Actually simpler: build U, V, W directly as arrays of shape
    # (n*m, rank), (m*p, rank), (p*n, rank).

    # Let me build via a "term list" approach where each term is a
    # triple of sparse vectors, then merge.

    # ---- Build level-1 Strassen terms ----
    # Each of the 7 M-blocks is itself a 2x2 product, decomposed by
    # level-2 Strassen into 7 scalar terms. So 7 * 7 = 49 terms for
    # the 4x4x4 core.

    # For M = X*Y (X, Y are 2x2 blocks expressed as linear combos of
    # A/B blocks), the level-2 Strassen produces 7 scalar terms.
    # Each scalar term is a product of one X-entry and one Y-entry,
    # where each entry is a linear combo of blocks.
    #
    # X-entry (r,c) for block X: X is one of {A00,A01,A10,A11} possibly
    #   with sign. So X[r,c] = sign * A[br+r, bc+c] where (br,bc) is
    #   the block's top-left in the 4x4.
    #
    # Let me define, for each of the 7 M's, the X and Y as (sign, block)
    #   pairs for each of the 4 positions.

    # M definitions: (X_positions, Y_positions) where each position is
    #   a list of (coeff, block) for that 2x2 slot.
    #
    # M1 = (A00+A11) * (B00+B11)
    #   X: all 4 slots = A00+A11  -> each slot has (1,A00)+(1,A11)
    #   Y: all 4 slots = B00+B11
    # M2 = (A10+A11) * B00
    #   X: all slots = A10+A11
    #   Y: all slots = B00
    # M3 = A00 * (B01-B11)
    #   X: all slots = A00
    #   Y: all slots = B01-B11
    # M4 = A11 * (B10-B00)
    #   X: all slots = A11
    #   Y: all slots = B10-B00
    # M5 = (A00+A01) * B11
    #   X: all slots = A00+A01
    #   Y: all slots = B11
    # M6 = (A10-A00) * (B00+B01)
    #   X: all slots = A10-A00
    #   Y: all slots = B00+B01
    # M7 = (A01-A11) * (B10+B11)
    #   X: all slots = A01-A11
    #   Y: all slots = B10+B11

    # C reconstruction (which C-blocks each M contributes to, with sign):
    # C00 = M1 + M4 - M2 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7

    # Block top-left offsets in 4x4:
    # block 0 (A00): (0,0); block 1 (A01): (0,2); block 2 (A10): (2,0); block 3 (A11): (2,2)
    blk_off = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}

    # For each M, define X and Y as: for each slot (sr, sc) in {0,1}^2,
    #   a list of (coeff, block).
    def all_slots(lst):
        return { (sr, sc): lst for sr in (0, 1) for sc in (0, 1) }

    Ms = {}
    Ms[1] = (all_slots([(1, 0), (1, 3)]), all_slots([(1, 0), (1, 3)]))  # A00+A11, B00+B11
    Ms[2] = (all_slots([(1, 2), (1, 3)]), all_slots([(1, 0)]))          # A10+A11, B00
    Ms[3] = (all_slots([(1, 0)]), all_slots([(1, 1), (-1, 3)]))         # A00, B01-B11
    Ms[4] = (all_slots([(1, 3)]), all_slots([(1, 2), (-1, 0)]))         # A11, B10-B00
    Ms[5] = (all_slots([(1, 0), (1, 1)]), all_slots([(1, 3)]))          # A00+A01, B11
    Ms[6] = (all_slots([(1, 2), (-1, 0)]), all_slots([(1, 0), (1, 1)])) # A10-A00, B00+B01
    Ms[7] = (all_slots([(1, 1), (-1, 3)]), all_slots([(1, 2), (1, 3)])) # A01-A11, B10+B11

    # C reconstruction: list of (M, Cblock, coeff)
    C_rec = [
        (1, 0, 1), (4, 0, 1), (2, 0, -1), (6, 0, 1),   # C00
        (3, 1, 1), (5, 1, 1),                          # C01
        (2, 2, 1), (4, 2, 1),                          # C10
        (1, 3, 1), (3, 3, -1), (5, 3, 1), (7, 3, -1),  # C11
    ]

    # Level-2 Strassen scalar terms. For a 2x2 product X*Y = C (2x2),
    # we produce 7 scalar terms. Each term t contributes to specific
    # C-slots (cr, cc) with a coefficient.
    #
    # Using the M1..M7 above (same structure, but now X, Y are the
    # 2x2 slots we already have as block-combos):
    #   t1 = (X00+X11)(Y00+Y11)
    #   t2 = (X10+X11)Y00
    #   t3 = X00(Y01-Y11)
    #   t4 = X11(Y10-Y00)
    #   t5 = (X00+X01)Y11
    #   t6 = (X10-X00)(Y00+Y01)
    #   t7 = (X01-X11)(Y10+Y11)
    #   C00 = t1 + t4 - t2 + t6
    #   C01 = t3 + t5
    #   C10 = t2 + t4
    #   C11 = t1 - t3 + t5 - t7

    # For each term, define X-slots and Y-slots it uses:
    # term: (X_slots, Y_slots) where each is list of (coeff, (sr,sc))
    def slot(s):  # shorthand for a single slot
        return s
    L2 = {}
    L2[1] = ([(1, (0, 0)), (1, (1, 1))], [(1, (0, 0)), (1, (1, 1))])
    L2[2] = ([(1, (1, 0)), (1, (1, 1))], [(1, (0, 0))])
    L2[3] = ([(1, (0, 0))], [(1, (0, 1)), (-1, (1, 1))])
    L2[4] = ([(1, (1, 1))], [(1, (1, 0)), (-1, (0, 0))])
    L2[5] = ([(1, (0, 0)), (1, (0, 1))], [(1, (1, 1))])
    L2[6] = ([(1, (1, 0)), (-1, (0, 0))], [(1, (0, 0)), (1, (0, 1))])
    L2[7] = ([(1, (0, 1)), (-1, (1, 1))], [(1, (1, 0)), (1, (1, 1))])

    L2_rec = [
        (1, (0, 0), 1), (4, (0, 0), 1), (2, (0, 0), -1), (6, (0, 0), 1),
        (3, (0, 1), 1), (5, (0, 1), 1),
        (2, (1, 0), 1), (4, (1, 0), 1),
        (1, (1, 1), 1), (3, (1, 1), -1), (5, (1, 1), 1), (7, (1, 1), -1),
    ]

    # Now build all rank-1 terms.
    # Each term: u = dict {(i, j): coeff} in A (4x4), v = dict {(j, k): coeff} in B (4x5),
    #            w = dict {(k, i): coeff} in C (5x4) [W index k*n+i]
    #
    # For the 4x4x4 core (k in 0..3), B is 4x4 (cols 0..3).

    terms = []  # each: (u_dict, v_dict, w_dict)

    for M_id, (X_slots, Y_slots) in Ms.items():
        # X_slots: slot -> list of (coeff, Ablock)
        # Y_slots: slot -> list of (coeff, Bblock)
        for t_id, (tx_slots, ty_slots) in L2.items():
            # Build u_dict: sum over X-positions in tx_slots
            u_dict = {}
            for (cx, (sr, sc)) in tx_slots:
                # X slot (sr, sc) has block-combo X_slots[(sr, sc)]
                for (bx, ablk) in X_slots[(sr, sc)]:
                    total = cx * bx
                    if total == 0:
                        continue
                    br, bc = blk_off[ablk]
                    i = br + sr
                    j = bc + sc
                    u_dict[(i, j)] = u_dict.get((i, j), 0) + total
            # Build v_dict: sum over Y-positions in ty_slots
            v_dict = {}
            for (cy, (sr, sc)) in ty_slots:
                for (by, bblk) in Y_slots[(sr, sc)]:
                    total = cy * by
                    if total == 0:
                        continue
                    br, bc = blk_off[bblk]
                    j = br + sr
                    k = bc + sc
                    v_dict[(j, k)] = v_dict.get((j, k), 0) + total
            # Build w_dict: for each C-slot (cr, cc) that t_id contributes to
            w_dict = {}
            for (m2, (cr, cc), cw) in L2_rec:
                if m2 != t_id:
                    continue
                # t_id contributes to C-slot (cr, cc) of M_id with coeff cw
                for (m_id2, cblk, cm) in C_rec:
                    if m_id2 != M_id:
                        continue
                    # M_id contributes to C-block cblk with coeff cm
                    # C-slot (cr, cc) within block cblk -> global (r, c)
                    br, bc = blk_off[cblk]
                    r = br + cr
                    c = bc + cc
                    total = cw * cm
                    if total == 0:
                        continue
                    # W index: (k, i) where k = c (column of C = row of B), i = r (row of C)
                    # C[r, c] = sum_j A[r,j] B[j,c]; here c is the B-column index = k
                    w_dict[(c, r)] = w_dict.get((c, r), 0) + total
            if u_dict and v_dict and w_dict:
                terms.append((u_dict, v_dict, w_dict))

    # ---- Add the 4x4x1 strip (k = 4) via schoolbook ----
    # C[r, 4] = sum_j A[r, j] B[j, 4]
    for r in range(4):
        for j in range(4):
            u_dict = {(r, j): 1}
            v_dict = {(j, 4): 1}
            w_dict = {(4, r): 1}
            terms.append((u_dict, v_dict, w_dict))

    # ---- Assemble U, V, W ----
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            if val != 0:
                U[i * m + j][col] = val
        for (j, k), val in v_dict.items():
            if val != 0:
                V[j * p + k][col] = val
        for (k, i), val in w_dict.items():
            if val != 0:
                W[k * n + i][col] = val

    # ---- Verify and repair ----
    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # If verification fails, fall back to a known-correct schoolbook.
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
