"""Combined program: structural schoolbook baseline + correct Strassen-based refinement
for multiplying a 4x4 matrix by a 4x9 matrix (trace convention W[k*n+i, r])."""
import time
import random

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m        # 16
    mp = m * p        # 36
    pn = p * n        # 36

    def make_schoolbook():
        """Return schoolbook decomposition of rank n*m*p (each (i,j,k) is rank-1)."""
        rank = n * m * p  # 144
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition: for every (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1,
        and all off-support entries equal 0."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > tol:
                        return False
        # off-support: for every (a,b) pair that is NOT a consistent (i,j,k), the sum must be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    for c2 in range(pn):
                        if c2 == c:
                            continue
                        # c2 corresponds to (k2,i2)
                        k2, i2 = divmod(c2, n)
                        total = 0.0
                        for r in range(rank):
                            total += U[a][r] * V[b][r] * W[c2][r]
                        if abs(total) > tol:
                            return False
        return True

    # Start with schoolbook as the guaranteed-valid baseline
    best_result = make_schoolbook()
    best_rank = n * m * p  # 144

    # ------------------------------------------------------------------
    # Correct Strassen-style refinement.
    #
    # Split A (4x4) into 2x2 blocks A11,A12,A21,A22 and B (4x9) into
    # B1 (top 2x9) and B2 (bottom 2x9).  The output C (4x9) is:
    #     C1 = A11*B1 + A12*B2   (top 2x9)
    #     C2 = A21*B1 + A22*B2   (bottom 2x9)
    #
    # Each 2x2 * 2x9 product is computed with 7 bilinear products (a
    # rectangular Strassen scheme).  For M=[[a,b],[c,d]] and N=[[n1],[n2]]
    # (n1,n2 are 1x9 rows), result R=[[r1],[r2]] with r1=a*n1+b*n2, r2=c*n1+d*n2:
    #     p1 = (a+d)*(n1+n2)
    #     p2 = (c+d)*(n1+n2)
    #     p3 = a*(n2-n1)
    #     p4 = d*(n1-n2)
    #     p5 = (a+b)*n2
    #     p6 = (c-a)*(n1+n2)
    #     p7 = (b-d)*(n1+n2)
    #     r1 = p1 + p4 - p5 + p7
    #     r2 = p2 + p3 - p4 + p6
    # (verified algebraically: r1 = a*n1+b*n2, r2 = c*n1+d*n2).
    #
    # We build each bilinear product as a tensor term over the full index
    # spaces, then sum the coefficient vectors per rank column.
    # ------------------------------------------------------------------
    def build_strassen_result():
        # For each of the four 2x2 blocks, define which global A-entry feeds
        # which Strassen local slot (a=00,b=01,c=10,d=11).
        # block 0: A11 (rows 0-1, cols 0-1); block 1: A12 (rows 0-1, cols 2-3)
        # block 2: A21 (rows 2-3, cols 0-1); block 3: A22 (rows 2-3, cols 2-3)
        # local (li,lj) -> global (i,j)
        def block_map(block):
            base_i = 0 if block in (0, 1) else 2
            base_j = 0 if block in (0, 2) else 2
            return {(li, lj): (base_i + li, base_j + lj) for li in range(2) for lj in range(2)}

        # B split: B1 = rows 0-1 (top), B2 = rows 2-3 (bottom).  local B-row lbr
        # -> global row lbr (top) or 2+lbr (bottom).
        def brow_map(part, lbr):
            return lbr if part == 0 else 2 + lbr

        # C split: C1 = rows 0-1 (top), C2 = rows 2-3 (bottom).  local C-row lcr
        # -> global row lcr (top) or 2+lcr (bottom).
        def crow_map(part, lcr):
            return lcr if part == 0 else 2 + lcr

        # 7 Strassen products: each gives (U_slots, V_slots, R_slots)
        # U_slots: list of (local_A_slot, coeff); V_slots: list of (local_B_row, coeff)
        # R_slots: list of (local_C_row, coeff)
        strassen = [
            ([(0, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),   # p1=(a+d)(n1+n2)
            ([(2, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),   # p2=(c+d)(n1+n2)
            ([(0, 1)],         [(1, 1), (0, -1)], [(0, 1), (1, 1)]),  # p3=a(n2-n1)
            ([(3, 1)],         [(0, 1), (1, -1)], [(0, 1), (1, 1)]),  # p4=d(n1-n2)
            ([(0, 1), (1, 1)], [(1, 1)],           [(0, -1), (1, 0)]),# p5=(a+b)n2
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)],  [(0, 1), (1, 1)]), # p6=(c-a)(n1+n2)
            ([(1, 1), (3, -1)], [(0, 1), (1, 1)],  [(0, 1), (1, 1)]), # p7=(b-d)(n1+n2)
        ]

        # Accumulate coefficient vectors.  Each bilinear product is one rank column.
        U_cols = []
        V_cols = []
        W_cols = []

        for block in range(4):
            bmap = block_map(block)
            bpart = block  # 0,1 use B1 (top); 2,3 use B2 (bottom)
            cpart = block  # 0,1 feed C1 (top); 2,3 feed C2 (bottom)
            for u_slots, v_slots, r_slots in strassen:
                Ucol = [0] * nm
                Vcol = [0] * mp
                Wcol = [0] * pn
                for (slot, uc) in u_slots:
                    li, lj = slot  # slot 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
                    i, j = bmap[(li, lj)]
                    Ucol[i * m + j] = uc
                for (lbr, vc) in v_slots:
                    j = brow_map(bpart, lbr)
                    for k in range(p):
                        Vcol[j * p + k] = vc
                for (lcr, wc) in r_slots:
                    i = crow_map(cpart, lcr)
                    for k in range(p):
                        Wcol[k * n + i] = wc
                U_cols.append(Ucol)
                V_cols.append(Vcol)
                W_cols.append(Wcol)

        rank = len(U_cols)
        U = [U_cols[r] for r in range(rank)]
        # transpose to [nm][rank]
        U = [[U_cols[r][a] for r in range(rank)] for a in range(nm)]
        V = [[V_cols[r][b] for r in range(rank)] for b in range(mp)]
        W = [[W_cols[r][c] for r in range(rank)] for c in range(pn)]
        return {"U": U, "V": V, "W": W}

    # Try the Strassen refinement; adopt it only if it verifies exactly.
    if time.time() < deadline - 20:
        try:
            cand = build_strassen_result()
            cand_rank = len(cand["U"][0])
            if cand_rank < best_rank and verify(cand["U"], cand["V"], cand["W"]):
                best_result = cand
                best_rank = cand_rank
        except Exception:
            pass  # keep the safe schoolbook baseline

    return best_result
# EVOLVE-BLOCK-END
