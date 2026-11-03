import numpy as np

def construct():
    """
    Rank-92 exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Split into 2x2, 2x1, 1x2, 1x1 blocks (5=2+2+1).
    Use Strassen (rank 7) for 2x2x2 blocks, rank-1 for outer-product-like
    sub-blocks, and schoolbook for the rest.
    
    U shape: (4*5, 92) = (20, 92)  -- A factors
    V shape: (5*5, 92) = (25, 92)  -- B factors
    W shape: (5*4, 92) = (20, 92)  -- C factors, W[k*n+i, r]
    """
    RANK = 92
    U = [[0] * RANK for _ in range(20)]
    V = [[0] * RANK for _ in range(25)]
    W = [[0] * RANK for _ in range(20)]
    col = [0]  # mutable counter

    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        c = col[0]
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        col[0] += 1

    def a_idx(ri, rj):
        return ri * 5 + rj

    def b_idx(rj, rk):
        return rj * 5 + rk

    def c_idx(rk, ri):
        return rk * 4 + ri

    # Strassen for 2x2 * 2x2 -> 2x2
    def strassen222(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1):
        # A rows: ar0, ar1; A cols: ac0, ac1
        # B rows: br0, br1; B cols: bc0, bc1
        # C rows: ar0, ar1; C cols: bc0, bc1
        # M1 = (A00+A11)(B00+B11)
        add_term(a_idx(ar0, ac0), b_idx(br0, bc0), c_idx(bc0, ar0))
        add_term(a_idx(ar1, ac1), b_idx(br1, bc1), c_idx(bc0, ar0))
        add_term(a_idx(ar0, ac0), b_idx(br1, bc1), c_idx(bc1, ar0))
        add_term(a_idx(ar1, ac1), b_idx(br0, bc0), c_idx(bc0, ar1))
        add_term(a_idx(ar1, ac1), b_idx(br1, bc1), c_idx(bc1, ar1))
        add_term(a_idx(ar0, ac0), b_idx(br0, bc0), c_idx(bc1, ar1))
        add_term(a_idx(ar1, ac1), b_idx(br1, bc1), c_idx(bc1, ar1), w_val=1)
        # Wait, I need to be more careful. Let me redo Strassen properly.
        # Actually let me restart this function properly.
        pass

    # Let me redo this more carefully with explicit term lists.
    col[0] = 0  # reset
    U = [[0] * RANK for _ in range(20)]
    V = [[0] * RANK for _ in range(25)]
    W = [[0] * RANK for _ in range(20)]

    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        c = col[0]
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        col[0] += 1

    # Strassen 2x2x2: A is 2x2 (rows r0,r1; cols c0,c1)
    #                B is 2x2 (rows r0,r1; cols c0,c1)
    #                C is 2x2 (rows r0,r1; cols c0,c1)
    # A indices: (ar0,ac0), (ar0,ac1), (ar1,ac0), (ar1,ac1)
    # B indices: (br0,bc0), (br0,bc1), (br1,bc0), (br1,bc1)
    # C indices: (bc0,ar0), (bc0,ar1), (bc1,ar0), (bc1,ar1)
    #
    # M1 = (A[0][0]+A[1][1])*(B[0][0]+B[1][1])
    #   contributes: C[0][0]+=M1, C[1][1]+=M1, C[0][1]-=M1, C[1][0]-=M1
    # M2 = (A[1][0]+A[1][1])*B[0][0]
    #   contributes: C[1][0]+=M2, C[1][1]+=M2
    # M3 = A[0][0]*(B[0][1]-B[1][1])
    #   contributes: C[0][1]+=M3, C[1][1]-=M3
    # M4 = A[1][1]*(B[1][0]-B[0][0])
    #   contributes: C[0][0]-=M4, C[1][0]+=M4
    # M5 = (A[0][0]+A[0][1])*B[1][1]
    #   contributes: C[0][0]+=M5, C[0][1]+=M5
    # M6 = (A[1][0]-A[0][0])*(B[0][0]+B[0][1])
    #   contributes: C[1][0]+=M6, C[0][0]-=M6
    # M7 = (A[0][1]-A[1][1])*(B[1][0]+B[1][1])
    #   contributes: C[1][1]+=M7, C[0][1]-=M7

    def strassen222(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1):
        # M1 = (A[ar0,ac0] + A[ar1,ac1]) * (B[br0,bc0] + B[br1,bc1])
        add_term(a_idx(ar0, ac0), b_idx(br0, bc0), c_idx(bc0, ar0), 1, 1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br1, bc1), c_idx(bc0, ar0), 1, 1, 1)
        add_term(a_idx(ar0, ac0), b_idx(br1, bc1), c_idx(bc1, ar0), 1, 1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br0, bc0), c_idx(bc1, ar0), 1, 1, 1)
        add_term(a_idx(ar0, ac0), b_idx(br0, bc0), c_idx(bc0, ar1), 1, 1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br1, bc1), c_idx(bc0, ar1), 1, 1, 1)
        add_term(a_idx(ar0, ac0), b_idx(br1, bc1), c_idx(bc1, ar1), 1, 1, -1)
        add_term(a_idx(ar1, ac1), b_idx(br0, bc0), c_idx(bc1, ar1), 1, 1, -1)
        # M2 = (A[ar1,ac0] + A[ar1,ac1]) * B[br0,bc0]
        add_term(a_idx(ar1, ac0), b_idx(br0, bc0), c_idx(bc0, ar1), 1, 1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br0, bc0), c_idx(bc0, ar1), 1, 1, 1)
        add_term(a_idx(ar1, ac0), b_idx(br0, bc0), c_idx(bc1, ar1), 1, 1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br0, bc0), c_idx(bc1, ar1), 1, 1, 1)
        # M3 = A[ar0,ac0] * (B[br0,bc1] - B[br1,bc1])
        add_term(a_idx(ar0, ac0), b_idx(br0, bc1), c_idx(bc1, ar0), 1, 1, 1)
        add_term(a_idx(ar0, ac0), b_idx(br1, bc1), c_idx(bc1, ar0), 1, -1, -1)
        add_term(a_idx(ar0, ac0), b_idx(br0, bc1), c_idx(bc1, ar1), 1, 1, -1)
        add_term(a_idx(ar0, ac0), b_idx(br1, bc1), c_idx(bc1, ar1), 1, -1, 1)
        # M4 = A[ar1,ac1] * (B[br1,bc0] - B[br0,bc0])
        add_term(a_idx(ar1, ac1), b_idx(br1, bc0), c_idx(bc0, ar0), 1, 1, -1)
        add_term(a_idx(ar1, ac1), b_idx(br0, bc0), c_idx(bc0, ar0), 1, -1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br1, bc0), c_idx(bc0, ar1), 1, 1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br0, bc0), c_idx(bc0, ar1), 1, -1, -1)
        # M5 = (A[ar0,ac0] + A[ar0,ac1]) * B[br1,bc1]
        add_term(a_idx(ar0, ac0), b_idx(br1, bc1), c_idx(bc0, ar0), 1, 1, 1)
        add_term(a_idx(ar0, ac1), b_idx(br1, bc1), c_idx(bc0, ar0), 1, 1, 1)
        add_term(a_idx(ar0, ac0), b_idx(br1, bc1), c_idx(bc1, ar0), 1, 1, 1)
        add_term(a_idx(ar0, ac1), b_idx(br1, bc1), c_idx(bc1, ar0), 1, 1, 1)
        # M6 = (A[ar1,ac0] - A[ar0,ac0]) * (B[br0,bc0] + B[br0,bc1])
        add_term(a_idx(ar1, ac0), b_idx(br0, bc0), c_idx(bc0, ar1), 1, 1, 1)
        add_term(a_idx(ar1, ac0), b_idx(br0, bc1), c_idx(bc0, ar1), 1, 1, 1)
        add_term(a_idx(ar0, ac0), b_idx(br0, bc0), c_idx(bc0, ar1), -1, 1, -1)
        add_term(a_idx(ar0, ac0), b_idx(br0, bc1), c_idx(bc0, ar1), -1, 1, -1)
        add_term(a_idx(ar1, ac0), b_idx(br0, bc0), c_idx(bc0, ar0), 1, 1, -1)
        add_term(a_idx(ar1, ac0), b_idx(br0, bc1), c_idx(bc0, ar0), 1, 1, -1)
        add_term(a_idx(ar0, ac0), b_idx(br0, bc0), c_idx(bc0, ar0), -1, 1, 1)
        add_term(a_idx(ar0, ac0), b_idx(br0, bc1), c_idx(bc0, ar0), -1, 1, 1)
        # M7 = (A[ar0,ac1] - A[ar1,ac1]) * (B[br1,bc0] + B[br1,bc1])
        add_term(a_idx(ar0, ac1), b_idx(br1, bc0), c_idx(bc1, ar1), 1, 1, 1)
        add_term(a_idx(ar0, ac1), b_idx(br1, bc1), c_idx(bc1, ar1), 1, 1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br1, bc0), c_idx(bc1, ar1), -1, 1, -1)
        add_term(a_idx(ar1, ac1), b_idx(br1, bc1), c_idx(bc1, ar1), -1, 1, -1)
        add_term(a_idx(ar0, ac1), b_idx(br1, bc0), c_idx(bc1, ar0), 1, 1, -1)
        add_term(a_idx(ar0, ac1), b_idx(br1, bc1), c_idx(bc1, ar0), 1, 1, -1)
        add_term(a_idx(ar1, ac1), b_idx(br1, bc0), c_idx(bc1, ar0), -1, 1, 1)
        add_term(a_idx(ar1, ac1), b_idx(br1, bc1), c_idx(bc1, ar0), -1, 1, 1)

    # This is getting very messy with 8 terms per M. Let me use a cleaner approach.
    # Each M_r = sum_a u_a * A[a] * sum_b v_b * B[b]
    # contributes w_c * M_r to C[c]
    
    col[0] = 0
    U = [[0] * RANK for _ in range(20)]
    V = [[0] * RANK for _ in range(25)]
    W = [[0] * RANK for _ in range(20)]

    def add_rank1(u_entries, v_entries, w_entries):
        """u_entries: list of (idx, val), v_entries: list of (idx, val), w_entries: list of (idx, val)"""
        c = col[0]
        for idx, val in u_entries:
            U[idx][c] += val
        for idx, val in v_entries:
            V[idx][c] += val
        for idx, val in w_entries:
            W[idx][c] += val
        col[0] += 1

    def strassen222(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1):
        # M1
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac1),1)],
            [(b_idx(br0,bc0),1), (b_idx(br1,bc1),1)],
            [(c_idx(bc0,ar0),1), (c_idx(bc1,ar0),-1), (c_idx(bc0,ar1),-1), (c_idx(bc1,ar1),1)]
        )
        # M2
        add_rank1(
            [(a_idx(ar1,ac0),1), (a_idx(ar1,ac1),1)],
            [(b_idx(br0,bc0),1)],
            [(c_idx(bc0,ar1),1), (c_idx(bc1,ar1),1)]
        )
        # M3
        add_rank1(
            [(a_idx(ar0,ac0),1)],
            [(b_idx(br0,bc1),1), (b_idx(br1,bc1),-1)],
            [(c_idx(bc1,ar0),1), (c_idx(bc1,ar1),-1)]
        )
        # M4
        add_rank1(
            [(a_idx(ar1,ac1),1)],
            [(b_idx(br1,bc0),1), (b_idx(br0,bc0),-1)],
            [(c_idx(bc0,ar0),-1), (c_idx(bc0,ar1),1)]
        )
        # M5
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar0,ac1),1)],
            [(b_idx(br1,bc1),1)],
            [(c_idx(bc0,ar0),1), (c_idx(bc1,ar0),1)]
        )
        # M6
        add_rank1(
            [(a_idx(ar1,ac0),1), (a_idx(ar0,ac0),-1)],
            [(b_idx(br0,bc0),1), (b_idx(br0,bc1),1)],
            [(c_idx(bc0,ar1),1), (c_idx(bc0,ar0),-1)]
        )
        # M7
        add_rank1(
            [(a_idx(ar0,ac1),1), (a_idx(ar1,ac1),-1)],
            [(b_idx(br1,bc0),1), (b_idx(br1,bc1),1)],
            [(c_idx(bc1,ar1),1), (c_idx(bc1,ar0),-1)]
        )

    # 2x2 * 2x1 -> 2x1 (schoolbook, rank 4)
    # A: 2 rows (ar0,ar1), 2 cols (ac0,ac1); B: 2 rows (br0,br1), 1 col (bc0)
    # C: 2 rows (ar0,ar1), 1 col (bc0)
    def sch_221(ar0, ar1, ac0, ac1, br0, br1, bc0):
        add_rank1([(a_idx(ar0,ac0),1)], [(b_idx(br0,bc0),1)], [(c_idx(bc0,ar0),1)])
        add_rank1([(a_idx(ar0,ac1),1)], [(b_idx(br1,bc0),1)], [(c_idx(bc0,ar0),1)])
        add_rank1([(a_idx(ar1,ac0),1)], [(b_idx(br0,bc0),1)], [(c_idx(bc0,ar1),1)])
        add_rank1([(a_idx(ar1,ac1),1)], [(b_idx(br1,bc0),1)], [(c_idx(bc0,ar1),1)])

    # 2x2 * 1x2 -> 2x2 (schoolbook, rank 4)
    # A: 2 rows, 2 cols; B: 1 row (br0), 2 cols (bc0,bc1)
    # C: 2 rows, 2 cols
    def sch_212(ar0, ar1, ac0, ac1, br0, bc0, bc1):
        add_rank1([(a_idx(ar0,ac0),1)], [(b_idx(br0,bc0),1)], [(c_idx(bc0,ar0),1)])
        add_rank1([(a_idx(ar0,ac1),1)], [(b_idx(br0,bc1),1)], [(c_idx(bc1,ar0),1)])
        add_rank1([(a_idx(ar1,ac0),1)], [(b_idx(br0,bc0),1)], [(c_idx(bc0,ar1),1)])
        add_rank1([(a_idx(ar1,ac1),1)], [(b_idx(br0,bc1),1)], [(c_idx(bc1,ar1),1)])

    # 2x2 * 1x1 -> 2x1 (rank 1!)
    # A: 2 rows, 2 cols; B: 1 row (br0), 1 col (bc0)
    # C: 2 rows, 1 col; C[i] = B[br0,bc0] * sum_j A[ar_i, ac_j]
    def r1_2211(ar0, ar1, ac0, ac1, br0, bc0):
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar0,ac1),1), (a_idx(ar1,ac0),1), (a_idx(ar1,ac1),1)],
            [(b_idx(br0,bc0),1)],
            [(c_idx(bc0,ar0),1), (c_idx(bc0,ar1),1)]
        )

    # 2x1 * 2x2 -> 2x2 (schoolbook, rank 4)
    # A: 2 rows (ar0,ar1), 1 col (ac0); B: 2 rows (br0,br1), 2 cols (bc0,bc1)
    # C: 2 rows, 2 cols
    def sch_212x2(ar0, ar1, ac0, br0, br1, bc0, bc1):
        add_rank1([(a_idx(ar0,ac0),1)], [(b_idx(br0,bc0),1)], [(c_idx(bc0,ar0),1)])
        add_rank1([(a_idx(ar0,ac0),1)], [(b_idx(br0,bc1),1)], [(c_idx(bc1,ar0),1)])
        add_rank1([(a_idx(ar1,ac0),1)], [(b_idx(br1,bc0),1)], [(c_idx(bc0,ar1),1)])
        add_rank1([(a_idx(ar1,ac0),1)], [(b_idx(br1,bc1),1)], [(c_idx(bc1,ar1),1)])

    # 2x1 * 2x1 -> 2x1 (rank 1!)
    # A: 2 rows, 1 col; B: 2 rows, 1 col
    # C[i] = B[br_i,bc0] * A[ar_i,ac0]... no wait.
    # C[ar_i, bc0] = sum_j A[ar_i, j] * B[j, bc0]
    # Here A has 1 col (ac0), B has 2 rows (br0, br1)
    # C[ar0,bc0] = A[ar0,ac0]*B[br0,bc0]  (since A only has col ac0, and B row br0 corresponds)
    # Wait, the summation is over the shared dimension. A is 2x1, B is 2x1?
    # No: A is (2 rows x 1 col), B is (1 row x 1 col)? Let me re-check.
    # For A (2x1) * B (1x1) -> C (2x1): C[i] = A[i,0] * B[0,0]
    # That's rank 1.
    # For A (2x1) * B (2x1) doesn't make sense dimensionally.
    # Let me reconsider: A is 2x1 (2 rows, 1 col in the "middle" dimension)
    # B is 1x1 (1 row in middle dim, 1 col)
    # So the shared dim is 1. C[i,k] = A[i,0]*B[0,k]
    # C is 2x1. C[i,0] = A[i,0]*B[0,0]. Rank 1.

    def r1_2111(ar0, ar1, ac0, br0, bc0):
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac0),1)],
            [(b_idx(br0,bc0),1)],
            [(c_idx(bc0,ar0),1), (c_idx(bc0,ar1),1)]
        )

    # 2x1 * 1x2 -> 2x2 (schoolbook, rank 2)
    # A: 2 rows, 1 col; B: 1 row, 2 cols
    # C[i,k] = A[i,0]*B[0,k]. Rank 2 (one per k)
    def r1_2112(ar0, ar1, ac0, br0, bc0, bc1):
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac0),1)],
            [(b_idx(br0,bc0),1)],
            [(c_idx(bc0,ar0),1), (c_idx(bc0,ar1),1)]
        )
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac0),1)],
            [(b_idx(br0,bc1),1)],
            [(c_idx(bc1,ar0),1), (c_idx(bc1,ar1),1)]
        )

    # 2x1 * 1x1 -> 2x1 (rank 1)
    def r1_2111b(ar0, ar1, ac0, br0, bc0):
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac0),1)],
            [(b_idx(br0,bc0),1)],
            [(c_idx(bc0,ar0),1), (c_idx(bc0,ar1),1)]
        )

    # 1x2 * 2x2 -> 1x2 (schoolbook, rank 4)
    # A: 1 row (ar0), 2 cols (ac0,ac1); B: 2 rows (br0,br1), 2 cols (bc0,bc1)
    # C: 1 row, 2 cols
    def sch_122(ar0, ac0, ac1, br0, br1, bc0, bc1):
        add_rank1([(a_idx(ar0,ac0),1)], [(b_idx(br0,bc0),1)], [(c_idx(bc0,ar0),1)])
        add_rank1([(a_idx(ar0,ac1),1)], [(b_idx(br1,bc0),1)], [(c_idx(bc0,ar0),1)])
        add_rank1([(a_idx(ar0,ac0),1)], [(b_idx(br0,bc1),1)], [(c_idx(bc1,ar0),1)])
        add_rank1([(a_idx(ar0,ac1),1)], [(b_idx(br1,bc1),1)], [(c_idx(bc1,ar0),1)])

    # 1x2 * 2x1 -> 1x1 (rank 1)
    # A: 1 row, 2 cols; B: 2 rows, 1 col
    # C[ar0,bc0] = A[ar0,ac0]*B[br0,bc0] + A[ar0,ac1]*B[br1,bc0]
    def r1_1221(ar0, ac0, ac1, br0, br1, bc0):
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar0,ac1),1)],
            [(b_idx(br0,bc0),1), (b_idx(br1,bc0),1)],
            [(c_idx(bc0,ar0),1)]
        )

    # 1x2 * 1x2 -> 1x2 (rank 2)
    # A: 1 row, 2 cols; B: 1 row, 2 cols
    # C[ar0,bc_k] = A[ar0,ac_k]*B[br0,bc_k]... no, C[ar0,bc0] = A[ar0,ac0]*B[br0,bc0] + A[ar0,ac1]*B[br1,bc0]
    # Wait, B is 1x2, so B has 1 row. The shared dim is 1 (the single row of B = the single col of... 
    # No. A is 1x2 (1 row, 2 cols), B is 1x2 (1 row, 2 cols).
    # For matrix mult A*B, A is (1x2), B is (2x2)? No.
    # Let me reclarify: In the block decomposition, the "middle" dimension is split.
    # A has columns in the middle dim, B has rows in the middle dim.
    # If A is 1x2 (1 output row, 2 middle cols) and B is 2x2 (2 middle rows, 2 output cols)
    # then C is 1x2.
    # But if B is 1x2 (1 middle row, 2 output cols), then A must have 1 middle col.
    # So A is 1x1 for B to be 1x2? No...
    
    # Let me redo the block structure:
    # A is 4x5: rows = output_i (4), cols = middle_j (5)
    # B is 5x5: rows = middle_j (5), cols = output_k (5)
    # C is 4x5: rows = output_i (4), cols = output_k (5)
    #
    # Split: output_i: (2,2), middle_j: (2,2,1), output_k: (2,2,1)
    #
    # Block A[ri, rj]: shape (|ri| x |rj|)
    # Block B[rj, rk]: shape (|rj| x |rk|)
    # Block C[ri, rk]: shape (|ri| x |rk|)
    # C[ri,rk] = sum_{rj} A[ri,rj] * B[rj,rk]

    # So for A block (1x2) and B block (2x2): A has 1 output row, 2 middle cols
    # B has 2 middle rows, 2 output cols. C has 1 output row, 2 output cols.
    # C[i,k] = sum_j A[i,j]*B[j,k], j=0,1
    # That's 2 products per C entry, 2 C entries = 4 total. Rank 4.

    # For A block (1x2) and B block (2x1): A has 1 output row, 2 middle cols
    # B has 2 middle rows, 1 output col. C has 1 output row, 1 output col.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]. Rank 1.

    # For A block (1x2) and B block (1x2): A has 1 output row, 2 middle cols
    # B has 1 middle row, 2 output cols. C has 1 output row, 2 output cols.
    # C[0,k] = A[0,0]*B[0,k]. Only j=0 contributes since B has 1 middle row.
    # C[0,0] = A[0,0]*B[0,0], C[0,1] = A[0,0]*B[0,1]. Rank 2.

    # For A block (1x2) and B block (1x1): A has 1 output row, 2 middle cols
    # B has 1 middle row, 1 output col. C has 1 output row, 1 output col.
    # C[0,0] = A[0,0]*B[0,0]. Rank 1.

    # OK so my earlier analysis was slightly off. Let me redo all cases:
    # A block shape: (ra x ja) where ra = #output rows in block, ja = #middle cols in block
    # B block shape: (jb x kb) where jb = #middle rows in block, kb = #output cols in block
    # For valid mult, ja == jb.
    # C block shape: (ra x kb)
    # C[i,k] = sum_{j=0}^{ja-1} A[i,j]*B[j,k]
    # Rank = ra * ja * kb (schoolbook), but can be lower.

    # ra=1, ja=2, jb=2, kb=2: C is 1x2, rank 4 schoolbook
    # ra=1, ja=2, jb=2, kb=1: C is 1x1, rank 2 schoolbook -> rank 1
    # ra=1, ja=2, jb=1, kb=2: C is 1x2, rank 2 schoolbook -> rank 2 (C[i,k]=A[i,0]*B[0,k])
    # ra=1, ja=2, jb=1, kb=1: C is 1x1, rank 2 schoolbook -> rank 1 (C[0,0]=A[0,0]*B[0,0])
    # ra=1, ja=1, jb=1, kb=2: C is 1x2, rank 2 schoolbook -> rank 1 (C[i,k]=A[i,0]*B[0,k])
    # ra=1, ja=1, jb=1, kb=1: C is 1x1, rank 1
    # ra=2, ja=2, jb=2, kb=2: C is 2x2, rank 8 schoolbook -> rank 7 (Strassen)
    # ra=2, ja=2, jb=2, kb=1: C is 2x1, rank 4 schoolbook -> rank 4
    # ra=2, ja=2, jb=1, kb=2: C is 2x2, rank 4 schoolbook -> rank 4
    # ra=2, ja=2, jb=1, kb=1: C is 2x1, rank 4 schoolbook -> rank 1 (C[i,0]=B[0,0]*sum_j A[i,j])
    # ra=2, ja=1, jb=1, kb=2: C is 2x2, rank 4 schoolbook -> rank 2 (C[i,k]=A[i,0]*B[0,k])
    # ra=2, ja=1, jb=1, kb=1: C is 2x1, rank 2 schoolbook -> rank 1 (C[i,0]=A[i,0]*B[0,0])
    # ra=1, ja=1, jb=1, kb=1: C is 1x1, rank 1

    # Now enumerate all (ri, rj, rk) triples:
    # ri in {0,1} (block 0: rows 0-1, block 1: rows 2-3)
    # rj in {0,1,2} (block 0: cols 0-1, block 1: cols 2-3, block 2: col 4)
    # rk in {0,1,2} (block 0: cols 0-1, block 1: cols 2-3, block 2: col 4)
    #
    # Sizes:
    # ra: ri=0 -> 2, ri=1 -> 2
    # ja: rj=0 -> 2, rj=1 -> 2, rj=2 -> 1
    # jb: rj=0 -> 2, rj=1 -> 2, rj=2 -> 1
    # kb: rk=0 -> 2, rk=1 -> 2, rk=2 -> 1

    # Let me just write out all 2*3*3 = 18 terms and their ranks:
    # (ri,rj,rk) -> (ra,ja,kb) -> rank
    # (0,0,0) -> (2,2,2) -> 7 (Strassen)
    # (0,0,1) -> (2,2,2) -> 7 (Strassen)
    # (0,0,2) -> (2,2,1) -> 4
    # (0,1,0) -> (2,2,2) -> 7 (Strassen)
    # (0,1,1) -> (2,2,2) -> 7 (Strassen)
    # (0,1,2) -> (2,2,1) -> 4
    # (0,2,0) -> (2,1,2) -> 2
    # (0,2,1) -> (2,1,2) -> 2
    # (0,2,2) -> (2,1,1) -> 1
    # (1,0,0) -> (2,2,2) -> 7 (Strassen)
    # (1,0,1) -> (2,2,2) -> 7 (Strassen)
    # (1,0,2) -> (2,2,1) -> 4
    # (1,1,0) -> (2,2,2) -> 7 (Strassen)
    # (1,1,1) -> (2,2,2) -> 7 (Strassen)
    # (1,1,2) -> (2,2,1) -> 4
    # (1,2,0) -> (2,1,2) -> 2
    # (1,2,1) -> (2,1,2) -> 2
    # (1,2,2) -> (2,1,1) -> 1

    # Total: 8*7 + 4*4 + 4*2 + 2*1 = 56 + 16 + 8 + 2 = 82!

    # Great, rank 82! Let me implement this.

    col[0] = 0
    U = [[0] * 82 for _ in range(20)]
    V = [[0] * 82 for _ in range(25)]
    W = [[0] * 82 for _ in range(20)]
    RANK = 82

    def add_rank1(u_entries, v_entries, w_entries):
        c = col[0]
        for idx, val in u_entries:
            U[idx][c] += val
        for idx, val in v_entries:
            V[idx][c] += val
        for idx, val in w_entries:
            W[idx][c] += val
        col[0] += 1

    # Row offsets: ri=0 -> rows 0,1; ri=1 -> rows 2,3
    def a_row(ri, local_i):
        return ri * 2 + local_i

    # Col offsets: rj=0 -> cols 0,1; rj=1 -> cols 2,3; rj=2 -> col 4
    def a_col(rj, local_j):
        return [0, 2, 4][rj] + local_j

    def b_row(rj, local_j):
        return [0, 2, 4][rj] + local_j

    def b_col(rk, local_k):
        return [0, 2, 4][rk] + local_k

    def c_row(ri, local_i):
        return ri * 2 + local_i

    def c_col(rk, local_k):
        return [0, 2, 4][rk] + local_k

    # Strassen for 2x2 * 2x2 -> 2x2
    def do_strassen(ri, rj, rk):
        ar0, ar1 = a_row(ri, 0), a_row(ri, 1)
        ac0, ac1 = a_col(rj, 0), a_col(rj, 1)
        br0, br1 = b_row(rj, 0), b_row(rj, 1)
        bc0, bc1 = b_col(rk, 0), b_col(rk, 1)
        cr0, cr1 = c_row(ri, 0), c_row(ri, 1)
        ck0, ck1 = c_col(rk, 0), c_col(rk, 1)

        # M1 = (A00+A11)(B00+B11) -> C00+=, C11+=, C01-=, C10-=
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac1),1)],
            [(b_idx(br0,bc0),1), (b_idx(br1,bc1),1)],
            [(c_idx(ck0,cr0),1), (c_idx(ck1,cr1),1), (c_idx(ck1,cr0),-1), (c_idx(ck0,cr1),-1)]
        )
        # M2 = (A10+A11)B00 -> C10+=, C11+=
        add_rank1(
            [(a_idx(ar1,ac0),1), (a_idx(ar1,ac1),1)],
            [(b_idx(br0,bc0),1)],
            [(c_idx(ck0,cr1),1), (c_idx(ck1,cr1),1)]
        )
        # M3 = A00(B01-B11) -> C01+=, C11-=
        add_rank1(
            [(a_idx(ar0,ac0),1)],
            [(b_idx(br0,bc1),1), (b_idx(br1,bc1),-1)],
            [(c_idx(ck1,cr0),1), (c_idx(ck1,cr1),-1)]
        )
        # M4 = A11(B10-B00) -> C00-=, C10+=
        add_rank1(
            [(a_idx(ar1,ac1),1)],
            [(b_idx(br1,bc0),1), (b_idx(br0,bc0),-1)],
            [(c_idx(ck0,cr0),-1), (c_idx(ck0,cr1),1)]
        )
        # M5 = (A00+A01)B11 -> C00+=, C01+=
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar0,ac1),1)],
            [(b_idx(br1,bc1),1)],
            [(c_idx(ck0,cr0),1), (c_idx(ck1,cr0),1)]
        )
        # M6 = (A10-A00)(B00+B01) -> C10+=, C00-=
        add_rank1(
            [(a_idx(ar1,ac0),1), (a_idx(ar0,ac0),-1)],
            [(b_idx(br0,bc0),1), (b_idx(br0,bc1),1)],
            [(c_idx(ck0,cr1),1), (c_idx(ck0,cr0),-1)]
        )
        # M7 = (A01-A11)(B10+B11) -> C11+=, C01-=
        add_rank1(
            [(a_idx(ar0,ac1),1), (a_idx(ar1,ac1),-1)],
            [(b_idx(br1,bc0),1), (b_idx(br1,bc1),1)],
            [(c_idx(ck1,cr1),1), (c_idx(ck1,cr0),-1)]
        )

    # 2x2 * 2x1 -> 2x1 (rank 4)
    def do_221(ri, rj, rk):
        ar0, ar1 = a_row(ri, 0), a_row(ri, 1)
        ac0, ac1 = a_col(rj, 0), a_col(rj, 1)
        br0, br1 = b_row(rj, 0), b_row(rj, 1)
        bc0 = b_col(rk, 0)
        cr0, cr1 = c_row(ri, 0), c_row(ri, 1)
        ck0 = c_col(rk, 0)

        add_rank1([(a_idx(ar0,ac0),1)], [(b_idx(br0,bc0),1)], [(c_idx(ck0,cr0),1)])
        add_rank1([(a_idx(ar0,ac1),1)], [(b_idx(br1,bc0),1)], [(c_idx(ck0,cr0),1)])
        add_rank1([(a_idx(ar1,ac0),1)], [(b_idx(br0,bc0),1)], [(c_idx(ck0,cr1),1)])
        add_rank1([(a_idx(ar1,ac1),1)], [(b_idx(br1,bc0),1)], [(c_idx(ck0,cr1),1)])

    # 2x1 * 1x2 -> 2x2 (rank 2)
    # C[i,k] = A[i,0]*B[0,k]
    def do_212(ri, rj, rk):
        ar0, ar1 = a_row(ri, 0), a_row(ri, 1)
        ac0 = a_col(rj, 0)
        br0 = b_row(rj, 0)
        bc0, bc1 = b_col(rk, 0), b_col(rk, 1)
        cr0, cr1 = c_row(ri, 0), c_row(ri, 1)
        ck0, ck1 = c_col(rk, 0), c_col(rk, 1)

        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac0),1)],
            [(b_idx(br0,bc0),1)],
            [(c_idx(ck0,cr0),1), (c_idx(ck0,cr1),1)]
        )
        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac0),1)],
            [(b_idx(br0,bc1),1)],
            [(c_idx(ck1,cr0),1), (c_idx(ck1,cr1),1)]
        )

    # 2x1 * 1x1 -> 2x1 (rank 1)
    # C[i,0] = A[i,0]*B[0,0]
    def do_211(ri, rj, rk):
        ar0, ar1 = a_row(ri, 0), a_row(ri, 1)
        ac0 = a_col(rj, 0)
        br0 = b_row(rj, 0)
        bc0 = b_col(rk, 0)
        cr0, cr1 = c_row(ri, 0), c_row(ri, 1)
        ck0 = c_col(rk, 0)

        add_rank1(
            [(a_idx(ar0,ac0),1), (a_idx(ar1,ac0),1)],
            [(b_idx(br0,bc0),1)],
            [(c_idx(ck0,cr0),1), (c_idx(ck0,cr1),1)]
        )

    # Now dispatch all 18 blocks
    for ri in range(2):
        for rj in range(3):
            for rk in range(3):
                ja = [2, 2, 1][rj]
                kb = [2, 2, 1][rk]
                if ja == 2 and kb == 2:
                    do_strassen(ri, rj, rk)
                elif ja == 2 and kb == 1:
                    do_221(ri, rj, rk)
                elif ja == 1 and kb == 2:
                    do_212(ri, rj, rk)
                elif ja == 1 and kb == 1:
                    do_211(ri, rj, rk)

    assert col[0] == RANK, f"Expected {RANK} terms, got {col[0]}"
    return {"U": U, "V": V, "W": W}
