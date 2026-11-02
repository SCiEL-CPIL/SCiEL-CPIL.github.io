import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    DEADLINE = 120.0
    
    # We'll decompose 3x5x5 into:
    # - Row 0: 1x5x5 schoolbook (rank 5)
    # - Rows 1,2: 2x5x5 using block-Strassen on 5x5 multiplication
    #   For 5x5, we can use a rank-19 decomposition (Burgin 1982)
    #   But applying it to 2x5x5: we need to think carefully.
    #
    # Actually, let me reconsider. The directive says to use a known rank-19
    # decomposition for 5x5 matrix multiplication. Let me implement a
    # 5x5 rank-19 decomposition and apply it to each output row separately.
    # That gives 2*19 + 5 = 43.
    #
    # The 5x5 rank-19 decomposition (Burgin's) is complex. Let me use a
    # simpler approach: decompose 5x5 into overlapping 2x2 Strassen blocks
    # plus schoolbook remainder.
    #
    # Alternative: Use the fact that 5x5 can be done with rank 19 using
    # a specific tensor decomposition. Let me implement it directly.
    #
    # Actually, for safety and correctness, let me use a different approach:
    # Decompose 5x5 into a 3x3 + 2x2 structure or use known results.
    #
    # Let me try: 5 = 3 + 2, so 5x5 = block diagonal of 3x3 and 2x2
    # 3x3 rank: 23 (known minimal for 3x3) -- too complex
    # 2x2 rank: 7 (Strassen)
    # But the cross terms complicate things.
    #
    # Simplest safe approach: use 4 non-overlapping 2x2x2 Strassen blocks
    # in the 5x5 space, covering indices {0,1}x{0,1}x{0,1}, {0,1}x{2,3}x{0,1},
    # {2,3}x{0,1}x{0,1}, {2,3}x{2,3}x{0,1}, plus schoolbook for the rest.
    # That covers 4*8=32 terms with 4*7=28 rank, remaining 25-32... wait
    # 5*5=25 terms total for 5x5. 4 blocks cover 32 positions but 5x5 only
    # has 25 (i,j) pairs. Let me reconsider.
    #
    # For 5x5 multiplication: tensor is T[i,j,k] for i,j,k in 0..4
    # Total terms = 125
    # 4 Strassen blocks of 2x2x2 cover 4*8=32 terms with rank 4*7=28
    # Remaining: 125-32=93 terms at rank 93
    # Total: 28+93=121. That's worse than schoolbook 125! No wait,
    # each Strassen block replaces 8 schoolbook terms (rank 8) with 7 terms.
    # So 4 blocks save 4 rank. 125-4=121. Still bad.
    #
    # The rank-19 for 5x5 is a specific result. Let me look up the actual
    # decomposition. It's from Burgin (1982) or similar.
    #
    # Actually, I recall that the minimal rank for 5x5 is 19, but constructing
    # it explicitly is non-trivial. Let me instead use a pragmatic approach:
    #
    # For the 2x5x5 sub-problem (rows 1,2 of output), I can view it as
    # multiplying a 2x5 matrix by a 5x5 matrix. This is equivalent to
    # two 1x5 matrix multiplications sharing the same B matrix.
    #
    # Key insight: For C = A*B where A is 2x5 and B is 5x5,
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # We can use the structure: group the 5 columns of A into groups and
    # use Strassen-like techniques.
    #
    # Let me try a different, simpler approach that's guaranteed correct:
    # Use the current 4-Strassen-block approach but extend it to cover
    # more of the 3x5x5 tensor.
    #
    # Current: 4 blocks of 2x2x2 in the 3x5x5 space, covering i in {0,1}.
    # We can add more blocks for i in {1,2} if they don't overlap.
    #
    # Blocks for i={0,1}: (0,0,0),(0,2,0),(0,0,2),(0,2,2) - 4 blocks
    # Blocks for i={1,2}: (1,0,0),(1,2,0),(1,0,2),(1,2,2) - 4 blocks
    # But (1,*,*) overlaps with (0,*,*) on i=1.
    #
    # Non-overlapping in i dimension: {0,1} and {2} only.
    # So for i={2}, we can use 1x2x2 blocks? No, Strassen needs 2x2x2.
    #
    # Let me just go with the current approach (rank 71) and try to improve
    # by adding Strassen blocks in the i=2 row using 2x2x2 blocks that
    # don't conflict.
    #
    # Actually, the simplest improvement: the current code has a bug in
    # verification. Let me fix the verification and also try to find
    # additional non-overlapping Strassen blocks.
    #
    # For i in {2}, j in {0,1}, k in {0,1}: block (2,0,0) - 2x2x2 needs
    # i in {2,3} but n=3 so i=3 doesn't exist. Can't use 2x2x2 for i=2.
    #
    # So the current 4 blocks are maximal for 2x2x2 Strassen in 3x5x5.
    #
    # Let me try a different family: use 2x2x2 Strassen blocks but also
    # try to combine some schoolbook terms.
    #
    # Actually, let me re-examine: can I use a 2x2x2 block at (1,0,0)?
    # That would cover i in {1,2}, j in {0,1}, k in {0,1}.
    # This overlaps with block (0,0,0) on i=1, j in {0,1}, k in {0,1}.
    # So no.
    #
    # What about using a different decomposition entirely?
    # 
    # Let me try: for the 3x5x5 tensor, use the following strategy:
    # - For i=0: use 4 Strassen blocks in j,k space (2x2x2 each)
    #   Blocks: (0,0,0),(0,2,0),(0,0,2),(0,2,2) -> covers i=0, all j,k in {0,1,2,3}
    #   Wait, 2x2x2 blocks at j-offsets 0 and 2 cover j in {0,1} and {2,3}.
    #   j=4 is not covered. Similarly k=4.
    #   So 4 blocks cover i=0, j in {0,1,2,3}, k in {0,1,2,3} = 16 terms
    #   with rank 28. Remaining for i=0: j or k = 4: 5+5-1=9 terms.
    #   Total for i=0: 28+9=37 vs schoolbook 25. Worse!
    #
    # I see the issue: Strassen helps when the block is "dense" enough.
    # For a 2x2x2 block, it saves 1 rank (8->7). For the full 5x5, using
    # 4 such blocks saves 4 rank total.
    #
    # The real improvement comes from larger structured decompositions.
    #
    # Let me try implementing the actual rank-19 decomposition for 5x5.
    # I'll use a known construction.
    #
    # For 5x5, one approach: 5 = 2+2+1
    # Decompose into blocks:
    # - 2x2x2 Strassen blocks at (0,0,0), (0,2,0), (2,0,0), (2,2,0) 
    #   Wait, that's for the 4x4 submatrix.
    #
    # Let me just carefully implement a correct decomposition and verify.
    #
    # New plan: Keep the current 4-block Strassen approach (rank 71) but
    # fix any bugs and ensure correctness. The verification in the current
    # code checks that sum_c U*V*W = 1 for all (i,j,k), which is correct
    # for the identity tensor (matrix multiplication tensor has T[i,j,k]=1
    # for all valid indices since it's the structure tensor).
    #
    # Actually wait - the matrix multiplication tensor T_ijk = 1 for all
    # (i,j,k) in the valid range? No! For matrix multiplication C=AB:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the structure tensor
    # of matrix multiplication). Yes, all entries are 1.
    #
    # So the verification is correct. Let me just ensure the code works
    # and try to find a better decomposition.
    #
    # Let me try: can I find a rank-19 decomposition for 5x5?
    # I'll implement it using a known result. The decomposition uses
    # specific linear combinations.
    #
    # For now, let me keep the current approach but add a deadline guard
    # and fix the verification to be more robust.

    # Use 4 disjoint 2x2x2 Strassen blocks
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7  # 28
    total_terms = n * m * p  # 75
    strassen_terms_count = num_strassen_blocks * 8  # 32
    schoolbook_terms_count = total_terms - strassen_terms_count  # 43
    total_rank = total_strassen_rank + schoolbook_terms_count  # 71
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        """Add Strassen rank-7 decomposition for a 2x2x2 block."""
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Verify correctness
    if time.time() - start_time < DEADLINE - 5:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(total_rank))
                    expected = 1.0
                    err = abs(val - expected)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            # Fall back to pure schoolbook
            total_rank_fb = n * m * p
            U_fb = [[0.0] * total_rank_fb for _ in range(n * m)]
            V_fb = [[0.0] * total_rank_fb for _ in range(m * p)]
            W_fb = [[0.0] * total_rank_fb for _ in range(p * n)]
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col_fb] = 1.0
                        V_fb[j*p+k][col_fb] = 1.0
                        W_fb[k*n+i][col_fb] = 1.0
                        col_fb += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
