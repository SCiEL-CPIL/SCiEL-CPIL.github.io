"""Strassen-based rank reduction for 3x4 @ 4x7 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 7]

def _strassen222():
    """Return (U,V,W) for 2x2x2 Strassen, rank 7."""
    n, m, p = 2, 2, 2
    # M1 = (A11+A22)*(B12+B21)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B21-B22)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U = []
    V = []
    W = []
    # U rows: i*m+j = i*2+j, i in 0..1, j in 0..1
    # V rows: j*p+k = j*2+k
    # W rows: k*n+i = k*2+i
    
    # M1: (A11+A22)*(B12+B21)
    # A11: i=0,j=0; A22: i=1,j=1
    # B12: j=0,k=1; B21: j=1,k=0
    U.append([1,0,0,0])  # A11
    V.append([0,1,0,0])  # B12
    W.append([1*2+0, 0, 0, 0, 0, 0, 0, 0])  # C11: k=1,i=0
    # Wait, W is a matrix of shape (p*n, rank) = (4, 7)
    # W[k*n+i, c]
    
    # Let me build this more carefully as arrays
    import numpy as np
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # M1 = (A11+A22)*(B12+B21) -> C11 += 1, C22 -= 1
    U[0,0] += 1  # A11
    U[3,0] += 1  # A22
    V[1,0] += 1  # B12
    V[2,0] += 1  # B21
    W[1*2+0, 0] += 1   # C11: k=1,i=0
    W[0*2+1, 0] -= 1   # C22: k=0,i=1
    
    # M2 = (A21+A22)*B11 -> C21 += 1, C22 -= 1
    U[2,1] += 1  # A21
    U[3,1] += 1  # A22
    V[0,1] += 1  # B11
    W[0*2+1, 1] += 1   # C21: k=0,i=1
    W[1*2+1, 1] -= 1   # C22: k=1,i=1
    
    # M3 = A11*(B11-B21) -> C12 += 1, C22 += 1
    U[0,2] += 1  # A11
    V[0,2] += 1  # B11
    V[2,2] -= 1  # B21
    W[0*2+0, 2] += 1   # C12: k=0,i=0
    W[1*2+1, 2] += 1   # C22: k=1,i=1
    
    # M4 = A22*(B21-B22) -> C11 += 1, C21 += 1
    U[3,3] += 1  # A22
    V[2,3] += 1  # B21
    V[3,3] -= 1  # B22
    W[1*2+0, 3] += 1   # C11: k=1,i=0
    W[0*2+1, 3] += 1   # C21: k=0,i=1
    
    # M5 = (A11+A12)*B22 -> C11 -= 1, C12 += 1
    U[0,4] += 1  # A11
    U[1,4] += 1  # A12
    V[3,4] += 1  # B22
    W[1*2+0, 4] -= 1   # C11: k=1,i=0
    W[0*2+0, 4] += 1   # C12: k=0,i=0
    
    # M6 = (A21-A11)*(B11+B12) -> C22 += 1
    U[2,5] += 1  # A21
    U[0,5] -= 1  # A11
    V[0,5] += 1  # B11
    V[1,5] += 1  # B12
    W[1*2+1, 5] += 1   # C22: k=1,i=1
    
    # M7 = (A12-A22)*(B21+B22) -> C11 += 1
    U[1,6] += 1  # A12
    U[3,6] -= 1  # A22
    V[2,6] += 1  # B21
    V[3,6] += 1  # B22
    W[1*2+0, 6] += 1   # C11: k=1,i=0
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = 120 - 5  # 5s safety margin
    
    n, m, p = 3, 4, 7
    # Schoolbook rank = 84
    # Use Strassen where possible to reduce rank
    
    # Strategy: split 3x4 @ 4x7 into blocks
    # A: 3x4, B: 4x7, C: 3x7
    # 
    # Split A into top 2 rows and bottom 1 row
    # A_top: 2x4, A_bot: 1x4
    # Split B into left 4 cols and right 3 cols  
    # B_left: 4x4, B_right: 4x3
    #
    # C_top_left = A_top @ B_left  (2x4)
    # C_top_right = A_top @ B_right (2x3)
    # C_bot_left = A_bot @ B_left  (1x4)
    # C_bot_right = A_bot @ B_right (1x3)
    
    # For each sub-problem, use Strassen decomposition
    # M(2,4,4): split 4x4 into four 2x2 blocks, use 4 Strassen decompositions
    # Total rank = 4 * 7 = 28
    # M(2,3,4): split into (2,2,2)+(2,1,2) = 7 + 4 = 11? Or use different approach
    # M(1,4,4): 1x4 @ 4x4 = 1x4, schoolbook rank 4, or use Strassen? 
    # Actually for 1xm @ mxp, rank is just p (each output element is independent)
    # Wait no: 1x4 @ 4x4: output is 1x4, each element is dot product of row of A with col of B
    # Rank = 4 (schoolbook is optimal for this case since no repeated subexpressions)
    
    # Let me think differently. Use the known Strassen decomposition for 2x2x2
    # and compose it.
    
    # For M(n,m,p), if n=2, m is even, p is even:
    # Split A into 2 x (m/2) blocks, B into (m/2) x (p/2) blocks
    # Use Strassen on each 2x2x2 sub-block
    
    # M(2,4,4): 
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    # Use 4 independent Strassen decompositions, rank = 4*7 = 28
    
    # M(2,4,3):
    # This is trickier. Split into M(2,2,2) + M(2,2,1) + M(2,2,2)? No.
    # Let's just use schoolbook for the odd dimensions or find a decomposition.
    # Actually, let's split B (4x3) into left 2 cols and right 1 col
    # M(2,4,2) + M(2,4,1)
    # M(2,4,2): A(2x4) @ B(4x2). Split A into 2x2 + 2x2, B into 2x2 + 2x2
    # = 2 Strassen decompositions, rank = 14
    # M(2,4,1): A(2x4) @ B(4x1). Output is 2x1. 
    # Schoolbook rank = 2*4*1 = 8
    # Can we do better? Each output element is a dot product.
    # C1 = sum of 4 products, C2 = sum of 4 products
    # Rank 8 schoolbook. Probably can't reduce much.
    
    # Let me just implement a general approach:
    # For each sub-block, if all dimensions are >= 2 and even where possible, use Strassen
    # Otherwise use schoolbook
    
    # Simpler approach: just use Strassen for 2x2x2 blocks and schoolbook elsewhere
    # and concatenate.
    
    # Let me build the full decomposition by decomposing into 2x2x2 Strassen blocks
    # and 1x1x1 schoolbook blocks.
    
    # Actually, the cleanest approach:
    # Split A(3x4) into three 1x4 rows? No, that gives rank = 3*4*7 = 84.
    
    # Better: Use the Strassen 7-multiplication formula for 2x2x2.
    # 
    # Split the problem as:
    # Rows 0,1 of A and all of B: M(2,4,7)
    # Row 2 of A and all of B: M(1,4,7)
    
    # M(2,4,7): Split B into 4+3 columns
    # M(2,4,4) + M(2,4,3)
    # M(2,4,4) = 4 * M(2,2,2) = 4*7 = 28
    # M(2,4,3): Split B into 2+1 cols
    # M(2,4,2) + M(2,4,1)
    # M(2,4,2) = 2 * M(2,2,2) = 14
    # M(2,4,1) = schoolbook = 8
    # Total for M(2,4,7) = 28 + 14 + 8 = 50
    
    # M(1,4,7) = schoolbook = 28
    
    # Total rank = 50 + 28 = 78
    
    # Let me implement this.
    
    sU, sV, sW = _strassen222()
    
    # Helper to add a Strassen block with offsets
    def add_strassen(U_list, V_list, W_list, rank_offset, i_off, j_off, k_off, i_dim, j_dim, k_dim):
        """Add a Strassen 2x2x2 decomposition at given offsets.
        i_off, i_dim: which rows of A (offset and dimension, should be 2)
        j_off, j_dim: which cols of A / rows of B (offset and dimension, should be 2)
        k_off, k_dim: which cols of B / rows of C (offset and dimension, should be 2)
        """
        for c in range(7):
            for i in range(4):
                val = sU[i][c]
                if val != 0:
                    idx = (i_off + i // 2) * 4 + (j_off + i % 2)  # i*m+j, m=4
                    U_list[idx][rank_offset + c] += val
            for j in range(4):
                val = sV[j][c]
                if val != 0:
                    idx = (j_off + j // 2) * 7 + (k_off + j % 2)  # j*p+k, p=7
                    V_list[idx][rank_offset + c] += val
            for k in range(4):
                val = sW[k][c]
                if val != 0:
                    idx = (k_off + k // 2) * 3 + (i_off + k % 2)  # k*n+i, n=3
                    W_list[idx][rank_offset + c] += val
        return rank_offset + 7
    
    # Initialize with rank 78
    total_rank = 78
    U = [[0] * total_rank for _ in range(12)]  # 3*4
    V = [[0] * total_rank for _ in range(28)]  # 4*7
    W = [[0] * total_rank for _ in range(21)]  # 7*3
    
    rank = 0
    
    # M(2,4,4): i_off=0, j_off=0, k_off=0
    # Split into 4 Strassen 2x2x2 blocks
    for jj in range(0, 4, 2):
        for kk in range(0, 4, 2):
            rank = add_strassen(U, V, W, rank, 0, jj, kk, 2, 2, 2)
    
    # M(2,4,2): i_off=0, j_off=0, k_off=4
    # Split into 2 Strassen 2x2x2 blocks
    for jj in range(0, 4, 2):
        rank = add_strassen(U, V, W, rank, 0, jj, 4, 2, 2, 2)
    
    # M(2,4,1): i_off=0, j_off=0, k_off=6
    # Schoolbook: 2*4*1 = 8
    for i in range(2):
        for j in range(4):
            U[i * 4 + j][rank] = 1
            V[j * 7 + 6][rank] = 1
            W[6 * 3 + i][rank] = 1
            rank += 1
    
    # M(1,4,7): i_off=2, j_off=0, k_off=0
    # Schoolbook: 1*4*7 = 28
    for j in range(4):
        for k in range(7):
            U[2 * 4 + j][rank] = 1
            V[j * 7 + k][rank] = 1
            W[k * 3 + 2][rank] = 1
            rank += 1
    
    assert rank == total_rank, f"Rank mismatch: {rank} vs {total_rank}"
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen since this is deterministic and fast
        pass
    
    return {"U": U, "V": V, "W": W}
