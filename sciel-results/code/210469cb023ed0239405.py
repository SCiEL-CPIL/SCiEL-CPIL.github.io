"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s kill

    n, m, p = DIMENSIONS  # 4, 5, 5
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]

    def strassen_2x2():
        """Return (U, V, W) for 2x2x2 Strassen decomposition.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Convention: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        rank = 7
        U = np.zeros((4, rank), dtype=np.float64)
        V = np.zeros((4, rank), dtype=np.float64)
        W = np.zeros((4, rank), dtype=np.float64)
        
        # A indices: (i,j) -> i*2+j, i,j in {0,1}
        # B indices: (j,k) -> j*2+k, j,k in {0,1}
        # C indices: (i,k) -> k*2+i (trace convention: W[k*n+i])
        #   n=2, so W[k*2+i]
        
        # P1 = (A11 + A22)(B11 + B22)
        # A11=(0,0), A22=(1,1), B11=(0,0), B22=(1,1)
        U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
        V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
        W[0*2+0, 0] = 1; W[0*2+1, 0] = 1; W[1*2+0, 0] = 1; W[1*2+1, 0] = 1
        # C11 += P1, C12 += P1, C21 += P1, C22 += P1
        
        # P2 = (A21 + A22)B11
        # A21=(1,0), A22=(1,1), B11=(0,0)
        U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
        V[0*2+0, 1] = 1
        W[0*2+1, 1] = 1; W[1*2+1, 1] = 1
        # C21 += P2, C22 += P2
        
        # P3 = A11(B12 - B22)
        # A11=(0,0), B12=(0,1), B22=(1,1)
        U[0*2+0, 2] = 1
        V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
        W[0*2+0, 2] = 1; W[0*2+1, 2] = 1
        # C11 += P3, C12 += P3
        
        # P4 = A22(B21 - B11)
        # A22=(1,1), B21=(1,0), B11=(0,0)
        U[1*2+1, 3] = 1
        V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
        W[0*2+1, 3] = 1; W[1*2+1, 3] = 1
        # C21 += P4, C22 += P4
        
        # P5 = (A11 + A12)B22
        # A11=(0,0), A12=(0,1), B22=(1,1)
        U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
        V[1*2+1, 4] = 1
        W[1*2+0, 4] = 1; W[1*2+1, 4] = 1
        # C21 += P5, C22 += P5
        
        # P6 = (A21 - A11)(B11 + B12)
        # A21=(1,0), A11=(0,0), B11=(0,0), B12=(0,1)
        U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
        V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
        W[0*2+0, 5] = 1; W[1*2+0, 5] = 1
        # C11 += P6, C21 += P6
        
        # P7 = (A12 - A22)(B21 + B22)
        # A12=(0,1), A22=(1,1), B21=(1,0), B22=(1,1)
        U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
        V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
        W[0*2+1, 6] = 1; W[1*2+1, 6] = 1
        # C12 += P7, C22 += P7
        
        return U, V, W

    def strassen_4x4():
        """Return (U, V, W) for 4x4x4 using Strassen on 2x2 blocks.
        U: (16, 49), V: (16, 49), W: (16, 49)
        Convention: U[i*4+j], V[j*4+k], W[k*4+i]
        """
        # The 4x4 matrix is split into four 2x2 blocks.
        # Block (a,b) where a,b in {0,1} covers rows/cols 2a..2a+1, 2b..2b+1.
        # For A (4x4): block (a,b) has indices i in 2a..2a+1, j in 2b..2b+1.
        #   Linear index: i*4+j
        # For B (4x4): block (b,c) has indices j in 2b..2b+1, k in 2c..2c+1.
        #   Linear index: j*4+k
        # For C (4x4): block (a,c) has indices i in 2a..2a+1, k in 2c..2c+1.
        #   Linear index: k*4+i (trace convention with n=4)
        
        # We use 7 Strassen products, each involving 2x2 block matrices.
        # Each 2x2x2 Strassen uses rank 7.
        # Total rank = 7 * 7 = 49.
        
        sU, sV, sW = strassen_2x2()
        
        rank = 49
        U = np.zeros((16, rank), dtype=np.float64)
        V = np.zeros((16, rank), dtype=np.float64)
        W = np.zeros((16, rank), dtype=np.float64)
        
        # Strassen formulas for 2x2 block matrices:
        # P1 = (A00 + A11)(B00 + B11)
        # P2 = (A10 + A11)B00
        # P3 = A00(B01 - B11)
        # P4 = A11(B10 - B00)
        # P5 = (A00 + A01)B11
        # P6 = (A10 - A00)(B00 + B01)
        # P7 = (A01 - A11)(B10 + B11)
        #
        # C00 = P1 + P3 + P6 - P4 - P5
        # C01 = P3 + P7
        # C10 = P2 + P5
        # C11 = P1 - P2 + P4 + P7
        
        # For each Strassen product s (0..6) and each scalar rank r (0..6):
        # We need to map block-level entries to scalar-level entries.
        
        # Let's define the block structure:
        # A block (a, b): rows 2a..2a+1, cols 2b..2b+1 of A
        # B block (b, c): rows 2b..2b+1, cols 2c..2c+1 of B  
        # C block (a, c): rows 2c..2c+1, cols 2a..2a+1 of C (due to trace convention)
        
        # For Strassen product s with scalar rank r:
        # The U part involves A blocks, the V part involves B blocks, the W part involves C blocks.
        
        # Let me define the block indices for each Strassen product:
        # s=0 (P1): A blocks {00, 11}, B blocks {00, 11}, C blocks {00, 01, 10, 11}
        # s=1 (P2): A blocks {10, 11}, B blocks {00}, C blocks {10, 11}
        # s=2 (P3): A blocks {00}, B blocks {01, 11}, C blocks {00, 01}
        # s=3 (P4): A blocks {11}, B blocks {10, 00}, C blocks {10, 11}
        # s=4 (P5): A blocks {00, 01}, B blocks {11}, C blocks {10, 11}
        # s=5 (P6): A blocks {10, 00}, B blocks {00, 01}, C blocks {00, 10}
        # s=6 (P7): A blocks {01, 11}, B blocks {10, 11}, C blocks {01, 11}
        
        # And the coefficients from strassen_2x2:
        # sU[s*2+i_local, r] gives the A-block coefficient
        # sV[s*2+j_local, r] gives the B-block coefficient  
        # sW[s*2+k_local, r] gives the C-block coefficient
        
        # Actually, let me just directly construct it.
        # For each of the 7 Strassen products and each of the 7 scalar ranks:
        
        for s in range(7):
            for r in range(7):
                col = s * 7 + r
                
                # A coefficients: sU gives 4x7, indexed by [i*2+j, r]
                # We need to map to 4x4 A blocks.
                # sU[0, r] -> A00 block, sU[1, r] -> A01 block, sU[2, r] -> A10 block, sU[3, r] -> A11 block
                # Wait, sU is (4, 7) with index i*2+j where i,j are 0..1 within the 2x2 block space.
                # So sU[i*2+j, r] is the coefficient for A_{i,j} in the 2x2 block matrix.
                
                a00 = sU[0*2+0, r]  # A00 block
                a01 = sU[0*2+1, r]  # A01 block
                a10 = sU[1*2+0, r]  # A10 block
                a11 = sU[1*2+1, r]  # A11 block
                
                b00 = sV[0*2+0, r]  # B00 block
                b01 = sV[0*2+1, r]  # B01 block
                b10 = sV[1*2+0, r]  # B10 block
                b11 = sV[1*2+1, r]  # B11 block
                
                c00 = sW[0*2+0, r]  # C00 block
                c01 = sW[0*2+1, r]  # C01 block
                c10 = sW[1*2+0, r]  # C10 block
                c11 = sW[1*2+1, r]  # C11 block
                
                # Now expand each block coefficient to scalar level.
                # A block (a, b) with coefficient val:
                #   For di, dj in 0..1: U[(2a+di)*4 + (2b+dj)] += val
                # B block (b, c) with coefficient val:
                #   For dj, dk in 0..1: V[(2b+dj)*4 + (2c+dk)] += val
                # C block (a, c) with coefficient val:
                #   For di, dk in 0..1: W[(2c+dk)*4 + (2a+di)] += val
                #     (because W index is k*4+i, and C block (a,c) has i in 2a..2a+1, k in 2c..2c+1)
                
                if a00 != 0:
                    for di in range(2):
                        for dj in range(2):
                            U[(2*0+di)*4 + (2*0+dj), col] += a00
                if a01 != 0:
                    for di in range(2):
                        for dj in range(2):
                            U[(2*0+di)*4 + (2*1+dj), col] += a01
                if a10 != 0:
                    for di in range(2):
                        for dj in range(2):
                            U[(2*1+di)*4 + (2*0+dj), col] += a10
                if a11 != 0:
                    for di in range(2):
                        for dj in range(2):
                            U[(2*1+di)*4 + (2*1+dj), col] += a11
                
                if b00 != 0:
                    for dj in range(2):
                        for dk in range(2):
                            V[(2*0+dj)*4 + (2*0+dk), col] += b00
                if b01 != 0:
                    for dj in range(2):
                        for dk in range(2):
                            V[(2*0+dj)*4 + (2*1+dk), col] += b01
                if b10 != 0:
                    for dj in range(2):
                        for dk in range(2):
                            V[(2*1+dj)*4 + (2*0+dk), col] += b10
                if b11 != 0:
                    for dj in range(2):
                        for dk in range(2):
                            V[(2*1+dj)*4 + (2*1+dk), col] += b11
                
                if c00 != 0:
                    for di in range(2):
                        for dk in range(2):
                            W[(2*0+dk)*4 + (2*0+di), col] += c00
                if c01 != 0:
                    for di in range(2):
                        for dk in range(2):
                            W[(2*1+dk)*4 + (2*0+di), col] += c01
                if c10 != 0:
                    for di in range(2):
                        for dk in range(2):
                            W[(2*0+dk)*4 + (2*1+di), col] += c10
                if c11 != 0:
                    for di in range(2):
                        for dk in range(2):
                            W[(2*1+dk)*4 + (2*1+di), col] += c11
        
        return U, V, W

    def schoolbook_block(n_block, m_block, p_block, offset_i, offset_j, offset_k, U, V, W, col_start):
        """Add schoolbook decomposition for a sub-block.
        A sub-block: n_block x m_block x p_block
        A indices: i in offset_i..offset_i+n_block-1, j in offset_j..offset_j+m_block-1
        B indices: j in offset_j..offset_j+m_block-1, k in offset_k..offset_k+p_block-1
        C indices: k in offset_k..offset_k+p_block-1, i in offset_i..offset_i+n_block-1
        """
        rank_block = n_block * m_block * p_block
        col = col_start
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[(offset_i + i) * m + (offset_j + j), col] = 1
                    V[(offset_j + j) * p + (offset_k + k), col] = 1
                    W[(offset_k + k) * n + (offset_i + i), col] = 1
                    col += 1
        return col_start + rank_block

    # Build the full 4x5x5 decomposition:
    # Part 1: 4x4x4 (j=0..3, k=0..3) - use Strassen, rank 49
    # Part 2: 4x4x1 (j=0..3, k=4) - schoolbook, rank 16
    # Part 3: 4x1x4 (j=4, k=0..3) - schoolbook, rank 16
    # Part 4: 4x1x1 (j=4, k=4) - schoolbook, rank 4
    # Total rank: 49 + 16 + 16 + 4 = 85
    
    # Wait, let me reconsider. Can I do better on the cross terms?
    # Part 2: 4x4x1: A is 4x4, B is 4x1. Product is 4x1. Rank = 4*4*1 = 16 for schoolbook.
    # Part 3: 4x1x4: A is 4x1, B is 1x4. Product is 4x4. Rank = 4*1*4 = 16 for schoolbook.
    # Part 4: 4x1x1: A is 4x1, B is 1x1. Product is 4x1. Rank = 4*1*1 = 4 for schoolbook.
    
    # Total: 49 + 16 + 16 + 4 = 85. Better than 100.
    
    # Can I improve further? 
    # Part 2 (4x4x1): This is like computing C_i = sum_j A_ij * B_j. 
    # This is a 4x4 matrix times a 4-vector. The tensor is (4,4,1).
    # Rank of (4,4,1) tensor: Since one dimension is 1, rank = 4*4 = 16. No improvement possible.
    
    # Part 3 (4x1x4): C_{i,k} = A_i * B_k. This is an outer product.
    # Rank of (4,1,4) tensor: rank = 4*4 = 16. No improvement possible.
    
    # So total rank = 85.
    
    total_rank = 49 + 16 + 16 + 4
    U_full = np.zeros((n*m, total_rank), dtype=np.float64)
    V_full = np.zeros((m*p, total_rank), dtype=np.float64)
    W_full = np.zeros((p*n, total_rank), dtype=np.float64)
    
    # Part 1: 4x4x4 using Strassen
    U_s, V_s, W_s = strassen_4x4()
    U_full[0:16, 0:49] = U_s
    V_full[0:16, 0:49] = V_s
    W_full[0:16, 0:49] = W_s
    
    # Part 2: 4x4x1 (j=0..3, k=4)
    # A: 4x4 (i=0..3, j=0..3), B: 4x1 (j=0..3, k=4), C: 1x4 (k=4, i=0..3)
    # W index: k*n+i = 4*4+i = 16+i
    col_start = 49
    for i in range(4):
        for j in range(4):
            for k in range(1):  # k=4
                U_full[i * 5 + j, col_start] = 1
                V_full[j * 5 + (4), col_start] = 1
                W_full[(4) * 4 + i, col_start] = 1
                col_start += 1
    
    # Part 3: 4x1x4 (j=4, k=0..3)
    # A: 4x1 (i=0..3, j=4), B: 1x4 (j=4, k=0..3), C: 4x4 (k=0..3, i=0..3)
    for i in range(4):
        for j in range(1):  # j=4
            for k in range(4):  # k=0..3
                U_full[i * 5 + (4), col_start] = 1
                V_full[(4) * 5 + k, col_start] = 1
                W_full[k * 4 + i, col_start] = 1
                col_start += 1
    
    # Part 4: 4x1x1 (j=4, k=4)
    for i in range(4):
        for j in range(1):  # j=4
            for k in range(1):  # k=4
                U_full[i * 5 + (4), col_start] = 1
                V_full[(4) * 5 + (4), col_start] = 1
                W_full[(4) * 4 + i, col_start] = 1
                col_start += 1
    
    # Verify col_start == total_rank
    assert col_start == total_rank, f"Column mismatch: {col_start} vs {total_rank}"
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in U_full]
    V_list = [[float(x) for x in row] for row in V_full]
    W_list = [[float(x) for x in row] for row in W_full]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
