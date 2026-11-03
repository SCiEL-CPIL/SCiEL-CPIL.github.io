import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10
    rank_school = n * m * p  # 40

    # Build trivial (schoolbook) decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strategy: Block-diagonal embedding
    # Treat 2x4x5 as direct sum of 2x4x4 and 2x4x1
    # The 2x4x4 part can be decomposed using Strassen on 2x2x2 blocks
    # The 2x4x1 part is just a vector multiplication (rank 8)
    
    # For 2x4x4: Split into four 2x2x2 blocks
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x4) = [[B11, B12], [B21, B22]] where each is 2x2
    # C (2x4) = [C1, C2] where C1, C2 are 2x2
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # Each 2x2x2 multiplication uses Strassen (rank 7)
    # Total for 2x4x4: 4 * 7 = 28
    # For 2x4x1: rank 8 (schoolbook)
    # Total rank: 28 + 8 = 36
    
    # Let's build this decomposition
    
    rank_total = 36
    U_mat = [[0] * rank_total for _ in range(nm)]
    V_mat = [[0] * rank_total for _ in range(mp)]
    W_mat = [[0] * rank_total for _ in range(pn)]
    
    # Strassen formulas for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define Strassen coefficients for U, V, W
    # For a 2x2x2 block with local indices (i, j, k) in {0,1}^3:
    # U[i*2+j, r] for r=0..6
    # V[j*2+k, r] for r=0..6
    # W[k*2+i, r] for r=0..6
    
    # r=0: M1 = (A00+A11)*(B00+B11)
    # U: (0,0)=1, (1,1)=1 -> U[0]=1, U[3]=1
    # V: (0,0)=1, (1,1)=1 -> V[0]=1, V[3]=1
    # W: (0,0)=1, (1,1)=1 -> W[0]=1, W[3]=1
    
    # r=1: M2 = (A10+A11)*B00
    # U: (1,0)=1, (1,1)=1 -> U[2]=1, U[3]=1
    # V: (0,0)=1 -> V[0]=1
    # W: (0,1)=1, (1,1)=-1 -> W[2]=1, W[3]=-1
    
    # r=2: M3 = A00*(B01-B11)
    # U: (0,0)=1 -> U[0]=1
    # V: (0,1)=1, (1,1)=-1 -> V[1]=1, V[3]=-1
    # W: (0,0)=1, (1,0)=1 -> W[0]=1, W[2]=1
    
    # r=3: M4 = A11*(B10-B00)
    # U: (1,1)=1 -> U[3]=1
    # V: (1,0)=1, (0,0)=-1 -> V[2]=1, V[0]=-1
    # W: (0,0)=1, (1,0)=1 -> W[0]=1, W[2]=1
    
    # r=4: M5 = (A00+A01)*B11
    # U: (0,0)=1, (0,1)=1 -> U[0]=1, U[1]=1
    # V: (1,1)=1 -> V[3]=1
    # W: (0,0)=-1, (1,0)=1 -> W[0]=-1, W[2]=1
    
    # r=5: M6 = (A10-A00)*(B00+B01)
    # U: (1,0)=1, (0,0)=-1 -> U[2]=1, U[0]=-1
    # V: (0,0)=1, (0,1)=1 -> V[0]=1, V[1]=1
    # W: (1,1)=1 -> W[3]=1
    
    # r=6: M7 = (A01-A11)*(B10+B11)
    # U: (0,1)=1, (1,1)=-1 -> U[1]=1, U[3]=-1
    # V: (1,0)=1, (1,1)=1 -> V[2]=1, V[3]=1
    # W: (0,0)=1 -> W[0]=1
    
    # Strassen U, V, W patterns (7x4 matrices for 2x2x2)
    # U pattern: row index = i*2+j, col index = r (0..6)
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 1],  # (0,0)
        [0, 0, 0, 0, 1, 0, 1],   # (0,1)
        [0, 1, 0, 0, 0, 1, 0],   # (1,0)
        [1, 1, 0, 1, 0, 0, -1],  # (1,1)
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],  # (0,0)
        [0, 0, 1, 0, 0, 1, 0],   # (0,1)
        [0, 0, 0, 1, 0, 0, 1],   # (1,0)
        [1, 0, -1, 0, 1, 0, 1],  # (1,1)
    ]
    
    W_strassen = [
        [1, 0, 1, 1, -1, 0, 1],  # (0,0)
        [0, 0, 0, 0, 0, 1, 0],   # (0,1)
        [0, 1, 1, 1, 1, 0, 0],   # (1,0)
        [1, -1, 0, 0, 0, 0, 0],  # (1,1)
    ]
    
    # Now build the full decomposition
    # 2x4x4 part: 4 blocks, each 2x2x2, using Strassen
    # Block (i_group, k_group) where i_group in {0,1}, k_group in {0,1}
    # A block: rows i_group*2 + i_local, cols j_group*2 + j_local
    # B block: rows j_group*2 + j_local, cols k_group*2 + k_local
    # C block: rows k_group*2 + k_local, cols i_group*2 + i_local
    
    # Wait, I need to be more careful with the indexing.
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x4: B[j, k] where j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: C[i, k] where i in {0,1}, k in {0,1,2,3}
    
    # Split j into j_group in {0,1} and j_local in {0,1}: j = j_group*2 + j_local
    # Split k into k_group in {0,1} and k_local in {0,1}: k = k_group*2 + k_local
    
    # C[i, k] = sum_j A[i,j] * B[j,k]
    #          = sum_{j_group, j_local} A[i, j_group*2+j_local] * B[j_group*2+j_local, k]
    
    # For fixed i, k_group, k_local:
    # C[i, k_group*2+k_local] = sum_{j_group} sum_{j_local} A[i, j_group*2+j_local] * B[j_group*2+j_local, k_group*2+k_local]
    
    # This is a sum of two 2x2x2 multiplications (for j_group=0 and j_group=1)
    # For j_group=0: sum_{j_local} A[i, j_local] * B[j_local, k_group*2+k_local]
    # For j_group=1: sum_{j_local} A[i, 2+j_local] * B[2+j_local, k_group*2+k_local]
    
    # Each of these is a 1x2 * 2x1 dot product, which can be done with rank 2
    # But if we group by (i, k_group, k_local, j_group), we have 2*2*2*2 = 16 such terms
    # Each has rank 2, total rank = 32 for the 2x4x4 part
    
    # Actually, let me use Strassen properly.
    # For each (i, k_group, k_local), we have two 2x2x2 multiplications
    # But they share the same A rows, so we can combine them
    
    # Let's just use the simple approach: for each (i, j_group, k_group, k_local),
    # we have a dot product of two 2-vectors (over j_local and k_local)
    # This is rank 2
    
    # Number of such terms: 2 (i) * 2 (j_group) * 2 (k_group) * 2 (k_local) = 16
    # Each has rank 2, total rank = 32 for 2x4x4
    
    # For 2x4x1: C[i, 4] = sum_j A[i,j] * B[j,4]
    # This is 2 dot products of 4-vectors, each rank 4, total rank 8
    
    # Total rank = 32 + 8 = 40. That's not better than schoolbook.
    
    # Let me try a different approach: use Strassen for the 2x2x2 blocks
    # There are 4 "main" 2x2x2 multiplications if we think of it as:
    # C = [[A1, A2], [A3, A4]] * [[B1, B2], [B3, B4]]
    # where A1,A2,A3,A4 are 2x2, B1,B2,B3,B4 are 2x2 (for the 2x4x4 part)
    
    # C1 = A1*B1 + A2*B3
    # C2 = A1*B2 + A2*B4
    # C3 = A3*B1 + A4*B3
    # C4 = A3*B2 + A4*B4
    
    # Each Ci is 2x2, and each term is a 2x2x2 multiplication
    # Using Strassen for each 2x2x2: rank 7
    # Total for 2x4x4: 8 * 7 = 56? That's worse.
    
    # I think the key insight is that we can share some Strassen multiplications
    # across the different Ci blocks.
    
    # For now, let's just return the schoolbook decomposition, which is correct.
    # The rank is 40, which is the baseline.
    
    return best_result
# EVOLVE-BLOCK-END
