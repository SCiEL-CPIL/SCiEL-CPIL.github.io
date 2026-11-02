"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    # Rank-minimal (or near-minimal) decomposition for 2x5x6 matrix multiplication.
    # We decompose the 2x5x6 tensor into a 2x1x6 part (rank 12) and a 2x4x6 part.
    # The 2x4x6 part is further decomposed into two 2x2x6 blocks.
    # Each 2x2x6 block uses 3 Strassen (2x2x2) instances, giving rank 21 per block.
    # Total rank = 12 + 21 + 21 = 54.
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank for _ in range(p * n)]  # 12 x 54
    
    # Strassen 2x2x2 decomposition coefficients
    # For 2x2x2, we need 7 terms.
    # The tensor T_{i,j,k} = 1 if i,j,k in {0,1}, 0 otherwise for the identity mapping.
    # Strassen uses the following 7 bilinear forms:
    # Let A, B be 2x2 matrices.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A00-A10)(B00+A11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U coefficients for 2x2x2 (row = i*m+j = 2i+j, col = term index)
    # A is 2x2, indexed as A[i,j]
    # U[2i+j, term]
    # M1: A00, A11 -> U[0,0]=1, U[3,0]=1
    # M2: A10, A11 -> U[2,1]=1, U[3,1]=1
    # M3: A00 -> U[0,2]=1
    # M4: A11 -> U[3,3]=1
    # M5: A00, A10 -> U[0,4]=1, U[2,4]=1
    # M6: A01, -A11 -> U[1,5]=1, U[3,5]=-1
    # M7: A00, -A10 -> U[0,6]=1, U[2,6]=-1
    
    U_2x2 = [
        [1, 0, 1, 0, 1, 0, 1],  # A00
        [0, 0, 0, 0, 0, 1, 0],  # A01
        [0, 1, 0, 0, 1, 0, -1], # A10
        [1, 1, 0, 1, 0, -1, 0], # A11
    ]
    
    # V coefficients for 2x2x2 (row = j*p+k = 2j+k, col = term index)
    # B is 2x2, indexed as B[j,k]
    # V[2j+k, term]
    # M1: B00, B11 -> V[0,0]=1, V[3,0]=1
    # M2: B00 -> V[0,1]=1
    # M3: B01, -B11 -> V[1,2]=1, V[3,2]=-1
    # M4: B10, -B00 -> V[2,3]=1, V[0,3]=-1
    # M5: B11 -> V[3,4]=1
    # M6: B10, B11 -> V[2,5]=1, V[3,5]=1
    # M7: B00, A11 (but for V it's B) -> B00, B11 -> V[0,6]=1, V[3,6]=1
    
    V_2x2 = [
        [1, 1, 0, -1, 0, 0, 1],  # B00
        [0, 0, 1, 0, 0, 0, 0],   # B01
        [0, 0, 0, 1, 0, 1, 0],   # B10
        [1, 0, -1, 0, 1, 1, 1],  # B11
    ]
    
    # W coefficients for 2x2x2 (row = k*n+i = 2k+i, col = term index)
    # C is 2x2, indexed as C[k,i]
    # W[2k+i, term]
    # C00 = M1 + M4 - M5 + M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    # C01 = M3 + M5 -> W[1,2]=1, W[1,4]=1
    # C10 = M2 + M4 -> W[2,1]=1, W[2,3]=1
    # C11 = M1 - M2 + M3 + M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    
    W_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0],  # C11 (k=1, i=1)
    ]
    
    # Part 1: 2x1x6 block (i in 0..1, j=0, k in 0..5)
    # Rank 12 (schoolbook for 2x1x6)
    col_offset = 0
    for i in range(2):
        for k in range(6):
            U[i * 5 + 0][col_offset] = 1
            V[0 * 6 + k][col_offset] = 1
            W[k * 2 + i][col_offset] = 1
            col_offset += 1
    # col_offset is now 12
    
    # Part 2: 2x4x6 block, decomposed into two 2x2x6 blocks
    # Block 1: i in 0..1, j in 0..1, k in 0..5 (3 Strassen instances for k=0-1, k=2-3, k=4-5)
    # Block 2: i in 0..1, j in 2..3, k in 0..5 (3 Strassen instances)
    
    # Helper to add a 2x2x2 Strassen instance at given offsets
    def add_strassen(i_base, j_base, k_base, col_start):
        """Add a 2x2x2 Strassen decomposition with global indices offset by (i_base, j_base, k_base)."""
        for t in range(7):
            # U: rows are i_base+0, i_base+1 with j=j_base
            # U[2i+j, t] for local 2x2, mapped to global
            for li in range(2):
                for lj in range(2):
                    if U_2x2[li * 2 + lj][t] != 0:
                        global_i = i_base + li
                        global_j = j_base + lj
                        U[global_i * 5 + global_j][col_start + t] = U_2x2[li * 2 + lj][t]
            
            for lj in range(2):
                for lk in range(2):
                    if V_2x2[lj * 2 + lk][t] != 0:
                        global_j = j_base + lj
                        global_k = k_base + lk
                        V[global_j * 6 + global_k][col_start + t] = V_2x2[lj * 2 + lk][t]
            
            for lk in range(2):
                for li in range(2):
                    if W_2x2[lk * 2 + li][t] != 0:
                        global_k = k_base + lk
                        global_i = i_base + li
                        W[global_k * 2 + global_i][col_start + t] = W_2x2[lk * 2 + li][t]
    
    # Block 1: j in 0..1
    # 3 Strassen instances for k=0-1, k=2-3, k=4-5
    add_strassen(0, 0, 0, 12)  # cols 12-18
    add_strassen(0, 0, 2, 19)  # cols 19-25
    add_strassen(0, 0, 4, 26)  # cols 26-32
    
    # Block 2: j in 2..3
    add_strassen(0, 2, 0, 33)  # cols 33-39
    add_strassen(0, 2, 2, 40)  # cols 40-46
    add_strassen(0, 2, 4, 47)  # cols 47-53
    
    # Total rank used: 54
    # col_offset would be 54, which matches rank.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
