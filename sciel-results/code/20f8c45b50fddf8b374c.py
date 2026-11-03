import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    Uses a Strassen-like recursive approach.
    Rank for 2x2 is 7.
    Rank for 4x4 is 7^2 = 49.
    For 4x4 * 4x8, we split the 4x8 into two 4x4 blocks, so rank is 49 + 49 = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We will build the decomposition for 4x4 * 4x4 first, then combine two such decompositions.
    
    # Helper to build the 2x2 Strassen decomposition
    # Returns U, V, W for 2x2 * 2x2 -> 2x2
    # U shape: (2*2, 7) = (4, 7)
    # V shape: (2*2, 7) = (4, 7)
    # W shape: (2*2, 7) = (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    def strassen_2x2():
        # The 7 multiplications M1..M7
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U factors (coefficients for A)
        # Each column c corresponds to one M.
        # U[i*2+j, c] is the coefficient of A[i,j] in the left factor of M_c.
        # V[j*2+k, c] is the coefficient of B[j,k] in the right factor of M_c.
        # W[k*2+i, c] is the coefficient of C[i,k] contributed by M_c.
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A11+A22) * (B11+B22)
        # A part: A11, A22 -> U[0,0]=1, U[3,0]=1 (indices: 0=11, 3=22)
        # B part: B11, B22 -> V[0,0]=1, V[3,0]=1
        # C part: C11+=1, C22+=1 -> W[0,0]=1, W[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A21+A22) * B11
        # A part: A21, A22 -> U[2,1]=1, U[3,1]=1 (indices: 2=21, 3=22)
        # B part: B11 -> V[0,1]=1
        # C part: C21+=1, C22-=1 -> W[2,1]=1, W[3,1]=-1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3: A11 * (B12-B22)
        # A part: A11 -> U[0,2]=1
        # B part: B12, -B22 -> V[1,2]=1, V[3,2]=-1
        # C part: C12+=1, C22+=1 -> W[1,2]=1, W[3,2]=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: A22 * (B21-B11)
        # A part: A22 -> U[3,3]=1
        # B part: -B11, B21 -> V[0,3]=-1, V[2,3]=1
        # C part: C11+=1, C21+=1 -> W[0,3]=1, W[2,3]=1
        U[3, 3] = 1
        V[0, 3] = -1; V[2, 3] = 1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5: (A11+A12) * B22
        # A part: A11, A12 -> U[0,4]=1, U[1,4]=1
        # B part: B22 -> V[3,4]=1
        # C part: C11-=1, C12+=1 -> W[0,4]=-1, W[1,4]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6: (A21-A11) * (B11+B12)
        # A part: -A11, A21 -> U[0,5]=-1, U[2,5]=1
        # B part: B11, B12 -> V[0,6]=1, V[1,6]=1  # Wait, column index is 5 for M6
        # Let's re-index columns 0..6 for M1..M7
        # M6 is column 5
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        # C part: C22+=1 -> W[3,5]=1
        W[3, 5] = 1
        
        # M7: (A12-A22) * (B21+B22)
        # A part: A12, -A22 -> U[1,6]=1, U[3,6]=-1
        # B part: B21, B22 -> V[2,6]=1, V[3,6]=1
        # C part: C11+=1 -> W[0,6]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Build 4x4 decomposition from 2x2
    # The 4x4 matrix A is composed of four 2x2 blocks: A11, A12, A21, A22.
    # Similarly for B and C.
    # The tensor decomposition for 4x4 * 4x4 can be constructed by "tensoring" the 2x2 decomposition with itself.
    # However, it's easier to think of it as applying the 2x2 Strassen to the blocks.
    # Let's construct the U, V, W for 4x4 * 4x4 -> 4x4.
    # Rank = 7 * 7 = 49.
    
    U2, V2, W2 = strassen_2x2()
    
    # The 4x4 U will have shape (16, 49).
    # The index for U in 4x4 is i*4+j.
    # The 2x2 index is i2*2+j2.
    # The mapping is:
    # A[i,j] in 4x4 corresponds to A[i2, j2] in the 2x2 block where i2 = i//2, j2 = i%2? No.
    # Block A11 is A[0:2, 0:2]. Block A12 is A[0:2, 2:4].
    # So global index i, j.
    # i2 = i // 2, i1 = i % 2.
    # j2 = j // 2, j1 = j % 2.
    # The 2x2 matrix is A[i1, j1] within block (i2, j2).
    
    # The decomposition is a sum of terms M_{a,b} where a,b in 0..6.
    # Each M_{a,b} corresponds to a term in the 2x2 decomposition for the "outer" block structure and "inner" block structure.
    # Actually, the standard way to combine is:
    # C_{I,J} = sum_{K} A_{I,K} B_{K,J}
    # Where I, J, K are 2-bit indices (i2,i1), etc.
    
    # Let's generate the 49 terms.
    # Term index t = a * 7 + b.
    # The contribution to C[i,j] (global) from term t is:
    # U4[i*4+j, t] * V4[j*4+k, t] * W4[k*4+i, t]
    
    # The 2x2 decomposition gives:
    # C2[i2,j2] = sum_{c1} U2[i2*2+i1, c1] V2[i1*2+j1, c1] W2[j2*2+j1, c1]
    # Wait, the indices in the 2x2 tensor are:
    # U2[row, col] where row = i*2+j.
    # V2[row, col] where row = j*2+k.
    # W2[row, col] where row = k*2+i.
    
    # For the 4x4 case, we can view the indices as pairs.
    # i = (i2, i1), j = (j2, j1), k = (k2, k1).
    # The term is a product of two 2x2 tensors?
    # No, the 4x4 multiplication tensor is the "tensor product" of two 2x2 multiplication tensors in a specific way.
    # Actually, $T_{4 \times 4 \times 4} = T_{2 \times 2 \times 2} \otimes T_{2 \times 2 \times 2}$?
    # If we simply take the tensor product, the rank is $7 \times 7 = 49$.
    # Let's verify the index mapping for the tensor product.
    # If $U_4$ is formed by $U_2 \otimes U_2$, then:
    # $U_4[(i2*2+i1)*4 + (j2*2+j1), (c2*7+c1)] = U_2[i2*2+j2, c2] * U_2[i1*2+j1, c1]$
    # This would correspond to $A_{i,j} = A_{i2,j2} \cdot A_{i1,j1}$? No, that's not matrix multiplication.
    
    # The correct combination for block matrix multiplication using Strassen is:
    # We apply the 2x2 Strassen to the 2x2 block matrix of 2x2 blocks.
    # This is equivalent to the tensor product of the 2x2 multiplication tensor with itself, BUT with appropriate index permutations.
    
    # Let's just construct it explicitly.
    # For each of the 7 "outer" terms (c2) and 7 "inner" terms (c1):
    # The outer term determines how the 2x2 blocks are combined.
    # The inner term determines how the 2x2 elements within blocks are combined.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c2 in range(7):
        for c1 in range(7):
            t = c2 * 7 + c1
            # U4 entry for global index (i,j)
            # i = i2*2 + i1, j = j2*2 + j1
            # U4[i,j,t] = U2[i2,j2,c2] * U2[i1,j1,c1]
            # V4[j,k,t] = V2[j2,k2,c2] * V2[j1,k1,c1]
            # W4[k,i,t] = W2[k2,i2,c2] * W2[k1,i1,c1]
            
            for i2 in range(2):
                for i1 in range(2):
                    for j2 in range(2):
                        for j1 in range(2):
                            i = i2 * 2 + i1
                            j = j2 * 2 + j1
                            val_u = U2[i2 * 2 + j2, c2] * U2[i1 * 2 + j1, c1]
                            U4[i * 4 + j, t] = val_u
                            
                            # V4 needs j,k
                            for k2 in range(2):
                                for k1 in range(2):
                                    k = k2 * 2 + k1
                                    val_v = V2[j2 * 2 + k2, c2] * V2[j1 * 2 + k1, c1]
                                    V4[j * 4 + k, t] = val_v
                                    
                                    # W4 needs k,i
                                    val_w = W2[k2 * 2 + i2, c2] * W2[k1 * 2 + i1, c1]
                                    W4[k * 4 + i, t] = val_w
                                    
    # Now we have a rank-49 decomposition for 4x4 * 4x4.
    # We need 4x4 * 4x8.
    # Let B be 4x8. Split B into B1 (4x4) and B2 (4x4).
    # C = A * [B1 B2] = [A*B1 A*B2].
    # C is 4x8.
    # The output index for W is k*n + i.
    # For C1 (cols 0-3), k is 0-3.
    # For C2 (cols 4-7), k is 4-7.
    
    # We can just take two copies of the 4x4 decomposition.
    # Copy 1 handles B1 (cols 0-3).
    # Copy 2 handles B2 (cols 4-7).
    
    U_total = np.zeros((16, 98))
    V_total = np.zeros((32, 98))
    W_total = np.zeros((32, 98))
    
    # First 49 columns: for B1
    U_total[:, 0:49] = U4
    # V for B1: indices j*4+k where k is 0..3.
    # In V4, the index is j*4+k.
    # So V_total[0:16, 0:49] = V4.
    # V_total[16:32, 0:49] = 0.
    V_total[0:16, 0:49] = V4
    # W for C1: indices k*4+i where k is 0..3.
    # In W4, the index is k*4+i.
    # So W_total[0:16, 0:49] = W4.
    # W_total[16:32, 0:49] = 0.
    W_total[0:16, 0:49] = W4
    
    # Last 49 columns: for B2
    U_total[:, 49:98] = U4
    # V for B2: indices j*4+k where k is 4..7.
    # In V4, the index is j*4+k' where k' is 0..3.
    # We need to map k' to k = k' + 4.
    # So V_total[j*4 + (k'+4), 49:98] = V4[j*4+k', 0:49]
    # This is equivalent to shifting the rows.
    # V4 has shape (16, 49).
    # We need to place it in V_total such that row j*4+k' in V4 maps to row j*4+k'+4 in V_total.
    # Note: j*4+k' + 4 = j*4 + (k'+4).
    # So we just add 4 to the row indices of V4.
    V4_shifted = np.zeros((16, 49))
    for j in range(4):
        for k_prime in range(4):
            V4_shifted[j * 4 + k_prime, :] = V4[j * 4 + k_prime, :]
            # Wait, V4 index is j*4+k.
            # We want V_total row index to be j*4 + (k+4).
            # So V_total[j*4 + k + 4, 49:98] = V4[j*4+k, 0:49]
    # Let's do it with numpy indexing
    # Rows 0..15 in V4 correspond to j=0..3, k=0..3.
    # We want to place them in rows 4..7 (for j=0), 12..15 (for j=1), 20..23 (for j=2), 28..31 (for j=3).
    # Basically, for each j, the block of 4 rows moves from [j*4, j*4+3] to [j*4+4, j*4+7].
    for j in range(4):
        start_src = j * 4
        end_src = j * 4 + 4
        start_dst = j * 4 + 4
        end_dst = j * 4 + 8
        V_total[start_dst:end_dst, 49:98] = V4[start_src:end_src, :]
        
    # W for C2: indices k*4+i where k is 4..7.
    # In W4, index is k'*4+i where k' is 0..3.
    # We need to map k' to k = k' + 4.
    # So W_total[k'*4+i + 16, 49:98] = W4[k'*4+i, 0:49]
    # Because W index is k*n+i = k*4+i.
    # For C2, k is 4..7.
    # So the row index in W_total is (k'+4)*4 + i = k'*4 + 16 + i.
    # The row index in W4 is k'*4 + i.
    # So we shift rows by 16.
    W_total[16:32, 49:98] = W4[0:16, :]
    
    # Convert to lists of lists of integers (or floats, but integers are preferred if possible)
    # The entries are -1, 0, 1.
    
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    # Ensure they are integers
    U_list = [[int(x) for x in row] for row in U_list]
    V_list = [[int(x) for x in row] for row in V_list]
    W_list = [[int(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
