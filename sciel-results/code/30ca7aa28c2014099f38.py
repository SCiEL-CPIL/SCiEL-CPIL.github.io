import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2
    # U2, V2, W2 are 4x7 matrices
    # Rows correspond to indices:
    # U2: i*2+j (i,j in 0..1) -> 0:00, 1:01, 2:10, 3:11
    # V2: j*2+k (j,k in 0..1) -> 0:00, 1:01, 2:10, 3:11
    # W2: k*2+i (k,i in 0..1) -> 0:00, 1:01, 2:10, 3:11
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00
    
    # Construct 4x4 decomposition using Kronecker product
    # The tensor for 4x4 matmul is T_4 = T_2 \otimes T_2
    # U_4 = U_2 \otimes U_2
    # V_4 = V_2 \otimes V_2
    # W_4 = W_2 \otimes W_2
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # The standard Kronecker product orders indices as (i1, i2, j1, j2) where 
    # total index = i1*4 + i2 for rows of A (n*m).
    # However, the standard matrix indexing for 4x4 is row = i*4 + j.
    # If we view the 4x4 matrix as a 2x2 block matrix of 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # Index (i,j) in 4x4 corresponds to block (i//2, j//2) and inner (i%2, j%2).
    # Standard flat index: i*4 + j = (i//2)*4 + (i%2)*2 + (j//2)*2 + (j%2) ? 
    # No. i*4 + j. 
    # Let i = 2*i_b + i_i, j = 2*j_b + j_i.
    # Index = (2*i_b + i_i)*4 + (2*j_b + j_i) = 8*i_b + 4*i_i + 2*j_b + j_i.
    # Kronecker product of vectors of length 4 (indices 0..3) results in length 16.
    # np.kron(a, b)[k*len(b) + l] = a[k]*b[l].
    # If a is U2 (indexed by i_b, j_b? No, U2 is 4x7. Rows are 0,1,2,3 corresponding to 00,01,10,11).
    # Let's check the mapping.
    # U2 row index r = 2*i_b + j_b.
    # U4 row index R = 4*r2 + r1 = 4*(2*i_b + j_b) + (2*i_i + j_i) = 8*i_b + 4*j_b + 2*i_i + j_i.
    # This is NOT the standard row-major order (8*i_b + 4*i_i + 2*j_b + j_i).
    # We need to permute the rows of U4, V4, W4 to match standard indexing.
    
    def permute_rows(M, n_dim1, n_dim2):
        # M has shape (n_dim1*n_dim2, rank)
        # Current order from kron: (d1_high, d1_low, d2_high, d2_low) -> index = 4*d1_high + 2*d1_low + 2*d2_high + d2_low ? 
        # Actually np.kron(A, B) where A is 4x7 and B is 4x7.
        # Row r in A (0..3), row s in B (0..3). New row 4*r + s.
        # A row r = 2*i_b + j_b. B row s = 2*i_i + j_i.
        # New row = 4*(2*i_b + j_b) + (2*i_i + j_i) = 8*i_b + 4*j_b + 2*i_i + j_i.
        # Target row = 8*i_b + 4*i_i + 2*j_b + j_i.
        
        n1, n2 = 2, 2
        shape = (n1, n2, n1, n2)
        # Reshape M from (16, rank) to (2, 2, 2, 2, rank)
        # Indices: i_b, j_b, i_i, j_i
        # Current index in M: 4*(2*i_b+j_b) + (2*i_i+j_i)
        # We want to reorder to i_b, i_i, j_b, j_i
        
        M_reshaped = M.reshape(n1, n2, n1, n2, -1)
        # Transpose axes: (i_b, j_b, i_i, j_i) -> (i_b, i_i, j_b, j_i)
        M_transposed = np.transpose(M_reshaped, (0, 2, 1, 3, 4))
        M_permuted = M_transposed.reshape(n1*n2*n1*n2, -1)
        return M_permuted

    U4_perm = permute_rows(U4, 2, 2)
    V4_perm = permute_rows(V4, 2, 2)
    # W4 uses indices (k, i). The tensor decomposition W[k*n+i, c] corresponds to C[i,k] contribution?
    # The tensor is T[i,j,k]. U[i,j,c], V[j,k,c], W[k,i,c].
    # W row index in standard is k*4 + i.
    # In W2, row index is k_b*2 + i_b.
    # In W4 (kron), row index is 4*(2*k_b + i_b) + (2*k_i + i_i) = 8*k_b + 4*i_b + 2*k_i + i_i.
    # Target row index is k*4 + i = (2*k_b+k_i)*4 + (2*i_b+i_i) = 8*k_b + 4*k_i + 2*i_b + i_i.
    # Same permutation logic applies.
    W4_perm = permute_rows(W4, 2, 2)
    
    # Convert to lists for output
    U_list = U4_perm.tolist()
    V_list = V4_perm.tolist()
    W_list = W4_perm.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
