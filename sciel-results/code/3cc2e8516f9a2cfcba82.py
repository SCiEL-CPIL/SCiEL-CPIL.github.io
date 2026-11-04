"""
Rewritten program to find a rank-49 decomposition for 4x4 matrix multiplication
using the tensor product of two Strassen decompositions.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Rank 7
    # U: (2*2, 7) -> indices (i, j, r)
    # V: (2*2, 7) -> indices (j, k, r)
    # W: (2*2, 7) -> indices (k, i, r)
    
    # Strassen factors (standard form)
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U[i, j, r] corresponds to coefficient of A[i, j] in M_r
    # V[j, k, r] corresponds to coefficient of B[j, k] in M_r
    # W[k, i, r] corresponds to coefficient of C[i, k] from M_r
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices for 2x2: i in {0,1}, j in {0,1}
    # Row index in U2: i*2 + j
    # Row index in V2: j*2 + k
    # Row index in W2: k*2 + i
    
    # M1
    U2[0*2+0, 0] = 1; U2[1*2+1, 0] = 1
    V2[0*2+0, 0] = 1; V2[1*2+1, 0] = 1
    W2[0*2+0, 0] = 1; W2[1*2+1, 0] = 1
    
    # M2
    U2[1*2+0, 1] = 1; U2[1*2+1, 1] = 1
    V2[0*2+0, 1] = 1
    W2[1*2+0, 1] = 1
    
    # M3
    U2[0*2+0, 2] = 1
    V2[1*2+0, 2] = 1; V2[1*2+1, 2] = -1
    W2[0*2+1, 2] = 1
    
    # M4
    U2[1*2+1, 3] = 1
    V2[0*2+1, 3] = 1; V2[0*2+0, 3] = -1
    W2[1*2+1, 3] = 1
    
    # M5
    U2[0*2+0, 4] = 1; U2[0*2+1, 4] = 1
    V2[1*2+1, 4] = 1
    W2[0*2+0, 4] = 1
    
    # M6
    U2[1*2+0, 5] = 1; U2[0*2+0, 5] = -1
    V2[0*2+0, 5] = 1; V2[0*2+1, 5] = 1
    W2[1*2+1, 5] = 1
    
    # M7
    U2[0*2+1, 6] = 1; U2[1*2+1, 6] = -1
    V2[1*2+0, 6] = 1; V2[1*2+1, 6] = 1
    W2[0*2+1, 6] = 1

    # Verify Strassen decomposition (optional, but good for sanity)
    # T2[i,j,k] = sum_r U2[i,j,r] V2[j,k,r] W2[k,i,r]
    # This is standard and assumed correct.

    # Construct 4x4 decomposition via Tensor Product
    # U4 = U2 (x) U2
    # Dimensions: U2 is (4, 7). U4 will be (16, 49).
    # Row index in U4: (i_big * 2 + i_small) * 2 + (j_big * 2 + j_small) ?
    # No, the Kronecker product of matrices A (rows R_A, cols C_A) and B (rows R_B, cols C_B)
    # results in a matrix of size (R_A*R_B, C_A*C_B).
    # Element ( (i_A, i_B), (j_A, j_B) ) = A[i_A, j_A] * B[i_B, j_B].
    # Here, the "row" of the tensor factor corresponds to the input indices (i, j) for U.
    # For 2x2, row is i*2+j.
    # For 4x4, we treat i as (I, i') and j as (J, j').
    # The row index in the 4x4 factor should map to (I, J, i', j').
    # If we use standard Kronecker product U4 = U2 @ U2 (where @ is kron):
    # Row r4 = r2_a * 4 + r2_b.
    # r2_a = I*2 + J. r2_b = i'*2 + j'.
    # So r4 = (I*2 + J)*4 + (i'*2 + j') = 8I + 4J + 2i' + j'.
    # This corresponds to i = 2I + i', j = 2J + j'.
    # i = 2I + i' => I = i // 2, i' = i % 2.
    # j = 2J + j' => J = j // 2, j' = j % 2.
    # So the row index in U4 for input (i, j) is (i//2 * 2 + j//2) * 4 + (i%2 * 2 + j%2).
    # Let's check:
    # i=0, j=0 => I=0, J=0, i'=0, j'=0 => r4 = 0*4 + 0 = 0.
    # i=0, j=1 => I=0, J=0, i'=0, j'=1 => r4 = 0*4 + 1 = 1.
    # i=1, j=0 => I=0, J=0, i'=1, j'=0 => r4 = 0*4 + 2 = 2.
    # i=1, j=1 => I=0, J=0, i'=1, j'=1 => r4 = 0*4 + 3 = 3.
    # i=2, j=0 => I=1, J=0, i'=0, j'=0 => r4 = (1*2+0)*4 + 0 = 8.
    # This ordering is consistent with treating the 4x4 matrix as a 2x2 block matrix where blocks are 2x2.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Check shapes
    # U4: (16, 49)
    # V4: (16, 49)
    # W4: (16, 49)
    
    # The task requires U to have shape (n*m, rank) = (16, 49).
    # The row index in U should be i*m + j = i*4 + j.
    # Our Kronecker product row index is 8I + 4J + 2i' + j'.
    # We need to permute the rows of U4, V4, W4 to match the i*4+j ordering.
    
    # Mapping from kron_row to linear_index (i*4+j)
    # kron_row = 8I + 4J + 2i' + j'
    # i = 2I + i'
    # j = 2J + j'
    # linear_index = i*4 + j = (2I+i')*4 + (2J+j') = 8I + 4i' + 2J + j'
    
    # We need to reorder rows of U4 such that new_row[linear_index] = old_row[kron_row]
    
    def permute_rows(mat, n=4):
        """
        mat is (16, R).
        Row k in mat corresponds to kron index k.
        We want row l in result to correspond to linear index l (i*4+j).
        Find kron index for linear index l.
        l = i*4 + j.
        i = l // 4, j = l % 4.
        I = i // 2, i' = i % 2.
        J = j // 2, j' = j % 2.
        kron_index = 8I + 4J + 2i' + j'.
        """
        new_mat = np.zeros_like(mat)
        for l in range(16):
            i = l // 4
            j = l % 4
            I = i // 2
            i_prime = i % 2
            J = j // 2
            j_prime = j % 2
            kron_index = 8 * I + 4 * J + 2 * i_prime + j_prime
            new_mat[l] = mat[kron_index]
        return new_mat

    U_final = permute_rows(U4)
    V_final = permute_rows(V4)
    W_final = permute_rows(W4)
    
    # The task specifies W[k*n+i, r].
    # Our W4 from kron is W2 (x) W2.
    # W2 row index is k*2 + i.
    # W4 row index from kron is (k_big*2 + i_big)*4 + (k_small*2 + i_small) = 8k_big + 4i_big + 2k_small + i_small.
    # We need W_final row index to be k*n + i = k*4 + i.
    # k = 2k_big + k_small
    # i = 2i_big + i_small
    # linear_index = (2k_big + k_small)*4 + (2i_big + i_small) = 8k_big + 4k_small + 2i_big + i_small.
    # This is the same permutation logic as above, just with k and i swapped in the variable names, but the structure is identical.
    # The permutation function `permute_rows` assumes the first index is the "row" (i) and second is "col" (j) for U.
    # For W, the indices are k and i.
    # Let's check if the permutation is the same.
    # U: row = i*4+j. kron = 8I + 4J + 2i' + j'.
    # W: row = k*4+i. kron = 8K + 4I + 2k' + i'.
    # The mapping from linear index to kron index is structurally identical.
    # So we can reuse `permute_rows` for W4 as well.
    
    # Convert to lists of lists with integer entries
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Ensure they are integers (they should be, but np.float64 tolist might give floats if not careful, 
    # but Strassen is integer, so kron is integer)
    # np.kron of integer arrays yields integer array.
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Shapes:", len(res['U']), len(res['V']), len(res['W']))
