"""Matrix multiplication 2x5x6 with Strassen-based rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# Strassen 2x2x2 decomposition
# M1 = (A11+A22)*(B11+B22)
# M2 = (A21+A22)*B11
# M3 = A11*(B12-B22)
# M4 = A22*(B21-B11)
# M5 = (A11+A12)*B22
# M6 = (A21-A11)*(B11+B12)
# M7 = (A12-A22)*(B21+B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def _strassen_block():
    """Return (U, V, W) for 2x2x2 Strassen, where U is (4,7), V is (4,7), W is (4,7).
    Indexing: U row = i*m+j (i in 0..1, j in 0..1), V row = j*p+k, W row = k*n+i (k in 0..1, i in 0..1)
    n=2, m=2, p=2
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A11+A22)*(B11+B22)
    U[0,0] = 1  # A11 (i=0,j=0)
    U[3,0] = 1  # A22 (i=1,j=1)
    V[0,0] = 1  # B11 (j=0,k=0)
    V[3,0] = 1  # B22 (j=1,k=1)
    W[0,0] = 1  # C11 (k=0,i=0)
    W[3,0] = 1  # C22 (k=1,i=1)
    
    # M2 = (A21+A22)*B11
    U[2,1] = 1  # A21 (i=1,j=0)
    U[3,1] = 1  # A22 (i=1,j=1)
    V[0,1] = 1  # B11 (j=0,k=0)
    W[1,1] = 1  # C21 (k=0,i=1)
    
    # M3 = A11*(B12-B22)
    U[0,2] = 1  # A11 (i=0,j=0)
    V[1,2] = 1  # B12 (j=0,k=1)
    V[3,2] = -1 # B22 (j=1,k=1)
    W[2,2] = 1  # C12 (k=1,i=0)
    
    # M4 = A22*(B21-B11)
    U[3,3] = 1  # A22 (i=1,j=1)
    V[2,3] = 1  # B21 (j=1,k=0)
    V[0,3] = -1 # B11 (j=0,k=0)
    W[0,3] = 1  # C11 (k=0,i=0)
    W[1,3] = 1  # C21 (k=0,i=1)
    
    # M5 = (A11+A12)*B22
    U[0,4] = 1  # A11 (i=0,j=0)
    U[1,4] = 1  # A12 (i=0,j=1)
    V[3,4] = 1  # B22 (j=1,k=1)
    W[0,4] = -1 # C11 (k=0,i=0)
    W[2,4] = 1  # C12 (k=1,i=0)
    
    # M6 = (A21-A11)*(B11+B12)
    U[2,5] = 1  # A21 (i=1,j=0)
    U[0,5] = -1 # A11 (i=0,j=0)
    V[0,5] = 1  # B11 (j=0,k=0)
    V[1,5] = 1  # B12 (j=0,k=1)
    W[3,5] = 1  # C22 (k=1,i=1)
    
    # M7 = (A12-A22)*(B21+B22)
    U[1,6] = 1  # A12 (i=0,j=1)
    U[3,6] = -1 # A22 (i=1,j=1)
    V[2,6] = 1  # B21 (j=1,k=0)
    V[3,6] = 1  # B22 (j=1,k=1)
    W[0,6] = 1  # C11 (k=0,i=0)
    W[3,6] = -1 # C22 (k=1,i=1)
    
    return U, V, W


def construct():
    """Construct rank-minimal decomposition for 2x5x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Schoolbook rank: n*m*p = 60
    # With Strassen on 2x2 blocks: we can apply Strassen where dimensions allow 2x2 sub-blocks.
    # A is 2x5, B is 5x6.
    # We can split A into columns: A = [A_left (2x2), A_mid (2x1), A_right (2x2)]
    # B = [B_top (2x6), B_mid (1x6), B_bottom (2x6)]
    # Then C = A_left @ B_top + A_mid @ B_mid + A_right @ B_bottom
    # Each 2x2x6 or 2x1x6 or 2x2x6 term can use Strassen on the 2x2 part.
    
    # Actually, let's think more carefully.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # i in 0..1, j in 0..4, k in 0..5
    
    # Strategy: Group the 5 columns of A (and 5 rows of B) into blocks:
    # Block 1: j=0,1 (2 columns) -> 2x2x6 product
    # Block 2: j=2 (1 column) -> 2x1x6 product (schoolbook: 12 terms)
    # Block 3: j=3,4 (2 columns) -> 2x2x6 product
    
    # For each 2x2x6 block, use Strassen: rank 7*6 = 42 instead of 2*2*6 = 24... 
    # Wait, that's worse. Strassen for 2x2x2 is rank 7 vs 8, saving 1 per 2x2x2.
    # For 2x2x6, we have 6 independent 2x2x2 multiplications (one per k), each saving 1 rank.
    # So 2x2x6: schoolbook = 2*2*6 = 24, Strassen = 7*6 = 42. That's worse!
    
    # I made an error. Strassen is for square matrices. For 2x2x6, we can't directly apply Strassen
    # because Strassen multiplies 2x2 by 2x2. 
    # 
    # But we can reshape: 2x2x6 is like multiplying (2x2) by (2x6), which is NOT what Strassen does.
    # Strassen multiplies (2x2) by (2x2) to get (2x2).
    #
    # So we need to think about this differently.
    # 
    # Alternative: Use Strassen on the 2x2 sub-blocks where both dimensions are 2.
    # If we have a 2x2x2 sub-block, Strassen saves 1 rank (7 vs 8).
    # 
    # In 2x5x6, how many 2x2x2 sub-blocks can we identify?
    # We need to partition the 5 middle dimension into groups of 2 (and maybe 1), 
    # and the 6 output dimension into groups of 2 (and maybe 2).
    # 
    # Middle dim 5: split as 2+2+1
    # Output dim 6: split as 2+2+2
    # 
    # For each combination of (middle_group, output_group):
    # If middle_group has size 2 and output_group has size 2: 2x2x2 -> Strassen, rank 7
    # If middle_group has size 2 and output_group has size 1: 2x2x1 -> schoolbook, rank 4
    # If middle_group has size 1 and output_group has size 2: 2x1x2 -> schoolbook, rank 4
    # If middle_group has size 1 and output_group has size 1: 2x1x1 -> schoolbook, rank 2
    #
    # Let me recount. Actually, the tensor is 2x5x6. We can view it as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # If we partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4,5}:
    # 
    # For j in {0,1}, k in {0,1}: 2x2x2 sub-tensor -> Strassen rank 7
    # For j in {0,1}, k in {2,3}: 2x2x2 sub-tensor -> Strassen rank 7
    # For j in {0,1}, k in {4,5}: 2x2x2 sub-tensor -> Strassen rank 7
    # For j in {2,3}, k in {0,1}: 2x2x2 sub-tensor -> Strassen rank 7
    # For j in {2,3}, k in {2,3}: 2x2x2 sub-tensor -> Strassen rank 7
    # For j in {2,3}, k in {4,5}: 2x2x2 sub-tensor -> Strassen rank 7
    # For j in {4}, k in {0,1}: 2x1x2 sub-tensor -> schoolbook rank 4
    # For j in {4}, k in {2,3}: 2x1x2 sub-tensor -> schoolbook rank 4
    # For j in {4}, k in {4,5}: 2x1x2 sub-tensor -> schoolbook rank 4
    #
    # Total: 6*7 + 3*4 = 42 + 12 = 54. Schoolbook is 60. Save 6.
    #
    # This matches the knowledge: "Applying Strassen's algorithm to 2x2x2 sub-blocks reduces the rank from 60 to 54"
    
    # Now let's construct this decomposition.
    # We'll build U, V, W of shapes (10, 54), (30, 54), (12, 54)
    
    n, m, p = 2, 5, 6
    rank = 54
    U = np.zeros((n*m, rank))  # (10, 54)
    V = np.zeros((m*p, rank))  # (30, 54)
    W = np.zeros((p*n, rank))  # (12, 54)
    
    col = 0
    # Strassen block template
    Su, Sv, Sw = _strassen_block()
    
    # Define the blocks
    # j_groups: [(0,2), (2,4), (4,5)]  -> indices 0,1; 2,3; 4
    # k_groups: [(0,2), (2,4), (4,6)]  -> indices 0,1; 2,3; 4,5
    
    j_groups = [(0, 2), (2, 4), (4, 5)]
    k_groups = [(0, 2), (2, 4), (4, 6)]
    
    for (j_start, j_end) in j_groups:
        j_size = j_end - j_start
        for (k_start, k_end) in k_groups:
            k_size = k_end - k_start
            
            if j_size == 2 and k_size == 2:
                # Strassen 2x2x2
                for c in range(7):
                    # U: rows for i in 0..1, j in j_start..j_start+1
                    # Su has shape (4,7), row = i*2+j
                    for ii in range(2):
                        for jj in range(2):
                            local_row = ii * 2 + jj
                            global_i = ii
                            global_j = j_start + jj
                            U[global_i * m + global_j, col] = Su[local_row, c]
                    
                    # V: rows for j in j_start..j_start+1, k in k_start..k_start+1
                    # Sv has shape (4,7), row = j*2+k
                    for jj in range(2):
                        for kk in range(2):
                            local_row = jj * 2 + kk
                            global_j = j_start + jj
                            global_k = k_start + kk
                            V[global_j * p + global_k, col] = Sv[local_row, c]
                    
                    # W: rows for k in k_start..k_start+1, i in 0..1
                    # Sw has shape (4,7), row = k*2+i
                    for kk in range(2):
                        for ii in range(2):
                            local_row = kk * 2 + ii
                            global_k = k_start + kk
                            global_i = ii
                            W[global_k * n + global_i, col] = Sw[local_row, c]
                    
                    col += 1
            
            elif j_size == 2 and k_size == 1:
                # Schoolbook 2x2x1: rank 4
                for ii in range(2):
                    for jj in range(2):
                        U[ii * m + (j_start + jj), col] = 1
                        V[(j_start + jj) * p + k_start, col] = 1
                        W[k_start * n + ii, col] = 1
                        col += 1
            
            elif j_size == 1 and k_size == 2:
                # Schoolbook 2x1x2: rank 4
                for ii in range(2):
                    for kk in range(2):
                        U[ii * m + j_start, col] = 1
                        V[j_start * p + (k_start + kk), col] = 1
                        W[(k_start + kk) * n + ii, col] = 1
                        col += 1
            
            elif j_size == 1 and k_size == 1:
                # Schoolbook 2x1x1: rank 2
                for ii in range(2):
                    U[ii * m + j_start, col] = 1
                    V[j_start * p + k_start, col] = 1
                    W[k_start * n + ii, col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to lists of ints/floats
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
