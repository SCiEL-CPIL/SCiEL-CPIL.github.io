import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build a list of (u_idx, v_idx, w_idx, coeff) for each rank-1 tensor.
    # u_idx is in 0..24 (5*5)
    # v_idx is in 0..24 (5*5)
    # w_idx is in 0..24 (5*5)
    # Note: The problem statement says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # Wait, the standard convention in the prompt is:
    # U: (n*m, rank) -> indices i, j for A?
    # V: (m*p, rank) -> indices j, k for B?
    # W: (p*n, rank) -> indices k, i for C?
    # Prompt: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies C[k, i]? Or C[i, k]?
    # Standard matrix mult: C[i, k] = sum_j A[i, j] B[j, k].
    # If W is indexed by k*n+i, then W corresponds to C[k, i]?
    # Let's check the seed code:
    # U[i * m + j][col] = 1
    # V[j * p + k][col] = 1
    # W[k * n + i][col] = 1
    # This implies:
    # A is indexed by (i, j). i in 0..n-1, j in 0..m-1.
    # B is indexed by (j, k). j in 0..m-1, k in 0..p-1.
    # C is indexed by (k, i). k in 0..p-1, i in 0..n-1.
    # So C[k, i] = sum_j A[i, j] B[j, k].
    # This is C = B^T A^T? Or C^T = A B?
    # If C[k, i] is the element, then C is p x n.
    # A is n x m. B is m x p.
    # A B is n x p.
    # (A B)^T is p x n.
    # (A B)^T[k, i] = (A B)[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor computes C = (A B)^T.
    # Or equivalently, it computes the product where the output is transposed.
    # We must follow this convention.
    
    terms = []
    
    # 1. Strassen 4x4 for the top-left 4x4 block of A and B.
    # This computes C_top_left = A_top_left * B_top_left.
    # A_top_left is A[0:4, 0:4].
    # B_top_left is B[0:4, 0:4].
    # C_top_left is C[0:4, 0:4] in the transposed convention?
    # C[k, i] for k,i in 0..3.
    # This corresponds to (A B)^T[0:4, 0:4].
    # Which is (A[0:4, 0:4] * B[0:4, 0:4])^T.
    # So we need the tensor for 4x4x4 multiplication.
    
    # Helper to add Strassen 2x2 terms
    # offset_a, offset_b, offset_c are the starting indices in the flattened arrays?
    # No, we need to map logical (i, j) to flat index.
    # A[i, j] -> i * m + j
    # B[j, k] -> j * p + k
    # C[k, i] -> k * n + i
    
    def add_strassen_2x2(a_off_i, a_off_j, b_off_j, b_off_k, c_off_k, c_off_i):
        # a_off_i, a_off_j: top-left corner of A block
        # b_off_j, b_off_k: top-left corner of B block
        # c_off_k, c_off_i: top-left corner of C block
        # A indices: (a_off_i + di, a_off_j + dj)
        # B indices: (b_off_j + dj, b_off_k + dk)
        # C indices: (c_off_k + dk, c_off_i + di)
        
        # M1 = (A11+A22)(B11+B22)
        # A11: (a_off_i, a_off_j)
        # A22: (a_off_i+1, a_off_j+1)
        # B11: (b_off_j, b_off_k)
        # B22: (b_off_j+1, b_off_k+1)
        # C11: (c_off_k, c_off_i)
        # C12: (c_off_k+1, c_off_i) -> C[k+1, i]
        # C21: (c_off_k, c_off_i+1) -> C[k, i+1]
        # C22: (c_off_k+1, c_off_i+1)
        
        # We add 7 terms for the M's, and then add them to C's.
        # But wait, the rank-1 tensors are for the M's.
        # The C's are linear combinations of M's.
        # So we need to store the M's as rank-1 terms, and then the coefficients for C.
        # This requires a two-step process or careful bookkeeping.
        # Since we are building U, V, W directly, we can just add the terms for the final C.
        # But Strassen works by computing M's first.
        # If we add M1 to C11, C22, etc., we need to know which C element gets which M.
        # Actually, the tensor decomposition is a sum of rank-1 tensors.
        # $C_{k,i} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$? No.
        # $C_{k,i} = \sum_r U_{row_A, r} V_{row_B, r} W_{row_C, r}$.
        # If we use Strassen, we introduce intermediate variables M_r.
        # This means we need to expand the bilinear form.
        # $C_{k,i} = \sum_{r \in \text{Strassen}} (\sum_{\text{contribs}} \dots)$.
        # It is easier to just generate the rank-1 terms for the final C directly by expanding the Strassen formulas.
        
        # Let's define the 7 products:
        # M1: A11*A22 part? No.
        # M1 = (A11 + A22) * (B11 + B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # This expands to 4 rank-1 terms.
        # M2 = (A21 + A22) * B11 = A21*B11 + A22*B11
        # M3 = A11 * (B12 - B22) = A11*B12 - A11*B22
        # M4 = A22 * (B21 - B11) = A22*B21 - A22*B11
        # M5 = (A11 + A12) * B22 = A11*B22 + A12*B22
        # M6 = (A21 - A11) * (B11 + B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
        # M7 = (A12 - A22) * (B21 + B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
        
        # Then:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We can substitute M's back into C's and collect terms for each (A_elem, B_elem, C_elem).
        # This is tedious but doable programmatically.
        # Alternatively, we can just use the schoolbook for 4x4 (rank 16) if Strassen is too complex to expand correctly in this framework?
        # No, Strassen is rank 7. Schoolbook is 16.
        # For 4x4, Strassen is 49. Schoolbook is 64.
        # Difference is 15.
        # Cross terms are 61.
        # Total Strassen: 49 + 61 = 110.
        # Total Schoolbook: 64 + 61 = 125.
        # So Strassen saves 15 ranks.
        
        # Let's implement the expansion.
        # We will use a dictionary to accumulate coefficients for each (a_idx, b_idx, c_idx) tuple.
        # Then we will convert this to rank-1 terms.
        # Note: Multiple (a,b,c) tuples might map to the same rank-1 term if they share structure?
        # No, each (a,b,c) is a unique rank-1 tensor in the standard basis.
        # The rank of the tensor is the number of non-zero entries in this dictionary?
        # No, the rank is the minimum number of rank-1 tensors to sum to the tensor.
        # The standard basis decomposition has rank equal to the number of non-zero entries.
        # Strassen reduces this number by combining terms.
        # To use Strassen, we must express the final C as a sum of 49 rank-1 tensors.
        # This means we need to find the U, V, W matrices for those 49 tensors.
        # This is equivalent to finding a decomposition of the 4x4x4 tensor of rank 49.
        # We can just generate the 49 terms explicitly.
        
        # Let's hardcode the 49 terms for 4x4x4 Strassen.
        # Or better, write a helper that takes the 2x2 Strassen formulas and applies them recursively?
        # Given time, let's just use the schoolbook for the 4x4 block?
        # If we use schoolbook for 4x4, rank is 64.
        # Total 125.
        # This is the same as the current program.
        # We need to implement Strassen to get 110.
        
        # Let's try to implement the expansion of Strassen 4x4 into rank-1 terms.
        # We will create a list of (a_flat, b_flat, c_flat, coeff).
        
        # For the 4x4 block:
        # A[0:4, 0:4], B[0:4, 0:4], C[0:4, 0:4].
        
        # We will define the 7 Strassen products for a 2x2 block,
        # and then apply them to the 4 sub-blocks of the 4x4 matrix.
        # This is complex.
        
        # Alternative: Just use the known rank-110 decomposition if available?
        # No, we must construct it.
        
        # Let's stick to the simple embedding but optimize the cross terms if possible.
        # Actually, 110 is a good target.
        # Let's implement the 4x4 Strassen expansion.
        
        pass

# EVOLVE-BLOCK-END
