import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # The task is to find a rank-minimal decomposition for 2x4 * 4x8.
    # The schoolbook rank is 2*4*8 = 64.
    # We can decompose this into 8 independent 2x2 * 2x2 multiplications.
    # The 4-dimension inner index is split into two 2-dim parts.
    # The 8-dimension outer index is split into four 2-dim parts.
    # The 2-dimension first index is kept as is.
    # So we have 2 (rows) * 2 (inner splits) * 4 (outer splits) = 16? 
    # Wait, let's look at the indices.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # i in 0..1 (2 values)
    # j in 0..3 (4 values)
    # k in 0..7 (8 values)
    
    # We can group j into j1 in {0,1} and j2 in {2,3}.
    # We can group k into k1 in {0,1}, k2 in {2,3}, k3 in {4,5}, k4 in {6,7}.
    # For each fixed i, the calculation is:
    # C[i, k_block] = A[i, j1] * B[j1, k_block] + A[i, j2] * B[j2, k_block]
    # This is a 2x2 matrix multiplication for each (i, k_block) pair?
    # No, it's a sum of two 2x2 matrix multiplications?
    
    # Let's re-evaluate the direct sum structure.
    # The tensor T has indices (i, j, k).
    # T[i, j, k] = delta(i, row) delta(j, col) ... actually T is the multiplication tensor.
    # T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C * T_{ijk} where T_{ijk}=1 if C_ik depends on A_ij B_jk.
    # Actually, the tensor is T = sum_{j} (sum_i e_i^A e_j^B) \otimes (sum_k e_j^B e_k^C)? No.
    # The standard tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k for the product C_ik = sum_j A_ij B_jk.
    
    # If we fix i and a block of k (size 2), and a block of j (size 2):
    # C[i, k_block] = sum_{j in j_block} A[i, j] B[j, k_block]
    # This is a matrix product of a (1x2) row vector A[i, j_block] and a (2x2) matrix B[j_block, k_block].
    # The result is a (1x2) vector.
    # This is not a full 2x2 matrix multiplication.
    
    # Let's look at it differently.
    # R(2, 4, 8) >= R(2, 2, 2) * (4/2) * (8/2)? No, rank is multiplicative in direct sums.
    # Is T_{2,4,8} isomorphic to T_{2,2,2} \oplus T_{2,2,2} \oplus ...?
    # Yes, because the supports are disjoint.
    # Let's verify the disjointness.
    # A term in the decomposition contributes to C[i, k] via A[i, j] B[j, k].
    # If we decompose T_{2,4,8} into 8 copies of T_{2,2,2}:
    # For each i in {0,1}, each j_block in {0,1} (where j_block 0 is j=0,1 and j_block 1 is j=2,3), 
    # and each k_block in {0,1,2,3} (where k_block b is k=2b, 2b+1).
    # The sub-tensor for (i, j_block, k_block) involves:
    # A indices: i fixed? No, T_{2,2,2} involves all 2 rows of A.
    # If we use Strassen on the 2x2 block, we need A[0:2, j_block] and B[j_block, k_block].
    # But A is 2x4, so A[0:2, j_block] is a 2x2 matrix.
    # B is 4x8, so B[j_block, k_block] is a 2x2 matrix.
    # The product is a 2x2 matrix C[i, k_block] for i in {0,1}?
    # Wait, if we use a 2x2 matrix multiplication, it computes C[0:2, k_block] = A[0:2, j_block] * B[j_block, k_block].
    # This computes 4 entries of C.
    # The entries are C[i, k] for i in {0,1} and k in {2b, 2b+1}.
    # These entries depend on A[i, j] for j in {2a, 2a+1} and B[j, k] for j in {2a, 2a+1}.
    # Do these entries depend on other A or B entries?
    # C[i, k] = sum_{j=0}^3 A[i, j] B[j, k].
    # If we sum over all 4 j_blocks (which is just the 2 j_blocks, wait, j has 4 elements, so 2 blocks of 2)
    # and all 4 k_blocks (k has 8 elements, so 4 blocks of 2).
    # Total blocks: 2 (i is not split, it's the row dim of the 2x2) ?
    # Let's be precise.
    # A is 2x4. B is 4x8.
    # Split J (4) into J1={0,1}, J2={2,3}.
    # Split K (8) into K1={0,1}, K2={2,3}, K3={4,5}, K4={6,7}.
    # For a fixed i? No, Strassen computes a 2x2 matrix product.
    # Let M_{a,b} be the product of A[:, J_a] (2x2) and B[J_b, K_c] (2x2).
    # This product computes a 2x2 matrix P_{a,b,c} where P_{a,b,c}[i, k_offset] = sum_{j in J_b} A[i, j] B[j, K_c + k_offset].
    # This partial product contributes to C[i, K_c + k_offset].
    # The total C[i, k] is the sum of these partial products over all b such that K_c is fixed?
    # No, k is fixed. K_c is determined by k. So for a fixed k, only one c is involved.
    # But j varies. So C[i, k] = sum_{b=0}^1 (A[:, J_b] * B[J_b, K_c])[i, k_offset].
    # This is a sum of two 2x2 matrix products?
    # No, it's a sum of two vectors (rows of the 2x2 product).
    # Actually, we can view the whole 2x4 * 4x8 multiplication as:
    # C = A * B
    # Let A = [A_1, A_2] where A_1, A_2 are 2x2.
    # Let B = [B_1; B_2] where B_1, B_2 are 2x8.
    # C = A_1 B_1 + A_2 B_2.
    # B_1 is 2x8, which can be split into [B_{1,1}, B_{1,2}, B_{1,3}, B_{1,4}] (2x2 each).
    # B_2 is 2x8, split into [B_{2,1}, B_{2,2}, B_{2,3}, B_{2,4}] (2x2 each).
    # So C = sum_{c=1}^4 (A_1 B_{1,c} + A_2 B_{2,c}).
    # Each term A_1 B_{1,c} is a 2x2 matrix multiplication.
    # There are 8 such terms: A_1 B_{1,c} and A_2 B_{2,c} for c=1..4.
    # Are they independent?
    # The output blocks are C_c (2x2).
    # C_c = A_1 B_{1,c} + A_2 B_{2,c}.
    # The inputs A_1, A_2, B_{1,c}, B_{2,c} are distinct for different c.
    # So the tensor is the direct sum of 8 tensors of the form A_1 B_{1,c} + A_2 B_{2,c}.
    # Wait, is A_1 B_{1,c} + A_2 B_{2,c} a standard matrix multiplication tensor?
    # No, it's the sum of two matrix multiplication tensors that share no output indices?
    # Actually, the output of A_1 B_{1,c} is the same 2x2 block as the output of A_2 B_{2,c}.
    # So they are not disjoint in output space.
    # However, we can combine them.
    # Let's consider the sub-problem for one output block c (2x2 output).
    # Inputs: A_1 (2x2), A_2 (2x2), B_{1,c} (2x2), B_{2,c} (2x2).
    # Output: C_c (2x2) = A_1 B_{1,c} + A_2 B_{2,c}.
    # This is not a standard matrix multiplication tensor T_{2,2,2}.
    # T_{2,2,2} computes C = A B.
    # Here we compute C = A_1 B_1 + A_2 B_2.
    # This is equivalent to multiplying a 4x2 matrix [A_1^T, A_2^T]? No.
    # It is equivalent to ( [A_1 A_2] * [B_{1,c}; B_{2,c}] ).
    # Let A' = [A_1 A_2] (2x4) and B' = [B_{1,c}; B_{2,c}] (4x2).
    # Then C_c = A' B'.
    # So each output block corresponds to a 2x4 * 4x2 matrix multiplication.
    # And there are 4 such blocks (c=1..4).
    # So the total tensor is T_{2,4,2} \oplus T_{2,4,2} \oplus T_{2,4,2} \oplus T_{2,4,2}.
    # So R(2,4,8) = 4 * R(2,4,2).
    # What is R(2,4,2)?
    # By symmetry R(2,4,2) = R(2,2,4).
    # And R(2,2,4) = 2 * R(2,2,2)?
    # Let's check R(2,2,4).
    # C (2x4) = A (2x2) * B (2x4).
    # Split B into B1, B2 (2x2).
    # C1 = A B1, C2 = A B2.
    # These are two independent 2x2 multiplications.
    # So R(2,2,4) = 2 * R(2,2,2) = 2 * 7 = 14.
    # Therefore R(2,4,8) = 4 * 14 = 56.
    
    # So the minimal rank is 56.
    # We need to construct the U, V, W matrices of rank 56.
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 56
    
    # We will use Strassen's algorithm for each of the 8 sub-multiplications of 2x2 * 2x2.
    # Wait, the decomposition is 4 blocks of 2x4 * 4x2.
    # Each 2x4 * 4x2 is two 2x2 * 2x2 multiplications.
    # So total 8 2x2 * 2x2 multiplications.
    
    # Strassen's algorithm for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We need to map these to the global indices.
    
    # For each of the 8 sub-problems:
    # Sub-problem s (0..7).
    # s corresponds to:
    #   block_c = s // 2 (0..3) -> output k block
    #   inner_b = s % 2 (0..1) -> which part of the 4x2 multiplication?
    #   Actually, let's stick to the 2x4 * 4x2 view.
    #   For each c in 0..3 (output block), we have a 2x4 * 4x2 multiplication.
    #   This 2x4 * 4x2 is split into two 2x2 * 2x2 multiplications:
    #   One using A[:, 0:2] (A1) and B[0:2, 2c:2c+2] (B1,c)
    #   One using A[:, 2:4] (A2) and B[2:4, 2c:2c+2] (B2,c)
    #   Let s = 2*c + b, where b=0 for A1 part, b=1 for A2 part.
    
    # For sub-problem s:
    # A_sub (2x2):
    #   if b==0: A_sub[i, j_local] = A[i, 2*j_local] (j_local 0..1 -> j 0,1)
    #   if b==1: A_sub[i, j_local] = A[i, 2*j_local + 2] (j_local 0..1 -> j 2,3)
    # B_sub (2x2):
    #   B_sub[j_local, k_local] = B[2*b + j_local, 2*c + k_local]
    # C_sub (2x2):
    #   C_sub[i, k_local] is part of C[i, 2*c + k_local].
    
    # We will fill U, V, W for each Strassen multiplication.
    
    for s in range(8):
        c = s // 2  # 0..3
        b = s % 2   # 0..1
        
        # Base index in the rank dimension for this sub-problem
        base_rank = s * 7
        
        # Define the mapping for A indices (U matrix)
        # U[row, col] corresponds to A[row] contribution.
        # A has 2x4 = 8 elements. Indexing: i*m + j = 2*i + j.
        # For A_sub (2x2), local index (i, j_l) maps to global (i, j_g).
        # j_g = 2*j_l + 2*b.
        
        # Define the mapping for B indices (V matrix)
        # B has 4x8 = 32 elements. Indexing: j*p + k = 4*j + k.
        # For B_sub (2x2), local index (j_l, k_l) maps to global (j_g, k_g).
        # j_g = 2*b + j_l.
        # k_g = 2*c + k_l.
        
        # Define the mapping for C indices (W matrix)
        # C has 8x2 = 16 elements. Indexing: k*n + i = 2*k + i.
        # For C_sub (2x2), local index (k_l, i) maps to global (k_g, i).
        # k_g = 2*c + k_l.
        
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # A indices in U (row = 2*i + j)
        # B indices in V (row = 4*j + k)
        # C indices in W (row = 2*k + i)
        
        # We will iterate through the 7 Strassen multiplications.
        
        # Helper to add to matrices
        def add_U(row, col, val):
            U[row][col] += val
        def add_V(row, col, val):
            V[row][col] += val
        def add_W(row, col, val):
            W[row][col] += val
            
        # M1
        r = base_rank + 0
        # A00 (i=0, j_l=0 -> j_g=2*0+2*b=2*b) -> row 2*0 + 2*b = 2*b
        add_U(2*b, r, 1)
        # A11 (i=1, j_l=1 -> j_g=2*1+2*b=2+2*b) -> row 2*1 + 2+2*b = 4+2*b
        add_U(4+2*b, r, 1)
        # B00 (j_l=0, k_l=0 -> j_g=2*b, k_g=2*c) -> row 4*2*b + 2*c = 8*b + 2*c
        add_V(8*b + 2*c, r, 1)
        # B11 (j_l=1, k_l=1 -> j_g=2*b+1, k_g=2*c+1) -> row 4*(2*b+1) + 2*c+1 = 8*b+4+2*c+1 = 8*b+5+2*c
        add_V(8*b + 5 + 2*c, r, 1)
        
        # M2
        r = base_rank + 1
        # A10 (i=1, j_l=0 -> j_g=2*b) -> row 2+2*b
        add_U(2+2*b, r, 1)
        # A11 (i=1, j_l=1 -> j_g=2+2*b) -> row 4+2*b
        add_U(4+2*b, r, 1)
        # B00 (j_l=0, k_l=0 -> j_g=2*b, k_g=2*c) -> row 8*b + 2*c
        add_V(8*b + 2*c, r, 1)
        
        # M3
        r = base_rank + 2
        # A00 (i=0, j_l=0 -> j_g=2*b) -> row 2*b
        add_U(2*b, r, 1)
        # B10 (j_l=1, k_l=0 -> j_g=2*b+1, k_g=2*c) -> row 4*(2*b+1) + 2*c = 8*b+4+2*c
        add_V(8*b + 4 + 2*c, r, 1)
        # B11 (j_l=1, k_l=1 -> j_g=2*b+1, k_g=2*c+1) -> row 8*b+5+2*c
        add_V(8*b + 5 + 2*c, r, -1)
        
        # M4
        r = base_rank + 3
        # A11 (i=1, j_l=1 -> j_g=2+2*b) -> row 4+2*b
        add_U(4+2*b, r, 1)
        # B01 (j_l=0, k_l=1 -> j_g=2*b, k_g=2*c+1) -> row 8*b + 2*c+1
        add_V(8*b + 2*c + 1, r, 1)
        # B00 (j_l=0, k_l=0 -> j_g=2*b, k_g=2*c) -> row 8*b + 2*c
        add_V(8*b + 2*c, r, -1)
        
        # M5
        r = base_rank + 4
        # A00 (i=0, j_l=0 -> j_g=2*b) -> row 2*b
        add_U(2*b, r, 1)
        # A01 (i=0, j_l=1 -> j_g=2+2*b) -> row 2+2*b
        add_U(2+2*b, r, 1)
        # B11 (j_l=1, k_l=1 -> j_g=2*b+1, k_g=2*c+1) -> row 8*b+5+2*c
        add_V(8*b + 5 + 2*c, r, 1)
        
        # M6
        r = base_rank + 5
        # A10 (i=1, j_l=0 -> j_g=2*b) -> row 2+2*b
        add_U(2+2*b, r, 1)
        # A00 (i=0, j_l=0 -> j_g=2*b) -> row 2*b
        add_U(2*b, r, -1)
        # B00 (j_l=0, k_l=0 -> j_g=2*b, k_g=2*c) -> row 8*b + 2*c
        add_V(8*b + 2*c, r, 1)
        # B01 (j_l=0, k_l=1 -> j_g=2*b, k_g=2*c+1) -> row 8*b + 2*c+1
        add_V(8*b + 2*c + 1, r, 1)
        
        # M7
        r = base_rank + 6
        # A01 (i=0, j_l=1 -> j_g=2+2*b) -> row 2+2*b
        add_U(2+2*b, r, 1)
        # A11 (i=1, j_l=1 -> j_g=2+2*b) -> row 4+2*b
        add_U(4+2*b, r, -1)
        # B10 (j_l=1, k_l=0 -> j_g=2*b+1, k_g=2*c) -> row 8*b+4+2*c
        add_V(8*b + 4 + 2*c, r, 1)
        # B11 (j_l=1, k_l=1 -> j_g=2*b+1, k_g=2*c+1) -> row 8*b+5+2*c
        add_V(8*b + 5 + 2*c, r, 1)
        
        # Now update W (output coefficients)
        # C11 = M1+M4-M5+M7 -> W[2*k1 + i1]
        # C12 = M3+M5 -> W[2*k2 + i1]
        # C21 = M2+M4 -> W[2*k1 + i2]
        # C22 = M1-M2+M3+M6 -> W[2*k2 + i2]
        
        # i1=0, i2=1
        # k1=0, k2=1
        # W index: 2*k_g + i
        
        # C11 (i=0, k_l=0 -> k_g=2*c)
        w_idx_11 = 2*(2*c) + 0
        W[w_idx_11][base_rank + 0] += 1  # M1
        W[w_idx_11][base_rank + 3] += 1  # M4
        W[w_idx_11][base_rank + 4] += -1 # M5
        W[w_idx_11][base_rank + 6] += 1  # M7
        
        # C12 (i=0, k_l=1 -> k_g=2*c+1)
        w_idx_12 = 2*(2*c+1) + 0
        W[w_idx_12][base_rank + 2] += 1  # M3
        W[w_idx_12][base_rank + 4] += 1  # M5
        
        # C21 (i=1, k_l=0 -> k_g=2*c)
        w_idx_21 = 2*(2*c) + 1
        W[w_idx_21][base_rank + 1] += 1  # M2
        W[w_idx_21][base_rank + 3] += 1  # M4
        
        # C22 (i=1, k_l=1 -> k_g=2*c+1)
        w_idx_22 = 2*(2*c+1) + 1
        W[w_idx_22][base_rank + 0] += 1  # M1
        W[w_idx_22][base_rank + 1] += -1 # M2
        W[w_idx_22][base_rank + 2] += 1  # M3
        W[w_idx_22][base_rank + 5] += 1  # M6

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
