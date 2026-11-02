import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 4, 4, 8
    
    # Build 2x2 Strassen decomposition (rank 7)
    # U: (4, 7) - A indices: U[i*2+j]
    # V: (4, 7) - B indices: V[j*2+k]  
    # W: (4, 7) - C indices: W[k*2+i]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22) -> C11 + C22
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A21+A22)(B11) -> C21
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # M3 = (A11)(B12-B22) -> C12
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # M4 = (A22)(B21-B11) -> C12 + C21
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[1, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A11+A12)(B22) -> -C11 + C12
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12) -> C22
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22) -> C11 + C22
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1
    
    # Verify 2x2 Strassen
    # For any A (2x2), B (2x2): sum_r U[:,r] * V[:,r] * W[:,r] should equal C = A*B in tensor form
    # Tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] should equal delta for C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Verify: T[i,j,k] should be 1 if i,j,k form a valid C[i,k] = A[i,j]*B[j,k] path, else 0
    # Actually T[i,j,k] = 1 iff the (i,j,k) triple contributes to C[i,k] via A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k), which means T is the identity tensor for matmul
    
    # Let's verify with random matrices
    A_test = np.array([[1,2],[3,4]], dtype=float)
    B_test = np.array([[5,6],[7,8]], dtype=float)
    C_test = A_test @ B_test
    
    # Reconstruct: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = 1 for all i,j,k (since every (i,j,k) contributes)
    # Sum_r U[i,r]*V[j,r]*W[k,r] should = 1 for all i,j,k
    
    T_recon = np.zeros((2,2,2))
    for r in range(7):
        T_recon += np.outer(np.outer(U2[:,r], V2[:,r]), W2[:,r])
    
    # T_recon[i,j,k] should be 1 for all i,j,k
    if not np.allclose(T_recon, 1.0):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now compose 2x2 into 4x4 for 4x4 * 4x4 (rank 49)
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each of these is a 2x2 * 2x2 multiplication, so we can use Strassen for each.
    # But that would be 4 * 7 = 28 terms, not 49.
    # Wait, that's not right. The block structure means:
    # C11 = A11*B11 + A12*B21 (sum of two 2x2 matmuls)
    # To compute this exactly, we need separate decompositions for A11*B11 and A12*B21.
    # So we need 7 + 7 = 14 terms for C11 alone.
    # Total: 4 blocks * 14 = 56? No, that's not right either.
    
    # Actually, the standard way to compose Strassen for 4x4 is:
    # Use 7 Strassen products, each involving 2x2 block combinations.
    # M1 = (A11+A22)(B11+B22) - but these are 2x2 matrices, so each M is a 2x2 matrix
    # M1 = M1_11, M1_12, M1_21, M1_22 (four 2x2 matrices)
    # To compute M1, we need 7 rank-1 terms for the 2x2 multiplication.
    # So total rank = 7 (outer) * 7 (inner) = 49.
    
    # Let's build this properly.
    # The 4x4 tensor has indices (i, j, k) where i,j,k in {0,1,2,3}
    # A[i,j], B[j,k], C[i,k]
    
    # Split: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # where i1,j1,k1 in {0,1} and i2,j2,k2 in {0,1}
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We want to decompose T as sum_r U[i,r]*V[j,r]*W[k,r]
    
    # Using Kronecker product structure:
    # T = T_outer ⊗ T_inner where T_outer is the 2x2x2 block tensor and T_inner is 2x2x2
    # But it's not exactly a Kronecker product because of the block structure.
    
    # Let me think differently. The 4x4 matmul tensor can be written as:
    # T[i,j,k] = 1 if the path i->j->k is valid
    # = delta[i1,i2][j1,j2][k1,k2] where i=2*i1+i2, j=2*j1+j2, k=2*k1+k2
    
    # The tensor factors as:
    # T = sum_{i1,j1,k1} E[i1,i2,j1,j2,k1,k2]
    # where E is the elementary tensor for block (i1,j1,k1)
    
    # Actually, let's just build the 49 terms explicitly using the composition rule.
    
    # For 4x4 * 4x4, using Strassen composition:
    # Outer level: 7 Strassen combinations of 2x2 blocks
    # Inner level: each 2x2 block multiplication uses 7 Strassen terms
    
    # So we have 7*7 = 49 rank-1 terms.
    
    # Let me define the outer Strassen combinations:
    # Each outer term r_out has:
    # - A_block_combination: a 2x2 matrix of coefficients for A blocks
    # - B_block_combination: a 2x2 matrix of coefficients for B blocks  
    # - C_block_combination: a 2x2 matrix of coefficients for C blocks
    
    # For M1 = (A11+A22)(B11+B22):
    # A_comb: [[1,0],[0,1]] (A11 and A22)
    # B_comb: [[1,0],[0,1]] (B11 and B22)
    # C_comb: [[1,0],[0,1]] (C11 and C22)
    
    # For M2 = (A21+A22)(B11):
    # A_comb: [[0,0],[1,1]] (A21 and A22)
    # B_comb: [[1,0],[0,0]] (B11)
    # C_comb: [[0,0],[1,0]] (C21)
    
    # For M3 = (A11)(B12-B22):
    # A_comb: [[1,0],[0,0]] (A11)
    # B_comb: [[0,1],[0,-1]] (B12 - B22)
    # C_comb: [[0,1],[0,0]] (C12)
    
    # For M4 = (A22)(B21-B11):
    # A_comb: [[0,0],[0,1]] (A22)
    # B_comb: [[-1,0],[1,0]] (B21 - B11)
    # C_comb: [[0,1],[1,0]] (C12 + C21)
    
    # For M5 = (A11+A12)(B22):
    # A_comb: [[1,1],[0,0]] (A11 + A12)
    # B_comb: [[0,0],[0,1]] (B22)
    # C_comb: [[-1,1],[0,0]] (-C11 + C12)
    
    # For M6 = (A21-A11)(B11+B12):
    # A_comb: [[-1,0],[1,0]] (A21 - A11)
    # B_comb: [[1,1],[0,0]] (B11 + B12)
    # C_comb: [[0,0],[0,1]] (C22)
    
    # For M7 = (A12-A22)(B21+B22):
    # A_comb: [[0,1],[-1,0]] (A12 - A22)
    # B_comb: [[0,0],[1,1]] (B21 + B22)
    # C_comb: [[1,0],[0,1]] (C11 + C22)
    
    outer_A = np.array([
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[-1,0]]
    ])
    
    outer_B = np.array([
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]]
    ])
    
    outer_C = np.array([
        [[1,0],[0,1]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
        [[0,1],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[0,1]]
    ])
    
    # Now for each outer term r_out, and each inner Strassen term r_in:
    # The combined term contributes to U, V, W
    
    # U has indices (i, r) where i = 2*i1 + i2, r = r_out * 7 + r_in
    # U[i, r] = outer_A[r_out, i1, j1] * U2[i2, r_in]
    # Wait, that's not quite right. Let me think more carefully.
    
    # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    
    # For a given outer term r_out and inner term r_in:
    # The contribution to T[2*i1+i2, 2*j1+j2, 2*k1+k2] is:
    # outer_A[r_out, i1, j1] * outer_B[r_out, j1, k1] * outer_C[r_out, k1, i1] * ... 
    # No, that's not right either.
    
    # Let me reconsider. The composition works as follows:
    # The 4x4 tensor T can be written as a sum over block indices:
    # T = sum_{i1,j1,k1} T_{i1,j1,k1} ⊗ (something)
    
    # Actually, the cleanest way is:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    # Write i = 2*i1+i2, etc.
    # Then T = sum_{i1,j1,k1,i2,j2,k2} E_{i1,i2} ⊗ E_{j1,j2} ⊗ E_{k1,k2}
    # where E_{a,b} is the elementary tensor with 1 at position (a,b)
    
    # This doesn't directly help. Let me just construct the 49 terms by:
    # For each outer r_out in 0..6:
    #   For each inner r_in in 0..6:
    #     r = r_out * 7 + r_in
    #     For each i1 in 0,1 and i2 in 0,1:
    #       i = 2*i1 + i2
    #       U[i, r] += outer_A[r_out, i1, ?] * U2[i2, r_in]
    #     
    # Hmm, I'm getting confused. Let me look at this differently.
    
    # The 4x4 matmul can be seen as:
    # C = A * B
    # where A, B, C are 4x4 matrices.
    # Partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each of these is a sum of two 2x2 matmuls.
    # To compute C00, we need to compute A00*B00 and A01*B10 separately.
    # Each 2x2 matmul has rank 7, so C00 needs 14 terms.
    # Total: 4 * 14 = 56 terms. But that's not 49.
    
    # The Strassen composition gives 49 because it shares some computations.
    # Let me just hardcode the 49 terms for 4x4x4.
    
    # Actually, let me just use the schoolbook for 4x4x4 (rank 64) and then 
    # extend to 4x4x8 (rank 128). That gives score 1.0.
    
    # To get rank 98, I need the proper Strassen composition.
    # Let me implement it correctly.
    
    # The key insight: the 4x4 tensor T satisfies:
    # T[i,j,k] = 1 for all i,j,k
    # We can write T as a sum of 49 rank-1 tensors using the Kronecker structure.
    
    # Let me define the 49 terms explicitly.
    
    # For 4x4 * 4x4, the rank-49 decomposition is:
    # T = sum_{r1=0}^6 sum_{r2=0}^6 U[:, r1*7+r2] ⊗ V[:, r1*7+r2] ⊗ W[:, r1*7+r2]
    
    # where:
    # U[2*i1+i2, r1*7+r2] = outer_A[r1, i1, j1] * U2[i2, r2]  -- NO, this is wrong
    
    # Let me think about what U, V, W should be.
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    # U[i, r] corresponds to coefficient of A[i,j] in the r-th term
    # V[j, r] corresponds to coefficient of B[j,k] in the r-th term
    # W[k, i, r] corresponds to coefficient of C[i,k] in the r-th term
    
    # For the r-th term (r = r1*7 + r2):
    # The term is: (sum_i U[i,r] * A[i,j]) * (sum_j V[j,r] * B[j,k]) * (sum_k W[k,i,r] * C[i,k])
    # Wait, that's not the right form.
    
    # The tensor decomposition is:
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where T[i,j,k] = 1 for all i,j,k (for 4x4*4x4)
    
    # So we need: sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
    
    # Using the block structure:
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    
    # T[2*i1+i2, 2*j1+j2, 2*k1+k2] = 1
    
    # We want to write this as sum_r U[2*i1+i2, r] * V[2*j1+j2, r] * W[2*k1+k2, r]
    
    # Let r = r1*7 + r2.
    # U[2*i1+i2, r1*7+r2] = a[r1, i1] * U2[i2, r2]
    # V[2*j1+j2, r1*7+r2] = b[r1, j1] * V2[j2, r2]  
    # W[2*k1+k2, r1*7+r2] = c[r1, k1] * W2[k2, r2]
    
    # Then: sum_r U[i,r]*V[j,r]*W[k,r] = sum_{r1,r2} a[r1,i1]*U2[i2,r2] * b[r1,j1]*V2[j2,r2] * c[r1,k1]*W2[k2,r2]
    # = sum_{r1} a[r1,i1]*b[r1,j1]*c[r1,k1] * sum_{r2} U2[i2,r2]*V2[j2,r2]*W2[k2,r2]
    # = sum_{r1} a[r1,i1]*b[r1,j1]*c[r1,k1] * 1  (since inner sum = 1)
    # = sum_{r1} a[r1,i1]*b[r1,j1]*c[r1,k1]
    
    # We need this to equal 1 for all i1,j1,k1.
    # So we need: sum_{r1=0}^6 a[r1,i1]*b[r1,j1]*c[r1,k1] = 1 for all i1,j1,k1 in {0,1}
    
    # This means the outer tensor (with indices i1,j1,k1) must also be the all-ones tensor.
    # So a, b, c must form a rank-7 decomposition of the 2x2x2 all-ones tensor.
    # But the 2x2x2 all-ones tensor has rank 1! It's just 1*1*1.
    
    # Wait, that can't be right. Let me reconsider.
    
    # Actually, T[i,j,k] = 1 for all i,j,k means the tensor is the all-ones tensor of size 4x4x4.
    # The all-ones tensor of size n x n x n has rank 1 (it's just the outer product of three all-ones vectors).
    # But that's not the matrix multiplication tensor!
    
    # I made a fundamental error. The matrix multiplication tensor T[i,j,k] is NOT all ones.
    # T[i,j,k] = 1 if the (i,j,k) triple is a valid path in the matmul, which is all triples.
    # But the tensor is defined as T[i,j,k] = delta for the coefficient in C[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i,j,k, meaning the tensor IS the all-ones tensor.
    
    # But that's wrong. The all-ones tensor has rank 1, but matmul has rank much higher.
    
    # I think I'm confusing the tensor definition. Let me reconsider.
    
    # The matrix multiplication tensor is:
    # T[i,j,k] = 1 for all i,j,k
    # This is correct! The tensor is the all-ones tensor of size (n*m) x (m*p) x (p*n)
    # No wait, that's not right either.
    
    # Let me re-read the problem. The tensor for multiplying A (n x m) by B (m x p) to get C (n x p) is:
    # T[i,j,k] where i indexes A (i = row*m + col), j indexes B (j = row*p + col), k indexes C (k = row*n + col)
    # T[i,j,k] = 1 if the (i,j,k) triple corresponds to a valid term in C = A*B
    
    # For C[row_c, col_c] = sum_j A[row_a, j] * B[j, col_b]
    # The tensor entry T[i, j, k] = 1 if i = row_a*m + j_idx, j = j_idx*p + col_b, k = row_c*n + col_c
    # where row_a = row_c, j_idx = j, col_b = col_c
    
    # So T[i, j, k] = 1 for all valid (i,j,k) triples, which is all triples.
    # So the tensor IS the all-ones tensor!
    
    # But that can't be right because the all-ones tensor has rank 1.
    
    # I think the issue is that the tensor is not the all-ones tensor. Let me think again.
    
    # The tensor T is defined such that:
    # C[k] = sum_{i,j} T[i,j,k] * A[i] * B[j]
    # where C[k] = C[row_c, col_c], A[i] = A[row_a, col_a], B[j] = B[row_b, col_b]
    
    # For C[row_c, col_c] = sum_j A[row_c, j] * B[j, col_c]
    # So T[i,j,k] = 1 if i = row_c*m + j, j = j*p + col_c, k = row_c*n + col_c
    # Otherwise T[i,j,k] = 0
    
    # So T[i,j,k] = 1 only for specific (i,j,k) triples, not all of them!
    # Specifically, T[i,j,k] = 1 if the row/col indices are consistent.
    
    # For n=4, m=4, p=8:
    # i = row_a * 4 + col_a, where row_a, col_a in {0,1,2,3}
    # j = col_a * 8 + col_b, where col_a in {0,1,2,3}, col_b in {0,...,7}
    # k = row_a * 8 + col_b, where row_a in {0,1,2,3}, col_b in {0,...,7}
    
    # T[i,j,k] = 1 if i = row_a*4+col_a, j = col_a*8+col_b, k = row_a*8+col_b for some row_a, col_a, col_b
    # Otherwise T[i,j,k] = 0
    
    # So T is NOT the all-ones tensor. It's a sparse tensor with specific structure.
    
    # This makes much more sense. The rank of this tensor is what we're trying to minimize.
    
    # OK so my earlier approach was wrong. Let me restart with the correct understanding.
    
    # For n=4, m=4, p=8:
    # U has shape (16, rank), V has shape (32, rank), W has shape (32, rank)
    # U[i, r] = coefficient of A[i] in term r
    # V[j, r] = coefficient of B[j] in term r
    # W[k, r] = coefficient of C[k] in term r
    
    # The constraint is: for all i, j, k:
    # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    
    # where T[i,j,k] = 1 if i = row_a*4+col_a, j = col_a*8+col_b, k = row_a*8+col_b
    # and T[i,j,k] = 0 otherwise.
    
    # For the schoolbook decomposition (rank 128):
    # For each (row_a, col_a, col_b):
    #   i = row_a*4 + col_a
    #   j = col_a*8 + col_b
    #   k = row_a*8 + col_b
    #   U[i, r] = 1, V[j, r] = 1, W[k, r] = 1
    
    # This gives rank 4*4*8 = 128.
    
    # For Strassen, we want to reduce this.
    
    # Given the complexity, let me just return the schoolbook decomposition for now.
    # The score will be 1.0, which matches the current program.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
