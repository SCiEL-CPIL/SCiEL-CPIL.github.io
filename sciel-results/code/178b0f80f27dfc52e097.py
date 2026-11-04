import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy: Use Strassen's algorithm on 2x2 blocks of 3x3 matrices.
    # 6x6 = 2x2 blocks of 3x3.
    # Strassen uses 7 multiplications of 3x3 matrices.
    # Each 3x3 multiplication uses schoolbook (rank 27).
    # Total rank = 7 * 27 = 189.
    
    def construct_3x3_schoolbook():
        n3 = 3
        rank3 = n3*n3*n3  # 27
        U3 = np.zeros((n3*n3, rank3))
        V3 = np.zeros((n3*n3, rank3))
        W3 = np.zeros((n3*n3, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i*n3*n3 + j*n3 + k
                    U3[i*n3+j, col] = 1.0
                    V3[j*n3+k, col] = 1.0
                    W3[k*n3+i, col] = 1.0
        return U3, V3, W3
    
    U3, V3, W3 = construct_3x3_schoolbook()
    r3 = U3.shape[1]  # 27
    
    # Strassen's 7 products for 2x2 block matrix multiplication:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pk = Mk * Nk where Mk, Nk are 3x3 matrices that are linear combinations
    # of the A and B blocks.
    
    # For each product Pk, we need to express Mk and Nk as linear combinations
    # of the 3x3 blocks, then use the 3x3 schoolbook decomposition.
    
    # The key insight: if Mk = sum_a alpha_a * A_block_a and Nk = sum_b beta_b * B_block_b,
    # then Pk = sum_{a,b} alpha_a * beta_b * (A_block_a * B_block_b)
    # But this doesn't work directly because the 3x3 multiplication is bilinear.
    
    # Actually, the correct approach:
    # For each Pk, Mk is a specific 3x3 matrix formed from A blocks.
    # Nk is a specific 3x3 matrix formed from B blocks.
    # We decompose Pk = Mk * Nk using the 3x3 tensor.
    # The U part for Pk: for each (i,j) in 3x3, U[i,j] = coefficient of A_block in Mk
    # The V part for Pk: for each (j,k) in 3x3, V[j,k] = coefficient of B_block in Nk
    # The W part for Pk: for each (k,i) in 3x3, W[k,i] = 1 (output goes to C_block)
    
    # Let me define the block structure more carefully.
    # A is 6x6, indexed by (row, col) where row, col in 0..5.
    # A11 = A[0:3, 0:3], A12 = A[0:3, 3:6]
    # A21 = A[3:6, 0:3], A22 = A[3:6, 3:6]
    # Similarly for B.
    # C11 = C[0:3, 0:3], C12 = C[0:3, 3:6]
    # C21 = C[3:6, 0:3], C22 = C[3:6, 3:6]
    
    # U has shape (36, rank). Row index = i*6 + j for A[i,j].
    # V has shape (36, rank). Row index = j*6 + k for B[j,k].
    # W has shape (36, rank). Row index = k*6 + i for C[i,k].
    
    # For a 3x3 block A[ri:ri+3, rj:rj+3]:
    #   Global U row for local (li, lj) = (ri+li)*6 + (rj+lj)
    # For a 3x3 block B[ri:ri+3, rj:rj+3]:
    #   Global V row for local (li, lj) = (ri+li)*6 + (rj+lj)
    # For a 3x3 block C[ri:ri+3, rj:rj+3]:
    #   Global W row for local (lk, li) = (rj+lk)*6 + (ri+li)
    
    # Define block positions (row_offset, col_offset) for 3x3 blocks:
    # Block 11: (0, 0), Block 12: (0, 3), Block 21: (3, 0), Block 22: (3, 3)
    
    def get_U_rows(block_row, block_col):
        """Get global U row indices for a 3x3 block at A[block_row:block_row+3, block_col:block_col+3]"""
        rows = []
        for li in range(3):
            for lj in range(3):
                rows.append((block_row + li) * 6 + (block_col + lj))
        return rows
    
    def get_V_rows(block_row, block_col):
        """Get global V row indices for a 3x3 block at B[block_row:block_row+3, block_col:block_col+3]"""
        rows = []
        for li in range(3):
            for lj in range(3):
                rows.append((block_row + li) * 6 + (block_col + lj))
        return rows
    
    def get_W_rows(block_row, block_col):
        """Get global W row indices for a 3x3 block at C[block_row:block_row+3, block_col:block_col+3]"""
        rows = []
        for lk in range(3):
            for li in range(3):
                rows.append((block_col + lk) * 6 + (block_row + li))
        return rows
    
    # Strassen products definition:
    # Each product is defined by:
    #   A coefficients: dict mapping (block_row, block_col) -> coefficient for A
    #   B coefficients: dict mapping (block_row, block_col) -> coefficient for B
    #   C output: list of (block_row, block_col, sign) for where the result goes
    
    strassen_products = [
        # P1 = (A11 + A22) * (B11 + B22)
        {
            'A': {(0,0): 1, (3,3): 1},
            'B': {(0,0): 1, (3,3): 1},
            'C': [(0,0, 1), (3,3, 1)]  # P1 contributes to C11 and C22
        },
        # P2 = (A21 + A22) * B11
        {
            'A': {(3,0): 1, (3,3): 1},
            'B': {(0,0): 1},
            'C': [(3,0, 1), (3,3, -1)]  # P2 contributes to C21 (+) and C22 (-)
        },
        # P3 = A11 * (B12 - B22)
        {
            'A': {(0,0): 1},
            'B': {(0,3): 1, (3,3): -1},
            'C': [(0,3, 1), (3,3, 1)]  # P3 contributes to C12 (+) and C22 (+)
        },
        # P4 = A22 * (B21 - B11)
        {
            'A': {(3,3): 1},
            'B': {(3,0): 1, (0,0): -1},
            'C': [(0,0, 1), (3,0, 1)]  # P4 contributes to C11 (+) and C21 (+)
        },
        # P5 = (A11 + A12) * B22
        {
            'A': {(0,0): 1, (0,3): 1},
            'B': {(3,3): 1},
            'C': [(0,0, -1), (0,3, 1)]  # P5 contributes to C11 (-) and C12 (+)
        },
        # P6 = (A21 - A11) * (B11 + B12)
        {
            'A': {(3,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,3): 1},
            'C': [(3,3, 1)]  # P6 contributes to C22 (+)
        },
        # P7 = (A12 - A22) * (B21 + B22)
        {
            'A': {(0,3): 1, (3,3): -1},
            'B': {(3,0): 1, (3,3): 1},
            'C': [(0,0, 1)]  # P7 contributes to C11 (+)
        },
    ]
    
    total_rank = 7 * r3
    U6 = np.zeros((nm, total_rank))
    V6 = np.zeros((mp, total_rank))
    W6 = np.zeros((pn, total_rank))
    
    for p_idx, prod in enumerate(strassen_products):
        col_start = p_idx * r3
        
        # For each element in the 3x3 multiplication:
        for i3 in range(3):
            for j3 in range(3):
                for k3 in range(3):
                    col = col_start + i3*9 + j3*3 + k3
                    
                    # U part: coefficient from A blocks
                    # For local (i3, j3) in the 3x3 result, we need A[i3, j3]
                    # But A is a linear combination of blocks.
                    # For each A block with coefficient alpha, add alpha * U3[i3*3+j3, local_col]
                    # Wait, U3[i3*3+j3, col] is the coefficient for A[i3,j3] in the 3x3 schoolbook.
                    # In schoolbook, U3[i3*3+j3, i3*9+j3*3+k3] = 1.
                    # So for our case, we need to sum over A blocks.
                    
                    u_coeff = 0.0
                    for (ar, ac), alpha in prod['A'].items():
                        # This A block contributes to local position (i3, j3)
                        # In the 3x3 schoolbook, U3[i3*3+j3, i3*9+j3*3+k3] = 1
                        # So the contribution is alpha * 1 = alpha
                        u_coeff += alpha
                    
                    v_coeff = 0.0
                    for (br, bc), beta in prod['B'].items():
                        v_coeff += beta
                    
                    # W part: the output goes to C blocks
                    # For local (k3, i3) in the 3x3 result
                    # In schoolbook, W3[k3*3+i3, i3*9+j3*3+k3] = 1
                    # So we need to add to the appropriate C block.
                    
                    # Actually, let me reconsider the structure.
                    # The 3x3 schoolbook decomposition has:
                    # U3[i*3+j, col] = 1 if col corresponds to (i,j,k)
                    # V3[j*3+k, col] = 1 if col corresponds to (i,j,k)
                    # W3[k*3+i, col] = 1 if col corresponds to (i,j,k)
                    
                    # For our Strassen product, the "A matrix" is a linear combination of blocks.
                    # So for each (i3, j3), the U value is the sum of coefficients of A blocks
                    # that contain the element at local position (i3, j3).
                    
                    # But wait - each A block is a full 3x3 block. So for a given local (i3, j3),
                    # ALL A blocks in the combination contribute to that position with their coefficient.
                    
                    # So u_coeff = sum of all alpha values (since all blocks have the same local structure)
                    # Similarly v_coeff = sum of all beta values.
                    
                    # This means for each (i3, j3, k3), U6[global_row, col] = u_coeff
                    # where global_row maps to the A element at position (i3, j3) in the combined A.
                    
                    # But that's not right either. Let me think again.
                    
                    # The 3x3 schoolbook tensor is: T[i,j,k] = 1 for all i,j,k in 0..2
                    # This means: for each (i,j,k), there's one term with:
                    #   U[i*3+j] = 1, V[j*3+k] = 1, W[k*3+i] = 1
                    
                    # Now, in Strassen, Pk = Mk * Nk where Mk and Nk are 3x3 matrices.
                    # Mk[i,j] = sum of alpha_a * A_block_a[i,j] for each block a
                    # Nk[j,k] = sum of beta_b * B_block_b[j,k] for each block b
                    
                    # The product (Mk * Nk)[i,k] = sum_j Mk[i,j] * Nk[j,k]
                    # = sum_j (sum_a alpha_a * A_a[i,j]) * (sum_b beta_b * B_b[j,k])
                    # = sum_{a,b} alpha_a * beta_b * sum_j A_a[i,j] * B_b[j,k]
                    
                    # So the tensor for Pk is:
                    # T_Pk[i,j,k] = sum_{a,b} alpha_a * beta_b * (A_a[i,j] * B_b[j,k])
                    
                    # This means for each (i,j,k), the coefficient is sum_{a,b} alpha_a * beta_b
                    # And the U part picks out A_a[i,j], V part picks out B_b[j,k], W part picks out C[i,k].
                    
                    # So the rank-1 terms are indexed by (a, b, i, j, k) where a ranges over A blocks
                    # and b ranges over B blocks.
                    
                    # But that would give rank = len(A_blocks) * len(B_blocks) * 27, which is too high.
                    
                    # Wait, I'm overcomplicating this. Let me re-read the Strassen algorithm.
                    
                    # In Strassen, we compute Pk = Mk * Nk where Mk and Nk are specific 3x3 matrices.
                    # The multiplication Mk * Nk is a standard 3x3 matrix multiplication.
                    # We can decompose this using the 3x3 schoolbook tensor.
                    
                    # The key is: Mk[i,j] is a specific value (a linear combination of A elements),
                    # and Nk[j,k] is a specific value (a linear combination of B elements).
                    
                    # In the tensor decomposition, we need:
                    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = (Mk * Nk)[i,k] for all i,k
                    # But we also need to account for the linear combinations.
                    
                    # Actually, the correct way:
                    # The full 6x6 multiplication tensor is:
                    # T[i,j,k] = 1 for all i,j,k (in the sense that C[i,k] = sum_j A[i,j]*B[j,k])
                    
                    # Strassen reorganizes this as:
                    # C = sum_p Pk where Pk are the 7 products, each contributing to specific C blocks.
                    
                    # For each Pk, we have:
                    # (Pk contribution to C)[i,k] = sum_j Mk[i,j] * Nk[j,k]
                    # where Mk[i,j] = sum_a alpha_a * A[global_a(i,j)]
                    # and Nk[j,k] = sum_b beta_b * B[global_b(j,k)]
                    
                    # So (Pk contribution to C)[i,k] = sum_{a,b,j} alpha_a * beta_b * A[ga(i,j)] * B[gb(j,k)]
                    
                    # This means the tensor terms for Pk are:
                    # For each (a, b, i, j, k):
                    #   U[ga(i,j)] += alpha_a
                    #   V[gb(j,k)] += beta_b  
                    #   W[gc(i,k)] += 1  (with appropriate sign from C block assignment)
                    
                    # But this gives rank = |A_blocks| * |B_blocks| * 27 per product, which is too much.
                    
                    # I think I'm making an error. Let me look at this differently.
                    
                    # The 3x3 schoolbook decomposition of Mk * Nk is:
                    # (Mk * Nk)[i,k] = sum_j Mk[i,j] * Nk[j,k]
                    # = sum_j (sum_r U3[i,j,r] * M[r]) * (sum_r V3[j,k,r] * N[r])  -- no, this is wrong.
                    
                    # The schoolbook decomposition says:
                    # (X * Y)[i,k] = sum_j X[i,j] * Y[j,k] = sum_{i,j,k} X[i,j] * Y[j,k] * delta[i,k]
                    # In tensor form: T[i,j,k] = 1, and C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
                    # Wait, that's not right either.
                    
                    # Let me be very explicit. The 3x3 matrix multiplication tensor has:
                    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
                    # And the computation is: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
                    
                    # So for the schoolbook, we have 27 rank-1 terms, one for each (i,j,k):
                    # Term (i,j,k): U[i*3+j] = 1, V[j*3+k] = 1, W[k*3+i] = 1
                    
                    # Now for Strassen, Pk = Mk * Nk.
                    # We want to express Mk * Nk using the same tensor structure.
                    # (Mk * Nk)[i,k] = sum_j Mk[i,j] * Nk[j,k]
                    
                    # If we use the schoolbook tensor, we get:
                    # sum_{i,j,k} Mk[i,j] * Nk[j,k] * T[i,j,k] where T[i,j,k]=1
                    # = sum_j Mk[i,j] * Nk[j,k] ✓
                    
                    # So the decomposition of Pk using schoolbook is:
                    # For each (i,j,k): U[i*3+j] = Mk[i,j], V[j*3+k] = Nk[j,k], W[k*3+i] = 1
                    
                    # But Mk[i,j] is a linear combination of A elements, and Nk[j,k] is a linear combination of B elements.
                    # So we need to expand these.
                    
                    # Mk[i,j] = sum_a alpha_a * A[ga(i,j)]
                    # Nk[j,k] = sum_b beta_b * B[gb(j,k)]
                    
                    # So the term (i,j,k) becomes:
                    # sum_a sum_b alpha_a * beta_b * A[ga(i,j)] * B[gb(j,k)] * W[k*3+i]
                    
                    # This means we need |A_blocks| * |B_blocks| terms per (i,j,k), giving
                    # 27 * |A_blocks| * |B_blocks| terms per product.
                    
                    # For P1: |A_blocks|=2, |B_blocks|=2, so 27*4 = 108 terms.
                    # Total would be much more than 189.
                    
                    # I think the correct interpretation is different. Let me reconsider.
                    
                    # In Strassen's algorithm, the "multiplication" Mk * Nk is treated as a single
                    # black-box multiplication of two 3x3 matrices. The rank of this multiplication
                    # is the rank of the 3x3 matrix multiplication tensor, which is at most 27 (schoolbook).
                    
                    # The key insight: Mk and Nk are NOT arbitrary 3x3 matrices. They are specific
                    # linear combinations of the input blocks. But in the tensor decomposition,
                    # we treat the multiplication Mk * Nk as a standard 3x3 multiplication.
                    
                    # The way to handle this:
                    # - The U part should encode which A elements go into Mk
                    # - The V part should encode which B elements go into Nk
                    # - The W part should encode where the result goes in C
                    
                    # For the schoolbook 3x3 multiplication of Mk * Nk:
                    # Term (i,j,k): U[i*3+j] picks out Mk[i,j], V[j*3+k] picks out Nk[j,k], W[k*3+i] = 1
                    
                    # Since Mk[i,j] = sum_a alpha_a * A[ga(i,j)], the U part for term (i,j,k) should be:
                    # For each a: U[ga(i,j)] += alpha_a (but this is a sum, not a product)
                    
                    # This is where it gets tricky. The tensor decomposition requires rank-1 terms,
                    # meaning each term has a single U vector, single V vector, single W vector.
                    
                    # I think the correct approach is:
                    # For each product Pk, and for each (i,j,k) in 0..2:
                    #   Create |A_blocks| * |B_blocks| rank-1 terms.
                    #   For each (a, b):
                    #     U[ga(i,j)] = alpha_a
                    #     V[gb(j,k)] = beta_b
                    #     W[gc(i,k)] = sign (from C block assignment)
                    
                    # But this gives too many terms.
                    
                    # Wait, I think I've been misunderstanding the structure. Let me re-read the problem.
                    
                    # The tensor for 6x6 matrix multiplication is:
                    # T[i,j,k] = 1 for all i,j,k in 0..5
                    # C[i,k] = sum_j A[i,j] * B[j,k]
                    
                    # Strassen's algorithm computes this using 7 products of 3x3 matrices.
                    # Each product Pk = Mk * Nk where Mk and Nk are 3x3 matrices formed from A and B blocks.
                    
                    # The contribution of Pk to C is:
                    # For each (i,k) in the appropriate C block: C[i,k] += (Mk * Nk)[i_local, k_local]
                    # where i_local, k_local are local indices within the 3x3 block.
                    
                    # And (Mk * Nk)[i_local, k_local] = sum_{j_local} Mk[i_local, j_local] * Nk[j_local, k_local]
                    
                    # So the full tensor can be written as:
                    # T[i,j,k] = sum_{Pk} sum_{j_local} [Mk[i_local, j_local] * Nk[j_local, k_local]] * delta[i in C_block]
                    
                    # Where Mk[i_local, j_local] = sum_a alpha_a * A[ga(i_local, j_local)]
                    # and Nk[j_local, k_local] = sum_b beta_b * B[gb(j_local, k_local)]
                    
                    # So T[i,j,k] = sum_{Pk} sum_{j_local} sum_a sum_b alpha_a * beta_b * A[ga(i_local,j_local)] * B[gb(j_local,k_local)] * delta[i in C_block]
                    
                    # This means the rank-1 terms are indexed by (Pk, a, b, i_local, j_local, k_local) where:
                    # U[ga(i_local, j_local)] = alpha_a
                    # V[gb(j_local, k_local)] = beta_b
                    # W[gc(i_local, k_local)] = 1 (with sign)
                    
                    # Number of terms per Pk: |A_blocks| * |B_blocks| * 27
                    # For P1: 2*2*27 = 108
                    # Total: sum over 7 products of |A|*|B|*27
                    # P1: 2*2*27=108, P2: 2*1*27=54, P3: 1*2*27=54, P4: 1*2*27=54, P5: 2*1*27=54, P6: 2*2*27=108, P7: 2*2*27=108
                    # Total: 108+54+54+54+54+108+108 = 540. That's worse than 216!
                    
                    # I'm clearly making an error in my understanding. Let me look at this from a different angle.
                    
                    # The rank of the 6x6 multiplication tensor is at most 189 using Strassen + schoolbook 3x3.
                    # This is a well-known result. So my approach must be wrong.
                    
                    # Let me reconsider. The key is that in Strassen's algorithm, the 7 products are
                    # of 3x3 MATRICES, not of individual elements. The rank of multiplying two 3x3 matrices
                    # is at most 27 (schoolbook). So 7 * 27 = 189.
                    
                    # The way this works in the tensor decomposition:
                    # - We treat each 3x3 matrix multiplication as a separate tensor.
                    # - The "input" to each 3x3 multiplication is a linear combination of the original A and B blocks.
                    # - The "output" of each 3x3 multiplication is added to specific C blocks.
                    
                    # In the tensor decomposition:
                    # For each Pk, we have a 3x3 multiplication tensor with rank 27.
                    # The U part of this tensor encodes how A blocks are combined to form Mk.
                    # The V part encodes how B blocks are combined to form Nk.
                    # The W part encodes where the result goes in C.
                    
                    # The crucial point: for a given (i,j,k) in the 3x3 tensor, the U value is NOT
                    # a sum over A blocks. Instead, each rank-1 term in the 3x3 tensor corresponds to
                    # a specific element of Mk and Nk.
                    
                    # But Mk[i,j] is a linear combination of A elements. So how do we represent this
                    # in a rank-1 term?
                    
                    # I think the answer is: we DON'T expand the linear combinations in the tensor.
                    # Instead, the U vector for a given (i,j,k) has non-zero entries at ALL the A block
                    # positions that contribute to Mk[i,j], with the appropriate coefficients.
                    
                    # But that would make the U vector dense, and the term would no longer be rank-1
                    # in the original A, B, C space.
                    
                    # I'm stuck. Let me try a completely different approach.
                    
                    # Alternative: Just use the schoolbook 6x6 decomposition (rank 216) and try to
                    # find a lower rank decomposition using optimization.
                    
                    # Or, let me look at how the inspirations achieved rank 189.
                    
                    # The inspirations say they use "block-wise Strassen decomposition for 6x6 matrix
                    # multiplication, achieving a rank of 189 (7 Strassen steps * 27 schoolbook 3x3
                    # multiplications)."
                    
                    # So the approach is: 7 Strassen steps, each using a 3x3 schoolbook multiplication.
                    # Total rank = 7 * 27 = 189.
                    
                    # The key must be that each "3x3 schoolbook multiplication" in the Strassen context
                    # is NOT the same as the standard 3x3 schoolbook. It's a modified version where
                    # the inputs are linear combinations of blocks.
                    
                    # Let me try to think about this more carefully.
                    
                    # In the standard 3x3 schoolbook:
                    # C[i,k] = sum_j A[i,j] * B[j,k]
                    # Tensor: T[i,j,k] = 1
                    # Rank-1 terms: for each (i,j,k), U[i*3+j]=1, V[j*3+k]=1, W[k*3+i]=1
                    
                    # In Strassen, for P1 = (A11+A22) * (B11+B22):
                    # Let M = A11+A22, N = B11+B22.
                    # P1[i,k] = sum_j M[i,j] * N[j,k]
                    # = sum_j (A11[i,j]+A22[i,j]) * (B11[j,k]+B22[j,k])
                    # = sum_j A11[i,j]*B11[j,k] + sum_j A11[i,j]*B22[j,k] + sum_j A22[i,j]*B11[j,k] + sum_j A22[i,j]*B22[j,k]
                    
                    # So P1 is the sum of 4 products of 3x3 matrices:
                    # A11*B11 + A11*B22 + A22*B11 + A22*B22
                    
                    # Each of these is a standard 3x3 matrix multiplication with rank 27.
                    # So P1 has rank at most 4*27 = 108.
                    
                    # But that's not 27. So the Strassen trick must be doing something else.
                    
                    # OH WAIT. I think I've been misunderstanding Strassen's algorithm.
                    
                    # In Strassen's algorithm, the 7 products are NOT expanded into individual
                    # element products. The point is that we compute 7 products of (n/2)x(n/2) matrices
                    # instead of 8 products. The "product" of two (n/2)x(n/2) matrices is a single
                    # operation that has rank equal to the rank of matrix multiplication of size n/2.
                    
                    # So for 6x6, we view it as 2x2 blocks of 3x3.
                    # We compute 7 products of 3x3 matrices.
                    # Each product of 3x3 matrices has rank at most 27 (schoolbook).
                    # Total rank = 7 * 27 = 189.
                    
                    # The key is: when we say "compute P1 = M * N" where M and N are 3x3 matrices,
                    # we're using a rank-27 decomposition of the 3x3 multiplication tensor.
                    # The inputs M and N are linear combinations of the original A and B blocks,
                    # but the multiplication itself is just a standard 3x3 multiplication.
                    
                    # In the tensor decomposition, this means:
                    # - The U part for P1 encodes how to form M from A blocks.
                    # - The V part for P1 encodes how to form N from B blocks.
                    # - The W part for P1 encodes where the result goes in C.
                    
                    # But how do we encode "M = A11 + A22" in the U part?
                    # The U part is a vector of length 36 (for 6x6).
                    # For P1, M[i,j] = A11[i,j] + A22[i,j] for i,j in 0..2.
                    # So U should have non-zero entries at the positions corresponding to A11 and A22.
                    
                    # Specifically, for the (i,j,k) term in the 3x3 schoolbook:
                    # U[i*3+j] in the 3x3 space corresponds to M[i,j].
                    # In the 6x6 space, M[i,j] = A11[i,j] + A22[i,j].
                    # So U6[global_row_for_A11(i,j)] += 1
                    #    U6[global_row_for_A22(i,j)] += 1
                    
                    # But this means the U vector has TWO non-zero entries, which is fine for a
                    # rank-1 term in the 6x6 tensor. The term is still rank-1 in the sense that
                    # it's a single outer product U ⊗ V ⊗ W.
                    
                    # So the decomposition is:
                    # For each Pk, for each (i,j,k) in 0..2:
                    #   U vector: has non-zero entries at all A block positions that contribute to Mk[i,j]
                    #   V vector: has non-zero entries at all B block positions that contribute to Nk[j,k]
                    #   W vector: has non-zero entries at all C block positions that receive the result
                    #             with the appropriate signs.
                    
                    # This gives 27 terms per Pk, for a total of 7*27 = 189 terms.
                    
                    # Let me implement this.
                    
                    break  # Break out of the inner loops, we'll redo this properly.
                
                break
            break
        
        # For each (i3, j3, k3) in the 3x3 schoolbook:
        for i3 in range(3):
            for j3 in range(3):
                for k3 in range(3):
                    col = col_start + i3*9 + j3*3 + k3
                    
                    # U part: Mk[i3, j3] is a linear combination of A blocks
                    for (ar, ac), alpha in prod['A'].items():
                        # Global row in U for A[ar+i3, ac+j3]
                        global_u_row = (ar + i3) * 6 + (ac + j3)
                        U6[global_u_row, col] += alpha
                    
                    # V part: Nk[j3, k3] is a linear combination of B blocks
                    for (br, bc), beta in prod['B'].items():
                        # Global row in V for B[br+j3, bc+k3]
                        global_v_row = (br + j3) * 6 + (bc + k3)
                        V6[global_v_row, col] += beta
                    
                    # W part: result goes to C blocks
                    for (cr, cc), sign in prod['C']:
                        # Global row in W for C[cr+i3, cc+k3]
                        # W row index = k*n + i = k3_global * 6 + i3_global
                        # where k3_global = cc + k3, i3_global = cr + i3
                        global_w_row = (cc + k3) * 6 + (cr + i3)
                        W6[global_w_row, col] += sign
    
    # Verify the decomposition
    def is_valid(U, V, W):
        U3 = U.reshape(6, 6, -1)
        V3 = V.reshape(6, 6, -1)
        W3 = W.reshape(6, 6, -1)
        try:
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            target = np.ones((6, 6, 6))
            return np.allclose(reconstructed, target, atol=1e-9)
        except:
            return False
    
    if not is_valid(U6, V6, W6):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m + j
                    jk = j*p + k
                    ki = k*n + i
                    col = i*m*p + j*p + k
                    U[idx, col] = 1.0
                    V[jk, col] = 1.0
                    W[ki, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
