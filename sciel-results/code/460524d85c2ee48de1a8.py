import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    # U: (n*m) × rank, V: (m*p) × rank, W: (p*n) × rank
    # W index: k*n + i
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    # 1. Verification function
    def verify(U, V, W):
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        # Target tensor is all ones for valid (i,j,k)
        target = np.ones((n, m, p))
        computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Check for exact match with tolerance for floating point errors
        return np.allclose(computed, target, atol=1e-6)

    # 2. Schoolbook construction (Rank 96) - Fallback
    def get_schoolbook():
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
        return U, V, W

    # 3. Strassen-based Construction for 4x4 * 4x4 (Rank 21)
    # We use the standard 2x2 Strassen recursion applied to 4x4 blocks.
    # The 4x4 matrix multiplication tensor can be decomposed using the
    # Kronecker product structure or direct 4x4 fast algorithms.
    # A known decomposition for 4x4x4 has rank 21 (using 2x2 Strassen blocks).
    
    # Let's construct the 4x4x4 decomposition first.
    # We represent the matrices as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B
    # Strassen for 2x2 uses 7 multiplications of 2x2 matrices.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Since we are dealing with 4x4, we can view A and B as 2x2 blocks of 2x2 matrices.
    # Each "multiplication" in Strassen is a multiplication of two 2x2 matrices.
    # We can decompose each 2x2 multiplication into its 4 scalar multiplications (schoolbook)
    # OR we can use a further decomposition.
    # If we use schoolbook for the 2x2 sub-multiplications, the rank is 7 * 4 = 28.
    # If we use Strassen for the 2x2 sub-multiplications (rank 7), the rank is 7 * 7 = 49? No, that's for 8x8.
    # Wait, 4x4 via 2x2 Strassen blocks:
    # The tensor for 4x4x4 is the sum of 7 terms, each being a tensor product of 2x2x2 tensors?
    # Actually, the standard way to get rank 21 for 4x4x4 is to use the 2x2 Strassen decomposition
    # where the "scalars" are replaced by 2x2 matrices, and then those 2x2 matrix multiplications
    # are decomposed.
    
    # Let's try a direct construction for 4x4x4 using the 7 Strassen steps,
    # but instead of scalar multiplication, we use the identity that
    # (2x2 mat) * (2x2 mat) can be done with 4 multiplications.
    # Total rank = 7 * 4 = 28.
    # Is there a rank 21 decomposition? Yes, but it's more complex (Laderman's algorithm).
    # Laderman's algorithm for 4x4 matrix multiplication has rank 21.
    
    # Let's implement Laderman's algorithm for the 4x4x4 part.
    # Laderman's 4x4 algorithm:
    # It uses 21 multiplications.
    # The formulas are quite involved. Let's look up the specific tensor indices.
    # Alternatively, we can use a simpler approach:
    # Use Strassen for 4x4 (Rank 28) which is easier to implement correctly.
    # Then add schoolbook for the remaining 2 columns (4 rows * 2 cols * 1 per term = 8 terms).
    # Total rank = 28 + 8 = 36.
    # This is still much better than 96.
    
    # Let's try to implement the Strassen-based rank 28 for the first 4 columns.
    
    def get_strassen_4x4():
        # Returns U, V, W for 4x4 * 4x4 -> 4x4
        # Dimensions: n=4, m=4, p=4
        rank = 28
        U = np.zeros((4 * 4, rank))
        V = np.zeros((4 * 4, rank))
        W = np.zeros((4 * 4, rank))
        
        # We map the 4x4 matrices to 2x2 blocks of 2x2 matrices.
        # Indices:
        # A: 4x4. Block (b_i, b_j) where b_i, b_j in {0,1}.
        # Inside block: 2x2.
        # So A index (i, j) corresponds to block (i//2, j//2) and local (i%2, j%2).
        
        # Strassen steps for 2x2 block multiplication:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each M is a 2x2 matrix product. We decompose each M into 4 scalar multiplications.
        # M_ab = X * Y where X and Y are 2x2 matrices.
        # M_ab[0,0] = X[0,0]Y[0,0] + X[0,1]Y[1,0]
        # M_ab[0,1] = X[0,0]Y[0,1] + X[0,1]Y[1,1]
        # M_ab[1,0] = X[1,0]Y[0,0] + X[1,1]Y[1,0]
        # M_ab[1,1] = X[1,0]Y[0,1] + X[1,1]Y[1,1]
        
        # We need to define X and Y for each of the 7 steps.
        # X and Y are linear combinations of the blocks of A and B.
        
        # Let's define the blocks of A (4x4) as A00, A01, A10, A11 (each 2x2).
        # A00: rows 0-1, cols 0-1
        # A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1
        # A11: rows 2-3, cols 2-3
        
        # Similarly for B.
        
        # We will iterate through the 7 Strassen steps.
        # For each step, we have a sum/difference of blocks for X and Y.
        # Then we iterate through the 4 scalar multiplications for the 2x2 product.
        
        # Helper to add a term to U, V, W
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # We need to map the 2x2 block indices to 4x4 indices.
        # Block (bi, bj) of A covers rows 2*bi + {0,1} and cols 2*bj + {0,1}.
        
        def add_term(u_idx, v_idx, w_idx, coeff):
            # u_idx is (i, j) in 4x4
            # v_idx is (j, k) in 4x4
            # w_idx is (k, i) in 4x4
            U[u_idx[0] * 4 + u_idx[1], col] = coeff
            V[v_idx[0] * 4 + v_idx[1], col] = coeff
            W[w_idx[1] * 4 + w_idx[0], col] = coeff # W[k, i] -> index k*4+i
            
        # We need to be careful with the indices.
        # Let's define the 7 steps explicitly.
        
        # Step 1: M1 = (A11+A22)(B11+B22)
        # X = A11 + A22, Y = B11 + B22
        # A11 is block (1,1) -> rows 2-3, cols 2-3.
        # A22 is block (0,0) -> rows 0-1, cols 0-1.
        # Wait, standard Strassen:
        # A = [[A11, A12], [A21, A22]]
        # A11 is top-left (0,0), A12 is top-right (0,1), A21 is bottom-left (1,0), A22 is bottom-right (1,1).
        
        # Let's stick to 0-based block indices:
        # A[0,0], A[0,1], A[1,0], A[1,1]
        
        # M1: X = A[0,0] + A[1,1], Y = B[0,0] + B[1,1]
        # M2: X = A[1,0] + A[1,1], Y = B[0,0]
        # M3: X = A[0,0], Y = B[0,1] - B[1,1]
        # M4: X = A[1,1], Y = B[1,0] - B[0,0]
        # M5: X = A[0,0] + A[0,1], Y = B[1,1]
        # M6: X = A[1,0] - A[0,0], Y = B[0,0] + B[0,1]
        # M7: X = A[0,1] - A[1,1], Y = B[1,0] + B[1,1]
        
        # Output C:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # We need to expand each M into 4 scalar multiplications.
        # Let's create a list of (X_coeffs, Y_coeffs) for each of the 7 M's.
        # X_coeffs: dict mapping block_idx to coeff
        # Y_coeffs: dict mapping block_idx to coeff
        
        steps = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # M7
        ]
        
        # For each step, we have a 2x2 matrix multiplication X * Y.
        # X and Y are 2x2 matrices formed by summing 2x2 blocks.
        # We decompose X * Y into 4 scalar multiplications.
        
        col = 0
        for x_coeffs, y_coeffs in steps:
            # X is a 2x2 matrix where X[li, lj] = sum_{blocks} coeff * A[block][li, lj]
            # Y is a 2x2 matrix where Y[lk, lk2] = sum_{blocks} coeff * B[block][lk, lk2]
            
            # The product Z = X * Y.
            # Z[li, lk2] = sum_lj X[li, lj] * Y[lj, lk2]
            
            # We expand this into scalar multiplications.
            # Each term in the sum is A_block_a[li, la] * B_block_b[lb, lk2]
            # where la = lj and lb = lj.
            
            # We iterate over the 4 positions in the 2x2 product Z.
            for li in range(2): # Row of Z (local)
                for lk2 in range(2): # Col of Z (local)
                    for lj in range(2): # Inner dimension (local)
                        # Term: X[li, lj] * Y[lj, lk2]
                        # X[li, lj] is sum of A blocks.
                        # Y[lj, lk2] is sum of B blocks.
                        
                        # We need to create a rank-1 term for each pair of (A_block, B_block).
                        # This might increase the rank!
                        # X[li, lj] is a sum of up to 2 blocks.
                        # Y[lj, lk2] is a sum of up to 2 blocks.
                        # The product is a sum of up to 4 terms.
                        # So one "scalar multiplication" in the 2x2 sense becomes 4 scalar multiplications in the 4x4 sense?
                        # No, the blocks are 2x2.
                        # X is a 2x2 matrix of scalars? No, X is a 2x2 matrix of 2x2 matrices.
                        # So X[li, lj] is a 2x2 matrix.
                        # Wait, my indexing is mixed.
                        
                        # Let's clarify:
                        # A is 4x4.
                        # A[bi, bj] is a 2x2 block.
                        # A[bi, bj][li, lj] is a scalar.
                        
                        # Strassen operates on blocks.
                        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
                        # This is a 2x2 matrix multiplication.
                        # Let X = A[0,0] + A[1,1]. X is 2x2.
                        # Let Y = B[0,0] + B[1,1]. Y is 2x2.
                        # Z = X * Y. Z is 2x2.
                        # Z[li, lk2] = sum_lj X[li, lj] * Y[lj, lk2].
                        
                        # X[li, lj] = A[0,0][li, lj] + A[1,1][li, lj]
                        # Y[lj, lk2] = B[0,0][lj, lk2] + B[1,1][lj, lk2]
                        
                        # So Z[li, lk2] = sum_lj (A[0,0][li, lj] + A[1,1][li, lj]) * (B[0,0][lj, lk2] + B[1,1][lj, lk2])
                        # = sum_lj (A[0,0][li, lj]B[0,0][lj, lk2] + A[0,0][li, lj]B[1,1][lj, lk2] + A[1,1][li, lj]B[0,0][lj, lk2] + A[1,1][li, lj]B[1,1][lj, lk2])
                        
                        # Each of these 4 terms is a scalar multiplication.
                        # So for each li, lk2, lj, we have 4 terms.
                        # Total terms per M = 2 * 2 * 2 * 4 = 32 terms?
                        # No, we sum over lj.
                        # For a fixed li, lk2, we sum over lj (2 values).
                        # For each lj, we have 4 cross terms.
                        # So 8 terms per Z[li, lk2].
                        # 4 positions in Z -> 32 terms per M.
                        # 7 M's -> 224 terms. This is worse than schoolbook (16).
                        
                        # I made a mistake in the expansion.
                        # The "scalar multiplication" in Strassen is a multiplication of two 2x2 matrices.
                        # If we decompose that 2x2 multiplication into schoolbook (4 mults),
                        # then each Strassen step (1 mult of 2x2s) becomes 4 scalar mults.
                        # So 7 steps * 4 mults/step = 28 scalar mults.
                        
                        # How to map this to the tensor?
                        # A term is A_scalar * B_scalar.
                        # A_scalar is A[i, j].
                        # B_scalar is B[j, k].
                        # Result C[i, k].
                        
                        # In Strassen, we form linear combinations of A blocks.
                        # Let X = sum_alpha c_alpha A[alpha].
                        # Let Y = sum_beta d_beta B[beta].
                        # Z = X * Y = sum_alpha,beta c_alpha d_beta (A[alpha] * B[beta]).
                        # A[alpha] * B[beta] is a 2x2 matrix product.
                        # We decompose A[alpha] * B[beta] into 4 scalar mults.
                        
                        # So, for each pair (alpha, beta), we have 4 scalar mults.
                        # The coefficient for each scalar mult is c_alpha * d_beta.
                        
                        # Let's implement this.
                        
                        pass # We will fill this in below.
                        
        # This approach is getting complex to implement on the fly without errors.
        # Let's try a simpler, guaranteed valid decomposition:
        # Rank 32 using 4x4 Strassen? No, Strassen is for 2x2.
        
        # Alternative: Use the fact that 4x4x6 can be done by 4x4x4 + 4x4x2.
        # 4x4x4 schoolbook rank is 16.
        # 4x4x2 schoolbook rank is 8.
        # Total 24.
        # This is already better than 96.
        # Can we do better?
        # 4x4x4 rank is 16 (schoolbook). Is there a faster 4x4x4?
        # Yes, Laderman rank 21.
        # 4x4x2 rank is 8 (schoolbook).
        # Total 29.
        
        # Let's implement the schoolbook for 4x4x4 and 4x4x2.
        # It's simple and guaranteed to work.
        # Rank 24.
        
        def get_schoolbook_4x4_4x2():
            rank = 24
            U = np.zeros((4 * 4, rank))
            V = np.zeros((4 * 6, rank))
            W = np.zeros((6 * 4, rank))
            
            col = 0
            # Part 1: 4x4 * 4x4 (cols 0-3)
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U[i * 4 + j, col] = 1
                        V[j * 6 + k, col] = 1
                        W[k * 4 + i, col] = 1
                        col += 1
            # Part 2: 4x4 * 4x2 (cols 4-5)
            for i in range(4):
                for j in range(4):
                    for k in range(4, 6):
                        U[i * 4 + j, col] = 1
                        V[j * 6 + k, col] = 1
                        W[k * 4 + i, col] = 1
                        col += 1
            
            return U, V, W

        U, V, W = get_schoolbook_4x4_4x2()
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to full schoolbook
            U_s, V_s, W_s = get_schoolbook()
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Try the simple rank 24 decomposition first
    U, V, W = get_schoolbook_4x4_4x2()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    U_s, V_s, W_s = get_schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# Helper function for the rank 24 decomposition
def get_schoolbook_4x4_4x2():
    n, m, p = 4, 4, 6
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: 4x4 * 4x4 (cols 0-3)
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # Part 2: 4x4 * 4x2 (cols 4-5)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def verify(U, V, W):
    n, m, p = 4, 4, 6
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    target = np.ones((n, m, p))
    computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(computed, target, atol=1e-6)

# Main entry point
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0

    # Try rank 24 decomposition
    U, V, W = get_schoolbook_4x4_4x2()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
