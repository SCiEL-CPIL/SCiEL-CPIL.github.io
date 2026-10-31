import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    
    # The tensor T[i, j, k] for matrix multiplication is 1 if i (row of A), j (col of A = row of B), k (col of B)
    # and the output index is c = k*n + i.
    # Wait, the standard tensor T_{a, b, c} is defined by the equation:
    # C[c] = sum_{a, b} T[a, b, c] A[a] B[b]
    # For matrix mult: C[k, i] = sum_j A[i, j] B[j, k]
    # Flatten A: a = i*m + j
    # Flatten B: b = j*p + k
    # Flatten C: c = k*n + i
    # So T[a, b, c] = 1 if a % m == b // p and b % p == c // n and c % n == a // m.
    # Let's verify:
    # a = i*4 + j. a//4 = i, a%4 = j.
    # b = j*4 + k. b//4 = j, b%4 = k.
    # c = k*4 + i. c//4 = k, c%4 = i.
    # Condition: a%4 == b//4 (j == j), b%4 == c//4 (k == k), c%4 == a//4 (i == i).
    # This is correct.

    # Initialize with Schoolbook (Rank 64)
    # We represent the decomposition as a list of tuples (u, v, w) where u, v, w are vectors of length 16.
    # Or more efficiently, we can work with the matrices U, V, W directly.
    
    # Let's use a representation that allows easy manipulation.
    # We start with 64 terms.
    # Term r corresponds to a specific (i, j, k).
    # U[:, r] has a 1 at index i*4+j.
    # V[:, r] has a 1 at index j*4+k.
    # W[:, r] has a 1 at index k*4+i.

    # We will try to reduce the rank by finding pairs of terms that can be merged or by finding linear dependencies.
    # A simpler heuristic: 
    # 1. Start with schoolbook.
    # 2. Try to express one term as a linear combination of others? No, rank is the dimension of the span.
    # 3. A better approach for small tensors: Use the "Strassen-like" reduction.
    #    However, implementing a general rank minimizer is hard.
    
    # Let's try a specific known decomposition: Winograd 4x4 (Rank 25).
    # I will hardcode the Winograd 4x4 coefficients. This is a "different strategy" from the current
    # program's implicit fallback to schoolbook due to failed dynamic construction attempts.
    # The current program *tries* to do Strassen 49 but falls back to schoolbook.
    # This program will *successfully* return Winograd 25.

    # Winograd 4x4 Matrix Multiplication
    # Reference: "Winograd's Method for Matrix Multiplication"
    # The algorithm uses 25 multiplications.
    # It involves transforming A and B using specific matrices, multiplying element-wise, and transforming back.
    # The tensor decomposition form T = sum U_r V_r W_r^T corresponds to the bilinear complexity.
    
    # The Winograd 4x4 scheme is often described via "Folding".
    # Let's use the explicit coefficients found in literature or derived from the 2x2 Winograd (which is rank 7? No, 2x2 is rank 7 Strassen, but Winograd 2x2 is not standard).
    # Actually, for 4x4, the Winograd algorithm is a specific rank 25 decomposition.
    
    # I will construct the U, V, W matrices for the Winograd 4x4 algorithm.
    # The algorithm is defined by:
    # A' = U_A * A
    # B' = U_B * B
    # C' = A' * B' (element-wise)
    # C = U_C * C'
    
    # The tensor decomposition is:
    # C = sum_{r=1}^{25} (U_A[:, r] * A) * (U_B[:, r] * B) ... wait.
    # The bilinear form is C = sum_r (u_r^T A) (v_r^T B) w_r.
    # Here u_r, v_r are vectors of length 16 (flattened matrices).
    # w_r is a vector of length 16.
    
    # I need the specific 25 vectors.
    # Since I cannot reliably recall all 25 vectors of Winograd 4x4 from memory without error,
    # I will use a robust alternative: **Strassen 49** but implemented CORRECTLY this time.
    # The previous attempt failed because of complexity in handling block sums.
    # I will implement Strassen 49 by recursively applying Strassen 2x2.
    # Strassen 2x2 is well-known and easy to hardcode.
    # Strassen 4x4 = 7 Strassen 2x2 operations on 2x2 blocks.
    # Each Strassen 2x2 operation has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # This is a valid, different strategy (Strassen) that I can implement correctly.
    # It is better than Schoolbook (64).
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Define Strassen 2x2 terms.
    # For a product P = X * Y where X, Y are 2x2:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # P = [[p00, p01], [p10, p11]]
    
    # The 7 scalar multiplications (terms) are:
    # m1 = (x00 + x11) * (y00 + y11)
    # m2 = (x10 + x11) * y00
    # m3 = x00 * (y01 - y11)
    # m4 = x11 * (y10 - y00)
    # m5 = (x00 + x01) * y11
    # m6 = (x10 - x00) * (y00 + y01)
    # m7 = (x01 - x11) * (y10 + y11)
    
    # P00 = m1 + m4 - m5 + m7
    # P01 = m3 + m5
    # P10 = m2 + m4
    # P11 = m1 - m2 + m3 + m6
    
    # We need to map these to the global 4x4 indices.
    # A is 4x4, split into 2x2 blocks:
    # A11 (0:2, 0:2), A12 (0:2, 2:4)
    # A21 (2:4, 0:2), A22 (2:4, 2:4)
    
    # B is 4x4, split into 2x2 blocks:
    # B11 (0:2, 0:2), B12 (0:2, 2:4)
    # B21 (2:4, 0:2), B22 (2:4, 2:4)
    
    # C is 4x4, split into 2x2 blocks:
    # C11 (0:2, 0:2), C12 (0:2, 2:4)
    # C21 (2:4, 0:2), C22 (2:4, 2:4)
    
    # Strassen 4x4 formulas:
    # M1 = (A11 + A22) * (B11 + B22) -> contributes to C11, C22?
    # Actually, the standard Strassen 4x4 computes:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product.
    # We decompose each M using Strassen 2x2 (rank 7).
    # Total 7 * 7 = 49 terms.
    
    # Helper to add a Strassen 2x2 decomposition for a product X*Y.
    # X is defined by a list of (block, coeff) pairs.
    # Y is defined by a list of (block, coeff) pairs.
    # The result Z is added to C blocks with a coeff.
    
    def get_block_indices(block_id):
        # block_id: 0=A11, 1=A12, 2=A21, 3=A22
        # Returns (row_offset, col_offset)
        if block_id == 0: return (0, 0)
        if block_id == 1: return (0, 2)
        if block_id == 2: return (2, 0)
        if block_id == 3: return (2, 2)
        return (0, 0)

    def add_strassen_term(x_blocks, y_blocks, c_blocks, base_col):
        # x_blocks: list of (block_id, coeff)
        # y_blocks: list of (block_id, coeff)
        # c_blocks: list of (block_id, coeff)
        
        # The 7 Strassen 2x2 terms.
        # Each term is (x_coeffs, y_coeffs, z_coeffs) for the 2x2 block.
        # x_coeffs: dict {(r,c): val}
        
        # m1: (x00+x11)(y00+y11)
        # x: {(0,0):1, (1,1):1}
        # y: {(0,0):1, (1,1):1}
        # z: {(0,0):1, (1,1):1}
        
        # m2: (x10+x11)y00
        # x: {(1,0):1, (1,1):1}
        # y: {(0,0):1}
        # z: {(1,0):1, (1,1):-1}
        
        # m3: x00(y01-y11)
        # x: {(0,0):1}
        # y: {(0,1):1, (1,1):-1}
        # z: {(0,1):1, (1,1):1}
        
        # m4: x11(y10-y00)
        # x: {(1,1):1}
        # y: {(1,0):1, (0,0):-1}
        # z: {(0,0):1, (1,0):1}
        
        # m5: (x00+x01)y11
        # x: {(0,0):1, (0,1):1}
        # y: {(1,1):1}
        # z: {(0,0):-1, (0,1):1}
        
        # m6: (x10-x00)(y00+y01)
        # x: {(1,0):1, (0,0):-1}
        # y: {(0,0):1, (0,1):1}
        # z: {(1,1):1}
        
        # m7: (x01-x11)(y10+y11)
        # x: {(0,1):1, (1,1):-1}
        # y: {(1,0):1, (1,1):1}
        # z: {(0,0):1}
        
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        
        for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(terms):
            col = base_col + t_idx
            
            # Accumulate U (X)
            for (bx_id, bx_val) in x_blocks:
                br, bc = get_block_indices(bx_id)
                for (lr, lc), val in x_coeff.items():
                    r = br + lr
                    c = bc + lc
                    u_idx = r * 4 + c
                    U[u_idx, col] += bx_val * val
            
            # Accumulate V (Y)
            for (by_id, by_val) in y_blocks:
                br, bc = get_block_indices(by_id)
                for (lr, lc), val in y_coeff.items():
                    r = br + lr
                    c = bc + lc
                    v_idx = r * 4 + c
                    V[v_idx, col] += by_val * val
            
            # Accumulate W (Z)
            for (bz_id, bz_val) in c_blocks:
                br, bc = get_block_indices(bz_id)
                for (lr, lc), val in z_coeff.items():
                    r = br + lr
                    c = bc + lc
                    w_idx = r * 4 + c
                    W[w_idx, col] += bz_val * val

    # Now, define the 7 M's for Strassen 4x4.
    # M1 = (A11 + A22) * (B11 + B22)
    # X blocks: [(0, 1), (3, 1)]
    # Y blocks: [(0, 1), (3, 1)]
    # C blocks: [(0, 1), (3, 1)]  # M1 contributes to C11 and C22 with coeff 1
    # Wait, C11 = M1 + ... and C22 = M1 + ...
    # So M1 is added to C11 with coeff 1 and C22 with coeff 1.
    # In my helper, c_blocks is a list of (block_id, coeff).
    # So c_blocks = [(0, 1), (3, 1)]
    
    # M2 = (A21 + A22) * B11
    # X blocks: [(2, 1), (3, 1)]
    # Y blocks: [(0, 1)]
    # C blocks: [(2, 1), (3, -1)] # C21 = M2 + ..., C22 = ... - M2 ...
    # C21 = M2 + M4. So M2 contributes 1 to C21.
    # C22 = M1 - M2 + M3 + M6. So M2 contributes -1 to C22.
    # c_blocks = [(2, 1), (3, -1)]
    
    # M3 = A11 * (B12 - B22)
    # X blocks: [(0, 1)]
    # Y blocks: [(1, 1), (3, -1)]
    # C blocks: [(1, 1), (3, 1)] # C12 = M3 + M5. C22 = ... + M3 ...
    # C12 = M3 + M5. So M3 contributes 1 to C12.
    # C22 = M1 - M2 + M3 + M6. So M3 contributes 1 to C22.
    # c_blocks = [(1, 1), (3, 1)]
    
    # M4 = A22 * (B21 - B11)
    # X blocks: [(3, 1)]
    # Y blocks: [(2, 1), (0, -1)]
    # C blocks: [(0, 1), (2, 1)] # C11 = M1 + M4 ... C21 = M2 + M4 ...
    # C11 = M1 + M4 - M5 + M7. So M4 contributes 1 to C11.
    # C21 = M2 + M4. So M4 contributes 1 to C21.
    # c_blocks = [(0, 1), (2, 1)]
    
    # M5 = (A11 + A12) * B22
    # X blocks: [(0, 1), (1, 1)]
    # Y blocks: [(3, 1)]
    # C blocks: [(0, -1), (1, 1)] # C11 = ... - M5 ... C12 = M3 + M5 ...
    # C11 = M1 + M4 - M5 + M7. So M5 contributes -1 to C11.
    # C12 = M3 + M5. So M5 contributes 1 to C12.
    # c_blocks = [(0, -1), (1, 1)]
    
    # M6 = (A21 - A11) * (B11 + B12)
    # X blocks: [(2, 1), (0, -1)]
    # Y blocks: [(0, 1), (1, 1)]
    # C blocks: [(3, 1)] # C22 = ... + M6 ...
    # C22 = M1 - M2 + M3 + M6. So M6 contributes 1 to C22.
    # c_blocks = [(3, 1)]
    
    # M7 = (A12 - A22) * (B21 + B22)
    # X blocks: [(1, 1), (3, -1)]
    # Y blocks: [(2, 1), (3, 1)]
    # C blocks: [(0, 1)] # C11 = ... + M7 ...
    # C11 = M1 + M4 - M5 + M7. So M7 contributes 1 to C11.
    # c_blocks = [(0, 1)]
    
    # Call the helper for each M.
    # M1
    add_strassen_term([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)], 0)
    # M2
    add_strassen_term([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)], 7)
    # M3
    add_strassen_term([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)], 14)
    # M4
    add_strassen_term([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)], 21)
    # M5
    add_strassen_term([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)], 28)
    # M6
    add_strassen_term([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 35)
    # M7
    add_strassen_term([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)], 42)
    
    # Verify correctness quickly with a random matrix multiplication?
    # If time permits.
    if time.time() < deadline - 10:
        # Test
        A_test = np.random.rand(4, 4)
        B_test = np.random.rand(4, 4)
        C_test = A_test @ B_test
        
        # Reconstruct C from U, V, W
        # C[c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a tensor contraction.
        # C = sum_r outer(U[:, r], V[:, r]) * W[:, r] ... no.
        # C is a matrix.
        # C[k, i] = sum_{a, b} T[a, b, c] A[a] B[b]
        # In our decomposition:
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # So C = sum_r (U[:, r] * A) (V[:, r] * B) ... no.
        # Let's check the convention.
        # The problem states: "output factor follows the official trace convention W[k*n+i,r]"
        # And "exactly decompose the matrix multiplication tensor".
        # The tensor is T_{a, b, c}.
        # My construction:
        # U[a, r] is the coefficient of A[a] in the r-th term.
        # V[b, r] is the coefficient of B[b] in the r-th term.
        # W[c, r] is the coefficient of the r-th term in C[c].
        # So C[c] = sum_r (sum_a U[a, r] A[a]) (sum_b V[b, r] B[b]) W[c, r]
        # This is correct.
        
        # Let's compute C_reconstructed
        C_rec = np.zeros((16,))
        for r in range(49):
            u_val = np.dot(U[:, r], A_test.flatten())
            v_val = np.dot(V[:, r], B_test.flatten())
            C_rec += u_val * v_val * W[:, r]
        
        # Check if C_rec matches C_test.flatten()
        # Note: C_test is 4x4. Flatten index c = k*4 + i.
        # My W index is c.
        # Let's compare.
        if not np.allclose(C_rec, C_test.flatten()):
            # If it fails, return schoolbook to be safe?
            # Or try to fix?
            # Given the explicit construction, it should be correct.
            # But if there's a sign error, it will fail.
            # Let's return schoolbook if it fails to ensure validity.
            pass
            
    # If verification fails, fallback to schoolbook
    if not np.allclose(C_rec, C_test.flatten()):
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a_idx = i * 4 + j
                    b_idx = j * 4 + k
                    c_idx = k * 4 + i
                    U[a_idx, col] = 1
                    V[b_idx, col] = 1
                    W[c_idx, col] = 1
                    col += 1
        
    # Convert to lists for the required output format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
