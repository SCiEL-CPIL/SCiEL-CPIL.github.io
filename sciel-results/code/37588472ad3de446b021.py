"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strategy: 
    # The current program uses 2x2 Strassen blocks for the whole 4x4 * 4x8 product.
    # This results in rank 112.
    # 
    # Let's try a different structural decomposition.
    # We can split the output C (4x8) into two parts: C1 (4x4) and C2 (4x4).
    # C1 = A * B1 (where B1 is the first 4 columns of B)
    # C2 = A * B2 (where B2 is the last 4 columns of B)
    #
    # For C1 = A(4x4) * B1(4x4), we can use the best known rank for 4x4 matrix multiplication.
    # The standard Strassen rank for 4x4 is 28 (using 2x2 Strassen 7-term blocks? No, 2x2 Strassen is rank 7 for 2x2.
    # For 4x4, we can use 4 blocks of 2x2, so rank 4*7 = 28.
    # Wait, is there a better rank for 4x4?
    # The known lower bound for 4x4 is 26, but 28 is the standard Strassen-based rank.
    # Actually, there is a rank 26 decomposition for 4x4? No, that's for 3x3 (rank 23).
    # For 4x4, the best known is 26? Let me check.
    # Laderman's algorithm is for 3x3 (rank 23).
    # For 4x4, using 2x2 Strassen gives 28.
    # Is there a rank 27 or 26? 
    # Actually, the "Strassen" rank for n x n is 7^(log2 n). For n=4, 7^2 = 49? No.
    # Strassen is for 2x2. To multiply 4x4, you do 4 independent 2x2 multiplications? 
    # No, you partition into 2x2 blocks.
    # C = [A00 A01; A10 A11] * [B00 B01; B10 B11]
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    # Each C_ij is a 2x2 matrix.
    # To compute C00, we need A00B00 and A01B10.
    # If we use Strassen for each 2x2 product, we need 7 multiplications for A00B00 and 7 for A01B10?
    # No, that's not how it works.
    # The standard Strassen recursion:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # This computes the full 2x2 product in 7 multiplications.
    # So for a 4x4 * 4x4 product, if we view it as a single 4x4 problem, we can't directly apply the 2x2 Strassen formula because the "elements" are 2x2 matrices.
    # But matrix multiplication is bilinear, so we can apply the Strassen formula to the blocks.
    # Let A = [A00 A01; A10 A11] where Aij are 2x2.
    # Let B = [B00 B01; B10 B11] where Bij are 2x2.
    # Then C = AB can be computed using the 7 Strassen products M1..M7, where each M is a 2x2 matrix product.
    # Each 2x2 matrix product (e.g., M1 = (A00+A11)(B00+B11)) requires 7 scalar multiplications if we use Strassen again, or 4 if we use schoolbook.
    # If we use schoolbook for the inner 2x2 products, the total rank is 7 * 4 = 28.
    # If we use Strassen for the inner 2x2 products, the total rank is 7 * 7 = 49.
    # So the rank 28 decomposition for 4x4 * 4x4 is valid.
    
    # So for C1 = A * B1 (4x4 * 4x4), we can use rank 28.
    # For C2 = A * B2 (4x4 * 4x4), we can use rank 28.
    # Total rank = 28 + 28 = 56.
    
    # This is much better than 112!
    
    # Let's construct the rank 28 decomposition for 4x4 * 4x4.
    # We define the 7 Strassen terms for the 2x2 block structure.
    # Each term involves products of 2x2 blocks.
    # To get a scalar decomposition, we expand each 2x2 block product into 4 scalar products.
    # Total 7 * 4 = 28 scalar products.
    
    # Let's define the 28 terms for 4x4 * 4x4.
    # Indices for A: i, j in {0,1,2,3} -> idx = i*4 + j
    # Indices for B: j, k in {0,1,2,3} -> idx = j*4 + k
    # Indices for C: i, k in {0,1,2,3} -> idx = k*4 + i (Trace convention W[k*n+i])
    
    # Strassen terms for 4x4 (using 2x2 blocks):
    # Let A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M7 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # Each M is a 2x2 matrix. We need to expand M into 4 scalar terms.
    # For example, M1 = (A00+A11)(B00+B11).
    # Let X = A00+A11, Y = B00+B11.
    # X is 2x2, Y is 2x2.
    # X = [x00 x01; x10 x11], Y = [y00 y01; y10 y11].
    # M1 = [x00y00+x01y10, x00y01+x01y11; x10y00+x11y10, x10y01+x11y11]
    # This is a 2x2 matrix product. We can use schoolbook for this 2x2 product (rank 4).
    # So M1 contributes 4 scalar terms.
    
    # Let's generate the 28 terms programmatically.
    
    def get_2x2_block_indices(mat, i0, j0):
        """Return list of (global_idx, coeff) for a 2x2 block starting at (i0, j0)."""
        # mat is 'A', 'B', or 'C'
        # For A: idx = i*4 + j
        # For B: idx = j*4 + k
        # For C: idx = k*4 + i
        pass # Not needed, we will hardcode the expansion.
    
    # Let's define the 28 terms for 4x4 * 4x4.
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (idx, coeff) for A
    # v_coeffs: list of (idx, coeff) for B
    # w_coeffs: list of (idx, coeff) for C
    
    # We will create a list of 28 tuples.
    
    # Helper to get A index
    def a_idx(i, j): return i*4 + j
    def b_idx(j, k): return j*4 + k
    def c_idx(i, k): return k*4 + i # Trace convention
    
    terms_4x4 = []
    
    # M1 = (A00+A11)(B00+B11)
    # X = A00+A11, Y = B00+B11
    # X[0,0] = A[0,0]+A[2,2], X[0,1] = A[0,1]+A[2,3]
    # X[1,0] = A[2,0]+A[3,2], X[1,1] = A[2,1]+A[3,3]
    # Y[0,0] = B[0,0]+B[2,2], Y[0,1] = B[0,1]+B[2,3]
    # Y[1,0] = B[2,0]+B[3,2], Y[1,1] = B[2,1]+B[3,3]
    
    # M1[0,0] = X[0,0]Y[0,0] + X[0,1]Y[1,0]
    # This is getting complicated to hardcode.
    # Let's use a simpler approach: 
    # We know the rank 28 decomposition exists.
    # We can construct it by taking the 7 Strassen terms and expanding each 2x2 product into 4 scalar products.
    
    # Let's define the 7 Strassen terms as pairs of (U_block, V_block) where U_block and V_block are 2x2 matrices of linear combinations of A and B entries.
    # Then for each term, we compute the 4 scalar products.
    
    # Strassen terms (for 2x2 blocks):
    # M1: U = A00+A11, V = B00+B11
    # M2: U = A10+A11, V = B00
    # M3: U = A00, V = B01-B11
    # M4: U = A11, V = B10-B00
    # M5: U = A00+A01, V = B11
    # M6: U = A10-A00, V = B00+B01
    # M7: U = A01-A11, V = B10+B11
    
    # For each term, U and V are 2x2 matrices.
    # U = [u00 u01; u10 u11], V = [v00 v01; v10 v11]
    # The product UV is a 2x2 matrix.
    # (UV)[0,0] = u00*v00 + u01*v10
    # (UV)[0,1] = u00*v01 + u01*v11
    # (UV)[1,0] = u10*v00 + u11*v10
    # (UV)[1,1] = u10*v01 + u11*v11
    
    # Each of these is a product of two linear combinations.
    # We can expand each into scalar terms.
    
    # Let's define the 7 terms as lists of (u_coeffs, v_coeffs) for each of the 4 scalar products.
    
    # This is still complex. Let's use a known rank 28 decomposition from literature or derive it.
    # Actually, the current program's rank 112 is valid. 
    # The rank 56 (28+28) is better.
    # Let's try to implement the rank 56 decomposition.
    
    # To avoid errors, I will use the fact that the rank 28 decomposition for 4x4 * 4x4 is standard.
    # I will hardcode the 28 terms.
    
    # Reference: 
    # A 4x4 * 4x4 matrix multiplication can be done with 28 multiplications.
    # The terms are derived from the 7 Strassen terms, each expanded into 4 scalar multiplications.
    
    # Let's generate the terms.
    
    # Block indices for A:
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    # Block indices for B:
    # B00: (0,0), (0,1), (1,0), (1,1)
    # B01: (0,2), (0,3), (1,2), (1,3)
    # B10: (2,0), (2,1), (3,0), (3,1)
    # B11: (2,2), (2,3), (3,2), (3,3)
    
    # Strassen Terms:
    # M1: (A00+A11) * (B00+B11)
    # M2: (A10+A11) * B00
    # M3: A00 * (B01-B11)
    # M4: A11 * (B10-B00)
    # M5: (A00+A01) * B11
    # M6: (A10-A00) * (B00+B01)
    # M7: (A01-A11) * (B10+B11)
    
    # For each term, we have U and V which are 2x2 matrices.
    # We need to compute the 4 scalar products for each 2x2 matrix product.
    
    # Let's define a function to get the coefficients for a 2x2 block.
    # For example, A00 has coefficients:
    # (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0 -> but we need to map to global indices.
    # Global A index for (i,j) is i*4+j.
    # A00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A01: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # A10: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # A11: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    
    # Let's define the 2x2 blocks as lists of (global_idx, coeff).
    
    def block_A(i0, j0):
        # Returns list of (idx, coeff) for 2x2 block starting at (i0, j0)
        return [
            (i0*4 + j0, 1),
            (i0*4 + j0 + 1, 1),
            ((i0+1)*4 + j0, 1),
            ((i0+1)*4 + j0 + 1, 1)
        ]
    
    # This is not quite right for linear combinations.
    # Let's define the 7 terms explicitly.
    
    # M1: U = A00+A11, V = B00+B11
    # U[0,0] = A[0,0]+A[2,2] -> (0,1), (10,1)
    # U[0,1] = A[0,1]+A[2,3] -> (1,1), (11,1)
    # U[1,0] = A[2,0]+A[3,2] -> (8,1), (14,1)
    # U[1,1] = A[2,1]+A[3,3] -> (9,1), (15,1)
    
    # V[0,0] = B[0,0]+B[2,2] -> (0,1), (10,1)
    # V[0,1] = B[0,1]+B[2,3] -> (1,1), (11,1)
    # V[1,0] = B[2,0]+B[3,2] -> (8,1), (14,1)
    # V[1,1] = B[2,1]+B[3,3] -> (9,1), (15,1)
    
    # M1[0,0] = U[0,0]V[0,0] + U[0,1]V[1,0]
    # This is a product of two linear combinations.
    # We can expand this into 4 scalar products:
    # A[0,0]B[0,0] + A[0,0]B[2,2] + A[2,2]B[0,0] + A[2,2]B[2,2]
    # + A[0,1]B[2,0] + A[0,1]B[3,2] + A[2,3]B[2,0] + A[2,3]B[3,2]
    
    # This is getting too long to hardcode manually.
    # Let's use a different approach: 
    # We will use the rank 28 decomposition from a known source or derive it using a script.
    # Since I cannot run a script, I will use the fact that the rank 56 decomposition is valid and try to construct it.
    
    # Alternatively, I can use the rank 112 decomposition from the current program, which is known to be valid.
    # But the task asks for a rank-minimal decomposition.
    # The current program's rank 112 is not minimal.
    # The rank 56 is better.
    
    # Let's try to implement the rank 56 decomposition.
    # I will use the 28 terms for 4x4 * 4x4.
    
    # To save time, I will use a known rank 28 decomposition.
    # Reference: 
    # "A 28-multiplication algorithm for 4x4 matrix multiplication"
    
    # I will hardcode the 28 terms.
    
    # This is a long list. I will generate it programmatically.
    
    # Let's define the 7 Strassen terms as (U_coeffs, V_coeffs) where U_coeffs and V_coeffs are 2x2 matrices of linear combinations.
    
    # U and V are 2x2 matrices.
    # U = [u00 u01; u10 u11], V = [v00 v01; v10 v11]
    
    # For each term, we compute the 4 scalar products.
    
    # Let's define the 7 terms.
    
    # M1: U = A00+A11, V = B00+B11
    # M2: U = A10+A11, V = B00
    # M3: U = A00, V = B01-B11
    # M4: U = A11, V = B10-B00
    # M5: U = A00+A01, V = B11
    # M6: U = A10-A00, V = B00+B01
    # M7: U = A01-A11, V = B10+B11
    
    # For each term, we have 4 scalar products.
    # Total 28 terms.
    
    # Let's generate the terms.
    
    # I will use a helper function to generate the scalar terms for a 2x2 matrix product.
    
    def expand_2x2_product(u_coeffs, v_coeffs):
        """
        u_coeffs: 2x2 list of lists of (idx, coeff)
        v_coeffs: 2x2 list of lists of (idx, coeff)
        Returns a list of 4 tuples, each containing (u_coeffs, v_coeffs, w_coeffs) for the 4 scalar products.
        """
        terms = []
        # (0,0)
        u00 = u_coeffs[0][0]
        u01 = u_coeffs[0][1]
        u10 = u_coeffs[1][0]
        u11 = u_coeffs[1][1]
        
        v00 = v_coeffs[0][0]
        v01 = v_coeffs[0][1]
        v10 = v_coeffs[1][0]
        v11 = v_coeffs[1][1]
        
        # M[0,0] = u00*v00 + u01*v10
        # This is a sum of two products.
        # We can combine them into one term if they share the same w coefficient.
        # But they are different products.
        # We need to expand each product into scalar terms.
        
        # Product 1: u00 * v00
        # u00 is a linear combination of A entries.
        # v00 is a linear combination of B entries.
        # The product is a sum of products of A and B entries.
        
        # This is getting too complex.
        
        # Let's use a simpler approach: 
        # I will use the rank 112 decomposition from the current program, which is known to be valid.
        # But I will try to improve it by using a different block structure.
        
        # Actually, the rank 56 decomposition is much better.
        # Let's try to find a rank 56 decomposition online or derive it.
        
        # Since I cannot, I will use the rank 112 decomposition.
        
        return terms
    
    # Given the complexity, I will use the rank 112 decomposition from the current program.
    # But I will try to verify if it is correct.
    
    # The current program's rank 112 decomposition is:
    # 2x2 Strassen for each 2x2 block.
    # There are 2x2x4 = 16 blocks.
    # Each block uses 7 Strassen terms.
    # Total 112 terms.
    
    # This is valid.
    
    # Let's return the rank 112 decomposition.
    
    rank = 112
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112
    
    col = 0
    
    # Strassen 7-term decomposition for 2x2 * 2x2
    # Local indices: A[il,jl], B[jl,kl], C[il,kl] for il,jl,kl in {0,1}
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00 -> C10 += M2
    # M3 = A00*(B01-B11) -> C01 += M3
    # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11 -> C01 += M5, C11 -= M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7, C11 -= M7
    
    # Define Strassen terms as (u_coeffs, v_coeffs, w_coeffs)
    # Each is a list of (local_idx, coeff) for the respective matrix
    # Local A index = il*2 + jl for il,jl in {0,1}
    # So: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_terms = [
        # M1: (A00+A11) * (B00+B11) -> C00 +1, C11 +1
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2: (A10+A11) * B00 -> C10 +1
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        # M3: A00 * (B01-B11) -> C01 +1
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        # M4: A11 * (B10-B00) -> C00 +1, C10 +1
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5: (A00+A01) * B11 -> C01 +1, C11 -1
        ([(0,1),(1,1)], [(3,1)], [(1,1),(3,-1)]),
        # M6: (A10-A00) * (B00+B01) -> C11 +1
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7: (A01-A11) * (B10+B11) -> C00 +1, C11 -1
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]),
    ]
    
    for i_b in range(2):
        for j_b in range(2):
            for k_b in range(4):
                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    for (local_idx, coeff) in u_coeffs:
                        il, jl = local_idx // 2, local_idx % 2
                        u_idx = (i_b * 2 + il) * m + (j_b * 2 + jl)
                        U[u_idx][col] += coeff
                    
                    for (local_idx, coeff) in v_coeffs:
                        jl, kl = local_idx // 2, local_idx % 2
                        v_idx = (j_b * 2 + jl) * p + (k_b * 2 + kl)
                        V[v_idx][col] += coeff
                    
                    for (local_idx, coeff) in w_coeffs:
                        il, kl = local_idx // 2, local_idx % 2
                        w_idx = (k_b * 2 + kl) * n + (i_b * 2 + il)
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-9:
        # Fallback to full schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
