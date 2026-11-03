"""Optimized matrix multiplication tensor decomposition for 4x4x8."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    schoolbook_rank = n * m * p  # 128
    
    # First, construct the schoolbook decomposition (always valid)
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Strategy: Utilize the block tensor structure to treat the 4x4x8 multiplication 
    # as two independent 4x4x4 multiplications. We will attempt to construct a 
    # decomposition of rank 40 (2 * 20) by reusing a rank-20 decomposition for 4x4x4.
    # The known rank-20 decomposition for 4x4x4 is complex to hardcode from scratch 
    # without errors, so we will implement a robust search that attempts to find a 
    # low-rank decomposition, falling back to schoolbook if it fails or takes too long.
    
    # However, hardcoding the specific rank-20 coefficients for 4x4x4 is error-prone 
    # in this context. A more reliable approach within the time budget is to use 
    # Strassen-like recursion or a known efficient algorithm.
    
    # Let's try a structured search for a rank-49 decomposition (2 * 49 = 98 for 4x4x8)
    # based on 2x2 Strassen blocks, which is easier to implement correctly than rank-20.
    # But even better, we can just return the schoolbook if we can't guarantee a lower rank
    # quickly. 
    
    # Wait, the prompt asks to "combine the strategy... with the strongest ideas".
    # The "strongest idea" from the inspiration is the rank-20 block reuse.
    # Since I cannot reliably generate the rank-20 coefficients in 140s via search 
    # (it's a huge non-linear problem), I will implement a heuristic that attempts 
    # to find a lower rank decomposition using a simple local search on the coefficients 
    # if possible, but given the difficulty, I will stick to a very safe fallback 
    # unless I can construct a specific known decomposition.
    
    # Actually, there is a well-known rank-49 decomposition for 4x4x4 using Strassen's 
    # algorithm on 2x2 blocks. 
    # Rank(4x4x4) <= 49.
    # Rank(4x4x8) <= 98.
    # This is a significant improvement over 128.
    
    # Let's implement the 2x2 Strassen decomposition for 4x4x4 and then duplicate it for 4x4x8.
    
    # Strassen's algorithm for 2x2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # B = [[B00, B01], [B10, B11]] where each block is 2x2.
    # C = [[C00, C01], [C10, C11]] where each block is 2x2.
    
    # Each multiplication of 2x2 matrices (e.g., A00*B00) can be done with rank 7 (Strassen) or rank 4 (schoolbook).
    # If we use Strassen for the inner 2x2 multiplications, the total rank is 7 (outer) * 7 (inner) = 49.
    
    # Let's build the rank-49 decomposition for 4x4x4.
    # We need to map the scalar indices to the block indices.
    # A[i,j] where i,j in 0..3.
    # Block A_ab corresponds to A[2a:2a+2, 2b:2b+2].
    
    # Let's define the decomposition for 4x4x4 first.
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49).
    
    # Inner Strassen (for 2x2x2) coefficients:
    # Let u, v, w be the indices for the inner 2x2x2 decomposition.
    # U_inner[i,j, u], V_inner[j,k, u], W_inner[k,i, u] for i,j,k in 0..1.
    
    # We will generate the 49 terms.
    # Outer Strassen terms M1..M7.
    # Each Mx is a product of two linear combinations of 2x2 blocks.
    # Each 2x2 block product is decomposed into 7 scalar multiplications.
    
    # This is complex to hardcode. Let's try a simpler approach: 
    # Use schoolbook for the inner blocks but Strassen for the outer?
    # Rank = 7 (outer) * 4 (inner schoolbook) = 28? No, that's not how it works.
    # The rank of the tensor is the number of scalar multiplications.
    # If we use Strassen for the outer 2x2 block structure, we have 7 "matrix multiplications" of 2x2 matrices.
    # Each 2x2 matrix multiplication requires rank 7 (Strassen) or 4 (schoolbook).
    # If we use schoolbook for the inner, total rank = 7 * 4 = 28.
    # Wait, is rank(4x4x4) <= 28?
    # Let's check.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]].
    # M1 = (A00+A11)(B00+B11) -> 4 mults
    # M2 = (A10+A11)B00 -> 4 mults
    # M3 = A00(B01-B11) -> 4 mults
    # M4 = A11(B10-B00) -> 4 mults
    # M5 = (A00+A01)B11 -> 4 mults
    # M6 = (A10-A00)(B00+B01) -> 4 mults
    # M7 = (A01-A11)(B10+B11) -> 4 mults
    # Total 28 scalar multiplications.
    # Then we add the results to get C.
    # This is a valid decomposition of rank 28 for 4x4x4.
    # Is it valid? Yes, because Strassen's algorithm is exact.
    # So rank(4x4x4) <= 28.
    # Then rank(4x4x8) <= 2 * 28 = 56.
    
    # This is much better than 128. Let's implement this rank-28 decomposition for 4x4x4.
    
    def construct_4x4x4_rank28():
        """Construct a rank-28 decomposition for 4x4x4 matrix multiplication."""
        n4, m4, p4 = 4, 4, 4
        rank = 28
        U = [[0] * rank for _ in range(n4 * m4)]  # 16 x 28
        V = [[0] * rank for _ in range(m4 * p4)]  # 16 x 28
        W = [[0] * rank for _ in range(p4 * n4)]  # 16 x 28
        
        # Inner block indices:
        # A[2a+di, 2b+dj] is in block A_ab at position (di, dj).
        # B[2b+di, 2c+dk] is in block B_bc at position (di, dk).
        # C[2a+di, 2c+dk] is in block C_ac at position (di, dk).
        
        # Outer Strassen terms:
        # Each term is a product of two linear combinations of blocks.
        # Let's define the coefficients for the outer Strassen.
        # Term t (0..6) corresponds to M_{t+1}.
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each term, we have 4 inner scalar multiplications (schoolbook for 2x2).
        # Inner schoolbook:
        # C_ab[di, dk] = sum_dj A_ab[di, dj] B_bc[dj, dk]
        # So for each (di, dj, dk), we have a scalar multiplication.
        
        # Let's iterate over the 7 outer terms and 4 inner mults.
        
        # Define the outer coefficients.
        # A_blocks: (a, b) where a,b in 0..1.
        # B_blocks: (b, c) where b,c in 0..1.
        # C_blocks: (a, c) where a,c in 0..1.
        
        # We need to map the outer term to the blocks involved.
        # For each outer term, we have a linear combination of A blocks and a linear combination of B blocks.
        # Actually, Strassen's terms are products of specific linear combinations.
        # Let's define the coefficients for A and B for each of the 7 terms.
        
        # A_coeffs[t][a][b] = coefficient of A_ab in the first factor of term t.
        # B_coeffs[t][b][c] = coefficient of B_bc in the second factor of term t.
        # W_coeffs[t][a][c] = coefficient of C_ac in the result of term t.
        
        A_coeffs = [
            # M1: (A00+A11)
            [[1, 0], [0, 1]],
            # M2: (A10+A11)
            [[0, 0], [1, 1]],
            # M3: A00
            [[1, 0], [0, 0]],
            # M4: A11
            [[0, 0], [0, 1]],
            # M5: (A00+A01)
            [[1, 1], [0, 0]],
            # M6: (A10-A00)
            [[-1, 0], [1, 0]],
            # M7: (A01-A11)
            [[0, 1], [0, -1]],
        ]
        
        B_coeffs = [
            # M1: (B00+B11)
            [[1, 0], [0, 1]],
            # M2: B00
            [[1, 0], [0, 0]],
            # M3: (B01-B11)
            [[0, 1], [0, -1]],
            # M4: (B10-B00)
            [[-1, 0], [1, 0]],
            # M5: B11
            [[0, 0], [0, 1]],
            # M6: (B00+B01)
            [[1, 1], [0, 0]],
            # M7: (B10+B11)
            [[0, 0], [1, 1]],
        ]
        
        W_coeffs = [
            # M1: C00 += M1, C11 += M1
            [[1, 0], [0, 1]],
            # M2: C10 += M2, C11 -= M2
            [[0, 0], [1, -1]],
            # M3: C01 += M3, C11 += M3
            [[0, 1], [0, 1]],
            # M4: C00 += M4, C10 += M4
            [[1, 0], [1, 0]],
            # M5: C00 -= M5, C01 += M5
            [[-1, 1], [0, 0]],
            # M6: C11 += M6
            [[0, 0], [0, 1]],
            # M7: C00 += M7
            [[1, 0], [0, 0]],
        ]
        
        # Now, for each outer term t (0..6) and each inner multiplication (di, dj, dk) (0..1, 0..1, 0..1):
        # We create a rank-1 tensor.
        # The scalar is A[2a+di, 2b+dj] * B[2b+dj, 2c+dk] * C[2a+di, 2c+dk]
        # But we need to sum over a, b, c with the coefficients.
        
        # Actually, the term is:
        # (sum_{a,b} A_coeffs[t][a][b] * A_ab) * (sum_{b,c} B_coeffs[t][b][c] * B_bc)
        # = sum_{a,b,c} A_coeffs[t][a][b] * B_coeffs[t][b][c] * A_ab * B_bc
        
        # And A_ab * B_bc is a 2x2 matrix product, which we decompose into 4 scalar mults.
        # A_ab[di, dj] * B_bc[dj, dk] contributes to C_ab[di, dk].
        
        # So for a fixed t, di, dj, dk:
        # We have a scalar multiplication of:
        # (sum_{a,b} A_coeffs[t][a][b] * A[2a+di, 2b+dj]) * (sum_{b,c} B_coeffs[t][b][c] * B[2b+dj, 2c+dk])
        
        # This is a product of two linear forms.
        # The U vector has 1s at positions (2a+di, 2b+dj) with coefficient A_coeffs[t][a][b].
        # The V vector has 1s at positions (2b+dj, 2c+dk) with coefficient B_coeffs[t][b][c].
        # The W vector has 1s at positions (2c+dk, 2a+di) with coefficient W_coeffs[t][a][c]... wait.
        
        # The output is C[2a+di, 2c+dk] += W_coeffs[t][a][c] * (A_ab[di,dj] * B_bc[dj,dk])
        
        # So for a fixed t, di, dj, dk:
        # col = t * 4 + di * 4 + dj * 4 + dk  # or some unique index
        # We need to add the contributions to U, V, W.
        
        col = 0
        for t in range(7):
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        # This is the (di, dj, dk)-th scalar multiplication in the t-th Strassen term.
                        
                        # U contributions:
                        # For each (a, b), U[2a+di, 2b+dj] += A_coeffs[t][a][b]
                        for a in range(2):
                            for b in range(2):
                                coeff_A = A_coeffs[t][a][b]
                                if coeff_A != 0:
                                    idx_U = (2 * a + di) * 4 + (2 * b + dj)
                                    U[idx_U][col] += coeff_A
                        
                        # V contributions:
                        # For each (b, c), V[2b+dj, 2c+dk] += B_coeffs[t][b][c]
                        for b in range(2):
                            for c in range(2):
                                coeff_B = B_coeffs[t][b][c]
                                if coeff_B != 0:
                                    idx_V = (2 * b + dj) * 4 + (2 * c + dk)
                                    V[idx_V][col] += coeff_B
                        
                        # W contributions:
                        # For each (a, c), W[2c+dk, 2a+di] += W_coeffs[t][a][c]
                        # Note: W index is [k*n + i] = [2c+dk, 2a+di] -> (2c+dk)*4 + (2a+di)
                        for a in range(2):
                            for c in range(2):
                                coeff_W = W_coeffs[t][a][c]
                                if coeff_W != 0:
                                    idx_W = (2 * c + dk) * 4 + (2 * a + di)
                                    W[idx_W][col] += coeff_W * coeff_A * coeff_B
                                    # Wait, the W coefficient in the decomposition is just the scalar that multiplies the product.
                                    # The product is (sum A...) * (sum B...).
                                    # The term in the tensor is sum_r U[...] V[...] W[...]
                                    # So W should be 1 if we absorb the coefficients into U and V?
                                    # No, the standard form is U * V * W.
                                    # If U has coeff_A and V has coeff_B, then U*V has coeff_A*coeff_B.
                                    # Then W must be 1 to get the correct coefficient for the product?
                                    # No, the output C gets coeff_A * coeff_B * W.
                                    # We want C to get W_coeffs[t][a][c] * (A_ab * B_bc).
                                    # So W should be W_coeffs[t][a][c].
                                    # But U and V already have the coefficients.
                                    # So the term is (coeff_A * A_ab) * (coeff_B * B_bc) * W.
                                    # We want this to equal W_coeffs[t][a][c] * A_ab * B_bc.
                                    # So W must be W_coeffs[t][a][c] / (coeff_A * coeff_B)?
                                    # This is problematic if coeff_A or coeff_B is 0.
                                    
                                    # Actually, the decomposition is:
                                    # T = sum_r U_r V_r W_r
                                    # Each term is a rank-1 tensor.
                                    # U_r[i,j] = coeff_A
                                    # V_r[j,k] = coeff_B
                                    # W_r[k,i] = coeff_W
                                    # Then the contribution to T[i,j,k] is coeff_A * coeff_B * coeff_W.
                                    # We want this to be 1 for the specific (i,j,k) corresponding to a,b,c,di,dj,dk?
                                    # No, the tensor T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
                                    # In the Strassen decomposition, the "term" is not a single A[i,j]B[j,k] but a linear combination.
                                    
                                    # Let's rethink.
                                    # The decomposition must satisfy:
                                    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
                                    
                                    # In our construction:
                                    # For a fixed t, di, dj, dk, we are creating a term r.
                                    # U[i,j,r] is non-zero only if i=2a+di, j=2b+dj for some a,b.
                                    # Specifically, U[2a+di, 2b+dj, r] = A_coeffs[t][a][b].
                                    # V[2b+dj, 2c+dk, r] = B_coeffs[t][b][c].
                                    # W[2c+dk, 2a+di, r] = W_coeffs[t][a][c].
                                    
                                    # Then the product U*V*W for a specific (i,j,k) = (2a+di, 2b+dj, 2c+dk) is:
                                    # A_coeffs[t][a][b] * B_coeffs[t][b][c] * W_coeffs[t][a][c].
                                    
                                    # We want the sum over all r (all t, di, dj, dk) to be 1 for each (i,j,k).
                                    # But for a fixed (i,j,k), only one (t, di, dj, dk) contributes?
                                    # No, i,j,k determine di,dj,dk and a,b,c.
                                    # i = 2a+di, j = 2b+dj, k = 2c+dk.
                                    # So a = i//2, di = i%2, etc.
                                    # So for a fixed (i,j,k), the sum is over t:
                                    # sum_t A_coeffs[t][a][b] * B_coeffs[t][b][c] * W_coeffs[t][a][c]
                                    
                                    # We need this sum to be 1.
                                    # Let's check for a specific (a,b,c).
                                    # sum_t A_coeffs[t][a][b] * B_coeffs[t][b][c] * W_coeffs[t][a][c] = 1?
                                    
                                    # This is the condition for Strassen's algorithm to be correct.
                                    # It is known to be correct, so this sum should be 1.
                                    
                                    # So we just set W[idx_W][col] = W_coeffs[t][a][c].
                                    # And U and V as defined.
                                    
                                    # But wait, I added coeff_A * coeff_B * coeff_W to W in the code above. That's wrong.
                                    # W should just be coeff_W.
                                    # Let's fix the code.
                        
                        col += 1
        
        return U, V, W
    
    # Let's implement the corrected version.
    def construct_4x4x4_rank28_correct():
        n4, m4, p4 = 4, 4, 4
        rank = 28
        U = [[0] * rank for _ in range(n4 * m4)]
        V = [[0] * rank for _ in range(m4 * p4)]
        W = [[0] * rank for _ in range(p4 * n4)]
        
        A_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]],
        ]
        
        B_coeffs = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]],
        ]
        
        W_coeffs = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, -1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
            [[-1, 1], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]],
        ]
        
        col = 0
        for t in range(7):
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        for a in range(2):
                            for b in range(2):
                                coeff_A = A_coeffs[t][a][b]
                                if coeff_A != 0:
                                    idx_U = (2 * a + di) * 4 + (2 * b + dj)
                                    U[idx_U][col] = coeff_A
                        
                                coeff_B = B_coeffs[t][b][c] if False else 0 # Placeholder
                                # Need to loop b,c for V and a,c for W separately?
                                # No, U depends on a,b. V depends on b,c. W depends on a,c.
                                # They are all part of the same rank-1 term.
                                # So we must set all of them for this col.
                        
                        # Let's restructure the loops to be cleaner.
                        # For a fixed t, di, dj, dk:
                        # U is determined by t, di, dj, a, b.
                        # V is determined by t, dj, dk, b, c.
                        # W is determined by t, di, dk, a, c.
                        
                        # We can just fill the vectors.
                        
                        # U vector
                        for a in range(2):
                            for b in range(2):
                                U[(2*a+di)*4 + (2*b+dj)][col] = A_coeffs[t][a][b]
                        
                        # V vector
                        for b in range(2):
                            for c in range(2):
                                V[(2*b+dj)*4 + (2*c+dk)][col] = B_coeffs[t][b][c]
                        
                        # W vector
                        for a in range(2):
                            for c in range(2):
                                W[(2*c+dk)*4 + (2*a+di)][col] = W_coeffs[t][a][c]
                        
                        col += 1
        
        return U, V, W
    
    # Now, we need to extend this to 4x4x8.
    # C is 4x8. We can split it into two 4x4 blocks: C_left (cols 0-3) and C_right (cols 4-7).
    # C_left = A * B_left, C_right = A * B_right.
    # We can use the same U for both, and duplicate V and W.
    # U shape (16, 56), V shape (32, 56), W shape (32, 56).
    
    U4, V4, W4 = construct_4x4x4_rank28_correct()
    
    # Verify U4, V4, W4
    def verify_4x4x4(U, V, W):
        n, m, p = 4, 4, 4
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(len(U[0])):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        return True
    
    if verify_4x4x4(U4, V4, W4):
        # Extend to 4x4x8
        rank = 56
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(32)]
        W = [[0] * rank for _ in range(32)]
        
        # Copy U4 to first 28 cols and second 28 cols
        for i in range(16):
            for r in range(28):
                U[i][r] = U4[i][r]
                U[i][r + 28] = U4[i][r]
        
        # Copy V4 to first 28 cols (rows 0-15) and second 28 cols (rows 16-31)
        for i in range(16):
            for r in range(28):
                V[i][r] = V4[i][r]
                V[i + 16][r + 28] = V4[i][r]
        
        # Copy W4 to first 28 cols (rows 0-15) and second 28 cols (rows 16-31)
        for i in range(16):
            for r in range(28):
                W[i][r] = W4[i][r]
                W[i + 16][r + 28] = W4[i][r]
        
        # Verify 4x4x8
        def verify_4x4x8(U, V, W):
            n, m, p = 4, 4, 8
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(len(U[0])):
                            val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if val != 1:
                            return False
            return True
        
        if verify_4x4x8(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
