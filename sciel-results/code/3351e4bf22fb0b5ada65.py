import time

def construct():
    deadline = time.time() + 130
    n, m, p = 4, 4, 5

    # Winograd F(4) algorithm for 4x4x4 matrix multiplication
    # Uses the cyclic group structure: index 0 = sum, indices 1,2,3 = DFT-like
    # This gives rank 12 for the 4x4x4 part
    #
    # The F(4) algorithm:
    # For each of 4 "frequencies" t in {0,1,2,3}:
    #   Compute 3 products: (A_t)(B_t) -> C_t
    # Each product is a 1x1 scalar multiplication
    # Total: 12 scalar multiplications
    #
    # The transform matrices:
    # F(4) has 4 points with coefficients:
    # t=0: [1, 1, 1, 1]  (sum)
    # t=1: [1, i, -1, -i] where i = imaginary unit... 
    # 
    # Actually, for real matrix multiplication, we use the real Winograd algorithm.
    # The F(4) algorithm over reals uses:
    # t=0: all ones (sum)
    # t=1: [1, 1, -1, -1]
    # t=2: [1, -1, -1, 1]  
    # t=3: [1, -1, 1, -1]
    #
    # Wait, that's not quite right either. Let me think more carefully.
    #
    # The Winograd F(4) algorithm for matrix multiplication:
    # We view the 4x4 multiplication as a convolution on Z_4.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Using the DFT on Z_4 (with real transforms):
    # The 4-point DFT has eigenvalues: 1, i, -1, -i
    # For real arithmetic, we pair conjugates:
    # t=0: real part (sum)
    # t=1,3: complex pair (use 2 real mults)
    # t=2: real part (alternating sum)
    #
    # So: 1 + 2 + 1 = 4 "frequency components"
    # But each needs a matrix multiply... 
    #
    # Actually, let me use a different approach. The known rank for 4x4x4 
    # via Winograd is 12. Let me construct it directly.
    #
    # The 12-term decomposition of M_{4,4,4}:
    # We use the fact that Z_4 has 4 idempotents in the group algebra.
    # The decomposition uses 4 "frequencies" each contributing 3 terms.
    
    # Let me try a different structural approach:
    # Use the block structure with 2x2 blocks and apply Strassen to get rank 49
    # for 4x4x4, plus 16 for 4x4x1, total 65.
    # But the directive says to use Winograd F(4) for rank 12 + schoolbook 4 = 16.
    
    # Let me carefully implement Winograd F(4).
    # The algorithm:
    # A is 4x4, B is 4x4, C = A*B is 4x4.
    # 
    # Step 1: Transform A along rows (for each column j, take 4-point transform of A[:,j])
    # Step 2: Transform B along columns (for each row j, take 4-point transform of B[j,:])
    # Step 3: Element-wise multiply the transformed matrices (4x4 element-wise)
    # Step 4: Inverse transform C along rows
    #
    # The 4-point DFT matrix (real form):
    # F = [1  1  1  1]
    #     [1  0 -1  0]  -- no, this isn't right
    #
    # Let me use the standard 4-point DFT:
    # W = e^{-2πi/4} = -i
    # F[k][j] = W^{kj} = (-i)^{kj}
    # F = [[1,  1,  1,  1],
    #      [1, -i, -1,  i],
    #      [1, -1,  1, -1],
    #      [1,  i, -1, -i]]
    #
    # For real inputs, we can use a real-valued transform.
    # The key insight: the 4-point DFT decomposes into:
    # - 1 real component (DC, k=0)
    # - 1 real component (Nyquist, k=2) 
    # - 1 complex pair (k=1, k=3)
    #
    # For the complex pair, we need 2 real multiplications per element.
    # So total real multiplications: 4*1 + 4*1 + 4*2 = 4 + 4 + 8 = 16? No...
    #
    # Actually, the Winograd algorithm counts the number of scalar multiplications.
    # For F(4): the number of multiplications is sum over k of (number of real mults for frequency k)
    # k=0: 4 mults (one per element in the 4-vector)
    # k=2: 4 mults
    # k=1,3: 8 mults (2 per element for the complex pair, 4 elements)
    # Total: 4 + 4 + 8 = 16? But the known rank is 12.
    
    # I think I'm overcomplicating this. Let me just use a known valid decomposition.
    # The schoolbook rank is 80. Let me try to get a better but definitely valid answer.
    
    # Actually, let me reconsider. The problem is 4x4 * 4x5.
    # Schoolbook: 4*4*5 = 80 terms.
    # 
    # A valid improvement: use Strassen for the 4x4x4 part (first 4 columns of B)
    # and schoolbook for the 4x4x1 part (last column of B).
    # Strassen rank for 4x4x4: 7^2 = 49 (recursive) or we can do better.
    # Actually, the rank of M_{4,4,4} is at most 49 by recursive Strassen.
    # Schoolbook for 4x4x1: 16 terms.
    # Total: 49 + 16 = 65.
    
    # But wait - can we do better than 65? The Winograd F(4) gives 12 for 4x4x4.
    # If that's correct, total would be 12 + 16 = 28.
    
    # Let me verify: is the rank of M_{4,4,4} really 12?
    # No! The rank of M_{2,2,2} is 7, and M_{4,4,4} has rank at most 7*7=49.
    # The lower bound is 4*4*4/... no, the trivial lower bound is 4*4*4=64? No.
    # The rank of M_{n,n,n} is at least n^2 (by a simple argument).
    # For n=4, rank >= 16. And rank <= 49 by Strassen.
    # The Winograd F(4) algorithm gives rank 12? That would be below the lower bound of 16!
    # So Winograd F(4) cannot give rank 12 for M_{4,4,4}.
    
    # I think the confusion is that Winograd F(4) is for convolution, not general matrix multiplication.
    # For general matrix multiplication, the rank is at least n^2 = 16 for n=4.
    
    # So the best we can do with Strassen is 49 + 16 = 65.
    # Or we can try to find a better decomposition.
    
    # Let me just implement the Strassen-based decomposition carefully.
    
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal r
        if r >= rank:
            raise ValueError("Rank overflow")
        U[u_idx][r] += coeff
        V[v_idx][r] += coeff
        W[w_idx][r] += coeff
        r += 1
    
    # Strassen 2x2x2 base case
    # For 2x2 matrices A, B -> C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # Wait, I need to double-check the Strassen formulas.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)A21  -- NO, this is wrong!
    # M5 = (A11+A12)B21
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    # Yes, M5 = (A11+A12)B21, which is (Lin A)(Lin B). Good.
    
    # For the 4x4x4 part, we use block Strassen.
    # A is 4x4, B is 4x4 (first 4 columns), C is 4x4 (first 4 columns of output).
    # Block into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen on the 2x2 block matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_i is a 2x2 matrix product, which we decompose using Strassen 2x2x2.
    # Each 2x2x2 decomposition has rank 7.
    # 7 M_i's * 7 = 49 terms for the 4x4x4 part.
    
    # Helper: decompose a 2x2x2 multiplication where the "A" matrix is a linear combo of A blocks
    # and the "B" matrix is a linear combo of B blocks.
    # a_coeffs: dict mapping (block_row, block_col) to coefficient for A blocks
    # b_coeffs: dict mapping (block_row, block_col) to coefficient for B blocks
    # c_target: dict mapping (block_row, block_col) to coefficient for C blocks
    def strassen_2x2(a_coeffs, b_coeffs, c_target):
        nonlocal r
        # a_coeffs: {(br, bc): coeff} for A[br*2:br*2+2, bc*2:bc*2+2]
        # b_coeffs: {(br, bc): coeff} for B[br*2:br*2+2, bc*2:bc*2+2]
        # c_target: {(br, bc): coeff} for C[br*2:br*2+2, bc*2:bc*2+2]
        
        # The 2x2 "A" matrix has entries that are linear combos of A sub-blocks.
        # A_local[0,0] = sum of a_coeffs[(br,bc)] * A[br*2+i, bc*2+j] for i,j in 0,1
        # Similarly for other entries.
        
        # We need to map the local 2x2 indices to global indices.
        # Local A[0,0] corresponds to A block (0,0), which is rows 0-1, cols 0-1.
        # Local A[0,1] corresponds to A block (0,1), rows 0-1, cols 2-3.
        # Local A[1,0] corresponds to A block (1,0), rows 2-3, cols 0-1.
        # Local A[1,1] corresponds to A block (1,1), rows 2-3, cols 2-3.
        
        # Similarly for B and C.
        
        # Strassen 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B10
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each M_i, the "A factor" is a linear combo of A blocks, and "B factor" is a linear combo of B blocks.
        # We need to decompose the product of two 2x2 matrices into 7 scalar multiplications.
        
        # Let's define the A-factor and B-factor as 2x2 matrices of coefficients.
        # A_factor[local_i][local_j] = sum of a_coeffs[(br,bc)] where the block maps to local (i,j)
        # Actually, each "entry" of the local 2x2 matrix is itself a 2x2 block of the original matrix.
        # So A_factor is a 2x2 matrix where each entry is a 2x2 matrix of coefficients.
        
        # Let me think differently. The local 2x2 multiplication is:
        # P = X * Y where X and Y are 2x2 matrices.
        # X[i][j] = sum over A blocks of a_coeffs * A_block
        # Y[i][j] = sum over B blocks of b_coeffs * B_block
        # P[i][j] = sum_k X[i][k] * Y[k][j]
        # And P contributes to C via c_target.
        
        # To decompose P = X*Y into scalar multiplications, we use Strassen on the 2x2 matrices X and Y.
        # But X and Y are themselves linear combos of the original A and B blocks.
        
        # Let me just directly enumerate the 7 Strassen terms for the local 2x2 multiplication.
        
        # Define helper to get the coefficient for a specific A sub-entry
        # A sub-entry at global position (row, col) where row = br*2+di, col = bc*2+dj
        # The coefficient is a_coeffs[(br,bc)]
        
        # Similarly for B and C.
        
        # Strassen terms for local 2x2:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y01-Y00)
        # M5 = (X00+X01)Y10
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Each X[i][j] is a 2x2 matrix of A entries.
        # X[i][j][di][dj] = a_coeffs.get((i,j), 0) * A[i*2+di][j*2+dj]
        
        # So for M1 = (X00+X11)(Y00+Y11):
        # The A factor has entries:
        # A_factor[di][dj] = a_coeffs.get((0,0),0)*A[di][dj] + a_coeffs.get((1,1),0)*A[2+di][2+dj]
        # The B factor has entries:
        # B_factor[di][dj] = b_coeffs.get((0,0),0)*B[di][dj] + b_coeffs.get((1,1),0)*B[2+di][2+dj]
        
        # The product A_factor * B_factor is a 2x2 matrix.
        # We need to decompose this 2x2 multiplication into 7 scalar multiplications.
        
        # But wait, A_factor and B_factor are 2x2 matrices, and we're multiplying them.
        # This is a 2x2x2 multiplication, which has rank 7.
        
        # So for each M_i, we add 7 terms.
        
        # Let me define a function to add a 2x2x2 Strassen decomposition for given A_factor and B_factor coefficients.
        
        def add_strassen_2x2(a_f, b_f, c_f):
            """
            a_f: 2x2 matrix of coefficients for A entries
            b_f: 2x2 matrix of coefficients for B entries  
            c_f: 2x2 matrix of coefficients for C entries
            Each is a dict mapping (row, col) to coefficient.
            """
            nonlocal r
            # Strassen 2x2:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B10
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # For each M_i, we need to decompose the product of two 2x2 matrices.
            # But actually, in this context, A00, A01, etc. are scalars (entries of the 2x2 matrix).
            # So M1 = (A00+A11)*(B00+B11) is a single scalar multiplication.
            
            # Wait, I'm confusing levels. Let me restart.
            
            # The local 2x2 multiplication P = X*Y where X, Y are 2x2 matrices of scalars.
            # Strassen decomposes this into 7 scalar multiplications.
            # Each scalar multiplication is of the form (linear combo of X entries) * (linear combo of Y entries).
            # And the result contributes to C entries.
            
            # So for M1 = (X00+X11)(Y00+Y11):
            # This is one scalar multiplication.
            # The X factor is X00+X11, which is a linear combo of X entries.
            # The Y factor is Y00+Y11, which is a linear combo of Y entries.
            
            # In our case, X entries are themselves linear combos of A blocks.
            # So X00 = sum of a_coeffs * A_block_entries.
            
            # Let me just directly construct the terms.
            
            # M1: A factor = X00+X11, B factor = Y00+Y11
            # X00 = sum over (br,bc) of a_f.get((br,bc),0) * A[br*2+di][bc*2+dj] for di,dj in 0,1
            # But X00 is a specific entry of the 2x2 matrix X.
            # X[0][0] corresponds to the top-left 2x2 block of the original A.
            # So X[0][0][di][dj] = a_f.get((0,0),0) * A[di][dj]
            
            # I think I need to be more careful. Let me define:
            # The "A matrix" in the local 2x2 multiplication has entries:
            # A_mat[0][0] = a_f.get((0,0),0)  (coefficient for A[0:2,0:2])
            # A_mat[0][1] = a_f.get((0,1),0)  (coefficient for A[0:2,2:4])
            # A_mat[1][0] = a_f.get((1,0),0)  (coefficient for A[2:4,0:2])
            # A_mat[1][1] = a_f.get((1,1),0)  (coefficient for A[2:4,2:4])
            
            # Similarly for B and C.
            
            # Strassen 2x2:
            # M1 = (A_mat[0][0]+A_mat[1][1]) * (B_mat[0][0]+B_mat[1][1])
            # This is a product of two scalars (linear combos of A and B entries).
            # In the tensor decomposition, this is ONE term.
            
            # So for each of the 7 Strassen terms, we add ONE term to the decomposition.
            
            # M1:
            a1 = a_f.get((0,0),0) + a_f.get((1,1),0)
            b1 = b_f.get((0,0),0) + b_f.get((1,1),0)
            c1 = c_f.get((0,0),0) + c_f.get((1,1),0)  # C00 = M1 + ...
            # Actually, C contributions are:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M6
            # C11 = M1 - M2 + M3 + M6
            
            # So M1 contributes +1 to C00 and +1 to C11.
            
            # Let me restructure. For each M_i, I know which C entries it contributes to and with what sign.
            
            # M1 -> C00 (+1), C11 (+1)
            # M2 -> C10 (+1), C11 (-1)
            # M3 -> C01 (+1), C11 (+1)
            # M4 -> C00 (+1)
            # M5 -> C00 (-1), C01 (+1)
            # M6 -> C10 (+1), C11 (+1)
            # M7 -> C00 (+1)
            
            # Wait, let me re-verify:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M6
            # C11 = M1 - M2 + M3 + M6
            
            # So:
            # M1: C00 +1, C11 +1
            # M2: C10 +1, C11 -1
            # M3: C01 +1, C11 +1
            # M4: C00 +1
            # M5: C00 -1, C01 +1
            # M6: C10 +1, C11 +1
            # M7: C00 +1
            
            # For each M_i, the A factor and B factor are:
            # M1: A=(0,0)+(1,1), B=(0,0)+(1,1)
            # M2: A=(1,0)+(1,1), B=(0,0)
            # M3: A=(0,0), B=(1,0)-(1,1)
            # M4: A=(1,1), B=(0,1)-(0,0)
            # M5: A=(0,0)+(0,1), B=(1,0)
            # M6: A=(1,0)-(0,0), B=(0,0)+(0,1)
            # M7: A=(0,1)-(1,1), B=(1,0)+(1,1)
            
            # Now, each "A factor" is a linear combo of A_mat entries, and each "B factor" is a linear combo of B_mat entries.
            # A_mat[br][bc] corresponds to the A block at (br,bc), which is a 2x2 submatrix.
            # So A_mat[br][bc] * A[br*2+di][bc*2+dj] for di,dj in 0,1.
            
            # For a given M_i, the A factor is sum over (br,bc) of coeff_A[br,bc] * A_mat[br,bc]
            # And A_mat[br,bc] is the coefficient for the entire 2x2 block.
            
            # In the tensor decomposition, each term is:
            # U[i*m+j][r] = sum over (br,bc) of coeff_A[br,bc] * (1 if (i,j) is in block (br,bc))
            # V[j'*p+k][r] = sum over (br,bc) of coeff_B[br,bc] * (1 if (j',k) is in block (br,bc))
            # W[k'*n+i'][r] = sum over (br,bc) of coeff_C[br,bc] * (1 if (i',k') is in block (br,bc))
            
            # But this is for the LOCAL 2x2 multiplication. The global indices are different.
            
            # I think I'm overcomplicating this. Let me just directly enumerate all 49 terms.
            
            # For the 4x4x4 part, we have 7 "block Strassen" multiplications.
            # Each block multiplication is of the form (sum of A blocks) * (sum of B blocks).
            # Each such product is a 2x2 matrix multiplication, which we decompose into 7 scalar multiplications.
            
            # Let me define the 7 block Strassen terms:
            # S1 = A00 + A11, T1 = B00 + B11, M1 = S1*T1
            # S2 = A10 + A11, T2 = B00, M2 = S2*T2
            # S3 = A00, T3 = B10 - B11, M3 = S3*T3
            # S4 = A11, T4 = B01 - B00, M4 = S4*T4
            # S5 = A00 + A01, T5 = B10, M5 = S5*T5
            # S6 = A10 - A00, T6 = B00 + B01, M6 = S6*T6
            # S7 = A01 - A11, T7 = B10 + B11, M7 = S7*T7
            
            # Each S_i is a 2x2 matrix (sum of 2x2 A blocks), and T_i is a 2x2 matrix.
            # M_i = S_i * T_i is a 2x2 matrix.
            # We decompose M_i into 7 scalar multiplications using Strassen 2x2.
            
            # For each M_i, the 7 scalar multiplications are:
            # m1_i = (S_i[0,0]+S_i[1,1]) * (T_i[0,0]+T_i[1,1])
            # m2_i = (S_i[1,0]+S_i[1,1]) * T_i[0,0]
            # m3_i = S_i[0,0] * (T_i[1,0]-T_i[1,1])
            # m4_i = S_i[1,1] * (T_i[0,1]-T_i[0,0])
            # m5_i = (S_i[0,0]+S_i[0,1]) * T_i[1,0]
            # m6_i = (S_i[1,0]-S_i[0,0]) * (T_i[0,0]+T_i[0,1])
            # m7_i = (S_i[0,1]-S_i[1,1]) * (T_i[1,0]+T_i[1,1])
            
            # And M_i[0,0] = m1_i + m4_i - m5_i + m7_i
            # M_i[0,1] = m3_i + m5_i
            # M_i[1,0] = m2_i + m6_i
            # M_i[1,1] = m1_i - m2_i + m3_i + m6_i
            
            # Now, S_i[di,dj] is a linear combo of A entries.
            # S_i[di,dj] = sum over blocks of coeff * A[br*2+di][bc*2+dj]
            
            # For example, S1[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
            # S1[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
            # etc.
            
            # Similarly for T_i.
            
            # So for m1_1 = (S1[0,0]+S1[1,1]) * (T1[0,0]+T1[1,1]):
            # S1[0,0]+S1[1,1] = (A[0,0]+A[2,2]) + (A[2,0]+A[0,2])... wait, no.
            # S1[1,1] = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
            # So S1[0,0]+S1[1,1] = A[0,0] + A[2,2] + A[1,1] + A[3,3]
            
            # This is getting very detailed. Let me just write a function that, given the block coefficients,
            # generates all the scalar terms.
            
            # I'll define the 7 block Strassen terms with their block coefficients.
            
            block_strassen = [
                # (S_coeffs, T_coeffs, C_contribution)
                # S_coeffs: {(br,bc): coeff} for A blocks
                # T_coeffs: {(br,bc): coeff} for B blocks
                # C_contribution: {(br,bc): coeff} for C blocks (which C block this M_i contributes to)
                ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
                ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),  # M2
                ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # M3
                ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1}),  # M4
                ({(0,0):1, (0,1):1}, {(1,0):1}, {(0,0):-1, (0,1):1}),  # M5
                ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),  # M6
                ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),  # M7
            ]
            
            for s_coeffs, t_coeffs, c_contrib in block_strassen:
                # For each of the 7 Strassen 2x2 terms:
                strassen_2x2_terms = [
                    # (S_local_coeffs, T_local_coeffs, C_local_contrib)
                    # S_local_coeffs: {(di,dj): coeff} for the local 2x2 S matrix
                    # T_local_coeffs: {(di,dj): coeff} for the local 2x2 T matrix
                    # C_local_contrib: {(di,dj): coeff} for the local 2x2 C matrix
                    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # m1
                    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),  # m2
                    ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # m3
                    ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1}),  # m4
                    ({(0,0):1, (0,1):1}, {(1,0):1}, {(0,0):-1, (0,1):1}),  # m5
                    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),  # m6
                    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),  # m7
                ]
                
                for s_local, t_local, c_local in strassen_2x2_terms:
                    # Now, the A factor is:
                    # sum over (br,bc) of s_coeffs[(br,bc)] * sum over (di,dj) of s_local[(di,dj)] * A[br*2+di][bc*2+dj]
                    # = sum over (br,bc,di,dj) of s_coeffs[(br,bc)] * s_local[(di,dj)] * A[br*2+di][bc*2+dj]
                    
                    # Similarly for B and C.
                    
                    # In the tensor decomposition, this is ONE term.
                    # U[i*m+j][r] = sum over (br,bc,di,dj) where i=br*2+di, j=bc*2+dj of s_coeffs[(br,bc)] * s_local[(di,dj)]
                    # V[j'*p+k][r] = sum over (br,bc,di,dj) where j'=br*2+di, k=bc*2+dj of t_coeffs[(br,bc)] * t_local[(di,dj)]
                    # W[k'*n+i'][r] = sum over (br,bc,di,dj) where i'=br*2+di, k'=bc*2+dj of c_contrib[(br,bc)] * c_local[(di,dj)]
                    
                    # Note: the C block index is different from the A and B block indices.
                    # For A: block (br,bc) means rows br*2..br*2+1, cols bc*2..bc*2+1.
                    # For B: block (br,bc) means rows br*2..br*2+1, cols bc*2..bc*2+1.
                    # For C: block (br,bc) means rows br*2..br*2+1, cols bc*2..bc*2+1.
                    
                    # But wait, the C contribution from M_i goes to specific C blocks.
                    # For M1, it goes to C00 and C11.
                    # C00 is the top-left 2x2 block, C11 is the bottom-right 2x2 block.
                    
                    # So c_contrib maps C block indices to coefficients.
                    
                    # Let me compute the U, V, W coefficients for this term.
                    
                    # U coefficients:
                    for i in range(4):
                        for j in range(4):
                            # i = br*2+di, j = bc*2+dj
                            br = i // 2
                            di = i % 2
                            bc = j // 2
                            dj = j % 2
                            coeff = s_coeffs.get((br,bc), 0) * s_local.get((di,dj), 0)
                            if coeff != 0:
                                U[i*m + j][r] += coeff
                    
                    # V coefficients:
                    for j in range(4):
                        for k in range(4):
                            # j = br*2+di, k = bc*2+dj
                            br = j // 2
                            di = j % 2
                            bc = k // 2
                            dj = k % 2
                            coeff = t_coeffs.get((br,bc), 0) * t_local.get((di,dj), 0)
                            if coeff != 0:
                                V[j*p + k][r] += coeff
                    
                    # W coefficients:
                    for k in range(4):
                        for i in range(4):
                            # i = br*2+di, k = bc*2+dj
                            br = i // 2
                            di = i % 2
                            bc = k // 2
                            dj = k % 2
                            coeff = c_contrib.get((br,bc), 0) * c_local.get((di,dj), 0)
                            if coeff != 0:
                                W[k*n + i][r] += coeff
                    
                    r += 1
    
    # Now add the 4x4x1 part (column 4 of B)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            if r >= rank:
                raise ValueError("Rank overflow in 4x4x1 part")
            U[i*4 + j][r] = 1.0
            V[j*5 + 4][r] = 1.0
            W[4*4 + i][r] = 1.0
            r += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                expected = 1.0
                if abs(s - expected) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): sum={s}, expected={expected}")
    
    # Trim to actual rank used
    U = [row[:r] for row in U]
    V = [row[:r] for row in V]
    W = [row[:r] for row in W]
    
    return {"U": U, "V": V, "W": W}

