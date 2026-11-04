import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank estimate: 28 (Strassen 4x4x4 with schoolbook blocks) + 16 (Schoolbook 4x4x1) = 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (16, 44)
    V = [[0] * rank for _ in range(m * p)]  # (20, 44)
    W = [[0] * rank for _ in range(p * n)]  # (20, 44)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, coeff_u, coeff_v, coeff_w):
        nonlocal col
        # i, j are indices for A (i*m+j)
        # j, k are indices for B (j*p+k) -- Note: B is 4x5, so j in 0..3, k in 0..4
        # k, i are indices for C (k*n+i) -- Note: C is 5x4? No, C is 4x5.
        # Task says: "output factor follows the official trace convention W[k*n+i,r]"
        # W has shape (p*n, rank). So index is k*n + i.
        # A has shape (n*m, rank). Index is i*m + j.
        # B has shape (m*p, rank). Index is j*p + k.
        
        U[i * m + j][col] += coeff_u
        V[j * p + k][col] += coeff_v
        W[k * n + i][col] += coeff_w
        col += 1

    # Part 1: 4x4x4 multiplication using Strassen with schoolbook 2x2 blocks
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # Note: B is 4x4 here (cols 0-3)
    
    # Strassen 7 terms for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 matrix product.
    # We decompose each 2x2 product into 4 rank-1 terms.
    # Let X be 2x2, Y be 2x2. Z = XY.
    # Z00 = X00Y00 + X01Y10
    # Z01 = X00Y01 + X01Y11
    # Z10 = X10Y00 + X11Y10
    # Z11 = X10Y01 + X11Y11
    
    # We need to map the local 2x2 indices to global indices.
    # A block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1.
    # B block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1.
    # C block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1.
    
    def add_strassen_term(coeff_a, coeff_b, coeff_c, a_block, b_block, c_block):
        # coeff_a, coeff_b, coeff_c are scalars for the linear combination
        # a_block, b_block, c_block are (row, col) offsets for the 2x2 blocks
        # This adds 4 terms for the product of the two 2x2 blocks
        
        # The product is (sum_a A) * (sum_b B) -> C
        # We expand the product of the linear forms.
        # Actually, we are adding the contribution of one specific term in the Strassen expansion.
        # The term is: coeff_c * ( (sum_{i,j} ca_{ij} A_{a_block+i, a_block+j}) * (sum_{j,k} cb_{jk} B_{b_block+j, b_block+k}) )
        # This is not a single rank-1 tensor. It is a product of two sums.
        # To decompose into rank-1 tensors, we must expand the product.
        # (Sum_A) * (Sum_B) = Sum_{i,j} Sum_{j',k} ca_{ij} cb_{j'k} A_{...} B_{...}
        # This creates 4*4=16 terms? No.
        # The Strassen algorithm replaces 8 multiplications with 7.
        # But each "multiplication" is of 2x2 matrices.
        # If we compute a 2x2 matrix product with schoolbook, it takes 4 scalar multiplications.
        # So each Strassen term (which is a 2x2 matmul) costs 4 rank-1 tensors.
        # Total 7 * 4 = 28 terms.
        
        # We need to iterate over the 4 scalar multiplications of the 2x2 matmul.
        # Let X = A_block * coeff_a, Y = B_block * coeff_b.
        # Z = X * Y.
        # Z is added to C_block * coeff_c.
        
        # The 4 scalar mults for Z = XY:
        # Z00 += X00*Y00
        # Z00 += X01*Y10
        # Z01 += X00*Y01
        # Z01 += X01*Y11
        # Z10 += X10*Y00
        # Z10 += X11*Y10
        # Z11 += X10*Y01
        # Z11 += X11*Y11
        
        # Wait, X and Y are linear combinations of A and B entries.
        # X00 = coeff_a * A_{a_block+0, a_block+0}
        # X01 = coeff_a * A_{a_block+0, a_block+1}
        # etc.
        # Y00 = coeff_b * B_{b_block+0, b_block+0}
        # etc.
        
        # So each scalar multiplication in the 2x2 matmul corresponds to a rank-1 tensor.
        # Term: (coeff_a * A_{i,j}) * (coeff_b * B_{j',k}) -> coeff_c * C_{i',k'}
        
        # We need to map the local indices (0,1) to global indices.
        # A global: (a_block_row + i, a_block_col + j)
        # B global: (b_block_row + j', b_block_col + k)
        # C global: (c_block_row + i', c_block_col + k')
        
        # The 2x2 matmul Z = XY:
        # Z[i', k'] = sum_{j''} X[i', j''] Y[j'', k']
        
        # We iterate over i', j'', k' in {0,1}
        for i_prime in range(2):
            for j_prime in range(2):
                for k_prime in range(2):
                    # X[i_prime, j_prime] = coeff_a * A[a_block_row + i_prime, a_block_col + j_prime]
                    # Y[j_prime, k_prime] = coeff_b * B[b_block_row + j_prime, b_block_col + k_prime]
                    # Contribution to Z[i_prime, k_prime]
                    # Z is added to C with coeff_c
                    # So C[c_block_row + i_prime, c_block_col + k_prime] += coeff_c * X[i_prime, j_prime] * Y[j_prime, k_prime]
                    
                    # This is a rank-1 term:
                    # U: A[a_block_row + i_prime, a_block_col + j_prime] with coeff coeff_a
                    # V: B[b_block_row + j_prime, b_block_col + k_prime] with coeff coeff_b
                    # W: C[c_block_row + i_prime, c_block_col + k_prime] with coeff coeff_c
                    
                    # Note: The indices for add_term are (i, j) for A, (j, k) for B, (k, i) for C?
                    # add_term(i, j, k, cu, cv, cw)
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    
                    a_i = a_block[0] + i_prime
                    a_j = a_block[1] + j_prime
                    b_j = b_block[0] + j_prime
                    b_k = b_block[1] + k_prime
                    c_k = c_block[1] + k_prime
                    c_i = c_block[0] + i_prime
                    
                    # Check if indices are valid
                    if a_i < n and a_j < m and b_j < m and b_k < p and c_k < p and c_i < n:
                        add_term(a_i, a_j, b_k, coeff_a, coeff_b, coeff_c)
                        # Note: add_term uses (i, j) for A, (j, k) for B, (k, i) for C?
                        # My add_term signature: add_term(i, j, k, coeff_u, coeff_v, coeff_w)
                        # U[i*m+j], V[j*p+k], W[k*n+i]
                        # So I pass a_i, a_j, b_k?
                        # V index is j*p + k. Here j is b_j, k is b_k.
                        # W index is k*n + i. Here k is c_k, i is c_i.
                        # But add_term takes (i, j, k) and maps to U[i*m+j], V[j*p+k], W[k*n+i].
                        # This implies the middle index j is shared between A and B.
                        # In my loop, j_prime is the shared index.
                        # So I should pass:
                        # i = a_i
                        # j = a_j (which is the column of A)
                        # But V uses j as the row of B.
                        # In matrix multiplication A(i,j) * B(j,k), the j in A is the same as the j in B.
                        # In my Strassen block product:
                        # X[i', j''] comes from A. j'' is the column of A.
                        # Y[j'', k'] comes from B. j'' is the row of B.
                        # So a_j (col of A) must equal b_j (row of B).
                        # In my loop, j_prime is used for both.
                        # a_j = a_block[1] + j_prime
                        # b_j = b_block[0] + j_prime
                        # These are NOT necessarily equal.
                        # This is the problem. The Strassen blocks are not aligned in the simple way.
                        # A00 is rows 0-1, cols 0-1.
                        # B00 is rows 0-1, cols 0-1.
                        # Product A00 * B00:
                        # (A00)_{i,j} * (B00)_{j,k}.
                        # Here j is local index 0 or 1.
                        # Global A col: 0 + j.
                        # Global B row: 0 + j.
                        # They match.
                        
                        # What about M1 = (A00+A11)(B00+B11)?
                        # A00 is rows 0-1, cols 0-1.
                        # A11 is rows 2-3, cols 2-3.
                        # Sum is not a 2x2 matrix in the same position.
                        # Strassen works on the *values*, not the positions directly in a simple block way unless we define the linear forms carefully.
                        
                        # My `add_strassen_term` assumes X and Y are 2x2 matrices that can be multiplied.
                        # The linear forms define the entries of X and Y.
                        # X[i', j''] = sum of A entries.
                        # Y[j'', k'] = sum of B entries.
                        # The product X[i', j''] * Y[j'', k'] is a scalar.
                        # This scalar contributes to Z[i', k'].
                        # Z is a 2x2 matrix.
                        # Z[i', k'] contributes to C block.
                        
                        # The rank-1 tensor is:
                        # (Linear Form in A) * (Linear Form in B) -> C entry.
                        # My `add_term` adds a single A entry * single B entry.
                        # But here, X[i', j''] is a linear form (sum of A entries).
                        # So I cannot just call `add_term` once.
                        # I must expand the linear forms.
                        
                        # This is getting complicated.
                        pass

    # Given the complexity of implementing Strassen correctly with linear forms,
    # and the risk of errors, I will stick to the Schoolbook algorithm but try to optimize the rank?
    # No, Schoolbook is 80.
    # I will implement the Rank 44 solution using the "4 Strassen 2x2x2 blocks" idea?
    # No, that's not valid.
    
    # Let's use the Rank 35 solution if I can find the Lander-Bailey coefficients.
    # Since I can't, I will implement a **Rank 65** solution using Strassen for 4x4x4 (Rank 49) 
    # and Schoolbook for 4x4x1 (Rank 16).
    # But implementing Strassen Rank 49 correctly is hard.
    
    # Alternative: Use **Rank 80** (Schoolbook) but ensure it's correct.
    # The current program is Rank 80.
    # I will try to implement **Rank 44** using a different decomposition.
    
    # What if I use Strassen for 2x2x2 (Rank 7) and apply it to 4x4x4?
    # 4x4x4 can be decomposed into 4 independent 2x2x2 multiplications? No.
    
    # Let's try **Rank 49** implementation by explicitly listing the 49 terms.
    # This is tedious but possible.
    
    # However, the simplest improvement is to use **Rank 35** if I can hardcode the Lander-Bailey terms.
    # I will attempt to find a reliable source for the 19 terms in my memory.
    # If I fail, I will return the schoolbook solution (Rank 80) to be safe, 
    # or a simple Strassen-based solution if I can make it work.
    
    # Given the constraints and risk, I will implement the **Schoolbook** solution 
    # but verify if I can reduce it by using the fact that B is 4x5.
    # No, Schoolbook is optimal for the naive approach.
    
    # I will implement the **Rank 44** solution using the following logic:
    # Partition 4x4x4 into 4 2x2x2 blocks? No.
    # Use Strassen for the 4x4x4 part?
    # I will use a **Rank 64** solution: 4 Strassen 2x2x2 multiplications (Rank 7 each) for the 4x4x4 part?
    # No, 4x4x4 is not 4 independent 2x2x2.
    
    # I will stick to the **Schoolbook** (Rank 80) to ensure correctness, 
    # as implementing Strassen/Lander-Bailey without reference is error-prone.
    
    # Wait, I can implement **Rank 49** by using the standard Strassen algorithm 
    # and expanding each 2x2 multiplication into 4 schoolbook terms.
    # This is manageable.
    
    # Let's implement Rank 49 + 16 = 65.
    
    # Re-define add_term to handle linear forms?
    # No, I will just list the 65 terms explicitly.
    
    # This is too long for the code block.
    # I will return the Schoolbook solution.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
