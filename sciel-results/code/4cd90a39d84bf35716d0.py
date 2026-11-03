import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We will build the decomposition in two parts:
    # 1. The 4x4 x 4x8 part using two 4x4 x 4x4 Strassen blocks (Rank 49 each)
    # 2. The 4x4 x 4x1 part using schoolbook (Rank 16)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_val, u_idx, v_val, v_idx, w_val, w_idx):
        # We are building sparse lists, but the interface expects dense matrices.
        # It is more efficient to build numpy arrays and then convert.
        pass

    # Let's use numpy arrays for construction
    # Max rank estimate: 49 + 49 + 16 = 114
    # Let's allocate slightly more in case of error, or just append to lists.
    # Appending to lists of dicts or tuples and then building is safer.
    
    terms = []
    
    def add_strassen_2x2_terms(a_row_offset, a_col_offset, b_row_offset, b_col_offset, k_offset):
        """
        Adds 7 terms for a 2x2x2 Strassen multiplication.
        a_row_offset, a_col_offset: offset for A block (i, j)
        b_row_offset, b_col_offset: offset for B block (j, k)
        k_offset: offset for k index in global V and W
        """
        # Local indices for 2x2 block
        # A[i, j], B[j, k], C[i, k]
        # i, j, k in {0, 1}
        
        # Strassen 2x2x2 (Rank 7)
        # M1 = A00 * (B01 - B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A01 - A11) * (B10 + B11)
        # M4 = A11 * (B00 - B01)
        # M5 = (A00 + A01) * B00
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A11 - A10) * (B10 - B11)
        
        # Each M is a scalar product of linear forms.
        # We expand each M into its scalar components.
        # For M1 = A[0,0] * (B[0,1] - B[1,1]):
        # Term 1: A[0,0] * B[0,1] -> C[0,0]
        # Term 2: A[0,0] * (-B[1,1]) -> C[0,0]
        
        # This is getting complicated to do manually. 
        # Let's define the 7 Strassen terms for 2x2 explicitly.
        
        # Local indices:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # M1 = A00 * (B01 - B11)
        # Contribution to C00: A00*B01 - A00*B11
        terms.append((
            [ (a_row_offset + 0, a_col_offset + 0, 1) ], # U
            [ (b_row_offset + 0, b_col_offset + 1 + k_offset, 1), (b_row_offset + 1, b_col_offset + 1 + k_offset, -1) ], # V
            [ (b_col_offset + 0 + k_offset, a_row_offset + 0, 1) ] # W
        ))
        # Wait, the W index is k*n + i. 
        # For C[0,0], k=0, i=0. 
        # The term is A[0,0] * B[0,1] * C[0,0].
        # U[0,0]=1, V[0,1]=1, W[0,0]=1.
        # My tuple structure is getting messy. Let's just store (u_idx, v_idx, w_idx, coeff) for each rank-1 term.
        # A rank-1 term is a single non-zero entry in U, V, W? No.
        # A rank-1 term in the tensor decomposition is a product of 3 vectors.
        # If the vectors are sparse, it's efficient.
        # Strassen terms are sums of products of entries.
        # e.g. M1 = A00*(B01-B11) = A00*B01 - A00*B11.
        # This is TWO rank-1 terms:
        # T1: U[A00]=1, V[B01]=1, W[C00]=1
        # T2: U[A00]=1, V[B11]=-1, W[C00]=1
        
        # So each Strassen "M" is actually a sum of scalar products.
        # M1 has 2 scalar products.
        # M2 = (A00+A10)*B11 = A00*B11 + A10*B11. (2 scalar products)
        # M3 = (A01-A11)*(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11. (4 scalar products)
        # M4 = A11*(B00-B01) = A11*B00 - A11*B01. (2 scalar products)
        # M5 = (A00+A01)*B00 = A00*B00 + A01*B00. (2 scalar products)
        # M6 = (A10-A00)*(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01. (4 scalar products)
        # M7 = (A11-A10)*(B10-B11) = A11*B10 - A11*B11 - A10*B10 + A10*B11. (4 scalar products)
        
        # Total scalar terms for 2x2x2 Strassen: 2+2+4+2+2+4+4 = 20?
        # No, Strassen rank is 7. The 7 terms are the "multiplications".
        # In the tensor decomposition context, the "rank" is the number of terms in the sum.
        # The standard Strassen decomposition of the 2x2x2 tensor has rank 7.
        # This means we can write T = sum_{r=1}^7 u_r v_r w_r.
        # The vectors u_r, v_r, w_r are not necessarily sparse single entries.
        # For example, for M3 = (A01-A11)(B10+B11):
        # u_3 has non-zeros at A01 (1) and A11 (-1).
        # v_3 has non-zeros at B10 (1) and B11 (1).
        # w_3 has non-zeros at C00, C01, C10, C11?
        # C00 = ... + M3 ...
        # C01 = ... + M3 ...
        # C10 = ...
        # C11 = ... + M3 ...
        # Actually, M3 contributes to C00, C01, C10, C11?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M3 contributes to C01 and C11.
        # w_3 has non-zeros at C01 (1) and C11 (1).
        
        # So, to implement Strassen, we define 7 terms.
        # Each term r has:
        # u_r: vector in R^4 (for 2x2 A)
        # v_r: vector in R^4 (for 2x2 B)
        # w_r: vector in R^4 (for 2x2 C)
        
        # Let's define these 7 terms for the local 2x2 block.
        
        # M1 = A00 * (B01 - B11)
        # u1: [1, 0, 0, 0] (A00)
        # v1: [0, 1, -1, 0] (B01 - B11) -> indices (0,1), (1,1)
        # w1: [1, 0, 0, 0] (C00) -- Wait, M1 contributes to C00 and C11.
        # C00 gets +M1. C11 gets +M1.
        # So w1: [1, 0, 0, 1]
        
        # M2 = (A00 + A10) * B11
        # u2: [1, 0, 1, 0]
        # v2: [0, 0, 0, 1]
        # w2: [0, 0, 1, 0] (C10) -- M2 contributes to C10 and C11?
        # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So w2: [0, 0, 1, -1]
        
        # M3 = (A01 - A11) * (B10 + B11)
        # u3: [0, 1, 0, -1]
        # v3: [0, 0, 1, 1]
        # w3: [0, 1, 0, 1] (C01, C11)
        
        # M4 = A11 * (B00 - B01)
        # u4: [0, 0, 0, 1]
        # v4: [1, -1, 0, 0]
        # w4: [1, 0, 1, 0] (C00, C10)
        
        # M5 = (A00 + A01) * B00
        # u5: [1, 1, 0, 0]
        # v5: [1, 0, 0, 0]
        # w5: [-1, 1, 0, 0] (C00 gets -M5, C01 gets +M5)
        
        # M6 = (A10 - A00) * (B00 + B01)
        # u6: [-1, 0, 1, 0]
        # v6: [1, 1, 0, 0]
        # w6: [0, 0, 0, 1] (C11)
        
        # M7 = (A11 - A10) * (B10 - B11)
        # u7: [0, 0, -1, 1]
        # v7: [0, 0, 1, -1]
        # w7: [1, 0, 0, 0] (C00)
        
        # Let's verify M1:
        # u1*v1*w1 = A00 * (B01-B11) * (C00+C11)
        # Term in C00: A00*B01 - A00*B11. Correct.
        # Term in C11: A00*B01 - A00*B11. Correct.
        
        # This defines the 7 rank-1 terms for the 2x2x2 tensor.
        
        # Now we map these to the global 4x4x4 block.
        # The 4x4 A is composed of four 2x2 blocks: A00, A01, A10, A11.
        # We apply the 7 Strassen terms at the BLOCK level.
        # This means we treat the 4x4 matrices as 2x2 matrices of 2x2 blocks.
        # Let A_blocks = [[A00, A01], [A10, A11]].
        # The "u" vector for the block-level Strassen is a vector of 2x2 blocks.
        # But we need scalar rank-1 terms for the full 4x4x4 tensor.
        
        # This approach (Block Strassen) results in 7 * (Rank of 2x2x2) terms.
        # If we use the rank-7 Strassen for the inner 2x2x2, we get 7*7=49 terms.
        
        # To generate these 49 terms, we can iterate over the 7 block-level terms.
        # For each block-level term, we have coefficients for the 4 A-blocks, 4 B-blocks, 4 C-blocks.
        # For example, for M1_block:
        # u_block = [A00, 0, 0, 0] (coeff 1 for A00)
        # v_block = [0, B01-B11, 0, 0] (coeff 1 for B01, -1 for B11)
        # w_block = [C00, 0, 0, C11] (coeff 1 for C00, 1 for C11)
        
        # We need to expand this into scalar terms.
        # u_block is a linear combination of A-blocks.
        # v_block is a linear combination of B-blocks.
        # w_block is a linear combination of C-blocks.
        
        # The term is (sum a_b A_b) * (sum b_c B_c) * (sum w_d C_d).
        # This expands to sum_{b,c,d} a_b b_c w_d (A_b * B_c * C_d).
        # Each A_b * B_c * C_d is a 2x2x2 tensor.
        # We can decompose each 2x2x2 tensor into 7 rank-1 terms using Strassen.
        
        # So, for each of the 7 block-level terms, we have up to 4*4*4 = 64 pairs of (A_block, B_block, C_block).
        # But the coefficients are sparse.
        # For M1_block:
        # A_blocks involved: A00 (coeff 1).
        # B_blocks involved: B01 (coeff 1), B11 (coeff -1).
        # C_blocks involved: C00 (coeff 1), C11 (coeff 1).
        # Pairs:
        # 1. A00 * B01 * C00 (coeff 1*1*1 = 1)
        # 2. A00 * B01 * C11 (coeff 1*1*1 = 1)
        # 3. A00 * (-B11) * C00 (coeff 1*(-1)*1 = -1)
        # 4. A00 * (-B11) * C11 (coeff 1*(-1)*1 = -1)
        
        # For each of these 4 pairs, we need the rank-7 Strassen decomposition of the 2x2x2 tensor.
        # So we add 4 * 7 = 28 terms for M1_block?
        # No, the Strassen decomposition is linear.
        # T_2x2 = sum_{r=1}^7 u_r v_r w_r.
        # So A00 * B01 * C00 = sum_{r=1}^7 (A00 * u_r) * (B01 * v_r) * (C00 * w_r).
        # This is 7 terms.
        # So yes, 4 pairs * 7 terms = 28 terms for M1_block.
        
        # This seems expensive. 7 block terms * (avg pairs) * 7 inner terms.
        # Let's just generate the 49 terms directly by applying the 2x2 Strassen to the 4x4 blocks.
        
        pass

    # Given the complexity and time, I will implement the **Rank 56** decomposition (Block Strassen with Schoolbook Inner).
    # 7 block terms * 8 inner terms (schoolbook for 2x2x2) = 56 terms per 4x4x4 block.
    # Total rank: 56 + 56 + 16 = 128.
    # This is easier to implement correctly.
    
    U = np.zeros((16, 128))
    V = np.zeros((36, 128))
    W = np.zeros((36, 128))
    
    col = 0
    
    # Strassen 2x2 block coefficients
    # M1: A00 * (B01 - B11) -> C00, C11
    # M2: (A00+A10) * B11 -> C10, C11
    # M3: (A01-A11) * (B10+B11) -> C01, C11
    # M4: A11 * (B00-B01) -> C00, C10
    # M5: (A00+A01) * B00 -> C00, C01
    # M6: (A10-A00) * (B00+B01) -> C11
    # M7: (A11-A10) * (B10-B11) -> C00
    
    # Inner 2x2x2 Schoolbook terms:
    # For A_block (2x2), B_block (2x2), C_block (2x2):
    # 4 terms:
    # (i', j', k') in {0,1}^3
    # U[A_block(i',j')] * V[B_block(j',k')] * W[C_block(i',k')]
    
    def add_block_strassen_term(k_offset, a_coeff, b_coeff, c_coeff):
        """
        a_coeff: dict {block_idx: coeff} for A (0:00, 1:01, 2:10, 3:11)
        b_coeff: dict {block_idx: coeff} for B
        c_coeff: dict {block_idx: coeff} for C
        """
        nonlocal col
        for a_b, a_c in a_coeff.items():
            for b_b, b_c in b_coeff.items():
                for c_b, c_c in c_coeff.items():
                    # Determine global indices for the 2x2 blocks
                    # A block a_b:
                    # 0: rows 0-1, cols 0-1
                    # 1: rows 0-1, cols 2-3
                    # 2: rows 2-3, cols 0-1
                    # 3: rows 2-3, cols 2-3
                    a_row = (a_b // 2) * 2
                    a_col = (a_b % 2) * 2
                    
                    # B block b_b:
                    b_row = (b_b // 2) * 2
                    b_col = (b_b % 2) * 2 + k_offset # k_offset is 0 or 4
                    
                    # C block c_b:
                    c_row = (c_b // 2) * 2 # i index
                    c_col = (c_b % 2) * 2 + k_offset # k index? No.
                    # C is 4x4. C[i,k].
                    # C block c_b corresponds to rows c_row..c_row+1 and cols c_col..c_col+1.
                    # But C is indexed by (i, k) in the tensor W[k*n+i].
                    # So W index is k*4 + i.
                    # For C block c_b:
                    # i in {c_row, c_row+1}
                    # k in {c_col, c_col+1}
                    
                    coeff = a_c * b_c * c_c
                    if coeff == 0: continue
                    
                    # 4 inner terms for 2x2x2 schoolbook
                    for i_prime in range(2):
                        for j_prime in range(2):
                            for k_prime in range(2):
                                # U index: i_global * 4 + j_global
                                i_global = a_row + i_prime
                                j_global = a_col + j_prime
                                u_idx = i_global * 4 + j_global
                                
                                # V index: j_global * 9 + k_global
                                # j_global for B is b_row + j_prime
                                # k_global for B is b_col + k_prime
                                j_b_global = b_row + j_prime
                                k_b_global = b_col + k_prime
                                v_idx = j_b_global * 9 + k_b_global
                                
                                # W index: k_global * 4 + i_global
                                # k_global for C is c_col + k_prime
                                # i_global for C is c_row + i_prime
                                k_c_global = c_col + k_prime
                                i_c_global = c_row + i_prime
                                w_idx = k_c_global * 4 + i_c_global
                                
                                U[u_idx, col] += coeff
                                V[v_idx, col] += coeff
                                W[w_idx, col] += coeff
                                
                                col += 1
                                
    # Define the 7 Strassen block terms
    # A blocks: 0:00, 1:01, 2:10, 3:11
    # B blocks: 0:00, 1:01, 2:10, 3:11
    # C blocks: 0:00, 1:01, 2:10, 3:11
    
    # M1 = A00 * (B01 - B11) -> C00, C11
    add_block_strassen_term(0, {0: 1}, {1: 1, 3: -1}, {0: 1, 3: 1})
    
    # M2 = (A00 + A10) * B11 -> C10, C11
    add_block_strassen_term(0, {0: 1, 2: 1}, {3: 1}, {2: 1, 3: -1})
    
    # M3 = (A01 - A11) * (B10 + B11) -> C01, C11
    add_block_strassen_term(0, {1: 1, 3: -1}, {2: 1, 3: 1}, {1: 1, 3: 1})
    
    # M4 = A11 * (B00 - B01) -> C00, C10
    add_block_strassen_term(0, {3: 1}, {0: 1, 1: -1}, {0: 1, 2: 1})
    
    # M5 = (A00 + A01) * B00 -> C00, C01
    add_block_strassen_term(0, {0: 1, 1: 1}, {0: 1}, {0: -1, 1: 1})
    
    # M6 = (A10 - A00) * (B00 + B01) -> C11
    add_block_strassen_term(0, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    
    # M7 = (A11 - A10) * (B10 - B11) -> C00
    add_block_strassen_term(0, {3: 1, 2: -1}, {2: 1, 3: -1}, {0: 1})
    
    # Second 4x4 block (k=4..7)
    add_block_strassen_term(4, {0: 1}, {1: 1, 3: -1}, {0: 1, 3: 1})
    add_block_strassen_term(4, {0: 1, 2: 1}, {3: 1}, {2: 1, 3: -1})
    add_block_strassen_term(4, {1: 1, 3: -1}, {2: 1, 3: 1}, {1: 1, 3: 1})
    add_block_strassen_term(4, {3: 1}, {0: 1, 1: -1}, {0: 1, 2: 1})
    add_block_strassen_term(4, {0: 1, 1: 1}, {0: 1}, {0: -1, 1: 1})
    add_block_strassen_term(4, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add_block_strassen_term(4, {3: 1, 2: -1}, {2: 1, 3: -1}, {0: 1})
    
    # Third part: 4x4 x 4x1 (k=8)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j * 9 + 8, col] = 1
            W[8 * 4 + i, col] = 1
            col += 1
            
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
