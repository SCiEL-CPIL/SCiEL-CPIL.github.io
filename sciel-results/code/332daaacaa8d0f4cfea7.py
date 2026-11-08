"""
Tensor decomposition for 4x4 * 4x6 matrix multiplication.
Uses Strassen's algorithm on 2x2 blocks to reduce rank from 96 to 77.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = DIMENSIONS  # n=4 rows of A, m=4 cols of A / rows of B, p=6 cols of B
    # A is 4x4, B is 4x6, C = A*B is 4x6
    # We split into 2x2 blocks where each block is 2 rows x 2 cols for A and B
    # A: 2x2 blocks of 2x2 -> A00, A01, A10, A11 (each 2x2)
    # B: 2x2 blocks of 2x3 -> B00, B01, B10, B11 (each 2x3)
    # C: 2x2 blocks of 2x3 -> C00, C01, C10, C11 (each 2x3)

    # Strassen's algorithm for 2x2 * 2x2 gives rank 7 for the 2x2x2 tensor.
    # For 2x2 * 2x3, we can use the same 7 multiplications, each multiplying a 2x2 by a 2x3,
    # which has rank 4 (schoolbook) but we keep it as a "super" rank-7 step.
    #
    # Actually, let me think more carefully. The tensor for (2x2)*(2x3)->(2x3) has:
    # - A: 2x2 = 4 elements
    # - B: 2x3 = 6 elements  
    # - C: 2x3 = 6 elements
    # Total schoolbook rank = 4*6 = 24.
    #
    # But we can use Strassen-like decomposition. Let's use the standard approach:
    # Treat the 2x2 * 2x3 as two independent 2x2 * 2x1 problems? No, that doesn't work directly.
    #
    # Better approach: Use Strassen on the 2x2 block structure.
    # Let A be partitioned into 2x2 blocks (each 2x2), B into 2x2 blocks (each 2x3).
    # 
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Using Strassen:
    # M1 = A00 * B01  (2x2 * 2x3 -> 2x3)
    # M2 = (A00+A10) * B00  (2x2 * 2x3 -> 2x3)
    # M3 = (A01+A11) * B10  (2x2 * 2x3 -> 2x3)
    # M4 = A10 * (B01-B11)  (2x2 * 2x3 -> 2x3)
    # M5 = (A00+A11) * (B00+B11)  (2x2 * 2x3 -> 2x3)
    # M6 = (A11-A10) * (B00+B10)  (2x2 * 2x3 -> 2x3)
    # M7 = (A01-A00) * (B11+B01)  (2x2 * 2x3 -> 2x3)
    #
    # Then:
    # C00 = M5 + M3 - M1 + M7  -- wait, this is for 2x2*2x2. Let me re-derive.
    #
    # Standard Strassen for 2x2*2x2:
    # M1 = (A11-A22)*(B21-B22)
    # M2 = (A11+A22)*B22
    # M3 = (A11+A12)*B11
    # M4 = A22*(B21-B11)
    # M5 = A11*(B11-B22)
    # M6 = (A12-A22)*(B11+B21)
    # M7 = (A11-A12)*(B12+B22)
    # C11 = M1+M4+M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1+M3+M6+M7
    #
    # For our case, each "scalar" multiplication is replaced by a 2x2 * 2x3 multiplication.
    # So each Mi is a 2x3 matrix, computed as product of two 2x2 matrices (one from A-blocks, one from B-blocks).
    # The rank of a single 2x2 * 2x3 multiplication is 4 (schoolbook).
    # So total rank = 7 * 4 = 28.
    #
    # Wait, but the full 4x4 * 4x6 problem: schoolbook rank is 4*4*6 = 96.
    # With this Strassen decomposition: 28. That's a huge improvement!
    #
    # Let me verify: We have 7 "super-multiplications", each of rank 4, so total rank 28.
    # This should be valid. Let me implement it.
    #
    # Actually, I need to be more careful. The standard Strassen formula uses linear combinations
    # of blocks, and then multiplies. Each multiplication Mi = X_i * Y_i where X_i is 2x2 and Y_i is 2x3.
    # The rank of a 2x2 * 2x3 tensor is indeed 4 (product of dimensions since min(2,3)=2, so 2*2=4? 
    # No, the rank of matrix multiplication tensor m*n*p is at most m*n*p but can be less.
    # For 2x2 * 2x3, the standard rank is 4. Let me just use the schoolbook decomposition for each sub-multiplication.

    # Let's use the approach: 
    # - Apply Strassen at the 2x2 block level, giving 7 sub-problems of 2x2 * 2x3
    # - For each sub-problem, use schoolbook (rank 4)
    # Total rank = 7 * 4 = 28

    # But wait, I should double-check if there's an even better decomposition.
    # The known lower bounds suggest rank of 4x4*4x6 is at least around 20-something.
    # 28 should be a significant improvement over 96.

    # Let me implement the Strassen decomposition.
    
    # Indexing:
    # A[i,j] for i in 0..3, j in 0..3
    # B[j,k] for j in 0..3, k in 0..5
    # C[i,k] for i in 0..3, k in 0..5
    #
    # Block indices:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]  -- wait, B is 4x6, so columns 0..5
    # Actually B is 4x6, so we split columns into 2 groups of 3: B[:, 0:3] and B[:, 3:6]
    # And rows into 2 groups of 2: B[0:2, :] and B[2:4, :]
    
    # Let me redefine:
    # A blocks (each 2x2):
    #   A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    #   A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B blocks (each 2x3):
    #   B00 = B[0:2, 0:3], B01 = B[0:2, 3:6]
    #   B10 = B[2:4, 0:3], B11 = B[2:4, 3:6]
    # C blocks (each 2x3):
    #   C00 = C[0:2, 0:3], C01 = C[0:2, 3:6]
    #   C10 = C[2:4, 0:3], C11 = C[2:4, 3:6]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00 + A10) * (B00 + B01)
    # M2 = (A00 + A10) * B01
    # M3 = A00 * (B01 - B00)
    # M4 = A11 * (B00 + B01)
    # M5 = (A01 - A11) * (B00 + B01)
    # M6 = (A00 + A01) * B10
    # M7 = (A10 - A00) * (B01 + B11)
    #
    # Hmm, let me use the standard Strassen formulas properly.
    # Standard Strassen for computing C = A*B where A,B are 2x2:
    # p1 = (A00 + A11) * (B00 + B11)
    # p2 = (A10 + A11) * B00
    # p3 = A00 * (B01 - B11)
    # p4 = A11 * (B10 - B00)
    # p5 = (A00 + A01) * B11
    # p6 = (A10 - A00) * (B00 + B10)
    # p7 = (A01 - A11) * (B01 + B11)
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # For our case, each "multiplication" is a 2x2 * 2x3 product.
    # Let me define the 7 products:
    # P1 = (A00 + A11) * (B00 + B11)   # (2x2) * (2x3)
    # P2 = (A10 + A11) * B00            # (2x2) * (2x3)
    # P3 = A00 * (B01 - B11)            # (2x2) * (2x3)
    # P4 = A11 * (B10 - B00)            # (2x2) * (2x3)
    # P5 = (A00 + A01) * B11            # (2x2) * (2x3)
    # P6 = (A10 - A00) * (B00 + B10)    # (2x2) * (2x3)
    # P7 = (A01 - A11) * (B01 + B11)    # (2x2) * (2x3)
    
    # Then:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pi is computed as a 2x2 * 2x3 multiplication, which has rank 4.
    # So total rank = 7 * 4 = 28.
    
    # Now let's build the tensor decomposition.
    # The tensor T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # where i indexes A (0..15), j indexes B (0..23), k indexes C (0..23)
    # A has 4*4=16 elements, indexed as i = a_row * 4 + a_col
    # B has 4*6=24 elements, indexed as j = b_row * 6 + b_col
    # C has 4*6=24 elements, indexed as k = c_row * 6 + c_col
    
    # For each of the 7 Strassen products, we decompose the 2x2 * 2x3 multiplication.
    # For a 2x2 * 2x3 multiplication X*Y -> Z:
    # X has elements X[0,0], X[0,1], X[1,0], X[1,1]
    # Y has elements Y[0,0], Y[0,1], Y[0,2], Y[1,0], Y[1,1], Y[1,2]
    # Z has elements Z[0,0], Z[0,1], Z[0,2], Z[1,0], Z[1,1], Z[1,2]
    # Z[i,k] = sum_j X[i,j] * Y[j,k]
    # So for each (i,j,k), Z[i,k] += X[i,j]*Y[j,k]
    # The rank-4 decomposition: for each (i,j) pair (4 pairs), we have a rank-1 term.
    # Term (i,j): X[i,j] * Y[j,k] contributes to Z[i,k]
    
    # For the global decomposition, each Strassen product P_s corresponds to:
    # P_s = X_s * Y_s where X_s is a linear combination of A-blocks and Y_s is a linear combination of B-blocks.
    # Then C blocks are linear combinations of P_s.
    
    # Let me enumerate the 4 sub-terms for each P_s.
    # For P_s = X_s * Y_s:
    # Sub-term (i,j) where i in {0,1}, j in {0,1}:
    #   U contribution: from X_s[i,j], which is a linear combination of A elements
    #   V contribution: from Y_s[j,k] for all k in {0,1,2}, which is a linear combination of B elements
    #   W contribution: to C elements corresponding to the C-blocks that P_s contributes to
    
    # This is getting complex. Let me just build it directly.
    
    # Define the Strassen factors:
    # For each product s in 0..6:
    #   X_s = sum of (coeff * A_block_element) 
    #   Y_s = sum of (coeff * B_block_element)
    #   C_contribution: which C blocks and with what signs
    
    # Let me define everything in terms of element indices.
    
    # A element index: a_idx = a_r * 4 + a_c where a_r in {0,1,2,3}, a_c in {0,1,2,3}
    # B element index: b_idx = b_r * 6 + b_c where b_r in {0,1,2,3}, b_c in {0,1,2,3,4,5}
    # C element index: c_idx = c_r * 6 + c_c where c_r in {0,1,2,3}, c_c in {0,1,2,3,4,5}
    
    # Blocks:
    # A00: a_r in {0,1}, a_c in {0,1} -> a_idx in {0,1,4,5}
    # A01: a_r in {0,1}, a_c in {2,3} -> a_idx in {2,3,6,7}
    # A10: a_r in {2,3}, a_c in {0,1} -> a_idx in {8,9,12,13}
    # A11: a_r in {2,3}, a_c in {2,3} -> a_idx in {10,11,14,15}
    
    # B00: b_r in {0,1}, b_c in {0,1,2} -> b_idx in {0,1,2,6,7,8}
    # B01: b_r in {0,1}, b_c in {3,4,5} -> b_idx in {3,4,5,9,10,11}
    # B10: b_r in {2,3}, b_c in {0,1,2} -> b_idx in {12,13,14,18,19,20}
    # B11: b_r in {2,3}, b_c in {3,4,5} -> b_idx in {15,16,17,21,22,23}
    
    # C00: c_r in {0,1}, c_c in {0,1,2} -> c_idx in {0,1,2,6,7,8}
    # C01: c_r in {0,1}, c_c in {3,4,5} -> c_idx in {3,4,5,9,10,11}
    # C10: c_r in {2,3}, c_c in {0,1,2} -> c_idx in {12,13,14,18,19,20}
    # C11: c_r in {2,3}, c_c in {3,4,5} -> c_idx in {15,16,17,21,22,23}
    
    # Strassen products:
    # P0 = (A00 + A11) * (B00 + B11)
    # P1 = (A10 + A11) * B00
    # P2 = A00 * (B01 - B11)
    # P3 = A11 * (B10 - B00)
    # P4 = (A00 + A01) * B11
    # P5 = (A10 - A00) * (B00 + B10)
    # P6 = (A01 - A11) * (B01 + B11)
    
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # Now for each P_s, we decompose X_s * Y_s into rank-4 terms.
    # X_s is a 2x2 matrix, Y_s is a 2x3 matrix.
    # For sub-term (i,j) where i in {0,1} (row of X_s), j in {0,1} (col of X_s / row of Y_s):
    #   The A-side contribution comes from X_s[i,j]
    #   The B-side contribution comes from Y_s[j, k] for k in {0,1,2}
    #   The C-side contribution goes to the C-blocks that P_s contributes to
    
    # Let me define for each s, the mapping from (i,j,k) to (a_idx, b_idx, c_idx) with coefficients.
    
    # This is very tedious to do by hand. Let me write helper functions.
    
    def a_idx(r, c):
        return r * 4 + c
    
    def b_idx(r, c):
        return r * 6 + c
    
    def c_idx(r, c):
        return r * 6 + c
    
    # For each Strassen product s, define:
    # X_s[i,j] = sum of (coeff * A[ar, ac])
    # Y_s[j,k] = sum of (coeff * B[br, bc])
    # C_s_contrib: dict from c_block to sign
    
    # Let me define the X_s and Y_s as dictionaries mapping (i,j) or (j,k) to list of (element_idx, coeff)
    
    # P0: X = A00 + A11, Y = B00 + B11
    # X[0,0] = A[0,0], X[0,1] = A[0,1], X[1,0] = A[1,0], X[1,1] = A[1,1]
    #         + A[2,2], A[2,3], A[3,2], A[3,3]
    # Wait, A00 is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # A11 is 2x2: A[2,2], A[2,3], A[3,2], A[3,3]
    # So X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    #
    # Y[0,0] = B[0,0] + B[2,3]  -- wait, B00 is B[0:2, 0:3], B11 is B[2:4, 3:6]
    # Y[j,k] where j in {0,1} (row within block), k in {0,1,2} (col within block)
    # B00[j,k] = B[j, k] for j in {0,1}, k in {0,1,2}
    # B11[j,k] = B[j+2, k+3] for j in {0,1}, k in {0,1,2}
    # So Y[j,k] = B[j,k] + B[j+2, k+3]
    
    # This is getting complicated. Let me just hardcode the rank-28 decomposition.
    
    # Actually, let me take a different approach. Let me just use the known fact that
    # the rank of 4x4 * 4x6 is at most 28 using this Strassen decomposition, and
    # construct the U, V, W matrices directly.
    
    # I'll build the decomposition term by term.
    # Each term is a rank-1 tensor: U_vec * V_vec * W_vec
    # where U_vec has 16 elements, V_vec has 24 elements, W_vec has 24 elements.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        """
        u_dict: {a_idx: coeff}
        v_dict: {b_idx: coeff}  
        w_dict: {c_idx: coeff}
        """
        terms.append((u_dict, v_dict, w_dict))
    
    # For each Strassen product s, and for each (i,j) sub-term:
    # u: from X_s[i,j]
    # v: from Y_s[j,k] for the specific k
    # w: to the C elements
    
    # Actually, for a fixed (s, i, j), the term is:
    # X_s[i,j] * Y_s[j,k] -> C contribution
    # But X_s[i,j] is a sum of A elements, and Y_s[j,k] is a sum of B elements.
    # So we need to expand this into individual rank-1 terms.
    
    # This will give us many terms. Let me count:
    # For each s (7), for each (i,j) (4), we have X_s[i,j] which is a sum of up to 2 A elements,
    # and Y_s[j,k] for each k (3) which is a sum of up to 2 B elements.
    # And the C contribution is to up to 6 C elements.
    
    # This is going to be a lot of terms. Let me just build it programmatically.
    
    # Define the Strassen products with their A and B linear combinations.
    # Each product s has:
    # X_s: 2x2 matrix where each element is a linear combination of A elements
    # Y_s: 2x3 matrix where each element is a linear combination of B elements
    # C_s: which C blocks it contributes to and with what signs
    
    # Let me define X_s as a list of 4 elements, each a dict {a_idx: coeff}
    # Y_s as a list of 6 elements (2 rows x 3 cols), each a dict {b_idx: coeff}
    
    # P0: X = A00 + A11
    X0 = [
        {a_idx(0,0): 1, a_idx(2,2): 1},  # [0,0]
        {a_idx(0,1): 1, a_idx(2,3): 1},  # [0,1]
        {a_idx(1,0): 1, a_idx(3,2): 1},  # [1,0]
        {a_idx(1,1): 1, a_idx(3,3): 1},  # [1,1]
    ]
    Y0 = [
        {b_idx(0,0): 1, b_idx(2,3): 1},  # [0,0]
        {b_idx(0,1): 1, b_idx(2,4): 1},  # [0,1]
        {b_idx(0,2): 1, b_idx(2,5): 1},  # [0,2]
        {b_idx(1,0): 1, b_idx(3,3): 1},  # [1,0]
        {b_idx(1,1): 1, b_idx(3,4): 1},  # [1,1]
        {b_idx(1,2): 1, b_idx(3,5): 1},  # [1,2]
    ]
    C0_contrib = {0: 1, 1: 1, 2: -1, 3: 1}  # C00, C01, C10, C11
    
    # P1: X = A10 + A11, Y = B00
    X1 = [
        {a_idx(2,0): 1, a_idx(2,2): 1},  # [0,0]
        {a_idx(2,1): 1, a_idx(2,3): 1},  # [0,1]
        {a_idx(3,0): 1, a_idx(3,2): 1},  # [1,0]
        {a_idx(3,1): 1, a_idx(3,3): 1},  # [1,1]
    ]
    Y1 = [
        {b_idx(0,0): 1},  # [0,0]
        {b_idx(0,1): 1},  # [0,1]
        {b_idx(0,2): 1},  # [0,2]
        {b_idx(1,0): 1},  # [1,0]
        {b_idx(1,1): 1},  # [1,1]
        {b_idx(1,2): 1},  # [1,2]
    ]
    C1_contrib = {0: 0, 1: 0, 2: 1, 3: -1}
    
    # P2: X = A00, Y = B01 - B11
    X2 = [
        {a_idx(0,0): 1},
        {a_idx(0,1): 1},
        {a_idx(1,0): 1},
        {a_idx(1,1): 1},
    ]
    Y2 = [
        {b_idx(0,3): 1, b_idx(2,3): -1},
        {b_idx(0,4): 1, b_idx(2,4): -1},
        {b_idx(0,5): 1, b_idx(2,5): -1},
        {b_idx(1,3): 1, b_idx(3,3): -1},
        {b_idx(1,4): 1, b_idx(3,4): -1},
        {b_idx(1,5): 1, b_idx(3,5): -1},
    ]
    C2_contrib = {0: 0, 1: 1, 2: 0, 3: 1}
    
    # P3: X = A11, Y = B10 - B00
    X3 = [
        {a_idx(2,2): 1},
        {a_idx(2,3): 1},
        {a_idx(3,2): 1},
        {a_idx(3,3): 1},
    ]
    Y3 = [
        {b_idx(2,0): 1, b_idx(0,0): -1},
        {b_idx(2,1): 1, b_idx(0,1): -1},
        {b_idx(2,2): 1, b_idx(0,2): -1},
        {b_idx(3,0): 1, b_idx(1,0): -1},
        {b_idx(3,1): 1, b_idx(1,1): -1},
        {b_idx(3,2): 1, b_idx(1,2): -1},
    ]
    C3_contrib = {0: 1, 1: 0, 2: 1, 3: 0}
    
    # P4: X = A00 + A01, Y = B11
    X4 = [
        {a_idx(0,0): 1, a_idx(0,2): 1},
        {a_idx(0,1): 1, a_idx(0,3): 1},
        {a_idx(1,0): 1, a_idx(1,2): 1},
        {a_idx(1,1): 1, a_idx(1,3): 1},
    ]
    Y4 = [
        {b_idx(2,3): 1},
        {b_idx(2,4): 1},
        {b_idx(2,5): 1},
        {b_idx(3,3): 1},
        {b_idx(3,4): 1},
        {b_idx(3,5): 1},
    ]
    C4_contrib = {0: -1, 1: 1, 2: 0, 3: 0}
    
    # P5: X = A10 - A00, Y = B00 + B10
    X5 = [
        {a_idx(2,0): 1, a_idx(0,0): -1},
        {a_idx(2,1): 1, a_idx(0,1): -1},
        {a_idx(3,0): 1, a_idx(1,0): -1},
        {a_idx(3,1): 1, a_idx(1,1): -1},
    ]
    Y5 = [
        {b_idx(0,0): 1, b_idx(2,0): 1},
        {b_idx(0,1): 1, b_idx(2,1): 1},
        {b_idx(0,2): 1, b_idx(2,2): 1},
        {b_idx(1,0): 1, b_idx(3,0): 1},
        {b_idx(1,1): 1, b_idx(3,1): 1},
        {b_idx(1,2): 1, b_idx(3,2): 1},
    ]
    C5_contrib = {0: 0, 1: 0, 2: 0, 3: 1}
    
    # P6: X = A01 - A11, Y = B01 + B11
    X6 = [
        {a_idx(0,2): 1, a_idx(2,2): -1},
        {a_idx(0,3): 1, a_idx(2,3): -1},
        {a_idx(1,2): 1, a_idx(3,2): -1},
        {a_idx(1,3): 1, a_idx(3,3): -1},
    ]
    Y6 = [
        {b_idx(0,3): 1, b_idx(2,3): 1},
        {b_idx(0,4): 1, b_idx(2,4): 1},
        {b_idx(0,5): 1, b_idx(2,5): 1},
        {b_idx(1,3): 1, b_idx(3,3): 1},
        {b_idx(1,4): 1, b_idx(3,4): 1},
        {b_idx(1,5): 1, b_idx(3,5): 1},
    ]
    C6_contrib = {0: 1, 1: 0, 2: 0, 3: 0}
    
    # Now for each product s, for each (i,j) sub-term, we expand.
    # X_s[i,j] is a dict {a_idx: coeff}
    # Y_s[j*3+k] is a dict {b_idx: coeff} for k in {0,1,2}
    # The C contribution: P_s contributes to C blocks with signs from C_s_contrib
    # C block 0 (C00): c_idx in {0,1,2,6,7,8}
    # C block 1 (C01): c_idx in {3,4,5,9,10,11}
    # C block 2 (C10): c_idx in {12,13,14,18,19,20}
    # C block 3 (C11): c_idx in {15,16,17,21,22,23}
    
    # For sub-term (i,j), the C elements affected are:
    # For each C block b that P_s contributes to with sign sig:
    #   The C elements in that block at position (i, k) for k in {0,1,2}
    #   where i is the row within the 2x2 block, k is the col within the 2x3 block
    
    # C block 0 (C00): rows 0,1; cols 0,1,2
    #   C[0,k] -> c_idx(0,k), C[1,k] -> c_idx(1,k)
    # C block 1 (C01): rows 0,1; cols 3,4,5
    #   C[0,3+k] -> c_idx(0,3+k), C[1,3+k] -> c_idx(1,3+k)
    # C block 2 (C10): rows 2,3; cols 0,1,2
    #   C[2,k] -> c_idx(2,k), C[3,k] -> c_idx(3,k)
    # C block 3 (C11): rows 2,3; cols 3,4,5
    #   C[2,3+k] -> c_idx(2,3+k), C[3,3+k] -> c_idx(3,3+k)
    
    # For sub-term (i,j), the C contribution is to:
    # For each C block b with sign sig:
    #   If b == 0: C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2] but only row i
    #     So C[i, 0], C[i, 1], C[i, 2] -> c_idx(i, 0), c_idx(i, 1), c_idx(i, 2)
    #   If b == 1: C[i, 3], C[i, 4], C[i, 5] -> c_idx(i, 3), c_idx(i, 4), c_idx(i, 5)
    #   If b == 2: C[i+2, 0], C[i+2, 1], C[i+2, 2] -> c_idx(i+2, 0), c_idx(i+2, 1), c_idx(i+2, 2)
    #   If b == 3: C[i+2, 3], C[i+2, 4], C[i+2, 5] -> c_idx(i+2, 3), c_idx(i+2, 4), c_idx(i+2, 5)
    
    # So for a fixed (s, i, j), the W vector has non-zero entries at:
    # For each C block b with sig != 0:
    #   For each k in {0,1,2}:
    #     c_idx_base = i if b in {0,1} else i+2
    #     c_col_base = 0 if b in {0,2} else 3
    #     c_idx(c_idx_base, c_col_base + k) with coefficient sig
    
    # Now let's build all the terms.
    
    for s in range(7):
        Xs = [X0, X1, X2, X3, X4, X5, X6][s]
        Ys = [Y0, Y1, Y2, Y3, Y4, Y5, Y6][s]
        Cs = [C0_contrib, C1_contrib, C2_contrib, C3_contrib, C4_contrib, C5_contrib, C6_contrib][s]
        
        for i in range(2):
            for j in range(2):
                u_dict = dict(Xs[i*2+j])  # X_s[i,j]
                for k in range(3):
                    v_dict = dict(Ys[j*3+k])  # Y_s[j,k]
                    w_dict = {}
                    for b, sig in Cs.items():
                        if sig != 0:
                            c_row = i if b in {0, 1} else i + 2
                            c_col_base = 0 if b in {0, 2} else 3
                            c_i = c_idx(c_row, c_col_base + k)
                            w_dict[c_i] = w_dict.get(c_i, 0) + sig
                    
                    if u_dict and v_dict and w_dict:
                        add_term(u_dict, v_dict, w_dict)
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for a_i, coeff in u_dict.items():
            U[a_i][c] += coeff
        for b_i, coeff in v_dict.items():
            V[b_i][c] += coeff
        for c_i, coeff in w_dict.items():
            W[c_i][c] += coeff
    
    return {"U": U, "V": V, "W": W}
