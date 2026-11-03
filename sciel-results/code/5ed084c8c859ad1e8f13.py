"""Matrix multiplication 4,4,6 via Strassen decomposition (rank 77)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-77 tensor decomposition for 4x4 * 4x6 multiplication.
    
    Decomposition:
      - A (4x4) * B_L (4x4) using Strassen: rank 49
      - A (4x4) * B_R (4x2) using 4 Strassen 2x2x2: rank 28
      Total: rank 77
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We'll build U (16 x rank), V (24 x rank), W (24 x rank)
    # U index: i*m + j (row i, col j of A)
    # V index: j*p + k (row j, col k of B)
    # W index: k*n + i (row k, col i of C)
    
    all_u_terms = []
    all_v_terms = []
    all_w_terms = []
    
    # Strassen 2x2x2: multiply X (2x2) * Y (2x2) = Z (2x2)
    # 7 multiplications:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10+Y11)
    # M4 = X11(Y01+Y10)  -- wait, standard Strassen:
    # Actually standard:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)(Y00+Y01)  -- no
    # Let me use the correct Strassen formulas:
    # Z00 = M1 - M4 + M6 - M7
    # Z01 = M3 + M4
    # Z10 = M1 + M5 - M6 - M7
    # Z11 = M2 - M3 + M5 + M7
    # where:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)  -- hmm, let me use a cleaner version
    
    # Standard Strassen 2x2:
    # P1 = (X00 + X11) * (Y00 + Y11)
    # P2 = (X10 + X11) * Y00
    # P3 = X00 * (Y01 - Y11)
    # P4 = X11 * (Y10 - Y00)
    # P5 = (X00 + X01) * Y11
    # P6 = (X10 - X00) * (Y00 + Y01)
    # P7 = (X01 - X11) * (Y10 + Y11)
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    def strassen_2x2_u_rows(x_indices):
        """For 2x2 matrix X with elements at given (i,j) positions, return U contribution for each of 7 Strassen mults.
        x_indices: list of 4 (i,j) pairs for X00, X01, X10, X11
        Returns list of 7 lists of (row, coeff) tuples
        """
        X00, X01, X10, X11 = x_indices
        # Each Pk is a product of two linear combos of X elements
        # For U side, we only care about which X elements with what coeffs
        P1 = [(X00, 1), (X11, 1)]
        P2 = [(X10, 1), (X11, 1)]
        P3 = [(X00, 1)]
        P4 = [(X11, 1)]
        P5 = [(X00, 1), (X01, 1)]
        P6 = [(X10, 1), (X00, -1)]
        P7 = [(X01, 1), (X11, -1)]
        return [P1, P2, P3, P4, P5, P6, P7]
    
    def strassen_2x2_v_rows(y_indices):
        """For 2x2 matrix Y with elements at given (j,k) positions, return V contribution for each of 7 Strassen mults.
        y_indices: list of 4 (j,k) pairs for Y00, Y01, Y10, Y11
        """
        Y00, Y01, Y10, Y11 = y_indices
        P1 = [(Y00, 1), (Y11, 1)]
        P2 = [(Y00, 1)]
        P3 = [(Y01, 1), (Y11, -1)]
        P4 = [(Y10, 1), (Y00, -1)]
        P5 = [(Y11, 1)]
        P6 = [(Y00, 1), (Y01, 1)]
        P7 = [(Y10, 1), (Y11, 1)]
        return [P1, P2, P3, P4, P5, P6, P7]
    
    def strassen_2x2_w_rows(z_indices):
        """For 2x2 result Z with elements at given (k,i) positions, return W contribution for each output element.
        z_indices: list of 4 (k,i) pairs for Z00, Z01, Z10, Z11
        Returns dict: output_pos -> list of (P_idx, coeff)
        """
        Z00, Z01, Z10, Z11 = z_indices
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        w_contrib = {}
        w_contrib[Z00] = [(0, 1), (3, 1), (4, -1), (6, 1)]
        w_contrib[Z01] = [(2, 1), (4, 1)]
        w_contrib[Z10] = [(1, 1), (3, 1)]
        w_contrib[Z11] = [(0, 1), (1, -1), (2, 1), (5, 1)]
        return w_contrib
    
    def add_strassen_2x2_block(x_idx, y_idx, z_idx, u, v, w):
        """Add one 2x2x2 Strassen multiplication to the global U, V, W.
        x_idx: 4 (i,j) for X
        y_idx: 4 (j,k) for Y  
        z_idx: 4 (k,i) for Z
        u, v, w: lists to append term-lists to
        """
        u_terms_block = strassen_2x2_u_rows(x_idx)
        v_terms_block = strassen_2x2_v_rows(y_idx)
        w_terms_block = strassen_2x2_w_rows(z_idx)
        
        for pk in range(7):
            u_term = [(row, coeff) for row, coeff in u_terms_block[pk]]
            v_term = [(row, coeff) for row, coeff in v_terms_block[pk]]
            # W: for each output position, add contribution from this Pk
            w_term = []
            for z_pos, contributions in w_terms_block.items():
                for p_idx, p_coeff in contributions:
                    if p_idx == pk:
                        w_term.append((z_pos, p_coeff))
            u.append(u_term)
            v.append(v_term)
            w.append(w_term)
    
    # Part 1: A (4x4) * B_L (4x4) -> C_L (4x4), k in 0..3
    # Use Strassen 4x4x4 via 7 blocks of 2x2x2 Strassen on 2x2 subblocks
    # A split into 4 2x2 blocks: A00 (i=0,1; j=0,1), A01 (i=0,1; j=2,3), A10 (i=2,3; j=0,1), A11 (i=2,3; j=2,3)
    # B_L split into 4 2x2 blocks: B00 (j=0,1; k=0,1), B01 (j=0,1; k=2,3), B10 (j=2,3; k=0,1), B11 (j=2,3; k=2,3)
    # C_L split into 4 2x2 blocks: C00 (k=0,1; i=0,1), C01 (k=0,1; i=2,3), C10 (k=2,3; i=0,1), C11 (k=2,3; i=2,3)
    
    # For 4x4x4 Strassen, we have 7 "big" multiplications, each of which is a 2x2x2 Strassen.
    # Big P1 = (A00+A11)(B00+B11)
    # Big P2 = (A10+A11)B00
    # Big P3 = A00(B01-B11)
    # Big P4 = A11(B10-B00)
    # Big P5 = (A00+A01)B11
    # Big P6 = (A10-A00)(B00+B01)
    # Big P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each "big" Pk is a 2x2 matrix product, decomposed via 2x2x2 Strassen (7 scalar mults each).
    # Total: 7 * 7 = 49 terms.
    
    # Define the 2x2 sub-block indices:
    # A blocks: A00: i in {0,1}, j in {0,1}; A01: i in {0,1}, j in {2,3}; A10: i in {2,3}, j in {0,1}; A11: i in {2,3}, j in {2,3}
    def a_block(i0, j0):
        return [(i0+di, j0+dj) for di in range(2) for dj in range(2)]
    
    def bl_block(j0, k0):
        return [(j0+dj, k0+dk) for dj in range(2) for dk in range(2)]
    
    def cl_block(k0, i0):
        return [(k0+dk, i0+di) for dk in range(2) for di in range(2)]
    
    A00 = a_block(0, 0)  # (0,0),(0,1),(1,0),(1,1)
    A01 = a_block(0, 2)  # (0,2),(0,3),(1,2),(1,3)
    A10 = a_block(2, 0)  # (2,0),(2,1),(3,0),(3,1)
    A11 = a_block(2, 2)  # (2,2),(2,3),(3,2),(3,3)
    
    B00 = bl_block(0, 0)  # (0,0),(0,1),(1,0),(1,1)
    B01 = bl_block(0, 2)  # (0,2),(0,3),(1,2),(1,3)
    B10 = bl_block(2, 0)  # (2,0),(2,1),(3,0),(3,1)
    B11 = bl_block(2, 2)  # (2,2),(2,3),(3,2),(3,3)
    
    C00 = cl_block(0, 0)  # (0,0),(0,1),(1,0),(1,1)
    C01 = cl_block(0, 2)  # (0,2),(0,3),(1,2),(1,3)
    C10 = cl_block(2, 0)  # (2,0),(2,1),(3,0),(3,1)
    C11 = cl_block(2, 2)  # (2,2),(2,3),(3,2),(3,3)
    
    # For each big Strassen Pk, we need to define the 2x2 X and Y matrices
    # and add 7 scalar-level Strassen terms.
    
    # Helper: add a 2x2x2 Strassen where X = sum of scaled A-blocks, Y = sum of scaled B-blocks, Z = scaled C-blocks
    def add_big_strassen(x_blocks, y_blocks, z_blocks, u, v, w):
        """x_blocks: dict mapping X00,X01,X10,X11 to list of (a_block, coeff)
        y_blocks: dict mapping Y00,Y01,Y10,Y11 to list of (b_block, coeff)
        z_blocks: dict mapping Z00,Z01,Z10,Z11 to list of (c_block, coeff)
        """
        # Expand to individual elements
        x_idx = {'X00': [], 'X01': [], 'X10': [], 'X11': []}
        for key in ['X00', 'X01', 'X10', 'X11']:
            for block, coeff in x_blocks.get(key, []):
                for ij in block:
                    x_idx[key].append((ij, coeff))
        
        y_idx = {'Y00': [], 'Y01': [], 'Y10': [], 'Y11': []}
        for key in ['Y00', 'Y01', 'Y10', 'Y11']:
            for block, coeff in y_blocks.get(key, []):
                for jk in block:
                    y_idx[key].append((jk, coeff))
        
        z_idx = {'Z00': [], 'Z01': [], 'Z10': [], 'Z11': []}
        for key in ['Z00', 'Z01', 'Z10', 'Z11']:
            for block, coeff in z_blocks.get(key, []):
                for ki in block:
                    z_idx[key].append((ki, coeff))
        
        # For 2x2x2 Strassen, X00 is a 1x1 "element" conceptually, but here each is a 2x2 block
        # So we treat the 2x2 Strassen at the block level: 7 block-level multiplications
        # Each block multiplication is X_block * Y_block where each is 2x2
        # We decompose each via 2x2x2 Strassen
        
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X10+X11)Y00
        # P3 = X00(Y01-Y11)
        # P4 = X11(Y10-Y00)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X01-X11)(Y10+Y11)
        
        # For each, define X_part (list of (element, coeff)) and Y_part
        # Then add 2x2x2 Strassen for that 2x2 x 2x2 product
        
        # Actually, each "block" here is 2x2, so X00 is a 2x2 matrix with specific elements.
        # The 7 Strassen terms operate on the 2x2 blocks.
        
        # Let me restructure: for each of the 7 big terms, I have a 2x2 X matrix and 2x2 Y matrix.
        # X matrix elements are 2x2 blocks. I need to flatten to individual (i,j) elements.
        
        # X00_block: 2x2 matrix with elements from x_idx['X00']
        # To do 2x2x2 Strassen on the 2x2 block product:
        # The 2x2 X has elements: top-left, top-right, bottom-left, bottom-right
        # These correspond to specific (i,j) positions.
        
        # For X00 (i in {0,1}, j in {0,1}):
        #   X00_00 = (0,0), X00_01 = (0,1), X00_10 = (1,0), X00_11 = (1,1)
        # For X01 (i in {0,1}, j in {2,3}):
        #   X01_00 = (0,2), X01_01 = (0,3), X01_10 = (1,2), X01_11 = (1,3)
        # etc.
        
        def get_block_elements(block_list):
            """Given list of (ij, coeff), organize into 2x2 block structure.
            Returns dict: '00', '01', '10', '11' -> list of ((i,j), coeff)
            """
            result = {'00': [], '01': [], '10': [], '11': []}
            for ij, coeff in block_list:
                i, j = ij
                # Determine position within 2x2 block based on parity
                di = i % 2
                dj = j % 2
                pos = f"{di}{dj}"
                result[pos].append((ij, coeff))
            return result
    
        # This is getting complex. Let me use a simpler approach:
        # Just enumerate all 7 big terms and for each, call add_strassen_2x2_block
        # with the appropriate 2x2 element indices.
        
        # For P1 = (A00+A11)(B00+B11):
        # X = A00+A11: a 2x2 "super" matrix where each element is 2x2
        # Actually no. A00 is 2x2, A11 is 2x2. Their sum is 2x2.
        # But in the 4x4 Strassen, X00 and X11 are 2x2 blocks of A.
        # (X00+X11) is a 2x2 matrix (element-wise sum of two 2x2 matrices).
        # Then (X00+X11) * (Y00+Y11) is a 2x2 * 2x2 = 2x2 multiplication.
        # This 2x2*2x2 is decomposed via 2x2x2 Strassen (7 scalar mults).
        
        # So for P1: X_2x2 has elements:
        #   X_2x2_00 = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
        #   X_2x2_01 = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
        #   X_2x2_10 = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
        #   X_2x2_11 = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
        
        # So the 2x2 X for the inner Strassen has elements at:
        # X00_inner: (0,0) with coeff 1, (2,2) with coeff 1
        # X01_inner: (0,1) with coeff 1, (2,3) with coeff 1
        # X10_inner: (1,0) with coeff 1, (3,2) with coeff 1
        # X11_inner: (1,1) with coeff 1, (3,3) with coeff 1
        
        # Similarly for Y and Z.
        
        # Let me define a helper to build the inner 2x2 Strassen inputs.
        
        def build_inner(x_blocks, y_blocks, z_blocks):
            """x_blocks: dict 'X00','X01','X10','X11' each -> list of (block, coeff)
            where block is a 2x2 list of (i,j)
            Returns x_inner: 4 lists for the 2x2 inner Strassen X matrix
            """
            x_inner = [[], [], [], []]  # X00, X01, X10, X11 of inner 2x2
            for key_idx, key in enumerate(['X00', 'X01', 'X10', 'X11']):
                for block, coeff in x_blocks.get(key, []):
                    for ij in block:
                        x_inner[key_idx].append((ij, coeff))
            return x_inner
        
        # For P1: (A00+A11)(B00+B11)
        x1 = {'X00': [(A00, 1), (A11, 1)], 'X01': [], 'X10': [], 'X11': []}
        # Wait, X00 of the inner 2x2 means the top-left element of the 2x2 result.
        # The 2x2 matrix (A00+A11) has:
        #   [0,0] = A[0,0]+A[2,2]
        #   [0,1] = A[0,1]+A[2,3]
        #   [1,0] = A[1,0]+A[3,2]
        #   [1,1] = A[1,1]+A[3,3]
        # So inner X00 (top-left) = A[0,0]+A[2,2]
        # inner X01 (top-right) = A[0,1]+A[2,3]
        # inner X10 (bottom-left) = A[1,0]+A[3,2]
        # inner X11 (bottom-right) = A[1,1]+A[3,3]
        
        # I think my abstraction is off. Let me just directly specify the 4 elements of the inner 2x2 X.
        
        def add_inner_strassen(x_elems, y_elems, z_elems, u, v, w):
            """x_elems: 4 lists, each list of ((i,j), coeff) for X00,X01,X10,X11 of 2x2 X
            y_elems: 4 lists for Y00,Y01,Y10,Y11
            z_elems: 4 lists for Z00,Z01,Z10,Z11
            """
            # Build the 2x2x2 Strassen
            # X00_inner elements
            X00_list = x_elems[0]
            X01_list = x_elems[1]
            X10_list = x_elems[2]
            X11_list = x_elems[3]
            
            # For 2x2x2 Strassen, we need the 4 "corners" of the 2x2 X
            # But each corner is a single (i,j) element (or sum with coefficients)
            # The Strassen 2x2x2 assumes X is a 2x2 matrix with specific elements.
            
            # P1_inner = (X00+X11)(Y00+Y11)
            # For U: X00 and X11 with coeff 1 each
            # But X00 might be a sum of multiple (i,j) with coefficients
            
            u_p1 = [(ij, c) for ij, c in X00_list] + [(ij, c) for ij, c in X11_list]
            u_p2 = [(ij, c) for ij, c in X10_list] + [(ij, c) for ij, c in X11_list]
            u_p3 = [(ij, c) for ij, c in X00_list]
            u_p4 = [(ij, c) for ij, c in X11_list]
            u_p5 = [(ij, c) for ij, c in X00_list] + [(ij, c) for ij, c in X01_list]
            u_p6 = [(ij, c) for ij, c in X10_list] + [(ij, -c) for ij, c in X00_list]
            u_p7 = [(ij, c) for ij, c in X01_list] + [(ij, -c) for ij, c in X11_list]
            
            Y00_list = y_elems[0]
            Y01_list = y_elems[1]
            Y10_list = y_elems[2]
            Y11_list = y_elems[3]
            
            v_p1 = [(jk, c) for jk, c in Y00_list] + [(jk, c) for jk, c in Y11_list]
            v_p2 = [(jk, c) for jk, c in Y00_list]
            v_p3 = [(jk, c) for jk, c in Y01_list] + [(jk, -c) for jk, c in Y11_list]
            v_p4 = [(jk, c) for jk, c in Y10_list] + [(jk, -c) for jk, c in Y00_list]
            v_p5 = [(jk, c) for jk, c in Y11_list]
            v_p6 = [(jk, c) for jk, c in Y00_list] + [(jk, c) for jk, c in Y01_list]
            v_p7 = [(jk, c) for jk, c in Y10_list] + [(jk, c) for jk, c in Y11_list]
            
            Z00_list = z_elems[0]
            Z01_list = z_elems[1]
            Z10_list = z_elems[2]
            Z11_list = z_elems[3]
            
            # W contributions: Z00 = P1+P4-P5+P7, Z01 = P3+P5, Z10 = P2+P4, Z11 = P1-P2+P3+P6
            w_p1 = [(ki, c) for ki, c in Z00_list] + [(ki, c) for ki, c in Z11_list]
            w_p2 = [(ki, c) for ki, c in Z10_list] + [(ki, -c) for ki, c in Z11_list]
            w_p3 = [(ki, c) for ki, c in Z01_list] + [(ki, c) for ki, c in Z11_list]
            w_p4 = [(ki, c) for ki, c in Z00_list] + [(ki, c) for ki, c in Z10_list]
            w_p5 = [(ki, -c) for ki, c in Z00_list] + [(ki, c) for ki, c in Z01_list]
            w_p6 = [(ki, c) for ki, c in Z11_list]
            w_p7 = [(ki, c) for ki, c in Z00_list]
            
            u.append(u_p1); v.append(v_p1); w.append(w_p1)
            u.append(u_p2); v.append(v_p2); w.append(w_p2)
            u.append(u_p3); v.append(v_p3); w.append(w_p3)
            u.append(u_p4); v.append(v_p4); w.append(w_p4)
            u.append(u_p5); v.append(v_p5); w.append(w_p5)
            u.append(u_p6); v.append(v_p6); w.append(w_p6)
            u.append(u_p7); v.append(v_p7); w.append(w_p7)
        
        # Now define the 7 big terms for 4x4x4 Strassen:
        
        # P1 = (A00+A11)(B00+B11) -> C00+=1, C01+=0, C10+=0, C11+=1 (with signs from C formulas)
        # Actually C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6
        # So P1 contributes to C00 with +1 and C11 with +1
        
        # For P1: X = A00+A11 (2x2), Y = B00+B11 (2x2), Z contributes to C00 and C11
        # Inner X elements:
        #   X00 = A[0,0]+A[2,2], X01 = A[0,1]+A[2,3], X10 = A[1,0]+A[3,2], X11 = A[1,1]+A[3,3]
        # Inner Y elements:
        #   Y00 = B[0,0]+B[2,0], Y01 = B[0,1]+B[2,1], Y10 = B[1,0]+B[3,0], Y11 = B[1,1]+B[3,1]
        #   Wait, B00 is (j=0,1; k=0,1): B[0,0],B[0,1],B[1,0],B[1,1]
        #   B11 is (j=2,3; k=2,3): B[2,2],B[2,3],B[3,2],B[3,3]
        #   B00+B11 element-wise: [0,0]=B[0,0]+B[2,2], [0,1]=B[0,1]+B[2,3], [1,0]=B[1,0]+B[3,2], [1,1]=B[1,1]+B[3,3]
        
        # Inner Z: C00 = (k=0,1; i=0,1), C11 = (k=2,3; i=2,3)
        # P1 contributes to both with +1
        # So Z00 (top-left of inner 2x2 Z) gets C00[0,0]=C[0,0] and C11[0,0]=C[2,2]
        # Hmm, the inner 2x2 Z is a 2x2 matrix where each element is itself 2x2.
        # I think I'm overcomplicating. 
        
        # Let me take a step back. The cleanest approach:
        # Just directly implement the 49 terms by explicit enumeration.
        
        return
    
    # OK let me just do it explicitly with numpy-like indexing.
    
    # I'll build the rank-77 decomposition using explicit term lists.
    # Term format: list of (U_rows, V_rows, W_rows) where each is list of (index, coeff)
    
    terms = []  # each term is (u_list, v_list, w_list)
    
    def make_term(u_list, v_list, w_list):
        terms.append((u_list, v_list, w_list))
    
    # Helper to create index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # ===== Part 1: 4x4x4 Strassen (49 terms) =====
    # A(i,j) * B_L(j,k) for k in 0..3
    
    # Strassen 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # Each Mk is a 2x2*2x2 product, decomposed via 2x2x2 Strassen (7 terms each)
    
    def add_strassen_2x2(x_elems, y_elems, z_out):
        """x_elems: dict '00','01','10','11' -> list of (u_index, coeff)
        y_elems: dict '00','01','10','11' -> list of (v_index, coeff)
        z_out: dict '00','01','10','11' -> list of (w_index, coeff)
        Adds 7 terms.
        """
        # P1 = (X00+X11)(Y00+Y11)
        u1 = x_elems['00'] + x_elems['11']
        v1 = y_elems['00'] + y_elems['11']
        w1 = z_out['00'] + z_out['11']  # P1 contributes to Z00 and Z11
        make_term(u1, v1, w1)
        
        # P2 = (X10+X11)Y00
        u2 = x_elems['10'] + x_elems['11']
        v2 = y_elems['00']
        w2 = z_out['10']  # P2 contributes to Z10
        make_term(u2, v2, w2)
        
        # P3 = X00(Y01-Y11)
        u3 = x_elems['00']
        v3 = y_elems['01'] + [(vi, -c) for vi, c in y_elems['11']]
        w3 = z_out['01']  # P3 contributes to Z01
        make_term(u3, v3, w3)
        
        # P4 = X11(Y10-Y00)
        u4 = x_elems['11']
        v4 = y_elems['10'] + [(vi, -c) for vi, c in y_elems['00']]
        w4 = z_out['00'] + z_out['10']  # P4 contributes to Z00 and Z10
        make_term(u4, v4, w4)
        
        # P5 = (X00+X01)Y11
        u5 = x_elems['00'] + x_elems['01']
        v5 = y_elems['11']
        w5 = [(wi, -c) for wi, c in z_out['00']] + z_out['01']  # P5: -Z00, +Z01
        make_term(u5, v5, w5)
        
        # P6 = (X10-X00)(Y00+Y01)
        u6 = x_elems['10'] + [(ui, -c) for ui, c in x_elems['00']]
        v6 = y_elems['00'] + y_elems['01']
        w6 = z_out['11']  # P6 contributes to Z11
        make_term(u6, v6, w6)
        
        # P7 = (X01-X11)(Y10+Y11)
        u7 = x_elems['01'] + [(ui, -c) for ui, c in x_elems['11']]
        v7 = y_elems['10'] + y_elems['11']
        w7 = z_out['00']  # P7 contributes to Z00
        make_term(u7, v7, w7)
    
    # Define A sub-blocks (as lists of u_indices)
    # A00: i in {0,1}, j in {0,1}
    A00_u = {'00': [(u_idx(0,0), 1)], '01': [(u_idx(0,1), 1)], '10': [(u_idx(1,0), 1)], '11': [(u_idx(1,1), 1)]}
    A01_u = {'00': [(u_idx(0,2), 1)], '01': [(u_idx(0,3), 1)], '10': [(u_idx(1,2), 1)], '11': [(u_idx(1,3), 1)]}
    A10_u = {'00': [(u_idx(2,0), 1)], '01': [(u_idx(2,1), 1)], '10': [(u_idx(3,0), 1)], '11': [(u_idx(3,1), 1)]}
    A11_u = {'00': [(u_idx(2,2), 1)], '01': [(u_idx(2,3), 1)], '10': [(u_idx(3,2), 1)], '11': [(u_idx(3,3), 1)]}
    
    # B_L sub-blocks (as lists of v_indices)
    # B00: j in {0,1}, k in {0,1}
    B00_v = {'00': [(v_idx(0,0), 1)], '01': [(v_idx(0,1), 1)], '10': [(v_idx(1,0), 1)], '11': [(v_idx(1,1), 1)]}
    B01_v = {'00': [(v_idx(0,2), 1)], '01': [(v_idx(0,3), 1)], '10': [(v_idx(1,2), 1)], '11': [(v_idx(1,3), 1)]}
    B10_v = {'00': [(v_idx(2,0), 1)], '01': [(v_idx(2,1), 1)], '10': [(v_idx(3,0), 1)], '11': [(v_idx(3,1), 1)]}
    B11_v = {'00': [(v_idx(2,2), 1)], '01': [(v_idx(2,3), 1)], '10': [(v_idx(3,2), 1)], '11': [(v_idx(3,3), 1)]}
    
    # C_L sub-blocks (as lists of w_indices)
    # C00: k in {0,1}, i in {0,1}
    C00_w = {'00': [(w_idx(0,0), 1)], '01': [(w_idx(0,1), 1)], '10': [(w_idx(1,0), 1)], '11': [(w_idx(1,1), 1)]}
    C01_w = {'00': [(w_idx(0,2), 1)], '01': [(w_idx(0,3), 1)], '10': [(w_idx(1,2), 1)], '11': [(w_idx(1,3), 1)]}
    C10_w = {'00': [(w_idx(2,0), 1)], '01': [(w_idx(2,1), 1)], '10': [(w_idx(3,0), 1)], '11': [(w_idx(3,1), 1)]}
    C11_w = {'00': [(w_idx(2,2), 1)], '01': [(w_idx(2,3), 1)], '10': [(w_idx(3,2), 1)], '11': [(w_idx(3,3), 1)]}
    
    # Now add the 7 big Strassen terms:
    
    # M1 = (A00+A11)(B00+B11) -> C00+, C11+
    X1 = {k: A00_u[k] + A11_u[k] for k in ['00','01','10','11']}
    Y1 = {k: B00_v[k] + B11_v[k] for k in ['00','01','10','11']}
    Z1 = {k: C00_w[k] for k in ['00','01','10','11']}  # will be modified
    # P1 of M1 contributes to C00 and C11, P4 to C00 and C10, P5 to -C00 and C01, P7 to C00
    # This is getting tangled. Let me just call add_strassen_2x2 with the right Z for each output.
    
    # Actually, each Mk produces a 2x2 result. The 2x2 result of Mk is added to specific C blocks.
    # For M1: result 2x2, added to C00 (top-left 2x2 of C) with +1 and C11 (bottom-right 2x2) with +1
    # So Z00 = C00, Z01 = C00's right half? No.
    
    # The 2x2 result of M1 has elements at:
    #   [0,0] -> C[0,0], [0,1] -> C[0,2], [1,0] -> C[1,0], [1,1] -> C[1,2]  (if adding to C00)
    #   AND [0,0] -> C[2,2], [0,1] -> C[2,3], [1,0] -> C[3,2], [1,1] -> C[3,3]  (if adding to C11)
    
    # So for M1, the Z 2x2 has:
    #   Z00 = C00['00'] + C11['00']  (i.e., w_idx(0,0) and w_idx(2,2))
    #   Z01 = C00['01'] + C11['01']
    #   Z10 = C00['10'] + C11['10']
    #   Z11 = C00['11'] + C11['11']
    
    Z1 = {
        '00': C00_w['00'] + C11_w['00'],
        '01': C00_w['01'] + C11_w['01'],
        '10': C00_w['10'] + C11_w['10'],
        '11': C00_w['11'] + C11_w['11']
    }
    add_strassen_2x2(X1, Y1, Z1)
    
    # M2 = (A10+A11)B00 -> C10+
    X2 = {k: A10_u[k] + A11_u[k] for k in ['00','01','10','11']}
    Y2 = B00_v
    Z2 = C10_w
    add_strassen_2x2(X2, Y2, Z2)
    
    # M3 = A00(B01-B11) -> C01+
    X3 = A00_u
    Y3 = {k: B01_v[k] + [(vi, -c) for vi, c in B11_v[k]] for k in ['00','01','10','11']}
    Z3 = C01_w
    add_strassen_2x2(X3, Y3, Z3)
    
    # M4 = A11(B10-B00) -> C00+, C10+
    X4 = A11_u
    Y4 = {k: B10_v[k] + [(vi, -c) for vi, c in B00_v[k]] for k in ['00','01','10','11']}
    Z4 = {
        '00': C00_w['00'],
        '01': C00_w['01'],
        '10': C10_w['00'] + C10_w['10'],
        '11': C10_w['11']
    }
    # Wait, M4's 2x2 result: [0,0]->C00[0,0], [0,1]->C00[0,1], [1,0]->C10[0,0], [1,1]->C10[1,1]?
    # No. M4 result 2x2: top-left goes to C00 top-left area, but C00 is 2x2.
    # M4 result[0,0] -> C[0,0], M4 result[0,1] -> C[0,1], M4 result[1,0] -> C[1,0], M4 result[1,1] -> C[1,1]
    # AND M4 result also contributes to C10: C10[0,0]=C[2,0], C10[0,1]=C[2,1], C10[1,0]=C[3,0], C10[1,1]=C[3,1]
    # Hmm, I think M4's 2x2 result maps to C00 (2x2) AND C10 (2x2) is not right.
    # C10 = M2 + M4. M2 is 2x2, M4 is 2x2. So M4's 2x2 adds to C10's 2x2.
    # And M4's 2x2 ALSO adds to C00's 2x2? No: C00 = M1+M4-M5+M7. So M4 adds to C00.
    # So M4's 2x2 result adds to both C00 (2x2) and C10 (2x2).
    # That means the 2x2 result's elements go to:
    #   [0,0] -> C00[0,0] AND C10[0,0]
    #   [0,1] -> C00[0,1] AND C10[0,1]
    #   [1,0] -> C00[1,0] AND C10[1,0]
    #   [1,1] -> C00[1,1] AND C10[1,1]
    
    Z4 = {
        '00': C00_w['00'] + C10_w['00'],
        '01': C00_w['01'] + C10_w['01'],
        '10': C00_w['10'] + C10_w['10'],
        '11': C00_w['11'] + C10_w['11']
    }
    add_strassen_2x2(X4, Y4, Z4)
    
    # M5 = (A00+A01)B11 -> -C00, +C01
    X5 = {k: A00_u[k] + A01_u[k] for k in ['00','01','10','11']}
    Y5 = B11_v
    Z5 = {
        '00': [(wi, -c) for wi, c in C00_w['00']],
        '01': [(wi, -c) for wi, c in C00_w['01']] + C01_w['00'],
        '10': [(wi, -c) for wi, c in C00_w['10']] + C01_w['10'],
        '11': [(wi, -c) for wi, c in C00_w['11']] + C01_w['11']
    }
    # Wait, C01 = M3 + M5. M5's 2x2 adds to C01.
    # C01[0,0]=C[0,2], C01[0,1]=C[0,3], C01[1,0]=C[1,2], C01[1,1]=C[1,3]
    # M5 result[0,0]->C01[0,0], M5 result[0,1]->C01[0,1], M5 result[1,0]->C01[1,0], M5 result[1,1]->C01[1,1]
    # And -C00: M5 result[0,0]->-C00[0,0], etc.
    
    Z5 = {
        '00': [(wi, -c) for wi, c in C00_w['00']] + C01_w['00'],
        '01': [(wi, -c) for wi, c in C00_w['01']] + C01_w['01'],
        '10': [(wi, -c) for wi, c in C00_w['10']] + C01_w['10'],
        '11': [(wi, -c) for wi, c in C00_w['11']] + C01_w['11']
    }
    add_strassen_2x2(X5, Y5, Z5)
    
    # M6 = (A10-A00)(B00+B01) -> C11+
    X6 = {k: A10_u[k] + [(ui, -c) for ui, c in A00_u[k]] for k in ['00','01','10','11']}
    Y6 = {k: B00_v[k] + B01_v[k] for k in ['00','01','10','11']}
    Z6 = C11_w
    add_strassen_2x2(X6, Y6, Z6)
    
    # M7 = (A01-A11)(B10+B11) -> C00+
    X7 = {k: A01_u[k] + [(ui, -c) for ui, c in A11_u[k]] for k in ['00','01','10','11']}
    Y7 = {k: B10_v[k] + B11_v[k] for k in ['00','01','10','11']}
    Z7 = C00_w
    add_strassen_2x2(X7, Y7, Z7)
    
    # ===== Part 2: 4x4x2 (28 terms) =====
    # A(4x4) * B_R(4x2) where B_R has k in {4,5}
    # Split into 4 2x2x2 Strassen multiplications:
    # Block 1: A[i=0,1; j=0,1] * B_R[j=0,1; k=4,5]
    # Block 2: A[i=0,1; j=2,3] * B_R[j=2,3; k=4,5]
    # Block 3: A[i=2,3; j=0,1] * B_R[j=0,1; k=4,5]
    # Block 4: A[i=2,3; j=2,3] * B_R[j=2,3; k=4,5]
    
    def add_strassen_4x4x2_block(a_i0, a_j0, b_j0, u_list, v_list, w_list):
        """Add one 2x2x2 Strassen for A[i=i0..i0+1, j=j0..j0+1] * B_R[j=j0..j0+1, k=4..5]"""
        # X 2x2: A elements
        x = {
            '00': [(u_idx(a_i0, a_j0), 1)],
            '01': [(u_idx(a_i0, a_j0+1), 1)],
            '10': [(u_idx(a_i0+1, a_j0), 1)],
            '11': [(u_idx(a_i0+1, a_j0+1), 1)]
        }
        # Y 2x2: B_R elements (j, k=4 or 5)
        y = {
            '00': [(v_idx(b_j0, 4), 1)],
            '01': [(v_idx(b_j0, 5), 1)],
            '10': [(v_idx(b_j0+1, 4), 1)],
            '11': [(v_idx(b_j0+1, 5), 1)]
        }
        # Z 2x2: C elements (k=4 or 5, i=a_i0 or a_i0+1)
        z = {
            '00': [(w_idx(4, a_i0), 1)],
            '01': [(w_idx(5, a_i0), 1)],
            '10': [(w_idx(4, a_i0+1), 1)],
            '11': [(w_idx(5, a_i0+1), 1)]
        }
        add_strassen_2x2(x, y, z)
    
    add_strassen_4x4x2_block(0, 0, 0, None, None, None)  # A[0:2,0:2] * B_R[0:2,4:6]
    add_strassen_4x4x2_block(0, 2, 2, None, None, None)  # A[0:2,2:4] * B_R[2:4,4:6]
    add_strassen_4x4x2_block(2, 0, 0, None, None, None)  # A[2:4,0:2] * B_R[0:2,4:6]
    add_strassen_4x4x2_block(2, 2, 2, None, None, None)  # A[2:4,2:4] * B_R[2:4,4:6]
    
    # Now assemble U, V, W from terms
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_list, v_list, w_list) in enumerate(terms):
        for idx, coeff in u_list:
            U[idx][col] += coeff
        for idx, coeff in v_list:
            V[idx][col] += coeff
        for idx, coeff in w_list:
            W[idx][col] += coeff
    
    # Convert to lists (should already be)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
