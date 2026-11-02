"""Rewritten program for matrix multiplication 4x4 * 4x8."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x8 matrix.
    
    Uses Strassen's algorithm: 4x4 * 4x4 has rank 49 (7x7 Strassen).
    Split 4x8 into two 4x4 matrices, giving total rank 98.
    
    Index conventions:
    - A is n x m = 4 x 4, indexed A[i,j] with i in [0,3], j in [0,3]
    - B is m x p = 4 x 8, indexed B[j,k] with j in [0,3], k in [0,7]
    - C is n x p = 4 x 8, indexed C[i,k] with i in [0,3], k in [0,7]
    - U has shape (n*m, rank) = (16, rank), U[i*m+j, r]
    - V has shape (m*p, rank) = (32, rank), V[j*p+k, r]
    - W has shape (p*n, rank) = (32, rank), W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen decomposition for 2x2 * 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = d(e+f)
    # M3 = (a+b)h
    # M4 = a(g-h)
    # M5 = (c+d)e
    # M6 = (a-c)(f+h)
    # M7 = (b-d)(g+e)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M6
    # C21 = M2 + M5
    # C22 = M1 - M3 + M2 + M6
    
    # For 2x2 * 2x2, the U, V, W vectors for each of the 7 terms:
    # U index: a=0, b=1, c=2, d=3
    # V index: e=0, f=1, g=2, h=3
    # W index: C11=0, C12=1, C21=2, C22=3
    
    strassen_2x2 = [
        # (U_vec, V_vec, W_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]),   # M2
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),  # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [1, 0, 0, 0]),  # M4
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 0, 1, 0]),  # M5
        ([1, 0, -1, 0], [0, 1, 0, 1], [0, 1, 0, 1]),  # M6
        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0]),  # M7
    ]
    
    # Now build the 4x4 * 4x4 decomposition using Strassen.
    # Split 4x4 into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    #
    # Similarly for B and C.
    #
    # Top-level Strassen for 4x4 * 4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = A22(B11 + B12)
    # M3 = (A11 + A12)B22
    # M4 = A11(B21 - B22)
    # M5 = (A21 + A22)B11
    # M6 = (A11 - A21)(B12 + B22)
    # M7 = (A12 - A22)(B21 + B11)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M6
    # C21 = M2 + M5
    # C22 = M1 - M3 + M2 + M6
    
    # Each Mi is a 2x2 * 2x2 multiplication, which we decompose using the 7 Strassen terms above.
    # So total rank = 7 * 7 = 49 for 4x4 * 4x4.
    
    # For 4x4 * 4x8, split B into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left, C_right = A * B_right
    # Total rank = 49 + 49 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    def add_decomposition(offset, U_mat, V_mat, W_mat):
        """
        Add a 4x4 * 4x4 Strassen decomposition starting at column offset.
        U_mat: 16 x 49 (for 4x4 A)
        V_mat: 16 x 49 (for 4x4 B)
        W_mat: 16 x 49 (for 4x4 C)
        """
        for col in range(49):
            U[:, offset + col] += U_mat[:, col]
            V[:, offset + col] += V_mat[:, col]
            W[:, offset + col] += W_mat[:, col]
    
    # Build one 4x4 * 4x4 Strassen decomposition
    def build_4x4_strassen():
        """
        Returns U_mat (16x49), V_mat (16x49), W_mat (16x49) for 4x4 * 4x4.
        """
        U_mat = [[0] * 49 for _ in range(16)]
        V_mat = [[0] * 49 for _ in range(16)]
        W_mat = [[0] * 49 for _ in range(16)]
        
        # Top-level Strassen terms for 4x4 * 4x4
        # Each term is a 2x2 * 2x2 multiplication
        # We need to express the 2x2 blocks in terms of the 4x4 entries
        
        # A block indices in 4x4:
        # A11: rows 0,1 cols 0,1 -> U indices 0,1,4,5
        # A12: rows 0,1 cols 2,3 -> U indices 2,3,6,7
        # A21: rows 2,3 cols 0,1 -> U indices 8,9,12,13
        # A22: rows 2,3 cols 2,3 -> U indices 10,11,14,15
        
        # B block indices in 4x4:
        # B11: rows 0,1 cols 0,1 -> V indices 0,1,4,5
        # B12: rows 0,1 cols 2,3 -> V indices 2,3,6,7
        # B21: rows 2,3 cols 0,1 -> V indices 8,9,12,13
        # B22: rows 2,3 cols 2,3 -> V indices 10,11,14,15
        
        # C block indices in 4x4:
        # C11: rows 0,1 cols 0,1 -> W indices 0,1,4,5
        # C12: rows 0,1 cols 2,3 -> W indices 2,3,6,7
        # C21: rows 2,3 cols 0,1 -> W indices 8,9,12,13
        # C22: rows 2,3 cols 2,3 -> W indices 10,11,14,15
        
        # For each top-level Strassen term, we have a 2x2 * 2x2 multiplication
        # We decompose each using the 7 Strassen terms for 2x2
        
        # Top-level term definitions:
        # Each term has:
        # - A block combination: (block1, coeff1, block2, coeff2) or just (block, coeff)
        # - B block combination: similar
        # - C block contributions: list of (block, coeff)
        
        # Let me define the 7 top-level terms more carefully.
        # For each term, I need to specify:
        # - Which 2x2 blocks of A are combined (with coefficients)
        # - Which 2x2 blocks of B are combined (with coefficients)
        # - Which 2x2 blocks of C get contributions (with coefficients)
        
        # Term 1: M1 = (A11 + A22)(B11 + B22)
        #   A: A11 + A22, B: B11 + B22
        #   C: C11 += M1, C22 += M1
        # Term 2: M2 = A22(B11 + B12)
        #   A: A22, B: B11 + B12
        #   C: C21 += M2, C22 += M2
        # Term 3: M3 = (A11 + A12)B22
        #   A: A11 + A12, B: B22
        #   C: C12 += M3, C22 -= M3
        # Term 4: M4 = A11(B21 - B22)
        #   A: A11, B: B21 - B22
        #   C: C11 += M4
        # Term 5: M5 = (A21 + A22)B11
        #   A: A21 + A22, B: B11
        #   C: C11 -= M5, C21 += M5
        # Term 6: M6 = (A11 - A21)(B12 + B22)
        #   A: A11 - A21, B: B12 + B22
        #   C: C12 += M6, C22 += M6
        # Term 7: M7 = (A12 - A22)(B21 + B11)
        #   A: A12 - A22, B: B21 + B11
        #   C: C11 += M7
        
        # For each term, I'll build the 2x2 * 2x2 decomposition.
        # The 2x2 A matrix for the term is a linear combination of A blocks.
        # The 2x2 B matrix for the term is a linear combination of B blocks.
        
        # Block index mapping:
        # A blocks: 0=A11, 1=A12, 2=A21, 3=A22
        # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
        # C blocks: 0=C11, 1=C12, 2=C21, 3=C22
        
        # U indices for each A block (2x2 block has 4 entries):
        # A11: [0, 1, 4, 5]
        # A12: [2, 3, 6, 7]
        # A21: [8, 9, 12, 13]
        # A22: [10, 11, 14, 15]
        
        # V indices for each B block:
        # B11: [0, 1, 4, 5]
        # B12: [2, 3, 6, 7]
        # B21: [8, 9, 12, 13]
        # B22: [10, 11, 14, 15]
        
        # W indices for each C block:
        # C11: [0, 1, 4, 5]
        # C12: [2, 3, 6, 7]
        # C21: [8, 9, 12, 13]
        # C22: [10, 11, 14, 15]
        
        a_block_indices = [
            [0, 1, 4, 5],   # A11
            [2, 3, 6, 7],   # A12
            [8, 9, 12, 13], # A21
            [10, 11, 14, 15] # A22
        ]
        b_block_indices = [
            [0, 1, 4, 5],   # B11
            [2, 3, 6, 7],   # B12
            [8, 9, 12, 13], # B21
            [10, 11, 14, 15] # B22
        ]
        c_block_indices = [
            [0, 1, 4, 5],   # C11
            [2, 3, 6, 7],   # C12
            [8, 9, 12, 13], # C21
            [10, 11, 14, 15] # C22
        ]
        
        # Define the 7 top-level terms
        # Each term: (A_blocks, B_blocks, C_contributions)
        # A_blocks: list of (block_idx, coeff)
        # B_blocks: list of (block_idx, coeff)
        # C_contributions: list of (block_idx, coeff)
        
        top_level_terms = [
            # Term 1: (A11 + A22)(B11 + B22) -> C11 += 1, C22 += 1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # Term 2: A22(B11 + B12) -> C21 += 1, C22 += 1
            ([(3, 1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]),
            # Term 3: (A11 + A12)B22 -> C12 += 1, C22 -= 1
            ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),
            # Term 4: A11(B21 - B22) -> C11 += 1
            ([(0, 1)], [(2, 1), (3, -1)], [(0, 1)]),
            # Term 5: (A21 + A22)B11 -> C11 -= 1, C21 += 1
            ([(2, 1), (3, 1)], [(0, 1)], [(0, -1), (2, 1)]),
            # Term 6: (A11 - A21)(B12 + B22) -> C12 += 1, C22 += 1
            ([(0, 1), (2, -1)], [(1, 1), (3, 1)], [(1, 1), (3, 1)]),
            # Term 7: (A12 - A22)(B21 + B11) -> C11 += 1
            ([(1, 1), (3, -1)], [(2, 1), (0, 1)], [(0, 1)]),
        ]
        
        for term_idx, (a_blocks, b_blocks, c_contribs) in enumerate(top_level_terms):
            # For this term, build the 2x2 * 2x2 Strassen decomposition
            # The effective 2x2 A matrix has entries that are linear combinations of A block entries
            # The effective 2x2 B matrix has entries that are linear combinations of B block entries
            
            # For each of the 7 Strassen sub-terms for 2x2 * 2x2:
            for sub_idx in range(7):
                col = term_idx * 7 + sub_idx
                u_sub, v_sub, w_sub = strassen_2x2[sub_idx]
                
                # u_sub[i] is the coefficient for the i-th entry of the 2x2 A matrix
                # v_sub[j] is the coefficient for the j-th entry of the 2x2 B matrix
                # w_sub[k] is the coefficient for the k-th entry of the 2x2 C matrix
                
                # For each entry in the 2x2 A matrix (i in 0..3):
                # u_sub[i] multiplies the i-th entry of the effective A
                # The i-th entry of the effective A is a linear combination of A block entries
                for i in range(4):
                    if u_sub[i] != 0:
                        # Add u_sub[i] * (linear combo of A blocks) to U
                        for block_idx, coeff in a_blocks:
                            for local_idx in range(4):
                                U_idx = a_block_indices[block_idx][local_idx]
                                U[U_idx][col] += u_sub[i] * coeff
                
                # For each entry in the 2x2 B matrix (j in 0..3):
                for j in range(4):
                    if v_sub[j] != 0:
                        for block_idx, coeff in b_blocks:
                            for local_idx in range(4):
                                V_idx = b_block_indices[block_idx][local_idx]
                                V[V_idx][col] += v_sub[j] * coeff
                
                # For each entry in the 2x2 C matrix (k in 0..3):
                for k in range(4):
                    if w_sub[k] != 0:
                        for block_idx, coeff in c_contribs:
                            for local_idx in range(4):
                                W_idx = c_block_indices[block_idx][local_idx]
                                W[W_idx][col] += w_sub[k] * coeff
        
        return U_mat, V_mat, W_mat
    
    # Build the left half (B columns 0-3)
    U_left, V_left, W_left = build_4x4_strassen()
    
    # For the right half (B columns 4-7), we need to shift V indices by 4
    # (since p=8, so B[j,k] for k in 4..7 has V index j*8 + k = j*8 + 4 + (k-4))
    # And W indices: C[i,k] for k in 4..7 has W index k*4 + i = (k-4+4)*4 + i = 4*4 + i + (k-4)*4
    # Wait, W index is k*n + i = k*4 + i
    # For left half: k in 0..3, W index = k*4 + i
    # For right half: k in 4..7, W index = k*4 + i = (k-4+4)*4 + i
    
    # Actually, let me just build two separate decompositions.
    # Left: A * B[0:4, 0:4] -> C[0:4, 0:4]
    # Right: A * B[0:4, 4:8] -> C[0:4, 4:8]
    
    # For the right half, V indices are shifted by 4 (since p=8, B[j,4+k] has index j*8 + 4 + k)
    # And W indices: C[i,4+k] has index (4+k)*4 + i = 16 + k*4 + i
    
    # Let me rebuild more carefully.
    
    # Clear and rebuild
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def build_4x4_strassen_for_offset(v_offset, w_offset):
        """
        Build 4x4 * 4x4 Strassen decomposition with V offset and W offset.
        v_offset: added to all V indices (for B column shift)
        w_offset: added to all W indices (for C column shift)
        """
        U_mat = [[0] * 49 for _ in range(16)]
        V_mat = [[0] * 49 for _ in range(16)]
        W_mat = [[0] * 49 for _ in range(16)]
        
        a_block_indices = [
            [0, 1, 4, 5],   # A11
            [2, 3, 6, 7],   # A12
            [8, 9, 12, 13], # A21
            [10, 11, 14, 15] # A22
        ]
        b_block_indices = [
            [0, 1, 4, 5],   # B11
            [2, 3, 6, 7],   # B12
            [8, 9, 12, 13], # B21
            [10, 11, 14, 15] # B22
        ]
        c_block_indices = [
            [0, 1, 4, 5],   # C11
            [2, 3, 6, 7],   # C12
            [8, 9, 12, 13], # C21
            [10, 11, 14, 15] # C22
        ]
        
        top_level_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(3, 1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),
            ([(0, 1)], [(2, 1), (3, -1)], [(0, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(0, -1), (2, 1)]),
            ([(0, 1), (2, -1)], [(1, 1), (3, 1)], [(1, 1), (3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (0, 1)], [(0, 1)]),
        ]
        
        for term_idx, (a_blocks, b_blocks, c_contribs) in enumerate(top_level_terms):
            for sub_idx in range(7):
                col = term_idx * 7 + sub_idx
                u_sub, v_sub, w_sub = strassen_2x2[sub_idx]
                
                for i in range(4):
                    if u_sub[i] != 0:
                        for block_idx, coeff in a_blocks:
                            for local_idx in range(4):
                                U_idx = a_block_indices[block_idx][local_idx]
                                U_mat[U_idx][col] += u_sub[i] * coeff
                
                for j in range(4):
                    if v_sub[j] != 0:
                        for block_idx, coeff in b_blocks:
                            for local_idx in range(4):
                                V_idx = b_block_indices[block_idx][local_idx] + v_offset
                                V_mat[V_idx][col] += v_sub[j] * coeff
                
                for k in range(4):
                    if w_sub[k] != 0:
                        for block_idx, coeff in c_contribs:
                            for local_idx in range(4):
                                W_idx = c_block_indices[block_idx][local_idx] + w_offset
                                W_mat[W_idx][col] += w_sub[k] * coeff
        
        return U_mat, V_mat, W_mat
    
    # Left half: B columns 0-3, C columns 0-3
    U_l, V_l, W_l = build_4x4_strassen_for_offset(0, 0)
    
    # Right half: B columns 4-7, C columns 4-7
    # V offset: B[j, 4+k] has index j*8 + 4 + k. The local B index is j*4 + k (for 4x4 B).
    # So V index = j*8 + 4 + k = (j*4 + k) + 4 + j*4... no.
    # Actually, for 4x4 B (columns 0-3), V index = j*4 + k.
    # For 4x4 B (columns 4-7), V index = j*8 + (4+k) = j*8 + 4 + k.
    # The difference is j*8 + 4 + k vs j*4 + k = j*4 + 4.
    # So the offset is not constant! It depends on j.
    
    # Let me handle this differently. I'll build the decomposition directly with the correct indices.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def build_full_decomposition():
        """Build the full 4x4 * 4x8 decomposition."""
        
        a_block_indices = [
            [0, 1, 4, 5],   # A11
            [2, 3, 6, 7],   # A12
            [8, 9, 12, 13], # A21
            [10, 11, 14, 15] # A22
        ]
        
        # For B, we have two 4x4 halves:
        # Left: B[j, k] for k in 0..3, V index = j*8 + k
        # Right: B[j, k] for k in 4..7, V index = j*8 + k
        
        # For a 4x4 B block (rows j1,j2, cols k1,k2):
        # Left half: V indices = [j1*8+k1, j1*8+k2, j2*8+k1, j2*8+k2]
        # Right half: V indices = [j1*8+k1+4, j1*8+k2+4, j2*8+k1+4, j2*8+k2+4]
        
        # B block indices for left half (cols 0-3):
        b_block_indices_left = [
            [0*8+0, 0*8+1, 1*8+0, 1*8+1],   # B11: rows 0,1 cols 0,1
            [0*8+2, 0*8+3, 1*8+2, 1*8+3],   # B12: rows 0,1 cols 2,3
            [2*8+0, 2*8+1, 3*8+0, 3*8+1],   # B21: rows 2,3 cols 0,1
            [2*8+2, 2*8+3, 3*8+2, 3*8+3],   # B22: rows 2,3 cols 2,3
        ]
        
        # B block indices for right half (cols 4-7):
        b_block_indices_right = [
            [0*8+4, 0*8+5, 1*8+4, 1*8+5],   # B11: rows 0,1 cols 4,5
            [0*8+6, 0*8+7, 1*8+6, 1*8+7],   # B12: rows 0,1 cols 6,7
            [2*8+4, 2*8+5, 3*8+4, 3*8+5],   # B21: rows 2,3 cols 4,5
            [2*8+6, 2*8+7, 3*8+6, 3*8+7],   # B22: rows 2,3 cols 6,7
        ]
        
        # C block indices:
        # C[i, k] has W index = k*4 + i
        # Left half (cols 0-3):
        c_block_indices_left = [
            [0*4+0, 0*4+1, 1*4+0, 1*4+1],   # C11: rows 0,1 cols 0,1
            [0*4+2, 0*4+3, 1*4+2, 1*4+3],   # C12: rows 0,1 cols 2,3
            [2*4+0, 2*4+1, 3*4+0, 3*4+1],   # C21: rows 2,3 cols 0,1
            [2*4+2, 2*4+3, 3*4+2, 3*4+3],   # C22: rows 2,3 cols 2,3
        ]
        
        # Right half (cols 4-7):
        c_block_indices_right = [
            [4*4+0, 4*4+1, 5*4+0, 5*4+1],   # C11: rows 0,1 cols 4,5
            [4*4+2, 4*4+3, 5*4+2, 5*4+3],   # C12: rows 0,1 cols 6,7
            [6*4+0, 6*4+1, 7*4+0, 7*4+1],   # C21: rows 2,3 cols 4,5
            [6*4+2, 6*4+3, 7*4+2, 7*4+3],   # C22: rows 2,3 cols 6,7
        ]
        
        top_level_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(3, 1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),
            ([(0, 1)], [(2, 1), (3, -1)], [(0, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(0, -1), (2, 1)]),
            ([(0, 1), (2, -1)], [(1, 1), (3, 1)], [(1, 1), (3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (0, 1)], [(0, 1)]),
        ]
        
        def add_half(offset, b_block_indices, c_block_indices):
            for term_idx, (a_blocks, b_blocks, c_contribs) in enumerate(top_level_terms):
                for sub_idx in range(7):
                    col = offset + term_idx * 7 + sub_idx
                    u_sub, v_sub, w_sub = strassen_2x2[sub_idx]
                    
                    for i in range(4):
                        if u_sub[i] != 0:
                            for block_idx, coeff in a_blocks:
                                for local_idx in range(4):
                                    U_idx = a_block_indices[block_idx][local_idx]
                                    U[U_idx][col] += u_sub[i] * coeff
                    
                    for j in range(4):
                        if v_sub[j] != 0:
                            for block_idx, coeff in b_blocks:
                                for local_idx in range(4):
                                    V_idx = b_block_indices[block_idx][local_idx]
                                    V[V_idx][col] += v_sub[j] * coeff
                    
                    for k in range(4):
                        if w_sub[k] != 0:
                            for block_idx, coeff in c_contribs:
                                for local_idx in range(4):
                                    W_idx = c_block_indices[block_idx][local_idx]
                                    W[W_idx][col] += w_sub[k] * coeff
        
        # Add left half (cols 0-3)
        add_half(0, b_block_indices_left, c_block_indices_left)
        
        # Add right half (cols 4-7)
        add_half(49, b_block_indices_right, c_block_indices_right)
    
    build_full_decomposition()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
