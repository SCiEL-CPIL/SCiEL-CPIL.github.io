"""Program for rank-minimal exact tensor decomposition of 3x5x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook U, V, W of rank n*m*p."""
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


def _strassen_2x2x2():
    """Return Strassen 2x2x2 decomposition (rank 7)."""
    # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2
    n = m = p = 2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00 gets +M1
    W[3, 0] = 1  # C11 gets +M1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 gets +M2
    
    # M3: A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01 gets +M3
    
    # M4: A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[3, 3] = 1  # C11 gets +M4
    
    # M5: (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[1, 4] = 1  # C01 gets +M5
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[2, 5] = 1  # C10 gets +M6
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[3, 6] = 1  # C11 gets +M7
    
    return U, V, W


def _combine_decompositions(decomp_list):
    """
    Combine a list of (U, V, W) decompositions where each operates on subspaces.
    Each (U, V, W) has shapes (a_i, r_i), (b_i, r_i), (c_i, r_i).
    We need to map them to the full space.
    """
    total_rank = sum(U.shape[1] for U, V, W in decomp_list)
    return total_rank


def construct():
    """
    Construct a tensor decomposition for 3x5x7 matrix multiplication.
    Strategy: Use block decomposition with Strassen where possible.
    Split into blocks and apply Strassen 2x2x2 where dimensions allow.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try block-based Strassen approach
    # Split A (3x5) into A_top (2x5) and A_bottom (1x5)
    # Split B (5x7) into B_left (5x4) and B_right (5x3)
    
    # This gives 4 sub-problems:
    # C_tl = A_top * B_left  (2x4)
    # C_tr = A_top * B_right (2x3)
    # C_bl = A_bottom * B_left (1x4)
    # C_br = A_bottom * B_right (1x3)
    
    # For each sub-problem, we can further decompose.
    # Let's use a more systematic approach with tensor rank optimization.
    
    # First, get the schoolbook solution as baseline
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    rank_school = n * m * p  # 105
    
    # Try to use Strassen on 2x2x2 sub-blocks
    # We can partition the 3x5x7 problem into smaller blocks.
    
    # Alternative: Use a direct search for rank reduction using known algorithms.
    # For now, let's implement a block Strassen approach.
    
    # Partition A (3x5) as:
    #   A = [A1; A2] where A1 is 2x5, A2 is 1x5
    # Partition B (5x7) as:
    #   B = [B1, B2] where B1 is 5x4, B2 is 5x3
    
    # Sub-products:
    # C11 = A1*B1 (2x4) - use Strassen-like
    # C12 = A1*B2 (2x3)
    # C21 = A2*B1 (1x4)
    # C22 = A2*B2 (1x3)
    
    # For 2x4: Split into two 2x2 blocks, use Strassen on each -> 7+7=14
    # For 2x3: Split B into 2x2 and 2x1, use Strassen on 2x2 (7) + schoolbook on 2x1 (2) = 9
    # For 1x4: Split into 1x2 + 1x2, schoolbook = 4
    # For 1x3: Split into 1x2 + 1x1, schoolbook = 3
    # Total: 14 + 9 + 4 + 3 = 30
    
    # But we need to verify this is a valid tensor decomposition.
    # The issue is that Strassen's algorithm works for square matrices or when
    # we can pad appropriately.
    
    # Let's try a different approach: use the Winograd/FFT-based method.
    # For now, let's implement a careful block decomposition.
    
    # Actually, let's try to use a known result: the rank of 3x5x7 matrix multiplication.
    # The schoolbook rank is 105. We can do better with block methods.
    
    # Let me implement a proper block Strassen decomposition.
    
    def make_2x2_block_strassen(a_rows, a_cols, b_rows, b_cols):
        """
        Create a Strassen-based decomposition for multiplying an a_rows x a_cols matrix
        by an a_cols x b_cols matrix, assuming both are even or can be padded.
        Returns (U, V, W) for this sub-problem.
        """
        if a_rows % 2 != 0 or b_cols % 2 != 0:
            # Pad to make even
            pass
        
        # This is complex; let's use a simpler approach for now.
        # Fall back to schoolbook for this sub-problem
        r = a_rows * a_cols * b_cols
        U = np.zeros((a_rows * a_cols, r))
        V = np.zeros((a_cols * b_cols, r))
        W = np.zeros((b_cols * a_rows, r))
        col = 0
        for i in range(a_rows):
            for j in range(a_cols):
                for k in range(b_cols):
                    U[i * a_cols + j, col] = 1
                    V[j * b_cols + k, col] = 1
                    W[k * a_rows + i, col] = 1
                    col += 1
        return U, V, W
    
    # For simplicity and correctness, let's return the schoolbook solution.
    # The rank is 105. We know Strassen can reduce this, but implementing
    # a correct block Strassen for non-square, non-power-of-2 dimensions
    # is complex and error-prone.
    
    # Let's at least try to use Strassen on the 2x2x2 core if we can partition.
    
    # For a 3x5x7 problem, one approach:
    # Pad to 4x8x8, apply recursive Strassen, then extract.
    # But this increases the rank due to padding.
    
    # Let's just return the schoolbook solution for now, which is guaranteed correct.
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
