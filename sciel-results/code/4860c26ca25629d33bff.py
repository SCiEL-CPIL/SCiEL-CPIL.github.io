import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n x m x p matrix multiplication tensor."""
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_comp, T_target, atol=tol)

def _build_strassen_222():
    """
    Build the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) with half-integer entries.
    
    Strassen's 7 products for 2x2:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    C11 = M1 - M2 + M4 + M5
    C12 = M3 + M5 - M6 - M7
    C21 = M2 + M4
    C22 = M1 - M3 + M7
    
    where:
    M1 = (a11 + a22)(b11 + b22)
    M2 = (a21 + a22)b11
    M3 = a11(b12 - b22)
    M4 = a22(b21 - b11)
    M5 = (a11 + a12)b22
    M6 = (a21 - a11)(b11 + b12)
    M7 = (a12 - a22)(b21 + b22)
    """
    # A indices: 0=a11, 1=a12, 2=a21, 3=a22
    # B indices: 0=b11, 1=b12, 2=b21, 3=b22
    # C indices: 0=c11, 1=c12, 2=c21, 3=c22
    
    # M1: (a11+a22)(b11+b22)
    # M2: (a21+a22)b11
    # M3: a11(b12-b22)
    # M4: a22(b21-b11)
    # M5: (a11+a12)b22
    # M6: (a21-a11)(b11+b12)
    # M7: (a12-a22)(b21+b22)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1  # a11, a22
    V[0, 0] = 1; V[3, 0] = 1  # b11, b22
    # C11 += M1, C22 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1  # a21, a22
    V[0, 1] = 1  # b11
    # C11 -= M2, C21 += M2
    W[0, 1] = -1; W[2, 1] = 1
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1  # a11
    V[1, 2] = 1; V[3, 2] = -1  # b12, -b22
    # C12 += M3, C22 -= M3
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1  # a22
    V[2, 3] = 1; V[0, 3] = -1  # b21, -b11
    # C11 += M4, C21 += M4
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1  # a11, a12
    V[3, 4] = 1  # b22
    # C11 += M5, C12 += M5
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1  # a21, -a11
    V[0, 5] = 1; V[1, 5] = 1  # b11, b12
    # C12 -= M6
    W[1, 5] = -1
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1  # a12, -a22
    V[2, 6] = 1; V[3, 6] = 1  # b21, b22
    # C12 -= M7, C22 += M7
    W[1, 6] = -1; W[3, 6] = 1
    
    return U, V, W

def _build_trivial_221():
    """
    Build the trivial (schoolbook) decomposition for 2x2x1 matrix multiplication.
    This computes a 2x2 matrix times a 2x1 vector.
    A = [[a11, a12], [a21, a22]], b = [b1, b2]
    c1 = a11*b1 + a12*b2
    c2 = a21*b1 + a22*b2
    
    Returns U, V, W each of shape (4, 4).
    A indices: 0=a11, 1=a12, 2=a21, 3=a22
    B indices: 0=b1, 1=b2
    C indices: 0=c1, 1=c2
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # c1 = a11*b1
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # c1 += a12*b2
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    # c2 = a21*b1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # c2 += a22*b2
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _build_2x4x7_rank50():
    """
    Build the rank-50 decomposition for 2x4x7 matrix multiplication.
    
    Strategy: Split the 4x7 multiplication into blocks.
    A is 2x4, B is 4x7.
    
    We can view this as:
    - The 4x4 part (columns 0-3 of B) can use a 2x2x2 Strassen-like structure
      if we split A into two 2x2 blocks and B's first 4 columns into two 2x2 blocks.
    - The remaining 4x3 part (columns 4-6 of B) uses schoolbook.
    
    More precisely, let's split:
    A = [A_left (2x2), A_right (2x2)] where A_left uses columns 0,1 and A_right uses columns 2,3
    B = [B_left (4x4), B_right (4x3)]
    
    Actually, let me think more carefully.
    
    For 2x4x7:
    A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    B is 4x7: B[j,k] for j in {0,1,2,3}, k in {0,...,6}
    C is 2x7: C[i,k] for i in {0,1}, k in {0,...,6}
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's split j into two groups: {0,1} and {2,3}.
    C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    
    For the first sum, we have a 2x2x7 multiplication (A_left 2x2, B_left 4x7 restricted to rows 0,1 which is 2x7).
    For the second sum, we have a 2x2x7 multiplication (A_right 2x2, B_right 4x7 restricted to rows 2,3 which is 2x7).
    
    Each 2x2x7 can be done with:
    - Split the 7 columns into 4+3.
    - The 2x2x4 part: use Strassen for 2x2x2 twice? No, 2x2x4 means multiplying 2x2 by 2x4.
      We can split the 2x4 into two 2x2 blocks and use Strassen on each.
      That's 7+7 = 14 operations for the 2x2x4 part.
    - The 2x2x3 part: schoolbook, 2*2*3 = 12 operations.
    
    So per 2x2x7 block: 14 + 12 = 26 operations.
    Two such blocks: 52 operations. That's worse than 56.
    
    Let me reconsider. The known result is 50.
    
    Alternative approach: 
    Split the 4x7 matrix B into four 2x2 blocks plus one 2x1 block? No, 4x7 doesn't split evenly.
    
    Let me try: Split A into two 2x2 blocks (columns 0-1, columns 2-3).
    Split B into:
    - B11 (2x2) from rows 0-1, cols 0-1
    - B12 (2x2) from rows 0-1, cols 2-3
    - B13 (2x2) from rows 0-1, cols 4-5
    - B14 (2x1) from rows 0-1, col 6
    - B21 (2x2) from rows 2-3, cols 0-1
    - B22 (2x2) from rows 2-3, cols 2-3
    - B23 (2x2) from rows 2-3, cols 4-5
    - B24 (2x1) from rows 2-3, col 6
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    For A_left (2x2, cols 0-1) times B's rows 0-1:
    C_left[i,k] = A_left[i,0]*B[0,k] + A_left[i,1]*B[1,k]
    This is 2x2x7.
    
    For A_right (2x2, cols 2-3) times B's rows 2-3:
    C_right[i,k] = A_right[i,0]*B[2,k] + A_right[i,1]*B[3,k]
    This is 2x2x7.
    
    Total C = C_left + C_right.
    
    For 2x2x7, we can use:
    - Strassen for the 2x2x2 part (cols 0-1): 7 ops
    - Strassen for the 2x2x2 part (cols 2-3): 7 ops
    - Schoolbook for the 2x2x2 part (cols 4-5): 4 ops
    - Schoolbook for the 2x2x1 part (col 6): 2 ops
    Total: 7+7+4+2 = 20 ops per 2x2x7.
    
    Two blocks: 40 ops. Plus we need to combine? No, they add directly to C.
    
    Wait, that gives 40, which is better than 50! But is it correct?
    
    Let me re-examine. The issue is that A is 2x4, not two independent 2x2 matrices.
    A = [A_left | A_right] where A_left is 2x2 (cols 0,1) and A_right is 2x2 (cols 2,3).
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    If I split B's rows into {0,1} and {2,3}:
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    The first term is A_left[i,:] * B_top[:,k] where B_top is 2x7 (rows 0,1 of B).
    The second term is A_right[i,:] * B_bottom[:,k] where B_bottom is 2x7 (rows 2,3 of B).
    
    So yes, it's two independent 2x2x7 multiplications added together.
    
    For each 2x2x7:
    Split the 7 columns into 2+2+2+1.
    - 2x2x2 (cols 0-1): Strassen, 7 ops
    - 2x2x2 (cols 2-3): Strassen, 7 ops
    - 2x2x2 (cols 4-5): Schoolbook, 4 ops
    - 2x2x1 (col 6): Schoolbook, 2 ops
    Total: 20 ops.
    
    Two blocks: 40 ops.
    
    But wait, the Strassen decomposition for 2x2x2 gives a 2x2 result. When we concatenate results from different column blocks, we just place them in the right positions. This should work.
    
    Let me verify the rank count:
    - Block 1 (A_left * B_top): 7 + 7 + 4 + 2 = 20
    - Block 2 (A_right * B_bottom): 7 + 7 + 4 + 2 = 20
    Total: 40.
    
    This seems too good. Let me check if there's an error.
    
    Actually, I think the issue is that the Strassen decomposition for 2x2x2 requires specific linear combinations. When we apply it to different column blocks, the U, V, W factors need to be consistent.
    
    Let me build it carefully.
    """
    n, m, p = 2, 4, 7
    A_dim = n * m  # 8
    B_dim = m * p  # 28
    C_dim = p * n  # 14
    
    # We'll build the decomposition by combining sub-decompositions.
    # Each sub-decomposition contributes to specific entries of U, V, W.
    
    # Let's use a list of (U_row_indices, V_row_indices, W_row_indices, values) tuples.
    # Actually, it's easier to build U, V, W directly.
    
    # Start with empty factor matrices
    # We'll determine the total rank first.
    
    # Block 1: A_left (2x2) * B_top (2x7)
    # A_left indices: A[0,0], A[0,1], A[1,0], A[1,1] -> global A indices 0,1,4,5
    # B_top indices: B[0,k], B[1,k] for k=0..6 -> global B indices 0..6, 7..13
    # C indices: C[0,k], C[1,k] for k=0..6 -> global C indices k*2+0, k*2+1 for k=0..6
    
    # Block 2: A_right (2x2) * B_bottom (2x7)
    # A_right indices: A[0,2], A[0,3], A[1,2], A[1,3] -> global A indices 2,3,6,7
    # B_bottom indices: B[2,k], B[3,k] for k=0..6 -> global B indices 14..20, 21..27
    # C indices: same as Block 1
    
    # For each block, we decompose the 2x2x7 into:
    # - 2x2x2 for cols 0-1: Strassen (7 ops)
    # - 2x2x2 for cols 2-3: Strassen (7 ops)
    # - 2x2x2 for cols 4-5: Schoolbook (4 ops)
    # - 2x2x1 for col 6: Schoolbook (2 ops)
    # Total: 20 ops per block, 40 total.
    
    # Let's build this step by step.
    # I'll create a helper to add a Strassen or schoolbook sub-decomposition.
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen_222(A_idx, B_idx, C_idx, col_offset):
        """
        Add a Strassen 2x2x2 decomposition.
        A_idx: [a11, a12, a21, a22] global indices
        B_idx: [b11, b12, b21, b22] global indices
        C_idx: [c11, c12, c21, c22] global indices
        col_offset: offset for column indices in B and C
        """
        nonlocal rank
        # M1 = (a11+a22)(b11+b22)
        U_list.append((A_idx[0], 1.0))
        U_list.append((A_idx[3], 1.0))
        V_list.append((B_idx[0], 1.0))
        V_list.append((B_idx[3], 1.0))
        W_list.append((C_idx[0], 1.0))
        W_list.append((C_idx[3], 1.0))
        rank += 1
        
        # M2 = (a21+a22)b11
        U_list.append((A_idx[2], 1.0))
        U_list.append((A_idx[3], 1.0))
        V_list.append((B_idx[0], 1.0))
        W_list.append((C_idx[0], -1.0))
        W_list.append((C_idx[2], 1.0))
        rank += 1
        
        # M3 = a11(b12-b22)
        U_list.append((A_idx[0], 1.0))
        V_list.append((B_idx[1], 1.0))
        V_list.append((B_idx[3], -1.0))
        W_list.append((C_idx[1], 1.0))
        W_list.append((C_idx[3], -1.0))
        rank += 1
        
        # M4 = a22(b21-b11)
        U_list.append((A_idx[3], 1.0))
        V_list.append((B_idx[2], 1.0))
        V_list.append((B_idx[0], -1.0))
        W_list.append((C_idx[0], 1.0))
        W_list.append((C_idx[2], 1.0))
        rank += 1
        
        # M5 = (a11+a12)b22
        U_list.append((A_idx[0], 1.0))
        U_list.append((A_idx[1], 1.0))
        V_list.append((B_idx[3], 1.0))
        W_list.append((C_idx[0], 1.0))
        W_list.append((C_idx[1], 1.0))
        rank += 1
        
        # M6 = (a21-a11)(b11+b12)
        U_list.append((A_idx[2], 1.0))
        U_list.append((A_idx[0], -1.0))
        V_list.append((B_idx[0], 1.0))
        V_list.append((B_idx[1], 1.0))
        W_list.append((C_idx[1], -1.0))
        rank += 1
        
        # M7 = (a12-a22)(b21+b22)
        U_list.append((A_idx[1], 1.0))
        U_list.append((A_idx[3], -1.0))
        V_list.append((B_idx[2], 1.0))
        V_list.append((B_idx[3], 1.0))
        W_list.append((C_idx[1], -1.0))
        W_list.append((C_idx[3], 1.0))
        rank += 1
    
    def add_schoolbook_222(A_idx, B_idx, C_idx):
        """
        Add a schoolbook 2x2x2 decomposition.
        """
        nonlocal rank
        # c11 = a11*b11
        U_list.append((A_idx[0], 1.0))
        V_list.append((B_idx[0], 1.0))
        W_list.append((C_idx[0], 1.0))
        rank += 1
        
        # c12 = a11*b12
        U_list.append((A_idx[0], 1.0))
        V_list.append((B_idx[1], 1.0))
        W_list.append((C_idx[1], 1.0))
        rank += 1
        
        # c21 = a21*b11
        U_list.append((A_idx[2], 1.0))
        V_list.append((B_idx[0], 1.0))
        W_list.append((C_idx[2], 1.0))
        rank += 1
        
        # c22 = a21*b12 + a22*b22 ... wait, schoolbook for 2x2x2 is 4 ops?
        # c11 = a11*b11
        # c12 = a11*b12
        # c21 = a21*b11
        # c22 = a22*b22
        # But that's only 4 products, and c12 also has a12*b22? No.
        # For 2x2x2: C = A*B
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # So we need 4 products:
        # M1 = A[0,0]*B[0,0]
        # M2 = A[0,1]*B[1,0]
        # M3 = A[0,0]*B[0,1]
        # M4 = A[0,1]*B[1,1]
        # M5 = A[1,0]*B[0,0]
        # M6 = A[1,1]*B[1,0]
        # M7 = A[1,0]*B[0,1]
        # M8 = A[1,1]*B[1,1]
        # That's 8 products for schoolbook 2x2x2!
        
        # I made an error. Schoolbook 2x2x2 is 8 ops, not 4.
        # Let me redo.
        
        # Actually, for the 2x2x3 part, I said 4 ops, but it should be 2*2*3 = 12 ops.
        # And for 2x2x1, it's 2*2*1 = 4 ops.
        
        # So per 2x2x7: 7 + 7 + 12 + 4 = 30 ops.
        # Two blocks: 60 ops. That's worse than 56!
        
        # I need to reconsider.
        pass
    
    # Let me restart with correct counts.
    # Schoolbook 2x2x2: 8 ops
    # Schoolbook 2x2x1: 4 ops
    
    # Per 2x2x7: 7 (Strassen) + 7 (Strassen) + 8 (Schoolbook for 2x2x2) + 4 (Schoolbook for 2x2x1) = 26
    # Two blocks: 52 ops. Still worse than 56? No, 52 < 56.
    
    # Wait, 52 is better than 56. Let me verify.
    # Trivial is 56. This gives 52. That's an improvement!
    
    # But the known best is 50. Let me see if I can do better.
    
    # Actually, for the 2x2x2 schoolbook parts, I can use Strassen if I want, but that's already done for the first two blocks.
    # For the third 2x2x2 block (cols 4-5), I could use Strassen too: 7 ops instead of 8.
    # Then per 2x2x7: 7+7+7+4 = 25. Two blocks: 50. That matches the known result!
    
    # So the strategy is:
    # - Split A into two 2x2 blocks.
    # - Split B's rows into two 2x7 blocks.
    # - For each 2x2x7, split columns into 2+2+2+1.
    # - Use Strassen for the first three 2x2x2 sub-blocks (7 ops each).
    # - Use schoolbook for the 2x2x1 sub-block (4 ops).
    # Total: 2*(7+7+7+4) = 50.
    
    # Let me implement this correctly.
    
    # Reset
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_strassen_222_correct(A_idx, B_idx, C_idx):
        """
        Add a Strassen 2x2x2 decomposition.
        A_idx: [a11, a12, a21, a22]
        B_idx: [b11, b12, b21, b22]
        C_idx: [c11, c12, c21, c22]
        """
        nonlocal rank
        # M1 = (a11+a22)(b11+b22)
        U_list.append((A_idx[0], 1.0))
        U_list.append((A_idx[3], 1.0))
        V_list.append((B_idx[0], 1.0))
        V_list.append((B_idx[3], 1.0))
        W_list.append((C_idx[0], 1.0))
        W_list.append((C_idx[3], 1.0))
        rank += 1
        
        # M2 = (a21+a22)b11
        U_list.append((A_idx[2], 1.0))
        U_list.append((A_idx[3], 1.0))
        V_list.append((B_idx[0], 1.0))
        W_list.append((C_idx[0], -1.0))
        W_list.append((C_idx[2], 1.0))
        rank += 1
        
        # M3 = a11(b12-b22)
        U_list.append((A_idx[0], 1.0))
        V_list.append((B_idx[1], 1.0))
        V_list.append((B_idx[3], -1.0))
        W_list.append((C_idx[1], 1.0))
        W_list.append((C_idx[3], -1.0))
        rank += 1
        
        # M4 = a22(b21-b11)
        U_list.append((A_idx[3], 1.0))
        V_list.append((B_idx[2], 1.0))
        V_list.append((B_idx[0], -1.0))
        W_list.append((C_idx[0], 1.0))
        W_list.append((C_idx[2], 1.0))
        rank += 1
        
        # M5 = (a11+a12)b22
        U_list.append((A_idx[0], 1.0))
        U_list.append((A_idx[1], 1.0))
        V_list.append((B_idx[3], 1.0))
        W_list.append((C_idx[0], 1.0))
        W_list.append((C_idx[1], 1.0))
        rank += 1
        
        # M6 = (a21-a11)(b11+b12)
        U_list.append((A_idx[2], 1.0))
        U_list.append((A_idx[0], -1.0))
        V_list.append((B_idx[0], 1.0))
        V_list.append((B_idx[1], 1.0))
        W_list.append((C_idx[1], -1.0))
        rank += 1
        
        # M7 = (a12-a22)(b21+b22)
        U_list.append((A_idx[1], 1.0))
        U_list.append((A_idx[3], -1.0))
        V_list.append((B_idx[2], 1.0))
        V_list.append((B_idx[3], 1.0))
        W_list.append((C_idx[1], -1.0))
        W_list.append((C_idx[3], 1.0))
        rank += 1
    
    def add_schoolbook_221(A_idx, B_idx, C_idx):
        """
        Add a schoolbook 2x2x1 decomposition.
        A is 2x2, B is 2x1, C is 2x1.
        A_idx: [a11, a12, a21, a22]
        B_idx: [b1, b2]
        C_idx: [c1, c2]
        """
        nonlocal rank
        # c1 = a11*b1 + a12*b2
        U_list.append((A_idx[0], 1.0))
        V_list.append((B_idx[0], 1.0))
        W_list.append((C_idx[0], 1.0))
        rank += 1
        
        U_list.append((A_idx[1], 1.0))
        V_list.append((B_idx[1], 1.0))
        W_list.append((C_idx[0], 1.0))
        rank += 1
        
        # c2 = a21*b1 + a22*b2
        U_list.append((A_idx[2], 1.0))
        V_list.append((B_idx[0], 1.0))
        W_list.append((C_idx[1], 1.0))
        rank += 1
        
        U_list.append((A_idx[3], 1.0))
        V_list.append((B_idx[1], 1.0))
        W_list.append((C_idx[1], 1.0))
        rank += 1
    
    # Block 1: A_left * B_top
    # A_left: A[0,0], A[0,1], A[1,0], A[1,1] -> global 0, 1, 4, 5
    # B_top: B[0,k], B[1,k] for k=0..6 -> global 0..6, 7..13
    # C: C[0,k], C[1,k] for k=0..6 -> global k*2+0, k*2+1
    
    A_left = [0, 1, 4, 5]
    
    # Strassen for cols 0-1
    B_top_01 = [0, 1, 7, 8]  # B[0,0], B[0,1], B[1,0], B[1,1]
    C_01 = [0, 1, 2, 3]  # C[0,0], C[0,1], C[1,0], C[1,1]
    add_strassen_222_correct(A_left, B_top_01, C_01)
    
    # Strassen for cols 2-3
    B_top_23 = [2, 3, 9, 10]  # B[0,2], B[0,3], B[1,2], B[1,3]
    C_23 = [4, 5, 6, 7]  # C[0,2], C[0,3], C[1,2], C[1,3]
    add_strassen_222_correct(A_left, B_top_23, C_23)
    
    # Strassen for cols 4-5
    B_top_45 = [4, 5, 11, 12]  # B[0,4], B[0,5], B[1,4], B[1,5]
    C_45 = [8, 9, 10, 11]  # C[0,4], C[0,5], C[1,4], C[1,5]
    add_strassen_222_correct(A_left, B_top_45, C_45)
    
    # Schoolbook for col 6
    B_top_6 = [6, 13]  # B[0,6], B[1,6]
    C_6 = [12, 13]  # C[0,6], C[1,6]
    add_schoolbook_221(A_left, B_top_6, C_6)
    
    # Block 2: A_right * B_bottom
    # A_right: A[0,2], A[0,3], A[1,2], A[1,3] -> global 2, 3, 6, 7
    # B_bottom: B[2,k], B[3,k] for k=0..6 -> global 14..20, 21..27
    
    A_right = [2, 3, 6, 7]
    
    # Strassen for cols 0-1
    B_bot_01 = [14, 15, 21, 22]  # B[2,0], B[2,1], B[3,0], B[3,1]
    C_01 = [0, 1, 2, 3]
    add_strassen_222_correct(A_right, B_bot_01, C_01)
    
    # Strassen for cols 2-3
    B_bot_23 = [16, 17, 23, 24]  # B[2,2], B[2,3], B[3,2], B[3,3]
    C_23 = [4, 5, 6, 7]
    add_strassen_222_correct(A_right, B_bot_23, C_23)
    
    # Strassen for cols 4-5
    B_bot_45 = [18, 19, 25, 26]  # B[2,4], B[2,5], B[3,4], B[3,5]
    C_45 = [8, 9, 10, 11]
    add_strassen_222_correct(A_right, B_bot_45, C_45)
    
    # Schoolbook for col 6
    B_bot_6 = [20, 27]  # B[2,6], B[3,6]
    C_6 = [12, 13]
    add_schoolbook_221(A_right, B_bot_6, C_6)
    
    # Now convert the lists to matrices
    total_rank = rank
    U = np.zeros((A_dim, total_rank))
    V = np.zeros((B_dim, total_rank))
    W = np.zeros((C_dim, total_rank))
    
    # We need to map the (index, value) pairs to the correct rank columns.
    # The issue is that U_list, V_list, W_list have different lengths for each rank term.
    # I need to restructure this.
    
    # Let me rebuild using a different approach: store each rank term as a tuple of (U_dict, V_dict, W_dict).
    
    # Actually, let me just build it directly with explicit rank indexing.
    
    U = np.zeros((A_dim, 50))
    V = np.zeros((B_dim, 50))
    W = np.zeros((C_dim, 50))
    
    r = 0
    
    # Block 1, Strassen cols 0-1
    # M1
    U[0, r] = 1; U[5, r] = 1
    V[0, r] = 1; V[8, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    # M2
    U[4, r] = 1; U[5, r] = 1
    V[0, r] = 1
    W[0, r] = -1; W[2, r] = 1
    r += 1
    # M3
    U[0, r] = 1
    V[1, r] = 1; V[8, r] = -1
    W[1, r] = 1; W[3, r] = -1
    r += 1
    # M4
    U[5, r] = 1
    V[7, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[2, r] = 1
    r += 1
    # M5
    U[0, r] = 1; U[1, r] = 1
    V[8, r] = 1
    W[0, r] = 1; W[1, r] = 1
    r += 1
    # M6
    U[4, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[1, r] = -1
    r += 1
    # M7
    U[1, r] = 1; U[5, r] = -1
    V[7, r] = 1; V[8, r] = 1
    W[1, r] = -1; W[3, r] = 1
    r += 1
    
    # Block 1, Strassen cols 2-3
    # M1
    U[0, r] = 1; U[5, r] = 1
    V[2, r] = 1; V[10, r] = 1
    W[4, r] = 1; W[7, r] = 1
    r += 1
    # M2
    U[4, r] = 1; U[5, r] = 1
    V[2, r] = 1
    W[4, r] = -1; W[6, r] = 1
    r += 1
    # M3
    U[0, r] = 1
    V[3, r] = 1; V[10, r] = -1
    W[5, r] = 1; W[7, r] = -1
    r += 1
    # M4
    U[5, r] = 1
    V[9, r] = 1; V[2, r] = -1
    W[4, r] = 1; W[6, r] = 1
    r += 1
    # M5
    U[0, r] = 1; U[1, r] = 1
    V[10, r] = 1
    W[4, r] = 1; W[5, r] = 1
    r += 1
    # M6
    U[4, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[5, r] = -1
    r += 1
    # M7
    U[1, r] = 1; U[5, r] = -1
    V[9, r] = 1; V[10, r] = 1
    W[5, r] = -1; W[7, r] = 1
    r += 1
    
    # Block 1, Strassen cols 4-5
    # M1
    U[0, r] = 1; U[5, r] = 1
    V[4, r] = 1; V[12, r] = 1
    W[8, r] = 1; W[11, r] = 1
    r += 1
    # M2
    U[4, r] = 1; U[5, r] = 1
    V[4, r] = 1
    W[8, r] = -1; W[10, r] = 1
    r += 1
    # M3
    U[0, r] = 1
    V[5, r] = 1; V[12, r] = -1
    W[9, r] = 1; W[11, r] = -1
    r += 1
    # M4
    U[5, r] = 1
    V[11, r] = 1; V[4, r] = -1
    W[8, r] = 1; W[10, r] = 1
    r += 1
    # M5
    U[0, r] = 1; U[1, r] = 1
    V[12, r] = 1
    W[8, r] = 1; W[9, r] = 1
    r += 1
    # M6
    U[4, r] = 1; U[0, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[9, r] = -1
    r += 1
    # M7
    U[1, r] = 1; U[5, r] = -1
    V[11, r] = 1; V[12, r] = 1
    W[9, r] = -1; W[11, r] = 1
    r += 1
    
    # Block 1, Schoolbook col 6
    # c1 = a11*b1 + a12*b2
    U[0, r] = 1
    V[6, r] = 1
    W[12, r] = 1
    r += 1
    U[1, r] = 1
    V[13, r] = 1
    W[12, r] = 1
    r += 1
    # c2 = a21*b1 + a22*b2
    U[4, r] = 1
    V[6, r] = 1
    W[13, r] = 1
    r += 1
    U[5, r] = 1
    V[13, r] = 1
    W[13, r] = 1
    r += 1
    
    # Block 2, Strassen cols 0-1
    # M1
    U[2, r] = 1; U[7, r] = 1
    V[14, r] = 1; V[22, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    # M2
    U[6, r] = 1; U[7, r] = 1
    V[14, r] = 1
    W[0, r] = -1; W[2, r] = 1
    r += 1
    # M3
    U[2, r] = 1
    V[15, r] = 1; V[22, r] = -1
    W[1, r] = 1; W[3, r] = -1
    r += 1
    # M4
    U[7, r] = 1
    V[21, r] = 1; V[14, r] = -1
    W[0, r] = 1; W[2, r] = 1
    r += 1
    # M5
    U[2, r] = 1; U[3, r] = 1
    V[22, r] = 1
    W[0, r] = 1; W[1, r] = 1
    r += 1
    # M6
    U[6, r] = 1; U[2, r] = -1
    V[14, r] = 1; V[15, r] = 1
    W[1, r] = -1
    r += 1
    # M7
    U[3, r] = 1; U[7, r] = -1
    V[21, r] = 1; V[22, r] = 1
    W[1, r] = -1; W[3, r] = 1
    r += 1
    
    # Block 2, Strassen cols 2-3
    # M1
    U[2, r] = 1; U[7, r] = 1
    V[16, r] = 1; V[24, r] = 1
    W[4, r] = 1; W[7, r] = 1
    r += 1
    # M2
    U[6, r] = 1; U[7, r] = 1
    V[16, r] = 1
    W[4, r] = -1; W[6, r] = 1
    r += 1
    # M3
    U[2, r] = 1
    V[17, r] = 1; V[24, r] = -1
    W[5, r] = 1; W[7, r] = -1
    r += 1
    # M4
    U[7, r] = 1
    V[23, r] = 1; V[16, r] = -1
    W[4, r] = 1; W[6, r] = 1
    r += 1
    # M5
    U[2, r] = 1; U[3, r] = 1
    V[24, r] = 1
    W[4, r] = 1; W[5, r] = 1
    r += 1
    # M6
    U[6, r] = 1; U[2, r] = -1
    V[16, r] = 1; V[17, r] = 1
    W[5, r] = -1
    r += 1
    # M7
    U[3, r] = 1; U[7, r] = -1
    V[23, r] = 1; V[24, r] = 1
    W[5, r] = -1; W[7, r] = 1
    r += 1
    
    # Block 2, Strassen cols 4-5
    # M1
    U[2, r] = 1; U[7, r] = 1
    V[18, r] = 1; V[26, r] = 1
    W[8, r] = 1; W[11, r] = 1
    r += 1
    # M2
    U[6, r] = 1; U[7, r] = 1
    V[18, r] = 1
    W[8, r] = -1; W[10, r] = 1
    r += 1
    # M3
    U[2, r] = 1
    V[19, r] = 1; V[26, r] = -1
    W[9, r] = 1; W[11, r] = -1
    r += 1
    # M4
    U[7, r] = 1
    V[25, r] = 1; V[18, r] = -1
    W[8, r] = 1; W[10, r] = 1
    r += 1
    # M5
    U[2, r] = 1; U[3, r] = 1
    V[26, r] = 1
    W[8, r] = 1; W[9, r] = 1
    r += 1
    # M6
    U[6, r] = 1; U[2, r] = -1
    V[18, r] = 1; V[19, r] = 1
    W[9, r] = -1
    r += 1
    # M7
    U[3, r] = 1; U[7, r] = -1
    V[25, r] = 1; V[26, r] = 1
    W[9, r] = -1; W[11, r] = 1
    r += 1
    
    # Block 2, Schoolbook col 6
    # c1 = a11*b1 + a12*b2
    U[2, r] = 1
    V[20, r] = 1
    W[12, r] = 1
    r += 1
    U[3, r] = 1
    V[27, r] = 1
    W[12, r] = 1
    r += 1
    # c2 = a21*b1 + a22*b2
    U[6, r] = 1
    V[20, r] = 1
    W[13, r] = 1
    r += 1
    U[7, r] = 1
    V[27, r] = 1
    W[13, r] = 1
    r += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = DIMENSIONS
    
    # Build the rank-50 decomposition
    U, V, W = _build_2x4x7_rank50()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # If verification fails, fall back to trivial
    rank = n * m * p
    U_triv = np.zeros((n * m, rank))
    V_triv = np.zeros((m * p, rank))
    W_triv = np.zeros((p * n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, r] = 1
                V_triv[j * p + k, r] = 1
                W_triv[k * n + i, r] = 1
                r += 1
    
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
