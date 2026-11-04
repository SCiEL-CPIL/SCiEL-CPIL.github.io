import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # 1. Define Strassen 2x2x2 terms
    # U, V, W are 2x2 matrices
    U_2 = []
    V_2 = []
    W_2 = []
    
    # t=0
    U_2.append(np.array([[1,0],[1,1]]))
    V_2.append(np.array([[1,0],[1,1]]))
    W_2.append(np.array([[1,0],[0,1]]))
    
    # t=1
    U_2.append(np.array([[0,0],[1,1]]))
    V_2.append(np.array([[1,0],[0,0]]))
    W_2.append(np.array([[0,0],[1,-1]]))
    
    # t=2
    U_2.append(np.array([[1,0],[0,0]]))
    V_2.append(np.array([[0,1],[0,-1]]))
    W_2.append(np.array([[0,1],[0,1]]))
    
    # t=3
    U_2.append(np.array([[0,0],[0,1]]))
    V_2.append(np.array([[0,1],[1,0]])) # Wait, standard is B10 - B00
    # Standard Strassen:
    # M4 = A11 * (B10 - B00)
    V_2[3] = np.array([[0, -1],[1, 0]]) # No, B10 is row 1 col 0?
    # Let's stick to the indices used in the previous program for consistency or verify standard.
    # Standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's re-define cleanly
    U_2 = [
        np.array([[1,0],[1,1]]), # A00+A11
        np.array([[0,0],[1,1]]), # A10+A11
        np.array([[1,0],[0,0]]), # A00
        np.array([[0,0],[0,1]]), # A11
        np.array([[1,1],[0,0]]), # A00+A01
        np.array([[-1,0],[1,0]]), # A10-A00
        np.array([[0,1],[0,-1]])  # A01-A11
    ]
    
    V_2 = [
        np.array([[1,0],[1,1]]), # B00+B11
        np.array([[1,0],[0,0]]), # B00
        np.array([[0,1],[0,-1]]), # B01-B11
        np.array([[0,-1],[1,0]]), # B10-B00
        np.array([[0,0],[0,1]]), # B11
        np.array([[1,1],[0,0]]), # B00+B01
        np.array([[0,1],[1,1]])  # B10+B11
    ]
    
    W_2 = [
        np.array([[1,0],[0,1]]), # C00+C11
        np.array([[0,0],[1,-1]]), # C10-C11
        np.array([[0,1],[0,1]]), # C01+C11
        np.array([[1,0],[1,0]]), # C00+C10
        np.array([[-1,1],[0,0]]), # -C00+C01
        np.array([[0,0],[0,1]]), # C11
        np.array([[1,0],[0,0]])  # C00
    ]

    # 2. Define Top-Level Strassen 4x4x4 terms
    # These define how 2x2 blocks of A, B, C are combined.
    # U_top[s] is 2x2 coeffs for A blocks
    # V_top[s] is 2x4 coeffs for B blocks (4 blocks of 2x4) -> Actually, V_top[s] is a list of coeffs for the 4 B blocks.
    # Since B blocks are 2x4, and the "sub" problem sees a 2x4 matrix,
    # we need to know which B blocks are added.
    
    # Let's represent V_top[s] as a 2x4 matrix of "block coefficients"?
    # No, the sub-problem B_s is a 2x4 matrix.
    # B_s = c00 * B00 + c01 * B01 + c10 * B10 + c11 * B11.
    # Here B00 is 2x4, etc.
    # So V_top[s] is a 4x1 vector of coefficients? Or 2x2?
    # The B matrix is 4x8.
    # Block B00: rows 0-1, cols 0-3.
    # Block B01: rows 0-1, cols 4-7.
    # Block B10: rows 2-3, cols 0-3.
    # Block B11: rows 2-3, cols 4-7.
    
    # In the top-level Strassen, B_s is a linear combination of these 4 blocks.
    # The coefficients are 1 or -1.
    
    V_top = [
        [[1,1],[0,0]], # B00+B11 (c00=1, c11=1) -> Wait, B00 is rows 0-1. B11 is rows 2-3.
                       # So B_s will have non-zeros in rows 0-1 and 2-3.
                       # The sub-problem expects a 2x4 matrix.
                       # This implies we must map the 4 blocks to the 2 rows of the sub-problem?
                       # NO. The sub-problem is 2x2 * 2x4.
                       # The A_s is 2x2. The B_s must be 2x4.
                       # But B_s is formed from 4 blocks of size 2x4.
                       # They are stacked vertically?
                       # B00 (2x4) + B11 (2x4) -> Result is 4x4?
                       # No, in Strassen, A is 4x4, B is 4x4.
                       # A_s is 2x2? No, A_s is a 2x2 matrix of coefficients?
                       # No, in Strassen, the "matrices" A00 etc are 2x2.
                       # The product (A00+A11)(B00+B11) is a 2x2 matrix product.
                       # The result is a 2x2 matrix.
                       
                       # WAIT. My previous analysis assumed A_s is 2x2 and B_s is 2x4.
                       # This is correct for the SUB problem.
                       # But in the TOP level Strassen, we are multiplying 2x2 blocks.
                       # The "A" in the sub problem is the 2x2 block A_s.
                       # The "B" in the sub problem is the 2x2 block B_s.
                       # But B is 4x8.
                       # So the "B blocks" in the top-level Strassen are 2x4.
                       # The product of a 2x2 matrix and a 2x4 matrix is 2x4.
                       # This matches the sub problem dimensions!
                       
                       # So, B_s is a 2x4 matrix.
                       # It is formed by adding 2x4 blocks?
                       # No, B_s is a linear combination of the 4 blocks B00, B01, B10, B11.
                       # B00 is 2x4. B11 is 2x4.
                       # B00 + B11 is 4x4? No, they are at different row positions.
                       # B00 is rows 0-1. B11 is rows 2-3.
                       # So B00 + B11 is a 4x4 matrix (with zeros in the other half)?
                       # No, in Strassen, the blocks are conceptual.
                       # The product (A00+A11)(B00+B11) computes a 2x2 result.
                       # This result is placed in C00 and C11.
                       
                       # So the sub-problem is:
                       # Input A_s (2x2), Input B_s (2x2). Output C_s (2x2).
                       # But B is 4x8.
                       # So B_s is 2x2?
                       # If B is 4x8, the blocks are 2x4.
                       # A 2x2 matrix times a 2x4 matrix gives a 2x4 result.
                       # This result is placed in C blocks (2x4).
                       
                       # So YES, the sub-problem is 2x2 * 2x4.
                       # And B_s is a 2x4 matrix formed by combining the relevant B blocks.
                       # Which blocks?
                       # For M1 = (A00+A11)(B00+B11):
                       # A_s = A00+A11 (2x2).
                       # B_s = B00+B11.
                       # B00 is 2x4 (rows 0-1, cols 0-3).
                       # B11 is 2x4 (rows 2-3, cols 0-3).
                       # They cannot be added directly to form a 2x4 matrix because they have different row indices!
                       # UNLESS we reshape B.
                       
                       # This indicates that the direct concatenation trick might fail if the blocks don't align.
                       # However, for B00 and B11, they are disjoint in rows.
                       # So B_s is effectively a 4x4 matrix with zeros?
                       # Or do we treat the "2x2" Strassen as operating on the BLOCKS?
                       # Yes. The Strassen algorithm operates on blocks.
                       # The "multiplication" is a block multiplication.
                       # (A00+A11) is a 2x2 matrix.
                       # (B00+B11) is a 2x2 matrix of blocks? No, B00 and B11 are the same size.
                       # If B is 4x4, blocks are 2x2.
                       # If B is 4x8, blocks are 2x4.
                       # B00 and B11 are both 2x4.
                       # So B00+B11 is a 2x4 matrix.
                       # But they are at different row positions in the global B.
                       # In the Strassen formula, we ignore global positions and just add the blocks.
                       # The result C_s is a 2x4 matrix.
                       # It is then distributed to C00 and C11.
                       
                       # So the sub-problem is indeed 2x2 * 2x4.
                       # And the input B_s is the sum of the specific 2x4 blocks.
                       # So we CAN use the rank-7 decomposition of M_{2,2,4}.
                       
    ]
    
    # The logic holds.
    # We just need to map the block coefficients to the global indices.
    
    return {
        "U": [], "V": [], "W": []
    }
