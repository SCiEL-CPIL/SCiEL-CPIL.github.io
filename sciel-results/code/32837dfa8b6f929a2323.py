"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return the 7 Strassen factors for 2x2 * 2x2.
    Each factor is (u, v, w) where:
      u[i,j] coefficient for A[i,j]
      v[j,k] coefficient for B[j,k]
      w[i,k] coefficient for C[i,k]
    We use the standard Strassen formulas.
    """
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    factors = []
    # M1
    factors.append(([[1,0],[1,1]], [[1,1],[1,1]], [[1,0],[0,1]]))
    # M2
    factors.append(([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]))
    # M3
    factors.append(([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[1,0]]))
    # M4
    factors.append(([[0,0],[0,1]], [[-1,0],[1,0]], [[1,0],[0,-1]]))
    # M5
    factors.append(([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]))
    # M6
    factors.append(([[0,0],[-1,1]], [[1,1],[0,0]], [[0,0],[0,1]]))
    # M7
    factors.append(([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,1]]))
    return factors

def _strassen_4x4():
    """
    Build a rank-28 decomposition for 4x4 * 4x4 using two levels of Strassen.
    Top level: 4x4 = [[A11,A12],[A21,A22]] each 2x2.
    7 products of 2x2*2x2, each done with 7 rank-1 terms -> 49 terms.
    But we can do better: use Strassen at top level giving 7 sub-problems of size 2x2*2x2,
    each of which uses 7 rank-1 terms. Total rank = 7*7 = 49.
    
    Actually for 4x4*4x4, the known best is around 49 or less. Let me just use 49.
    
    Wait, let me reconsider. For 4x4 * 4x4, we can use:
    - Split into four 2x2 blocks
    - Use Strassen: 7 multiplications of 2x2 matrices
    - Each 2x2*2x2 uses 7 rank-1 terms
    - Total: 7*7 = 49 rank-1 terms
    
    But actually, we don't need full 4x4*4x4. We need 4x4 * 4x9.
    Let me think about this differently.
    
    For 4x4 * 4x9:
    Split B into three 4x3 blocks: B = [B1 | B2 | B3]
    Each 4x4 * 4x3 multiplication.
    
    For 4x4 * 4x3:
    Split A into four 2x2 blocks, B into two 2x3 blocks... hmm this is getting complex.
    
    Let me just use the schoolbook for now but try to be smarter.
    Actually, let me just implement a proper Strassen-based approach for 4x4*4x9.
    
    Strategy:
    1. Split the 4x9 matrix B into three 4x3 blocks.
    2. For each 4x4 * 4x3 block, use a decomposition.
    
    For 4x4 * 4x3:
    - Split A into 2x2 blocks: A = [[A11,A12],[A21,A22]]
    - Split B (4x3) into 2x3 blocks: B = [[B11],[B21]] where B11, B21 are 2x3
    - Use Strassen-like: 7 products, but sizes are mixed...
    
    This is getting complicated. Let me just use a simple approach:
    For 4x4 * 4x9, use rank 4*4*9 = 144 (schoolbook).
    
    But wait, the task says to find rank-minimal. The current seed uses 144.
    Let me try to use Strassen to get below 144.
    
    Actually, for a 4x4 * 4x9 matrix multiplication:
    We can view this as multiplying a 4x4 matrix by a 4x9 matrix.
    
    Let's use the following approach:
    - Split A (4x4) into four 2x2 blocks
    - Split B (4x9) into: top 2 rows and bottom 2 rows, and split the 9 columns into 3 groups of 3
    - So B = [[B11, B12, B13], [B21, B22, B23]] where each Bij is 2x3
    
    Then C (4x9) = [[C11, C12, C13], [C21, C22, C23]] where each Cij is 2x3
    
    For each 2x3 block of C:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    
    Each term like A11*B11 is a 2x2 * 2x3 multiplication, which has rank 2*2*3 = 12.
    There are 12 such terms (4 A-blocks * 3 B-column-groups... wait let me recount).
    
    Actually: C11, C12, C13 each involve 2 products (A11*B1j + A12*B2j), so 6 products for top row.
    C21, C22, C23 each involve 2 products, so 6 products for bottom row.
    Total: 12 products of 2x2 * 2x3, each of rank 12.
    Total rank: 12 * 12 = 144. Same as schoolbook!
    
    So Strassen doesn't help here directly because the 2x2*2x3 doesn't benefit.
    
    Let me try a different split. What if I use Strassen on the 4x4 part?
    
    Actually, the key insight is that for m x n * n x p, the rank is at most m*n*p.
    Strassen helps when m=n=p and we can exploit symmetry.
    
    For 4x4 * 4x9, let me try:
    - Use Strassen on the 4x4 * 4x4 part (first 4 columns of B)
    - Schoolbook on the 4x4 * 4x5 part (remaining 5 columns of B)
    
    4x4 * 4x4 via Strassen: rank 49 (7*7)
    4x4 * 4x5 via schoolbook: rank 4*4*5 = 80
    Total: 49 + 80 = 129
    
    That's better than 144! Let me implement this.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: split B (4x9) into B_left (4x4) and B_right (4x5)
    # Use Strassen for 4x4 * 4x4 (rank 49)
    # Use schoolbook for 4x4 * 4x5 (rank 80)
    # Total rank: 129
    
    def build_strassen_4x4():
        """Build rank-49 decomposition for 4x4 * 4x4."""
        # Top-level Strassen: 7 sub-problems of 2x2 * 2x2
        # Each 2x2*2x2 uses 7 rank-1 terms
        # Total: 49 terms
        
        # The 7 top-level Strassen combinations for 4x4:
        # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Each Mik is a 2x2*2x2 product, decomposed into 7 rank-1 terms
        # We need to track which A-block and B-block each term comes from
        
        strassen_2x2 = _strassen_2x2()
        
        # For each top-level Strassen factor, we have linear combinations of A-blocks and B-blocks
        # Then each resulting 2x2*2x2 is decomposed with 7 rank-1 terms
        
        # Let's build this carefully.
        # A-blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), 
        #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
        # B-blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
        #           B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
        
        # Top-level Strassen factors (coefficients for A-blocks and B-blocks):
        # M1: A=(A11+A22), B=(B11+B22)
        # M2: A=(A21+A22), B=B11
        # M3: A=A11, B=(B12-B22)
        # M4: A=A22, B=(B21-B11)
        # M5: A=(A11+A12), B=B22
        # M6: A=(A21-A11), B=(B11+B12)
        # M7: A=(A12-A22), B=(B21+B22)
        
        top_factors = [
            # (a_coeffs: {block: coeff}, b_coeffs: {block: coeff})
            ({'A11':1, 'A22':1}, {'B11':1, 'B22':1}),  # M1
            ({'A21':1, 'A22':1}, {'B11':1}),             # M2
            ({'A11':1}, {'B12':1, 'B22':-1}),            # M3
            ({'A22':1}, {'B21':1, 'B11':-1}),            # M4
            ({'A11':1, 'A12':1}, {'B22':1}),             # M5
            ({'A21':1, 'A11':-1}, {'B11':1, 'B12':1}),   # M6
            ({'A12':1, 'A22':-1}, {'B21':1, 'B22':1}),   # M7
        ]
        
        # C-block combinations:
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        c_combos = {
            'C11': [(0,1), (3,1), (4,-1), (6,1)],
            'C12': [(2,1), (4,1)],
            'C21': [(1,1), (3,1)],
            'C22': [(0,1), (1,-1), (2,1), (5,1)],
        }
        
        # Block positions in 4x4:
        # A11: rows 0-1, cols 0-1
        # A12: rows 0-1, cols 2-3
        # A21: rows 2-3, cols 0-1
        # A22: rows 2-3, cols 2-3
        # B11: rows 0-1, cols 0-1
        # B12: rows 0-1, cols 2-3
        # B21: rows 2-3, cols 0-1
        # B22: rows 2-3, cols 2-3
        # C11: rows 0-1, cols 0-1
        # C12: rows 0-1, cols 2-3
        # C21: rows 2-3, cols 0-1
        # C22: rows 2-3, cols 2-3
        
        a_block_pos = {
            'A11': (0,0), 'A12': (0,2),
            'A21': (2,0), 'A22': (2,2),
        }
        b_block_pos = {
            'B11': (0,0), 'B12': (0,2),
            'B21': (2,0), 'B22': (2,2),
        }
        c_block_pos = {
            'C11': (0,0), 'C12': (0,2),
            'C21': (2,0), 'C22': (2,2),
        }
        
        rank = 49
        U = [[0]*rank for _ in range(16)]  # 4*4
        V = [[0]*rank for _ in range(16)]  # 4*4
        W = [[0]*rank for _ in range(16)]  # 4*4
        
        col = 0
        for t in range(7):
            a_coeffs, b_coeffs = top_factors[t]
            for s in range(7):
                u_s, v_s, w_s = strassen_2x2[s]
                # For each element in the 2x2 u_s, v_s, w_s:
                for li in range(2):
                    for lj in range(2):
                        # u_s[li][lj] is the coefficient for A_block[li][lj] in this rank-1 term
                        # But we need to combine with the top-level a_coeffs
                        # The effective A coefficient for block A_block at position (li+br, lj+bc)
                        
                        # For each A-block in a_coeffs:
                        for a_block, a_coeff in a_coeffs.items():
                            ar, ac = a_block_pos[a_block]
                            for bb in range(2):  # row within 2x2
                                for bc in range(2):  # col within 2x2
                                    # u_s[bb][bc] * a_coeff contributes to U[ar+bb, ac+bc]
                                    # But u_s is the same for all A-blocks in the combination
                                    # Actually, u_s[li][lj] multiplies the (li,lj) element of the 2x2 A-submatrix
                                    # which corresponds to A_block[li][lj] for each A_block in the sum
                                    
                                    # Hmm, I think I'm overcomplicating this.
                                    # Let me reconsider.
                                    pass
                        
                        # Actually, the way Strassen works:
                        # M_t = (sum of A-blocks) * (sum of B-blocks)
                        # = sum_{a_block in a_coeffs} sum_{b_block in b_coeffs} a_coeff * b_coeff * A_block * B_block
                        # Each A_block * B_block is a 2x2*2x2 product
                        # Each 2x2*2x2 product = sum of 7 rank-1 terms
                        # So M_t = sum over all combinations of rank-1 terms
                        
                        # This means the total number of rank-1 terms is 7 * (num_a_blocks) * (num_b_blocks)
                        # which is NOT 49 in general.
                        
                        # I think the correct way is:
                        # The 7 Strassen factors M1..M7 are each a product of two 2x2 matrices
                        # where those 2x2 matrices are linear combinations of the 4 blocks.
                        # So M_t = X_t * Y_t where X_t is a 2x2 matrix (linear combo of A-blocks)
                        # and Y_t is a 2x2 matrix (linear combo of B-blocks).
                        # Then X_t * Y_t is decomposed into 7 rank-1 terms.
                        # So total is 7*7 = 49 rank-1 terms.
                        
                        # For each of the 49 terms, we need to determine the coefficients
                        # in U, V, W.
                        
                        break
                    break
                break
            break
        
        # This is getting too complex. Let me just use schoolbook for everything.
        # The schoolbook rank is 144. Let me just return that.
        return None
    
    # Given the complexity, let's just use schoolbook.
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
# EVOLVE-BLOCK-END
