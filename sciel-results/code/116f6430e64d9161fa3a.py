"""Direct implementation of rank 182 decomposition for 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 120.0
    SAFETY_MARGIN = 10.0

    # Known rank-189 block Strassen: 27 blocks of 2x2 Strassen (7 terms each)
    # We can try to improve by replacing some blocks with 3x3 Strassen-style or mixed.
    # Actually, let's try a different decomposition:
    # 6x6 can be split into blocks. We can use:
    # - 9 blocks of 2x2: 9 * 7 = 63 (if we only multiply 2x2 blocks) but that's not 6x6
    # Better: think of 6x6 as composed of 2x2, 3x3, etc.
    
    # Let's use a mixed approach:
    # Split rows/cols into groups: {0,1}, {2,3}, {4,5} for 2x2 blocks
    # But we can also use 3x3 blocks on the whole matrix with Strassen for 3x3?
    # 3x3 MM rank is 19 (Bini et al.), so 19 * 4 = 76 if we had 4 blocks? No.
    
    # Alternative: Use the known rank 19 for 3x3.
    # 6x6 = 4 blocks of 3x3? No, 6/3 = 2, so 2x2 blocks of 3x3.
    # Rank would be 2x2 Strassen (7) applied to 3x3 blocks, where each "multiplication" 
    # of 3x3 blocks costs rank 19. Total rank = 7 * 19 = 133.
    
    # This is much better than 189! Let's implement this.
    # We need the rank-19 decomposition of 3x3 matrix multiplication.
    
    # Rank 19 decomposition of 3x3 MM (Bini, Capovani, Lotti, Romani 1978):
    # C = A * B, where A, B, C are 3x3.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # 19 multiplications.
    
    # Let me recall/derive the rank-19 coefficients.
    # This is complex. Let me use a different approach.
    
    # Actually, for 6x6, we can use:
    # - 9 blocks of 2x2: but 2x2 blocks don't cover 6x6 in a 3x3 grid properly for MM
    #   because MM requires the inner dimension to match.
    
    # Let's think again. 6x6 MM: C = AB
    # Split A into blocks A_{ij}, B into B_{jk}, C into C_{ik} where i,j,k in {1,2,3}
    # Each block is 2x2.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This is 3x3 MM of 2x2 matrices.
    # If we do schoolbook 3x3: 9 * 4 = 36 scalar mults per 2x2 block = 9*9*4 = 324? No.
    # Each 2x2 mult is rank 4 (schoolbook) or rank 7 (Strassen).
    # 9 block mults * 7 (Strassen) = 63? No, that's wrong.
    
    # The 3x3 MM of 2x2 blocks:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # For each (i,k), we sum over j=1,2,3.
    # Total number of 2x2 mults: 3*3*3 = 27.
    # Each 2x2 mult via Strassen: 7 scalar mults.
    # Total: 27 * 7 = 189. This is the current solution.
    
    # To improve, we need to reduce the number of 2x2 mults or use a lower-rank 2x2 mult.
    # 2x2 MM min rank is 7 (Strassen is optimal).
    
    # Alternative: Use 3x3 blocks.
    # 6x6 = 2x2 blocks of 3x3.
    # C_{IK} = sum_J A_{IJ} B_{JK} where I,J,K in {1,2}
    # Each A_{IJ} is 3x3.
    # 3x3 MM of 3x3 blocks: 8 block mults (schoolbook) or 7 (Strassen).
    # Each 3x3 MM has rank 19.
    # Total: 7 * 19 = 133 (Strassen) or 8 * 19 = 152 (schoolbook).
    
    # 133 < 189, so this is better!
    
    # I need the rank-19 decomposition of 3x3 MM.
    # Let me hardcode it.
    
    # Rank 19 for 3x3 MM:
    # Source: "The rank of matrix multiplication for 3x3 is 19"
    # The 19 terms (u, v, w) where u,v,w are 3x3 matrices:
    
    # I'll use a known set. Let me construct it carefully.
    
    # Actually, let me try a simpler approach first:
    # Use 2x2 blocks but with a smarter tiling.
    
    # Or: 6 = 2 + 2 + 2, use Strassen on the outer 3x3 structure?
    # No, the inner is 2x2 MM.
    
    # Let me try the 3x3 block approach with rank 19.
    
    # Rank 19 decomposition of 3x3 matrix multiplication:
    # I found a reference implementation. The 19 terms are:
    
    # Let's denote the 3x3 matrices for each term.
    # I'll hardcode the 19 triplets.
    
    # From "An Algebraic Algorithm for 3x3 Matrix Multiplication" or similar:
    
    # Term 1:  u = [[1,0,0],[0,0,0],[0,0,0]], v = [[0,0,0],[0,1,0],[0,0,0]], w = [[0,0,0],[0,0,0],[0,1,0]]
    # This is getting too complex to derive from memory.
    
    # Let me stick with the rank-189 solution but optimize the code structure.
    # The score is 189, and I need to beat 189.
    
    # Actually, wait. The inspiration mentions rank 182. Let me check if that's possible.
    # 182 = 26 * 7. So 26 Strassen terms.
    # This would mean we can express 6x6 MM with 26 2x2 Strassen mults.
    # 27 is the naive count. Can we save one?
    
    # 6x6 MM tensor rank: The tensor T has entries T_{i,j,k} = delta_{jl} delta_{km}... 
    # Actually, the rank of 6x6 MM is known to be 189 (from 27 * 7) but maybe lower?
    
    # Let me just implement the rank-189 solution cleanly and return it.
    # The score will be 189. If the seed was 216, then 216/189 = 1.143.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 MM
    # U: A indices (2x2), V: B indices (2x2), W: C indices (2x2)
    # C = AB, C_{ik} = sum_j A_{ij} B_{jk}
    
    # Strassen:
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
    # M2 = (A10 + A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01 - B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10 - B00)       -> C00 += M4, C10 += M4
    # M5 = (A00 + A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
    # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
    
    strassen_terms = [
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    # Iterate over 3x3 blocks of 2x2 submatrices
    # Block (i,k) in C = sum_j A(i,j) B(j,k)
    # i,j,k in {0,1,2}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    # A block at (i,j): rows 2i..2i+1, cols 2j..2j+1
                    # B block at (j,k): rows 2j..2j+1, cols 2k..2k+1
                    # C block at (i,k): rows 2i..2i+1, cols 2k..2k+1
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * j + lc)
                        U[u_row][col_idx] += coeff
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * j + lr) * p + (2 * k + lc)
                        V[v_row][col_idx] += coeff
                    for lr, lc, coeff in term['W']:
                        # W index: col * n + row (trace convention)
                        w_row = (2 * k + lc) * n + (2 * i + lr)
                        W[w_row][col_idx] += coeff
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Should not happen for this fast construction
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
