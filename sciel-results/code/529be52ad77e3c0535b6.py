import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _build_decomp():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication.
    Uses 2x2 Strassen on 3x3 blocks.
    Inner 3x3 uses schoolbook (rank 27).
    """
    n, m, p = 6, 6, 6
    bs = 3
    rank = 7 * 27
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 definitions
    # A_block, B_block for each of 7 products
    s_defs = [
        ((0, 1), (0, 1)),   # (A00+A11, B00+B11)
        ((2, 1), (0, 1)),   # (A10+A11, B00) -- Wait, Strassen is:
    ]
    
    # Correct Strassen 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Mapping block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_2x2 = [
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)]},
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)]},
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)]},
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)]},
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)]},
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)]},
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)]},
    ]
    
    # Reconstruction: C_block = sum gamma_s * P_s
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    recon = {
        0: [(0, 1), (3, 1), (4, -1), (6, 1)],
        1: [(2, 1), (4, 1)],
        2: [(1, 1), (3, 1)],
        3: [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    r = 0
    for s in range(7):
        def_a = strassen_2x2[s]['A']
        def_b = strassen_2x2[s]['B']
        
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # Fill U
                    for blk, coeff in def_a:
                        br, bc = blk // 2, blk % 2
                        i_g = br * bs + u
                        k_g = bc * bs + v
                        U[i_g * m + k_g, r] += coeff
                    
                    # Fill V
                    for blk, coeff in def_b:
                        br, bc = blk // 2, blk % 2
                        k_g = br * bs + v
                        j_g = bc * bs + j
                        V[k_g * p + j_g, r] += coeff
                    
                    # Fill W
                    for c_blk in range(4):
                        coeffs = recon[c_blk]
                        g = 0
                        for idx, val in coeffs:
                            if idx == s:
                                g = val
                                break
                        if g != 0:
                            br, bc = c_blk // 2, c_blk % 2
                            i_g = br * bs + u
                            j_g = bc * bs + j
                            W[j_g * n + i_g, r] += g
                    r += 1
                    
    assert r == rank
    return {'U': U, 'V': V, 'W': W}

def construct():
    """
    Returns the factor matrices for the decomposition.
    """
    result = _build_decomp()
    # Convert to lists if required, though numpy arrays are often accepted
    # The task says "factor matrices of shapes... with integer/half-integer entries"
    # Numpy arrays are fine.
    return result
# EVOLVE-BLOCK-END
