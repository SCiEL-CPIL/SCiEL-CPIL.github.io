"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy: Split p=10 into 5 blocks of size 2. Each block is M_{2,3,2}
    with rank 11 (7 Strassen for 2x2x2 part + 4 outer product terms).
    Total rank = 5 * 11 = 55.
    
    The output factor W follows the convention W[k*n+i, r].
    """
    deadline = time.time() + 140.0

    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks of size 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 55

    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 55

    # For each 2x3x2 block, we need 11 terms.
    # A: i in {0,1}, j in {0,1,2} -> U index = i*3+j
    # B: j in {0,1,2}, k in {0,1} -> V index = j*2+k_local (within block)
    # C: i in {0,1}, k in {0,1} -> W index = k*2+i
    
    # Strassen 7 terms for the 2x2x2 sub-problem (j=0,1):
    # A_sub: i in {0,1}, j in {0,1}
    # B_sub: j in {0,1}, k in {0,1}
    # C: i in {0,1}, k in {0,1}
    
    # Standard Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C01 += M4
    # M5 = (A00+A01)B11 -> C01 += M5, C00 -= M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Let me verify Strassen:
    # C00 = A00B00 + A01B10
    #   = M1 - M5 + M4 + M7 - M2? Let me recheck.
    #   
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    #
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let me verify C00 = M1 + M4 - M5 + M7:
    # M1 = A00B00+A00B11+A11B00+A11B11
    # M4 = A11B10-A11B00
    # -M5 = -A00B11-A01B11
    # M7 = A01B10+A01B11-A11B10-A11B11
    # Sum = A00B00+A00B11+A11B00+A11B11 + A11B10-A11B00 - A00B11-A01B11 + A01B10+A01B11-A11B10-A11B11
    # = A00B00 + (A00B11-A00B11) + (A11B00-A11B00) + (A11B11-A11B11) + (A11B10-A11B10) + (-A01B11+A01B11) + A01B10
    # = A00B00 + A01B10 ✓
    
    # C01 = M3 + M5:
    # M3 = A00B01-A00B11
    # M5 = A00B11+A01B11
    # Sum = A00B01 - A00B11 + A00B11 + A01B11 = A00B01 + A01B11 ✓
    
    # C10 = M2 + M6:
    # M2 = A10B00+A11B00
    # M6 = A10B00+A10B01-A00B00-A00B01
    # Sum = A10B00+A11B00+A10B00+A10B01-A00B00-A00B01
    # = 2*A10B00 + A11B00 + A10B01 - A00B00 - A00B01
    # Expected: A10B00 + A11B01
    # This is WRONG!
    
    # I have the Strassen formula wrong. Let me use the correct one.
    # Correct Strassen:
    # C10 = M2 + M6 where M6 = (A10-A00)(B00+B01)
    # M2 = (A10+A11)B00 = A10B00+A11B00
    # M6 = (A10-A00)(B00+B01) = A10B00+A10B01-A00B00-A00B01
    # Sum = 2A10B00+A11B00+A10B01-A00B00-A00B01
    # Expected C10 = A10B00+A11B01
    # Still wrong.
    
    # Let me look up the correct Strassen formula.
    # C = [[C00,C01],[C10,C11]] where C = AB
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # Let me recheck C10 = M2 + M6:
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M2+M6 = 2A10B00 + A11B00 + A10B01 - A00B00 - A00B01
    # Expected: A10B00 + A11B10
    # These are not equal! So the standard Strassen formula I'm using is incorrect.
    
    # The correct Strassen decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M5 + M4 + M7
    # C01 = M5 + M3
    # C10 = M6 + M2
    # C11 = M1 - M2 + M3 + M6
    
    # I keep getting the same thing. Let me just verify numerically.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[19,22],[43,50]]
    # C00 = 1*5+2*7 = 5+14 = 19
    # C01 = 1*6+2*8 = 6+16 = 22
    # C10 = 3*5+4*7 = 15+28 = 43
    # C11 = 3*6+4*8 = 18+32 = 50
    
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 7*5 = 35
    # M3 = 1*(6-8) = 1*(-2) = -2
    # M4 = 4*(7-5) = 4*2 = 8
    # M5 = (1+2)*8 = 3*8 = 24
    # M6 = (3-1)*(5+6) = 2*11 = 22
    # M7 = (2-4)*(7+8) = (-2)*15 = -30
    
    # C00 = M1 - M5 + M4 + M7 = 65 - 24 + 8 + (-30) = 65-24+8-30 = 19 ✓
    # C01 = M5 + M3 = 24 + (-2) = 22 ✓
    # C10 = M6 + M2 = 22 + 35 = 57 ✗ (expected 43)
    # C11 = M1 - M2 + M3 + M6 = 65 - 35 + (-2) + 22 = 50 ✓
    
    # C10 is wrong! 57 != 43. So the Strassen formula for C10 is not M6+M2.
    
    # Let me find the correct formula. We need C10 = 43.
    # M1=65, M2=35, M3=-2, M4=8, M5=24, M6=22, M7=-30
    # 43 = ?
    # M2 + M4 = 35+8 = 43 ✓!
    # So C10 = M2 + M4?
    # Let me verify: M2 = (A10+A11)B00, M4 = A11(B10-B00)
    # M2+M4 = A10B00+A11B00+A11B10-A11B00 = A10B00+A11B10 ✓
    
    # So the correct Strassen formulas are:
    # C00 = M1 - M5 + M4 + M7
    # C01 = M5 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Wait, but M4 is used in both C00 and C10. That's fine for tensor decomposition.
    
    # Let me double-check C11 = M1 - M2 + M3 + M6:
    # M1-M2+M3+M6 = 65-35-2+22 = 50 ✓
    
    # Great! Now I have the correct Strassen decomposition.
    
    # For the 2x3x2 block, the 11 terms are:
    # 7 Strassen terms (for j=0,1 part) + 4 outer product terms (for j=2)
    
    # Strassen terms with correct W coefficients:
    # M1: U=[(0,1),(4,1)], V=[(0,1),(1,1)], W=[(0,1),(3,1)]  # C00+=M1, C11+=M1
    # M2: U=[(3,1),(4,1)], V=[(0,1)], W=[(1,1)]  # C10+=M2
    # M3: U=[(0,1)], V=[(1,1),(1,-1)], W=[(2,1)]  # C01+=M3
    # M4: U=[(4,1)], V=[(1,1),(0,-1)], W=[(0,1),(1,1)]  # C00+=M4, C10+=M4
    # M5: U=[(0,1),(1,1)], V=[(1,1)], W=[(0,-1),(2,1)]  # C00-=M5, C01+=M5
    # M6: U=[(3,1),(0,-1)], V=[(0,1),(1,1)], W=[(3,1)]  # C11+=M6
    # M7: U=[(1,1),(4,-1)], V=[(1,1),(1,1)], W=[(0,1)]  # C00+=M7
    # Wait, M7 = (A01-A11)(B10+B11), so V should be B10 and B11.
    # B10 is j=1,k=0 -> V index 1*2+0=2, B11 is j=1,k=1 -> V index 1*2+1=3
    
    # Let me redo with proper V indices (j*2+k_local):
    # B00: j=0,k=0 -> V idx 0
    # B01: j=0,k=1 -> V idx 1
    # B10: j=1,k=0 -> V idx 2
    # B11: j=1,k=1 -> V idx 3
    # B20: j=2,k=0 -> V idx 4
    # B21: j=2,k=1 -> V idx 5
    
    # W indices (k*2+i):
    # C00: k=0,i=0 -> W idx 0
    # C01: k=1,i=0 -> W idx 2
    # C10: k=0,i=1 -> W idx 1
    # C11: k=1,i=1 -> W idx 3
    
    # U indices (i*3+j):
    # A00: i=0,j=0 -> U idx 0
    # A01: i=0,j=1 -> U idx 1
    # A02: i=0,j=2 -> U idx 2
    # A10: i=1,j=0 -> U idx 3
    # A11: i=1,j=1 -> U idx 4
    # A12: i=1,j=2 -> U idx 5
    
    terms_11 = [
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        {'U': [(0,1), (4,1)], 'V': [(0,1), (3,1)], 'W': [(0,1), (3,1)]},
        # M2 = (A10+A11)B00 -> C10+=1
        {'U': [(3,1), (4,1)], 'V': [(0,1)], 'W': [(1,1)]},
        # M3 = A00(B01-B11) -> C01+=1
        {'U': [(0,1)], 'V': [(1,1), (3,-1)], 'W': [(2,1)]},
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        {'U': [(4,1)], 'V': [(2,1), (0,-1)], 'W': [(0,1), (1,1)]},
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        {'U': [(0,1), (1,1)], 'V': [(3,1)], 'W': [(0,-1), (2,1)]},
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        {'U': [(3,1), (0,-1)], 'V': [(0,1), (1,1)], 'W': [(3,1)]},
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        {'U': [(1,1), (4,-1)], 'V': [(2,1), (3,1)], 'W': [(0,1)]},
        # Outer: A02*B20 -> C00+=1
        {'U': [(2,1)], 'V': [(4,1)], 'W': [(0,1)]},
        # Outer: A02*B21 -> C01+=1
        {'U': [(2,1)], 'V': [(5,1)], 'W': [(2,1)]},
        # Outer: A12*B20 -> C10+=1
        {'U': [(5,1)], 'V': [(4,1)], 'W': [(1,1)]},
        # Outer: A12*B21 -> C11+=1
        {'U': [(5,1)], 'V': [(5,1)], 'W': [(3,1)]}
    ]
    
    col_offset = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        for term in terms_11:
            col = col_offset
            for u_idx, val in term['U']:
                U[u_idx][col] += val
            for v_idx, val in term['V']:
                # v_idx is local: j*2+k_local
                j = v_idx // 2
                k_local = v_idx % 2
                k_global = b * block_size + k_local
                V[j * p + k_global][col] += val
            for w_idx, val in term['W']:
                # w_idx is local: k_local*2+i
                k_local = w_idx // 2
                i = w_idx % 2
                k_global = b * block_size + k_local
                W[k_global * n + i][col] += val
            col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
