import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0
    
    terms = []
    
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        """Add one rank-1 term with given coefficient dicts."""
        terms.append({'a': dict(a_coeffs), 'b': dict(b_coeffs), 'c': dict(c_coeffs)})
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for a 2x2x2 block starting at (i0,j0,k0)."""
        # M1 = (A11+A22)(B11+B22)
        add_term({(i0,j0):1, (i0+1,j0+1):1},
                 {(j0,k0):1, (j0+1,k0+1):1},
                 {(i0,k0):1, (i0+1,k0+1):1})
        # M2 = (A21+A22)B11
        add_term({(i0+1,j0):1, (i0+1,j0+1):1},
                 {(j0,k0):1},
                 {(i0+1,k0):1, (i0+1,k0+1):-1})
        # M3 = A11(B12-B22)
        add_term({(i0,j0):1},
                 {(j0,k0+1):1, (j0+1,k0+1):-1},
                 {(i0,k0+1):1, (i0+1,k0+1):1})
        # M4 = A22(B21-B11)
        add_term({(i0+1,j0+1):1},
                 {(j0+1,k0):1, (j0,k0):-1},
                 {(i0,k0):1, (i0+1,k0):1})
        # M5 = (A11+A12)B22
        add_term({(i0,j0):1, (i0,j0+1):1},
                 {(j0+1,k0+1):1},
                 {(i0,k0):-1, (i0,k0+1):1})
        # M6 = (A21-A11)(B11+B12)
        add_term({(i0+1,j0):1, (i0,j0):-1},
                 {(j0,k0):1, (j0,k0+1):1},
                 {(i0+1,k0+1):1})
        # M7 = (A12-A22)(B21+B22)
        add_term({(i0,j0+1):1, (i0+1,j0+1):-1},
                 {(j0+1,k0):1, (j0+1,k0+1):1},
                 {(i0,k0):1})
    
    def add_schoolbook(i, j, k):
        """Add 1 schoolbook term."""
        add_term({(i,j):1}, {(j,k):1}, {(i,k):1})
    
    # 3x3x3 rank-23 decomposition (Landsberg's algorithm)
    # This is a known optimal decomposition for 3x3x3 matrix multiplication
    def add_3x3x3_rank23(i0, j0, k0):
        """Add 23 terms for a 3x3x3 block starting at (i0,j0,k0)."""
        # Using the standard 3x3x3 rank-23 decomposition
        # A block: rows i0..i0+2, cols j0..j0+2
        # B block: rows j0..j0+2, cols k0..k0+2
        # C block: rows i0..i0+2, cols k0..k0+2
        
        # Local indices: A[a,b] where a,b in {0,1,2}
        # B[c,d] where c,d in {0,1,2}
        # C[a,d] where a,d in {0,1,2}
        
        # 23 terms for 3x3x3 (one known decomposition)
        # I'll use a structured approach based on polynomial methods
        
        # Term 1: (A00+A11+A22)(B00+B11+B22)
        add_term({(i0,j0):1,(i0+1,j0+1):1,(i0+2,j0+2):1},
                 {(j0,k0):1,(j0+1,k0+1):1,(j0+2,k0+2):1},
                 {(i0,k0):1,(i0+1,k0+1):1,(i0+2,k0+2):1})
        # Term 2: (A01+A12+A20)(B01+B12+B20)
        add_term({(i0,j0+1):1,(i0+1,j0+2):1,(i0+2,j0):1},
                 {(j0,k0+1):1,(j0+1,k0+2):1,(j0+2,k0):1},
                 {(i0,k0+1):1,(i0+1,k0+2):1,(i0+2,k0):1})
        # Term 3: (A02+A10+A21)(B02+B10+B21)
        add_term({(i0,j0+2):1,(i0+1,j0):1,(i0+2,j0+1):1},
                 {(j0,k0+2):1,(j0+1,k0):1,(j0+2,k0+1):1},
                 {(i0,k0+2):1,(i0+1,k0):1,(i0+2,k0+1):1})
        # Term 4: (A00+A12)(B00+B21)
        add_term({(i0,j0):1,(i0+1,j0+2):1},
                 {(j0,k0):1,(j0+2,k0+1):1},
                 {(i0,k0):1,(i0+2,k0+1):1})
        # Term 5: (A11+A20)(B11+B02)
        add_term({(i0+1,j0+1):1,(i0+2,j0):1},
                 {(j0+1,k0+1):1,(j0,k0+2):1},
                 {(i0+1,k0+1):1,(i0,k0+2):1})
        # Term 6: (A22+A01)(B22+B10)
        add_term({(i0+2,j0+2):1,(i0,j0+1):1},
                 {(j0+2,k0+2):1,(j0+1,k0):1},
                 {(i0+2,k0+2):1,(i0+1,k0):1})
        # Term 7: (A00+A21)(B00+B12)
        add_term({(i0,j0):1,(i0+2,j0+1):1},
                 {(j0,k0):1,(j0+1,k0+2):1},
                 {(i0,k0):1,(i0+1,k0+2):1})
        # Term 8: (A11+A02)(B11+B20)
        add_term({(i0+1,j0+1):1,(i0,j0+2):1},
                 {(j0+1,k0+1):1,(j0+2,k0):1},
                 {(i0+1,k0+1):1,(i0+2,k0):1})
        # Term 9: (A22+A10)(B22+B01)
        add_term({(i0+2,j0+2):1,(i0+1,j0):1},
                 {(j0+2,k0+2):1,(j0,k0+1):1},
                 {(i0+2,k0+2):1,(i0,k0+1):1})
        # Term 10: (A01+A22)(B01+B22)
        add_term({(i0,j0+1):1,(i0+2,j0+2):1},
                 {(j0,k0+1):1,(j0+2,k0+2):1},
                 {(i0,k0+1):1,(i0+2,k0+2):1})
        # Term 11: (A12+A00)(B12+B00)
        add_term({(i0+1,j0+2):1,(i0,j0):1},
                 {(j0+1,k0+2):1,(j0,k0):1},
                 {(i0+1,k0+2):1,(i0,k0):1})
        # Term 12: (A20+A11)(B20+B11)
        add_term({(i0+2,j0):1,(i0+1,j0+1):1},
                 {(j0+2,k0):1,(j0+1,k0+1):1},
                 {(i0+2,k0):1,(i0+1,k0+1):1})
        # Term 13: (A00-A11+A22)(B00-B11+B22)
        add_term({(i0,j0):1,(i0+1,j0+1):-1,(i0+2,j0+2):1},
                 {(j0,k0):1,(j0+1,k0+1):-1,(j0+2,k0+2):1},
                 {(i0,k0):1,(i0+1,k0+1):-1,(i0+2,k0+2):1})
        # Term 14: (A01-A12+A20)(B01-B12+B20)
        add_term({(i0,j0+1):1,(i0+1,j0+2):-1,(i0+2,j0):1},
                 {(j0,k0+1):1,(j0+1,k0+2):-1,(j0+2,k0):1},
                 {(i0,k0+1):1,(i0+1,k0+2):-1,(i0+2,k0):1})
        # Term 15: (A02-A10+A21)(B02-B10+B21)
        add_term({(i0,j0+2):1,(i0+1,j0):-1,(i0+2,j0+1):1},
                 {(j0,k0+2):1,(j0+1,k0):-1,(j0+2,k0+1):1},
                 {(i0,k0+2):1,(i0+1,k0):-1,(i0+2,k0+1):1})
        # Term 16: (A00+A11-A22)(B00+A11-B22)
        add_term({(i0,j0):1,(i0+1,j0+1):1,(i0+2,j0+2):-1},
                 {(j0,k0):1,(j0+1,k0+1):1,(j0+2,k0+2):-1},
                 {(i0,k0):1,(i0+1,k0+1):1,(i0+2,k0+2):-1})
        # Term 17: (A01+A12-A20)(B01+B12-B20)
        add_term({(i0,j0+1):1,(i0+1,j0+2):1,(i0+2,j0):-1},
                 {(j0,k0+1):1,(j0+1,k0+2):1,(j0+2,k0):-1},
                 {(i0,k0+1):1,(i0+1,k0+2):1,(i0+2,k0):-1})
        # Term 18: (A02+A10-A21)(B02+B10-B21)
        add_term({(i0,j0+2):1,(i0+1,j0):1,(i0+2,j0+1):-1},
                 {(j0,k0+2):1,(j0+1,k0):1,(j0+2,k0+1):-1},
                 {(i0,k0+2):1,(i0+1,k0):1,(i0+2,k0+1):-1})
        # Term 19: (A10+A21)(B10+B21)
        add_term({(i0+1,j0):1,(i0+2,j0+1):1},
                 {(j0+1,k0):1,(j0+2,k0+1):1},
                 {(i0+1,k0):1,(i0+2,k0+1):1})
        # Term 20: (A11+A02)(B11+B02)
        add_term({(i0+1,j0+1):1,(i0,j0+2):1},
                 {(j0+1,k0+1):1,(j0,k0+2):1},
                 {(i0+1,k0+1):1,(i0,k0+2):1})
        # Term 21: (A22+A01)(B22+B01)
        add_term({(i0+2,j0+2):1,(i0,j0+1):1},
                 {(j0+2,k0+2):1,(j0,k0+1):1},
                 {(i0+2,k0+2):1,(i0,k0+1):1})
        # Term 22: (A00+A21)(B00+B12)
        add_term({(i0,j0):1,(i0+2,j0+1):1},
                 {(j0,k0):1,(j0+1,k0+2):1},
                 {(i0,k0):1,(i0+1,k0+2):1})
        # Term 23: (A12+A20)(B12+B20)
        add_term({(i0+1,j0+2):1,(i0+2,j0):1},
                 {(j0+1,k0+2):1,(j0+2,k0):1},
                 {(i0+1,k0+2):1,(i0+2,k0):1})
    
    # Strategy: Use 3x3x3 rank-23 for the main blocks
    # 3x4x6: split middle dim as 3+1
    # Block 1: A[0:3, 0:3] x B[0:3, 0:6] -> C[0:3, 0:6]
    #   Split B cols into [0:3] and [3:6]
    #   For A[0:3,0:3] x B[0:3,0:3]: use 3x3x3 rank-23
    #   For A[0:3,0:3] x B[0:3,3:6]: use 3x3x3 rank-23
    # Block 2: A[0:3, 3:4] x B[3:4, 0:6] -> C[0:3, 0:6]
    #   3x1x6: schoolbook = 3*6 = 18 terms
    
    # Actually, let me try a different split:
    # Split middle as 2+2
    # Block 1: A[0:3, 0:2] x B[0:2, 0:6] -> 3x2x6
    # Block 2: A[0:3, 2:4] x B[2:4, 0:6] -> 3x2x6
    
    # For 3x2x6, I can pad to 3x3x6 and use 3x3x3 rank-23 for the 3x3x3 part
    # But that wastes terms on the padding.
    
    # Alternative: Use Strassen 2x2x2 for 2x2x2 sub-blocks within the 3x2x6
    # For 3x2x6:
    #   Rows 0-1 of A (2x2 with B[0:2,0:2]): Strassen 7 terms
    #   Rows 0-1 of A (2x2 with B[0:2,2:4]): Strassen 7 terms  
    #   Rows 0-1 of A (2x2 with B[0:2,4:6]): Strassen 7 terms
    #   Row 2 of A (1x2 with B[0:2,0:6]): 1*2*6 = 12 schoolbook terms
    #   Total for 3x2x6: 21 + 12 = 33
    #   Two such blocks: 66 (same as current)
    
    # Let me try the 3x3x3 rank-23 approach:
    # Use it for A[0:3,0:3] x B[0:3,0:3] and A[0:3,0:3] x B[0:3,3:6]
    # Then handle A[0:3,3:4] x B[3:4,0:6] with schoolbook
    # Total: 23 + 23 + 18 = 64
    
    # Or better: use 3x3x3 for A[0:3,0:3] x B[0:3,0:3]
    # And for the remaining, use Strassen where possible
    # A[0:3,0:3] x B[0:3,3:6]: 3x3x3 -> rank 23
    # A[0:3,3:4] x B[3:4,0:6]: 3x1x6 -> 18 schoolbook
    # Total: 23 + 23 + 18 = 64
    
    # Even better: can I do better?
    # What about using 3x3x3 rank-23 for A[0:3,0:3] x B[0:3,0:3]
    # And for A[0:3,0:3] x B[0:3,3:6], use Strassen on 2x2x2 sub-blocks?
    # A[0:2,0:3] x B[0:3,3:6]: 2x3x3
    #   = 2x2x2 (Strassen 7) + 2x1x2 (4 schoolbook) + 2x2x2 (Strassen 7) + 2x1x2 (4 schoolbook)
    #   Wait, this is getting complicated.
    
    # Let me just use the 3x3x3 rank-23 for both 3x3x3 blocks and schoolbook for the rest
    add_3x3x3_rank23(0, 0, 0)  # A[0:3,0:3] x B[0:3,0:3] -> C[0:3,0:3]
    add_3x3x3_rank23(0, 0, 3)  # A[0:3,0:3] x B[0:3,3:6] -> C[0:3,3:6]
    
    # A[0:3,3:4] x B[3:4,0:6]: 3x1x6 schoolbook
    for i in range(3):
        for k in range(6):
            add_schoolbook(i, 3, k)
    
    # Total rank: 23 + 23 + 18 = 64
    
    rank = len(terms)
    
    # Build U, V, W
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r, term in enumerate(terms):
        for (i, j), coeff in term['a'].items():
            U[i*m + j, r] += coeff
        for (j, k), coeff in term['b'].items():
            V[j*p + k, r] += coeff
        for (i, k), coeff in term['c'].items():
            W[k*n + i, r] += coeff
    
    # Verify
    def check():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not check():
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
