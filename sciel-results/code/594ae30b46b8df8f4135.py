"""Search for rank-minimal exact tensor decomposition of 2x3 @ 3x8 matrix multiply."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145  # safety margin under 150s

    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def strassen_decomposition():
        """
        Decompose 2x2 x 2x2 x 2x2 matrix multiplication using Strassen's algorithm.
        Returns U, V, W as lists of lists of Fractions with rank 7.
        
        Strassen's algorithm:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B12 - B22)
        M4 = A22(B21 - B11)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # Each M_t is a product of two linear combinations of A and B entries.
        # We need to express each M_t as sum_c U[c] * V[c] where U[c] is a linear form in A, V[c] in B.
        # Actually, Strassen uses 7 multiplications. Each multiplication M_t = (linear_A_t)(linear_B_t).
        # So rank is 7.
        
        # A has entries A[0,0], A[0,1], A[1,0], A[1,1] -> index i*m+j = i*2+j
        # B has entries B[0,0], B[0,1], B[1,0], B[1,1] -> index j*p+k = j*2+k
        # C has entries C[0,0], C[0,1], C[1,0], C[1,1] -> index k*n+i = k*2+i
        
        # Let's define the 7 linear forms for A and B.
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11
        
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # U[row][c] = coefficient of A[row] in the linear form for multiplication c
        # V[row][c] = coefficient of B[row] in the linear form for multiplication c
        # W[row][c] = coefficient of C[row] that receives the product M_c
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: A00+A11, B00+B11
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        # C11 += M1, C22 += M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: A10+A11, B00
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1  # B00
        # C21 += M2, C22 -= M2
        W[2][1] = 1; W[3][1] = -1
        
        # M3: A00, B01-B11
        U[0][2] = 1  # A00
        V[1][2] = 1; V[3][2] = -1  # B01, B11
        # C12 += M3, C22 += M3
        W[1][2] = 1; W[3][2] = 1
        
        # M4: A11, B10-B00
        U[3][3] = 1  # A11
        V[2][3] = 1; V[0][3] = -1  # B10, B00
        # C11 += M4, C21 += M4
        W[0][3] = 1; W[2][3] = 1
        
        # M5: A00+A01, B11
        U[0][4] = 1; U[1][4] = 1  # A00, A01
        V[3][4] = 1  # B11
        # C11 -= M5, C12 += M5
        W[0][4] = -1; W[1][4] = 1
        
        # M6: A10-A00, B00+B01
        U[2][5] = 1; U[0][5] = -1  # A10, A00
        V[0][5] = 1; V[1][5] = 1  # B00, B01
        # C22 += M6
        W[3][5] = 1
        
        # M7: A01-A11, B10+B11
        U[1][6] = 1; U[3][6] = -1  # A01, A11
        V[2][6] = 1; V[3][6] = 1  # B10, B11
        # C11 += M7
        W[0][6] = 1
        
        # Convert to Fractions
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        
        return Uf, Vf, Wf

    def combine_decompositions(decomps):
        """
        Combine multiple decompositions. Each decomp is (U, V, W) for a sub-problem.
        We need to map the local indices to global indices.
        This is complex, so let's handle the specific 2x3x8 case manually.
        """
        pass

    # Let's construct the 2x3x8 decomposition directly.
    # Split A (2x3) into A_left (2x2) and A_right (2x1).
    # Split B (3x8) into B_left (2x8) and B_right (1x8).
    # C = A_left * B_left + A_right * B_right
    
    # For A_left (2x2) * B_left (2x8):
    # Split B_left into 4 blocks of 2x2: B00, B01, B02, B03 (each 2x2)
    # C_left = A_left * B00 + A_left * B01 + A_left * B02 + A_left * B03
    # Each A_left * B_block is a 2x2x2 multiplication, rank 7.
    # Total rank for this part: 4 * 7 = 28.
    
    # For A_right (2x1) * B_right (1x8):
    # C_right[i,k] = A_right[i,0] * B_right[0,k]
    # This is an outer product of a 2-vector and an 8-vector.
    # We need 2*8 = 16 multiplications (no sharing possible since each product is distinct).
    # Rank 16.
    
    # Total rank: 28 + 16 = 44.
    
    def construct_2x3x8():
        rank = 44
        U = [[0]*rank for _ in range(n*m)]  # 6 x 44
        V = [[0]*rank for _ in range(m*p)]  # 24 x 44
        W = [[0]*rank for _ in range(p*n)]  # 16 x 44
        
        # Part 1: A_left (2x2) * B_left (2x8), using Strassen for each 2x2 block of B.
        # A_left entries: A[0,0], A[0,1], A[1,0], A[1,1] -> global indices 0,1,2,3
        # B_left entries: B[0,0..7], B[1,0..7] -> global indices 0..7, 8..15
        # C_left entries: C[0,0..1], C[1,0..1] for each block -> we'll fill in as we go.
        
        # For each of the 4 blocks of B_left (each 2x2):
        # Block s (s=0,1,2,3): B_left[:, 2s:2s+2]
        # The output C_left[:, 2s:2s+2]
        
        for s in range(4):
            # Get the Strassen decomposition for this block
            U_s, V_s, W_s = strassen_decomposition()
            
            # Map local indices to global indices.
            # Local A index la -> global A index la (since A_left is the first 2 columns)
            # Local B index lb -> global B index 2s + lb (since B_left is the first 2 rows, and block s is columns 2s, 2s+1)
            # Wait, B is 3x8. B_left is rows 0,1 (2 rows) and columns 0..7 (8 columns).
            # B_left[j,k] has global index j*8+k.
            # For block s, B_left[:, 2s:2s+2] has entries B_left[0,2s], B_left[0,2s+1], B_left[1,2s], B_left[1,2s+1].
            # Global indices: 0*8+2s = 2s, 0*8+2s+1 = 2s+1, 1*8+2s = 8+2s, 1*8+2s+1 = 8+2s+1.
            # Local B indices: 0=B00, 1=B01, 2=B10, 3=B11.
            # So local 0 -> global 2s, local 1 -> global 2s+1, local 2 -> global 8+2s, local 3 -> global 8+2s+1.
            
            # Local C index lc -> global C index.
            # C_left[:, 2s:2s+2] has entries C[0,2s], C[0,2s+1], C[1,2s], C[1,2s+1].
            # Global C index: k*n+i = k*2+i.
            # C[0,2s] -> 2s*2+0 = 4s, C[0,2s+1] -> (2s+1)*2+0 = 4s+2, C[1,2s] -> 2s*2+1 = 4s+1, C[1,2s+1] -> (2s+1)*2+1 = 4s+3.
            # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11.
            # So local 0 -> global 4s, local 1 -> global 4s+2, local 2 -> global 4s+1, local 3 -> global 4s+3.
            
            base = s * 7
            for la in range(4):
                for c in range(7):
                    if U_s[la][c] != 0:
                        U[la][base+c] = U_s[la][c]
            
            b_map = {0: 2*s, 1: 2*s+1, 2: 8+2*s, 3: 8+2*s+1}
            for lb in range(4):
                for c in range(7):
                    if V_s[lb][c] != 0:
                        V[b_map[lb]][base+c] = V_s[lb][c]
            
            c_map = {0: 4*s, 1: 4*s+2, 2: 4*s+1, 3: 4*s+3}
            for lc in range(4):
                for c in range(7):
                    if W_s[lc][c] != 0:
                        W[c_map[lc]][base+c] = W_s[lc][c]
        
        # Part 2: A_right (2x1) * B_right (1x8), schoolbook.
        # A_right entries: A[0,2], A[1,2] -> global indices 4, 5
        # B_right entries: B[2,0..7] -> global indices 16..23
        # C_right entries: C[0,0..7], C[1,0..7] -> global indices 0..7, 8..15
        
        base = 28
        for i in range(2):
            for k in range(8):
                c = base + i*8 + k
                U[4+i][c] = 1
                V[16+k][c] = 1
                W[k*2+i][c] = 1
        
        # Convert to Fractions
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        
        return {"U": Uf, "V": Vf, "W": Wf}

    best = construct_2x3x8()
    best_rank = 44

    # Verify the decomposition
    def verify_decomp(decomp):
        U = decomp["U"]
        V = decomp["V"]
        W = decomp["W"]
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    if not verify_decomp(best):
        # Fallback to schoolbook
        best = schoolbook()
        best_rank = 48

    return best
# EVOLVE-BLOCK-END
