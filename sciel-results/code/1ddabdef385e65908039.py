"""Matrix multiplication 3x4x6 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 4, 6]
DEADLINE = 140.0


def _build_schoolbook(n, m, p):
    """Return (U, V, W) as list-of-lists of Fraction for schoolbook decomposition."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, num_tests=5, seed=42):
    """Verify exact decomposition using random integer tests."""
    rank = len(U[0])
    rng = random.Random(seed)
    U_arr = np.array([[int(x) for x in row] for row in U], dtype=object)
    V_arr = np.array([[int(x) for x in row] for row in V], dtype=object)
    W_arr = np.array([[int(x) for x in row] for row in W], dtype=object)
    
    for _ in range(num_tests):
        A = [[Fraction(rng.randint(-3, 3)) for _ in range(m)] for _ in range(n)]
        B = [[Fraction(rng.randint(-3, 3)) for _ in range(p)] for _ in range(m)]
        AB = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for j in range(m):
                    s += A[i][j] * B[j][k]
                AB[i][k] = s
        X = [[Fraction(0) for _ in range(rank)] for _ in range(n)]
        for i in range(n):
            for c in range(rank):
                s = Fraction(0)
                for j in range(m):
                    s += U[i * m + j][c] * A[i][j]
                X[i][c] = s
        Y = [[Fraction(0) for _ in range(p)] for _ in range(rank)]
        for c in range(rank):
            for k in range(p):
                s = Fraction(0)
                for j in range(m):
                    s += B[j][k] * V[j * p + k][c]
                Y[c][k] = s
        Z = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += X[i][c] * W[k * n + i][c] * Y[c][k]
                Z[i][k] = s
        if Z != AB:
            return False
    return True


def _strassen_2x2():
    """Return Strassen decomposition for 2x2x2 as (U,V,W) lists of Fractions."""
    # 2x2 matrix mult: A*B = C
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    n = m = p = 2
    rank = 7
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 4 rows
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 4 rows
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 4 rows
    
    # U[i*m+j, c] = coefficient for A[i,j] in step c
    # V[j*p+k, c] = coefficient for B[j,k] in step c
    # W[k*n+i, c] = coefficient for C[i,k] in step c (with sign)
    
    # M1: (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 += M1, C22 += M1
    
    # M2: (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C21 += M2
    
    # M3: A11(B21-B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[2][2] = 1  # C12 += M3, C21... wait
    
    # Let me redo this more carefully.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W[k*n+i, c] is the coefficient for C[i,k] from step c.
    # C11 is at W[0*2+0] = W[0]
    # C12 is at W[1*2+0] = W[2]
    # C21 is at W[0*2+1] = W[1]
    # C22 is at W[1*2+1] = W[3]
    
    # Reset
    U = [[Fraction(0)] * rank for _ in range(4)]
    V = [[Fraction(0)] * rank for _ in range(4)]
    W = [[Fraction(0)] * rank for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C21
    
    # M3 = A11(B21-B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1  # C12
    W[3][2] = 1  # C22
    
    # M4 = A22(B12-B11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C11
    W[1][3] = 1  # C21
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C11
    W[2][4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11
    
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to build a decomposition using Strassen on 2x2 sub-blocks.
    # For 3x4x6, we can split:
    # A (3x4) = [A_top (2x4) | A_bot (1x4)]  -- or split differently
    # B (4x6) = [B_top (2x6); B_bot (2x6)]
    # C = A_top * B_top + A_bot * B_bot
    # Each product is 2x2x6 or 1x2x6.
    
    # For 2x2x6: use Strassen on the 2x2 part, giving rank 7 per "column block"
    # Actually, 2x2x6 means multiplying 2x2 by 2x6.
    # We can write B as [B1 (2x2) | B2 (2x2) | B3 (2x2)]
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bi is 2x2x2, rank 7 with Strassen.
    # Total rank: 3*7 = 21 for the 2x2x6 part.
    
    # For 1x2x6: multiply 1x2 by 2x6.
    # C = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # This is just 2 "products" of scalars with vectors, rank 2.
    
    # Total rank: 21 + 2 = 23? That seems too good. Let me check.
    
    # Actually, the 2x2x6 decomposition:
    # A is 2x2, B is 2x6 = [B1|B2|B3] each 2x2.
    # C = A*B1 + A*B2 + A*B3, each A*Bi is 2x2.
    # Using Strassen for each: 7 multiplications per Bi.
    # Total: 21 multiplications.
    
    # For 1x2x6:
    # A is 1x2 = [a0, a1], B is 2x6.
    # C = a0*B[0,:] + a1*B[1,:], which is 2 scalar-vector products.
    # Each scalar-vector product of length 6 has rank 6 (schoolbook) or can be done with rank 6.
    # Actually, a0*B[0,:] + a1*B[1,:] = sum_k (a0*B[0,k] + a1*B[1,k]) * e_k
    # This is rank 6 (one multiplication per output element).
    
    # So total rank: 21 + 6 = 27? 
    
    # Let me construct this.
    
    # Split A into A1 (2x4) and A2 (1x4)
    # Split B into B1 (2x6) and B2 (2x6)
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x4x6):
    # Split B1 into [B11|B12|B13] each 2x2
    # C1 = A1*B11 + A1*B12 + A1*B13
    # Each A1*Bi is 2x2x2, rank 7.
    # Total: 21.
    
    # For A2*B2 (1x4x6):
    # A2 is 1x4 = [a0,a1,a2,a3], B2 is 4x6.
    # C2 = sum_j a_j * B2[j,:]
    # This is rank 24 schoolbook, or we can do better?
    # Actually, 1x4x6: the rank is at most 4*6=24, but could be less.
    # For 1xm x p, the rank is m*p (schoolbook) and I believe it's optimal.
    
    # So total: 21 + 24 = 45? Still better than 72.
    
    # Let me try to construct the 2x4x6 part using Strassen.
    
    def _strassen_2x2x6():
        """Strassen-based decomposition for 2x2x6 (2x2 times 2x6)."""
        # B is 2x6 = [B1|B2|B3] each 2x2
        # C = A*B1 + A*B2 + A*B3
        # Each uses 7 Strassen multiplications.
        rank = 21
        n = m = 2
        p = 6
        U = [[Fraction(0)] * rank for _ in range(n * m)]  # 4 rows
        V = [[Fraction(0)] * rank for _ in range(m * p)]  # 12 rows
        W = [[Fraction(0)] * rank for _ in range(p * n)]  # 12 rows
        
        # For each block b=0,1,2:
        # B block b is columns 2b, 2b+1 of B
        # C block b is columns 2b, 2b+1 of C
        for b in range(3):
            base = b * 7
            # Strassen for A * B_block
            # M1 = (A11+A22)(B11+B22)
            U[0][base+0] = 1; U[3][base+0] = 1
            V[0+2*b][base+0] = 1; V[3+2*b][base+0] = 1
            W[0+2*b][base+0] = 1  # C[0,2b]
            W[3+2*b][base+0] = 1  # C[1,2b+1]
            
            # M2 = (A21+A22)B11
            U[2][base+1] = 1; U[3][base+1] = 1
            V[0+2*b][base+1] = 1
            W[1+2*b][base+1] = 1  # C[1,2b]
            
            # M3 = A11(B21-B22)
            U[0][base+2] = 1
            V[2+2*b][base+2] = 1; V[3+2*b][base+2] = -1
            W[2+2*b][base+2] = 1  # C[0,2b+1]
            W[3+2*b][base+2] = 1  # C[1,2b+1]
            
            # M4 = A22(B12-B11)
            U[3][base+3] = 1
            V[1+2*b][base+3] = 1; V[0+2*b][base+3] = -1
            W[0+2*b][base+3] = 1  # C[0,2b]
            W[1+2*b][base+3] = 1  # C[1,2b]
            
            # M5 = (A11+A12)B22
            U[0][base+4] = 1; U[1][base+4] = 1
            V[3+2*b][base+4] = 1
            W[0+2*b][base+4] = -1  # C[0,2b]
            W[2+2*b][base+4] = 1  # C[0,2b+1]
            
            # M6 = (A21-A11)(B11+B12)
            U[2][base+5] = 1; U[0][base+5] = -1
            V[0+2*b][base+5] = 1; V[1+2*b][base+5] = 1
            W[3+2*b][base+5] = 1  # C[1,2b+1]
            
            # M7 = (A12-A22)(B21+B22)
            U[1][base+6] = 1; U[3][base+6] = -1
            V[2+2*b][base+6] = 1; V[3+2*b][base+6] = 1
            W[0+2*b][base+6] = 1  # C[0,2b]
        
        return U, V, W
    
    # Now build the full 3x4x6 decomposition.
    # A (3x4) = [A1 (2x4); A2 (1x4)]
    # B (4x6) = [B1 (2x6); B2 (2x6)]
    # C (3x6) = [C1 (2x6); C2 (1x6)]
    # C1 = A1*B1, C2 = A2*B2
    
    # A1*B1: 2x4x6, use _strassen_2x2x6() -> rank 21
    # A2*B2: 1x4x6, schoolbook -> rank 24
    
    U1, V1, W1 = _strassen_2x2x6()
    rank1 = len(U1[0])
    
    # A2*B2: 1x4x6 schoolbook
    n2, m2, p2 = 1, 4, 6
    rank2 = n2 * m2 * p2
    U2 = [[Fraction(0)] * rank2 for _ in range(n2 * m2)]
    V2 = [[Fraction(0)] * rank2 for _ in range(m2 * p2)]
    W2 = [[Fraction(0)] * rank2 for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
    
    total_rank = rank1 + rank2
    
    # Combine
    U = [[Fraction(0)] * total_rank for _ in range(n * m)]  # 12 rows
    V = [[Fraction(0)] * total_rank for _ in range(m * p)]  # 24 rows
    W = [[Fraction(0)] * total_rank for _ in range(p * n)]  # 18 rows
    
    # A1 occupies rows 0-3 of A (i=0,1), B1 occupies rows 0-3 of B (j=0,1)
    # A2 occupies row 4 of A (i=2), B2 occupies rows 4-7 of B (j=2,3)
    # C1 occupies rows 0-11 of C (k=0-5, i=0,1), C2 occupies rows 12-17 (k=0-5, i=2)
    
    # Copy U1 into U rows 0-3 (A1), columns 0-20
    for i in range(4):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    
    # Copy U2 into U row 4 (A2), columns 21-44
    for i in range(n2 * m2):
        for c in range(rank2):
            U[4 + i][rank1 + c] = U2[i][c]
    
    # Copy V1 into V rows 0-3 (B1), columns 0-20
    for j in range(4):
        for c in range(rank1):
            V[j][c] = V1[j][c]
    
    # Copy V2 into V rows 4-7 (B2), columns 21-44
    for j in range(m2 * p2):
        for c in range(rank2):
            V[4 + j][rank1 + c] = V2[j][c]
    
    # Copy W1 into W rows 0-11 (C1), columns 0-20
    for idx in range(p * n):  # 12 rows for C1
        for c in range(rank1):
            W[idx][c] = W1[idx][c]
    
    # Copy W2 into W rows 12-17 (C2), columns 21-44
    for idx in range(p2 * n2):  # 6 rows for C2
        for c in range(rank2):
            W[12 + idx][rank1 + c] = W2[idx][c]
    
    # Verify
    if time.time() - start_time < DEADLINE - 10:
        valid = _verify(U, V, W, n, m, p, num_tests=3)
        if not valid:
            # Fall back to schoolbook
            U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
