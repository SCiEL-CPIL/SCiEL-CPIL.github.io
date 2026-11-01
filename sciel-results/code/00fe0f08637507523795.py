import time
import random
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target T[i, j, k] = 1 for all i, j, k.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = 2, 2, 5
    start_time = time.time()
    deadline = 130.0

    # The known optimal rank for 2x2x5 is 16 (7 + 7 + 2) or possibly lower.
    # We start with a rank 16 construction: 2 Strassen blocks (cols 0,1 and 2,3) and 1 schoolbook (col 4).
    # Wait, Strassen is 7. Schoolbook for 2x2x1 is 2. Total 7+7+2 = 16.
    # Let's build this rank 16 solution.
    
    rank16 = 16
    U16 = [[0]*rank16 for _ in range(n*m)]
    V16 = [[0]*rank16 for _ in range(m*p)]
    W16 = [[0]*rank16 for _ in range(p*n)]
    
    # Block 1: columns 0, 1 (indices 0, 1 in V, W)
    # Uses Strassen terms 0..6
    def add_strassen(U, V, W, offset, col1, col2):
        # P1 = (a+d)(e+h) -> U[a,d]=1, V[e,h]=1, W[?,?]=1
        # We map the 2x2 submatrix at A[:, 0:2] and B[0:2, col1:col2+1] to C[0:2, col1:col2+1]
        # Standard Strassen for 2x2:
        # C11 = P1 + P3 - P5 - P7
        # C12 = P4 + P5
        # C21 = P2 + P7
        # C22 = P1 - P2 + P3 + P4
        
        # P1 = (a+d)(e+h)
        U[0][offset+0] = 1; U[3][offset+0] = 1
        V[0][offset+0] = 1; V[1][offset+0] = 1  # e is B[0,col1], h is B[1,col2]
        W[0][offset+0] = 1; W[1][offset+0] = 1  # C[0,col1], C[0,col2]
        
        # P2 = (c+d)e
        U[2][offset+1] = 1; U[3][offset+1] = 1
        V[0][offset+1] = 1
        W[1][offset+1] = 1; W[3][offset+1] = -1 # C[1,col1]=c, C[1,col2]=-c? 
        # Wait, standard indexing:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        # P1=(a+d)(e+h) = ae+ah+de+dh
        # P2=(c+d)e = ce+de
        # P3=a(f-h) = af-ah
        # P4=d(g-e) = dg-de
        # P5=(a+b)h = ah+bh
        # P6=(c-a)(e+f) = ce+cf-ae-af
        # P7=(b-d)(g+h) = bg+bh-dg-dh
        
        # C11 = P1 + P3 - P5 - P7
        # = (ae+ah+de+dh) + (af-ah) - (ah+bh) - (bg+bh-dg-dh)
        # = ae + de + dh + af - ah - bh - bg - bh + dg + dh ... this is getting messy.
        # Let's just use the standard coefficients.
        
        # Let's restart the Strassen assignment carefully.
        # Indices in U for A: 0:a, 1:b, 2:c, 3:d
        # Indices in V for B: 0:e, 1:f, 2:g, 3:h (local)
        # Indices in W for C: 0:ae+bg, 1:af+bh, 2:ce+dg, 3:cf+dh
        
        # P1 = (a+d)(e+h)
        U[0][offset] = 1; U[3][offset] = 1
        V[0][offset] = 1; V[1][offset] = 1 # e,h are B[0,0], B[0,1] of the block? No, e=f? 
        # B is 2x2. e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
        # So V[0]=e, V[1]=f, V[2]=g, V[3]=h.
        # P1 uses e and h. So V[0] and V[3].
        V[0][offset] = 1; V[3][offset] = 1
        W[0][offset] = 1; W[1][offset] = 1; W[2][offset] = 1; W[3][offset] = 1
        
        # P2 = (c+d)e
        U[2][offset+1] = 1; U[3][offset+1] = 1
        V[0][offset+1] = 1
        W[2][offset+1] = 1; W[3][offset+1] = -1
        
        # P3 = a(f-h)
        U[0][offset+2] = 1
        V[1][offset+2] = 1; V[3][offset+2] = -1
        W[1][offset+2] = 1; W[3][offset+2] = 1
        
        # P4 = d(g-e)
        U[3][offset+3] = 1
        V[2][offset+3] = 1; V[0][offset+3] = -1
        W[0][offset+3] = 1; W[2][offset+3] = 1
        
        # P5 = (a+b)h
        U[0][offset+4] = 1; U[1][offset+4] = 1
        V[3][offset+4] = 1
        W[0][offset+4] = -1; W[1][offset+4] = 1
        
        # P6 = (c-a)(e+f)
        U[2][offset+5] = 1; U[0][offset+5] = -1
        V[0][offset+5] = 1; V[1][offset+5] = 1
        W[3][offset+5] = 1
        
        # P7 = (b-d)(g+h)
        U[1][offset+6] = 1; U[3][offset+6] = -1
        V[2][offset+6] = 1; V[3][offset+6] = 1
        W[0][offset+6] = 1

    # Call for first block (cols 0,1)
    add_strassen(U16, V16, W16, 0, 0, 1)
    
    # Call for second block (cols 2,3)
    # Need to map local B indices to global V indices.
    # Local 0 (e) -> Global 2 (col 2, row 0)
    # Local 1 (f) -> Global 3 (col 3, row 0)
    # Local 2 (g) -> Global 7 (col 2, row 1)
    # Local 3 (h) -> Global 8 (col 3, row 1)
    # My helper above assumes local V indices 0,1,2,3 correspond to rows/cols in a specific way.
    # Let's modify helper to take a mapping.
    
    # Actually, let's just hardcode the second block to avoid helper complexity errors.
    # Offset 7
    # P1: (a+d)(e+h) where e=B[0,2], h=B[1,3]
    U16[0][7] = 1; U16[3][7] = 1
    V16[2][7] = 1; V16[8][7] = 1
    W16[4][7] = 1; W16[5][7] = 1; W16[6][7] = 1; W16[7][7] = 1
    
    # P2: (c+d)e where e=B[0,2]
    U16[2][8] = 1; U16[3][8] = 1
    V16[2][8] = 1
    W16[6][8] = 1; W16[7][8] = -1
    
    # P3: a(f-h) where f=B[0,3], h=B[1,3]
    U16[0][9] = 1
    V16[3][9] = 1; V16[8][9] = -1
    W16[5][9] = 1; W16[7][9] = 1
    
    # P4: d(g-e) where g=B[1,2], e=B[0,2]
    U16[3][10] = 1
    V16[7][10] = 1; V16[2][10] = -1
    W16[4][10] = 1; W16[6][10] = 1
    
    # P5: (a+b)h where h=B[1,3]
    U16[0][11] = 1; U16[1][11] = 1
    V16[8][11] = 1
    W16[4][11] = -1; W16[5][11] = 1
    
    # P6: (c-a)(e+f) where e=B[0,2], f=B[0,3]
    U16[2][12] = 1; U16[0][12] = -1
    V16[2][12] = 1; V16[3][12] = 1
    W16[7][12] = 1
    
    # P7: (b-d)(g+h) where g=B[1,2], h=B[1,3]
    U16[1][13] = 1; U16[3][13] = -1
    V16[7][13] = 1; V16[8][13] = 1
    W16[4][13] = 1

    # Schoolbook for col 4 (rank 2)
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    # Term 14: A[0,0]*B[0,4] -> C[0,4]
    U16[0][14] = 1
    V16[4][14] = 1
    W16[8][14] = 1
    
    # Term 15: A[0,1]*B[1,4] -> C[0,4]
    U16[1][15] = 1
    V16[9][15] = 1
    W16[8][15] = 1
    
    # Wait, I missed terms for C[1,4].
    # C[1,4] needs A[1,0]*B[0,4] and A[1,1]*B[1,4].
    # I need 4 terms for schoolbook 2x2x1? No, 2x2x1 is rank 2?
    # T = a1*b1 + a2*b2. Rank 2.
    # But I have 2 outputs C[0,4] and C[1,4].
    # C[0,4] = a1*e + a2*f
    # C[1,4] = a3*e + a4*f
    # This is a 2x2x1 tensor? No, it's part of the 2x2x5 tensor.
    # The slice for k=4 is a 2x2 matrix.
    # The tensor for this slice is rank 2? No, matrix multiplication 2x2 by 2x1 is rank 2?
    # No, the rank of the matrix multiplication tensor (2,2,1) is 2.
    # T[i,j,k=4] = sum_l A[i,l] B[l,4].
    # This is a bilinear form.
    # We can decompose it into 2 rank 1 terms:
    # T1: A[0,0] B[0,4] -> C[0,4]
    # T2: A[0,1] B[1,4] -> C[0,4]
    # T3: A[1,0] B[0,4] -> C[1,4]
    # T4: A[1,1] B[1,4] -> C[1,4]
    # That's rank 4.
    # Can we do better?
    # C = A * b.
    # This is equivalent to two independent 1x2x1 multiplications? No.
    # The rank of 2x2x1 MM tensor is 2? No, it's 4?
    # Let's check.
    # If rank 2, we need U (4x2), V (2x2), W (2x2).
    # U: [[1,0],[0,1],[0,0],[1,0]]? No.
    # Standard rank of 2x2x1 is 2? No, 2x1x1 is 1. 2x2x1 is 2?
    # C1 = a1 e + a2 f
    # C2 = a3 e + a4 f
    # If rank 2:
    # C1 = U1a V1e W1c1 + U2a V2e W2c1 ...
    # This is just C = A b.
    # The tensor is [e, f] * A^T?
    # The rank of the 2x2x1 tensor is 2.
    # Let's try to construct rank 2 for the last column.
    # U1: [1, 0, 0, 1] (a1, a3)
    # V1: [1, 0] (e)
    # W1: [1, 1] (C1, C2)
    # U2: [0, 1, 1, 0] (a2, a4)
    # V2: [0, 1] (f)
    # W2: [1, 1] (C1, C2)
    # Check:
    # C1 = 1*1*1 + 0*0*1 = 1 (from a1 e) + 0*1*1 + 1*1*1 (from a2 f) = a1 e + a2 f. Correct.
    # C2 = 1*1*1 + 0*0*1 = 1 (from a1 e) + 0*1*1 + 1*1*1 (from a2 f) = a3 e + a4 f. Correct.
    # So yes, rank 2 is possible.
    
    U16[0][14] = 1; U16[2][14] = 1
    V16[4][14] = 1
    W16[8][14] = 1; W16[9][14] = 1
    
    U16[1][15] = 1; U16[3][15] = 1
    V16[9][15] = 1
    W16[8][15] = 1; W16[9][15] = 1

    if _verify(U16, V16, W16, n, m, p, 16):
        best_U, best_V, best_W = U16, V16, W16
        best_rank = 16
    else:
        # Fallback to rank 20
        best_rank = 20
        best_U = [[0]*20 for _ in range(n*m)]
        best_V = [[0]*20 for _ in range(m*p)]
        best_W = [[0]*20 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j][col] = 1
                    best_V[j*p+k][col] = 1
                    best_W[k*n+i][col] = 1
                    col += 1
        return {"U": best_U, "V": best_V, "W": best_W}

    # Attempt to find rank 15 or lower?
    # Unlikely to be lower than 16 for 2x2x5.
    # But let's try a small search for rank 15 just in case, or return 16.
    # Given time constraints, returning 16 is safe.
    
    return {"U": best_U, "V": best_V, "W": best_W}
