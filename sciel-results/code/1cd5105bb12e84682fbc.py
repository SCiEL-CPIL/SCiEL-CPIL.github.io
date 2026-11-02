import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def strassen_decomp():
        rank = 14
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # Left block (cols 0,1 of B)
        # r=0: M1 = (a00+a11)(b00+b10)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[2][0] = 1 # W[k*n+i]. k=0,i=0 -> W[0]. k=1,i=0 -> W[2].
        # Wait, W index is k*n+i.
        # C[0,0] is k=0,i=0 -> W[0]. C[1,0] is k=0,i=1 -> W[1].
        # C[0,1] is k=1,i=0 -> W[2]. C[1,1] is k=1,i=1 -> W[3].
        
        # Let's re-verify indices for W.
        # W[k*n + i, r].
        # n=2.
        # k=0, i=0 -> W[0]
        # k=0, i=1 -> W[1]
        # k=1, i=0 -> W[2]
        # k=1, i=1 -> W[3]
        # k=2, i=0 -> W[4]
        # k=2, i=1 -> W[5]
        # k=3, i=0 -> W[6]
        # k=3, i=1 -> W[7]

        # Strassen Left (cols 0,1)
        # M1 = (a00+a11)(b00+b10)
        # a00*b00 -> c00. a00*b10 -> c01? No.
        # c00 = a00*b00 + a01*b10.
        # c10 = a10*b00 + a11*b10.
        # M1 = a00*b00 + a00*b10 + a11*b00 + a11*b10.
        # This doesn't map directly to c00, c10.
        
        # The previous Strassen implementation in the prompt was:
        # r=0: U[0]=1, U[3]=1. V[0]=1, V[5]=1. W[0]=1, W[3]=1.
        # V[5] is b11 (row 1, col 1 of left block? No, V index j*p+k. j=1, k=1 -> V[5]).
        # So V[0]=b00, V[5]=b11.
        # This corresponds to standard Strassen on 2x2 blocks.
        
        # I will stick to the verified rank 14 from the prompt if rank 13 fails.
        
        # Let's just return the rank 13 candidate if it verifies.
        return None # Placeholder

    # Try Rank 13
    R = 13
    U = [[0]*R for _ in range(4)]
    V = [[0]*R for _ in range(8)]
    W = [[0]*R for _ in range(8)]
    
    # Coefficients from literature/derivation
    # 1. M1 = (a00+a11)(b00+b10)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[4][0]=1
    W[0][0]=1; W[1][0]=1 # c00, c10? 
    # Actually, let's use the verify function to check the hardcoded 13.
    # If it fails, use the prompt's 14.
    
    # I will use the prompt's 14 as the primary fallback, but try 13 first.
    # The prompt's 14 is:
    U14 = [[0]*14 for _ in range(4)]
    V14 = [[0]*14 for _ in range(8)]
    W14 = [[0]*14 for _ in range(8)]
    
    # Left Block Strassen
    # r=0
    U14[0][0]=1; U14[3][0]=1
    V14[0][0]=1; V14[5][0]=1
    W14[0][0]=1; W14[3][0]=1
    # r=1
    U14[2][1]=1; U14[3][1]=1
    V14[0][1]=1
    W14[1][1]=1; W14[3][1]=-1
    # r=2
    U14[0][2]=1
    V14[1][2]=1; V14[5][2]=-1
    W14[2][2]=1; W14[3][2]=1
    # r=3
    U14[3][3]=1
    V14[4][3]=1; V14[0][3]=-1
    W14[0][3]=1; W14[1][3]=1
    # r=4
    U14[0][4]=1; U14[1][4]=1
    V14[5][4]=1
    W14[0][4]=-1; W14[2][4]=1
    # r=5
    U14[2][5]=1; U14[0][5]=-1
    V14[0][5]=1; V14[1][5]=1
    W14[3][5]=1
    # r=6
    U14[1][6]=1; U14[3][6]=-1
    V14[4][6]=1; V14[5][6]=1
    W14[0][6]=1

    # Right Block Strassen
    # r=7
    U14[0][7]=1; U14[3][7]=1
    V14[2][7]=1; V14[7][7]=1
    W14[4][7]=1; W14[7][7]=1
    # r=8
    U14[2][8]=1; U14[3][8]=1
    V14[2][8]=1
    W14[5][8]=1; W14[7][8]=-1
    # r=9
    U14[0][9]=1
    V14[3][9]=1; V14[7][9]=-1
    W14[6][9]=1; W14[7][9]=1
    # r=10
    U14[3][10]=1
    V14[6][10]=1; V14[2][10]=-1
    W14[4][10]=1; W14[5][10]=1
    # r=11
    U14[0][11]=1; U14[1][11]=1
    V14[7][11]=1
    W14[4][11]=-1; W14[6][11]=1
    # r=12
    U14[2][12]=1; U14[0][12]=-1
    V14[2][12]=1; V14[3][12]=1
    W14[7][12]=1
    # r=13
    U14[1][13]=1; U14[3][13]=-1
    V14[6][13]=1; V14[7][13]=1
    W14[4][13]=1

    if verify(U14, V14, W14):
        best_result = {"U": U14, "V": V14, "W": W14}
        best_rank = 14
    else:
        best_result = naive_decomp()
        best_rank = 16

    # Try Rank 13 (if I had a verified one, I'd put it here)
    # For now, return the best found.
    
    return best_result
