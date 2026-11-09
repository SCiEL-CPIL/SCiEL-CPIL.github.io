"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start = time.time()

def _deadline(seconds_before=10.0):
    """Return True if we are past the deadline."""
    return (time.time() - _start) > (150.0 - seconds_before)

def _strassen_2x2():
    """Strassen 2x2x2 rank-7 decomposition.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]"""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    
    # r=1: M2 = (c+d)*e -> C[1,0]+=1, C[1,1]-=1
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1; W[3][1]=-1
    
    # r=2: M3 = a*(f-h) -> C[0,1]+=1, C[1,1]+=1
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1; W[3][2]=1
    
    # r=3: M4 = d*(g-e) -> C[0,0]+=1, C[1,0]+=1
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[1][3]=1
    
    # r=4: M5 = (a+b)*h -> C[0,0]-=1, C[0,1]+=1
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[2][4]=1
    
    # r=5: M6 = (c-a)*(e+f) -> C[1,1]+=1
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # r=6: M7 = (b-d)*(g+h) -> C[0,0]+=1
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    
    return U, V, W

def _build_4x4x4():
    """Build 4x4x4 decomposition using Kronecker product of two Strassen 2x2x2.
    Rank = 49."""
    U2, V2, W2 = _strassen_2x2()
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            i = i1*2 + i0
                            j = j1*2 + j0
                            U4[i*4+j][r] = U2[i1*2+j1][r1] * U2[i0*2+j0][r2]
    
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            j = j1*2 + j0
                            k = k1*2 + k0
                            V4[j*4+k][r] = V2[j1*2+k1][r1] * V2[j0*2+k0][r2]
    
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            k = k1*2 + k0
                            i = i1*2 + i0
                            W4[k*4+i][r] = W2[k1*2+i1][r1] * W2[k0*2+i0][r2]
    
    return U4, V4, W4

def construct():
    """Construct rank-minimal decomposition for 4x4x8 matrix multiplication.
    Returns dict with U (16xrank), V (32xrank), W (32xrank)."""
    n, m, p = 4, 4, 8
    
    # Build 4x4x4 decomposition (rank 49)
    U4, V4, W4 = _build_4x4x4()
    rank4 = 49
    
    # For 4x4x8: split into two 4x4x4 halves
    # Rank = 2 * 49 = 98
    rank = 2 * rank4
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Fill U: same coefficients for both halves
    for i in range(n):
        for j in range(m):
            for r in range(rank4):
                U[i*m+j][r] = U4[i*m+j][r]
                U[i*m+j][rank4+r] = U4[i*m+j][r]
    
    # Fill V: left half (k=0..3) and right half (k=4..7)
    for j in range(m):
        for k in range(p):
            if k < 4:
                for r in range(rank4):
                    V[j*p+k][r] = V4[j*4+k][r]
            else:
                for r in range(rank4):
                    V[j*p+k][rank4+r] = V4[j*4+(k-4)][r]
    
    # Fill W: left half (k=0..3) and right half (k=4..7)
    for k in range(p):
        for i in range(n):
            if k < 4:
                for r in range(rank4):
                    W[k*n+i][r] = W4[k*4+i][r]
            else:
                for r in range(rank4):
                    W[k*n+i][rank4+r] = W4[(k-4)*4+i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
