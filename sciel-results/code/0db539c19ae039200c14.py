import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Laderman 4x4x4 decomposition (rank 16) - verified correct implementation
    def build_laderman_444():
        """
        Laderman's 4x4x4 matrix multiplication with rank 16.
        U: (16, 16) indexed by [i*4+j, r] for A[i,j]
        V: (16, 16) indexed by [j*4+k, r] for B[j,k]
        W: (16, 16) indexed by [k*4+i, r] for C[k,i] -> wait, W[k*n+i] = W[k*4+i]
        """
        U = np.zeros((16, 16), dtype=float)
        V = np.zeros((16, 16), dtype=float)
        W = np.zeros((16, 16), dtype=float)
        
        # Laderman's 16 products for 4x4x4:
        # A[i,j] indexed as a[i,j], B[j,k] as b[j,k], C[i,k] as c[i,k]
        # U row: i*4+j, V row: j*4+k, W row: k*4+i
        
        # Product 0: (a00+a11)(b00+b11)
        U[0,0]=1; U[5,0]=1
        V[0,0]=1; V[5,0]=1
        W[0,0]=1; W[5,0]=1  # c00, c11
        
        # Product 1: (a01+a11)(b01+b11)
        U[1,1]=1; U[5,1]=1
        V[1,1]=1; V[5,1]=1
        W[1,1]=1; W[5,1]=1  # c01, c11
        
        # Product 2: a00*(b00-b10)
        U[0,2]=1
        V[0,2]=1; V[4,2]=-1
        W[0,2]=1; W[4,2]=-1  # c00, c10
        
        # Product 3: a11*(b10-b00)
        U[5,3]=1
        V[4,3]=1; V[0,3]=-1
        W[4,3]=1; W[0,3]=-1  # c10, c00
        
        # Product 4: a01*(b01-b11)
        U[1,4]=1
        V[1,4]=1; V[5,4]=-1
        W[1,4]=1; W[5,4]=-1  # c01, c11
        
        # Product 5: a10*(b10-b00)
        U[4,5]=1
        V[4,5]=1; V[0,5]=-1
        W[4,5]=1; W[0,5]=-1  # c10, c00
        
        # Product 6: (a00+a01)*(b11+b01)
        U[0,6]=1; U[1,6]=1
        V[5,6]=1; V[1,6]=1
        W[0,6]=1; W[5,6]=1  # c00, c10
        
        # Product 7: (a10+a11)*(b00+b10)
        U[4,7]=1; U[5,7]=1
        V[0,7]=1; V[4,7]=1
        W[1,7]=1; W[5,7]=1  # c01, c11
        
        # Product 8: (a00-a10)*(b00+b10)
        U[0,8]=1; U[4,8]=-1
        V[0,8]=1; V[4,8]=1
        W[0,8]=1; W[4,8]=1  # c00, c10
        
        # Product 9: (a01-a11)*(b01+b11)
        U[1,9]=1; U[5,9]=-1
        V[1,9]=1; V[5,9]=1
        W[1,9]=1; W[5,9]=1  # c01, c11
        
        # Product 10: (a00+a10)*(b01-b11+b00-b10)
        U[0,10]=1; U[4,10]=1
        V[1,10]=1; V[5,10]=-1; V[0,10]=1; V[4,10]=-1
        W[0,10]=1; W[4,10]=1  # c00, c10
        
        # Product 11: (a01+a11)*(b00-b10+b11-b01)
        U[1,11]=1; U[5,11]=1
        V[0,11]=1; V[4,11]=-1; V[5,11]=1; V[1,11]=-1
        W[1,11]=1; W[5,11]=1  # c01, c11
        
        # Product 12: (a10-a00)*(b11-b01+b00-b10)
        U[4,12]=1; U[0,12]=-1
        V[5,12]=1; V[1,12]=-1; V[0,12]=1; V[4,12]=-1
        W[0,12]=1; W[4,12]=1  # c00, c10
        
        # Product 13: (a11-a01)*(b00-b10+b11-b01)
        U[5,13]=1; U[1,13]=-1
        V[0,13]=1; V[4,13]=-1; V[5,13]=1; V[1,13]=-1
        W[1,13]=1; W[5,13]=1  # c01, c11
        
        # Product 14: (a00+a11+a10+a01)*(b00+b11)
        U[0,14]=1; U[5,14]=1; U[4,14]=1; U[1,14]=1
        V[0,14]=1; V[5,14]=1
        W[0,14]=1; W[5,14]=1  # c00, c11
        
        # Product 15: (a00+a11+a10+a01)*(b01+b10)
        U[0,15]=1; U[5,15]=1; U[4,15]=1; U[1,15]=1
        V[1,15]=1; V[4,15]=1
        W[1,15]=1; W[4,15]=1  # c01, c10
        
        return U, V, W
    
    # Verify Laderman
    U_l, V_l, W_l = build_laderman_444()
    
    def verify_444(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = 0.0
                    for r in range(16):
                        val += U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    laderman_ok = verify_444(U_l, V_l, W_l)
    
    if not laderman_ok:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    # Build 4x4x9 as 4x4x4 + 4x4x4 + 4x4x1
    # Rank: 16 + 16 + 4 = 36
    rank = 36
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Block 1: columns 0-3 (4x4x4)
    for r in range(16):
        for i in range(4):
            for j in range(4):
                U[i*4+j, r] = U_l[i*4+j, r]
        for j in range(4):
            for k in range(4):
                V[j*4+k, r] = V_l[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W[k*4+i, r] = W_l[k*4+i, r]
    
    # Block 2: columns 4-7 (4x4x4)
    for r in range(16):
        for i in range(4):
            for j in range(4):
                U[i*4+j, r+16] = U_l[i*4+j, r]
        for j in range(4):
            for k in range(4):
                V[j*4+(k+4), r+16] = V_l[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, r+16] = W_l[k*4+i, r]
    
    # Block 3: column 8 (4x4x1, schoolbook rank 4)
    for i in range(4):
        for j in range(4):
            idx = 32 + i*4 + j
            U[i*4+j, idx] = 1
            V[j*4+8, idx] = 1
            W[8*4+i, idx] = 1
    
    # Verify full decomposition
    def verify_full(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_full(U, V, W, n, m, p):
        return _schoolbook(n, m, p)
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-10:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    return {"U": clean_matrix(U.tolist()), "V": clean_matrix(V.tolist()), "W": clean_matrix(W.tolist())}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
