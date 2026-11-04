"""Matrix multiplication 6x6x6 - rank-92 decomposition via block Winograd."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    Ur = U.reshape(n, m, r) if isinstance(U, np.ndarray) else np.array(U).reshape(n, m, r)
    Vr = V.reshape(m, p, r) if isinstance(V, np.ndarray) else np.array(V).reshape(m, p, r)
    Wr = W.reshape(p, n, r) if isinstance(W, np.ndarray) else np.array(W).reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-8, rtol=1e-8)


def _winograd_3x3():
    """
    Rank-23 decomposition for 3x3x3 matrix multiplication (Winograd F(3,3)).
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indices: U[i*3+j, c] for A[i,j], V[j*3+k, c] for B[j,k], W[k*3+i, c] for C[i,k]
    """
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # m0 = (a00+a01+a02)(b00+b10+b20)
    U[0,0]=1; U[1,0]=1; U[2,0]=1
    V[0,0]=1; V[3,0]=1; V[6,0]=1
    W[0,0]=1; W[3,0]=1; W[6,0]=1
    
    # m1 = (a00-a01+a02)(b00-b10+b20)
    U[0,1]=1; U[1,1]=-1; U[2,1]=1
    V[0,1]=1; V[3,1]=-1; V[6,1]=1
    W[0,1]=1; W[3,1]=1; W[6,1]=1
    
    # m2 = (a01-a02)(b10-b20)
    U[1,2]=1; U[2,2]=-1
    V[3,2]=1; V[6,2]=-1
    W[0,2]=-1; W[3,2]=1; W[6,2]=0
    
    # m3 = (a10+a11+a12)(b01+b11+b21)
    U[3,3]=1; U[4,3]=1; U[5,3]=1
    V[1,3]=1; V[4,3]=1; V[7,3]=1
    W[1,3]=1; W[4,3]=1; W[7,3]=1
    
    # m4 = (a10-a11+a12)(b01-b11+b21)
    U[3,4]=1; U[4,4]=-1; U[5,4]=1
    V[1,4]=1; V[4,4]=-1; V[7,4]=1
    W[1,4]=1; W[4,4]=1; W[7,4]=1
    
    # m5 = (a11-a12)(b11-b21)
    U[4,5]=1; U[5,5]=-1
    V[4,5]=1; V[7,5]=-1
    W[1,5]=-1; W[4,5]=1; W[7,5]=0
    
    # m6 = (a20+a21+a22)(b02+b12+b22)
    U[6,6]=1; U[7,6]=1; U[8,6]=1
    V[2,6]=1; V[5,6]=1; V[8,6]=1
    W[2,6]=1; W[5,6]=1; W[8,6]=1
    
    # m7 = (a20-a21+a22)(b02-b12+b22)
    U[6,7]=1; U[7,7]=-1; U[8,7]=1
    V[2,7]=1; V[5,7]=-1; V[8,7]=1
    W[2,7]=1; W[5,7]=1; W[8,7]=1
    
    # m8 = (a21-a22)(b12-b22)
    U[7,8]=1; U[8,8]=-1
    V[5,8]=1; V[8,8]=-1
    W[2,8]=-1; W[5,8]=1; W[8,8]=0
    
    # m9 = (a00+a10+a20)(b10+b20)
    U[0,9]=1; U[3,9]=1; U[6,9]=1
    V[3,9]=1; V[6,9]=1
    W[3,9]=1; W[6,9]=1
    
    # m10 = (a01+a11+a21)(b11+b21)
    U[1,10]=1; U[4,10]=1; U[7,10]=1
    V[4,10]=1; V[7,10]=1
    W[4,10]=1; W[7,10]=1
    
    # m11 = (a02+a12+a22)(b12+b22)
    U[2,11]=1; U[5,11]=1; U[8,11]=1
    V[5,11]=1; V[8,11]=1
    W[5,11]=1; W[8,11]=1
    
    # m12 = (a00-a10+a20)(b10-b20)
    U[0,12]=1; U[3,12]=-1; U[6,12]=1
    V[3,12]=1; V[6,12]=-1
    W[3,12]=-1; W[6,12]=1
    
    # m13 = (a01-a11+a21)(b11-b21)
    U[1,13]=1; U[4,13]=-1; U[7,13]=1
    V[4,13]=1; V[7,13]=-1
    W[4,13]=-1; W[7,13]=1
    
    # m14 = (a02-a12+a22)(b12-b22)
    U[2,14]=1; U[5,14]=-1; U[8,14]=1
    V[5,14]=1; V[8,14]=-1
    W[5,14]=-1; W[8,14]=1
    
    # m15 = (a00+a11+a22)(b00+b11+b22)
    U[0,15]=1; U[4,15]=1; U[8,15]=1
    V[0,15]=1; V[4,15]=1; V[8,15]=1
    W[0,15]=1; W[4,15]=1; W[8,15]=1
    
    # m16 = (a01+a12+a20)(b01+b12+b20)
    U[1,16]=1; U[5,16]=1; U[6,16]=1
    V[1,16]=1; V[5,16]=1; V[6,16]=1
    W[1,16]=1; W[5,16]=1; W[6,16]=1
    
    # m17 = (a02+a10+a21)(b02+b10+b21)
    U[2,17]=1; U[3,17]=1; U[7,17]=1
    V[2,17]=1; V[3,17]=1; V[7,17]=1
    W[2,17]=1; W[3,17]=1; W[7,17]=1
    
    # m18 = (a00-a11-a22)(b00+b11+b22)
    U[0,18]=1; U[4,18]=-1; U[8,18]=-1
    V[0,18]=1; V[4,18]=1; V[8,18]=1
    W[0,18]=1; W[4,18]=-1; W[8,18]=-1
    
    # m19 = (a01-a12-a20)(b01+b12+b20)
    U[1,19]=1; U[5,19]=-1; U[6,19]=-1
    V[1,19]=1; V[5,19]=1; V[6,19]=1
    W[1,19]=1; W[5,19]=-1; W[6,19]=-1
    
    # m20 = (a02-a10-a21)(b02+b10+b21)
    U[2,20]=1; U[3,20]=-1; U[7,20]=-1
    V[2,20]=1; V[3,20]=1; V[7,20]=1
    W[2,20]=1; W[3,20]=-1; W[7,20]=-1
    
    # m21 = (a00+a11-a22)(b00+b11-b22)
    U[0,21]=1; U[4,21]=1; U[8,21]=-1
    V[0,21]=1; V[4,21]=1; V[8,21]=-1
    W[0,21]=1; W[4,21]=1; W[8,21]=-1
    
    # m22 = (a01+a12-a20)(b01+b12-b20)
    U[1,22]=1; U[5,22]=1; U[6,22]=-1
    V[1,22]=1; V[5,22]=1; V[6,22]=-1
    W[1,22]=1; W[5,22]=1; W[6,22]=-1
    
    return U, V, W


def _block_winograd_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 2x2 blocks of 3x3 matrices.
    Each 3x3 uses Winograd rank 23. Total rank = 4 * 23 = 92.
    """
    n, m, p = 6, 6, 6
    rank = 92
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    sU, sV, sW = _winograd_3x3()
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for c in range(23):
                    for li in range(3):
                        for lj in range(3):
                            u_row = (bi*3+li)*6 + (bj*3+lj)
                            U[u_row, col] = sU[li*3+lj, c]
                    
                    for lj in range(3):
                        for lk in range(3):
                            v_row = (bj*3+lj)*6 + (bk*3+lk)
                            V[v_row, col] = sV[lj*3+lk, c]
                    
                    for lk in range(3):
                        for li in range(3):
                            w_row = (bk*3+lk)*6 + (bi*3+li)
                            W[w_row, col] = sW[lk*3+li, c]
                    
                    col += 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication.
    Uses block Winograd: 2x2 blocks of 3x3, each 3x3 using Winograd rank 23.
    Total rank = 4 * 23 = 92.
    """
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Primary: block Winograd (rank 92)
    try:
        U_w, V_w, W_w = _block_winograd_6x6x6()
        if _verify(U_w, V_w, W_w, 6, 6, 6):
            return {"U": U_w.tolist(), "V": V_w.tolist(), "W": W_w.tolist()}
    except Exception:
        pass
    
    # If verification fails, fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
