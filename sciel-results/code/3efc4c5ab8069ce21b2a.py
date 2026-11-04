"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    
    # Vectorized Kronecker construction
    # U4[i*4+j, c0*7+c1] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
    # We can use np.kron or explicit loops which are fast enough for 16x49
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            u_col = U2[:, c0] * U2[:, c1]
            # Map 2x2 indices to 4x4 indices
            # i in 0..3, j in 0..3
            # i//2, j//2 are high bits, i%2, j%2 are low bits
            # Index in U2 is (i//2)*2 + (j//2) ? No.
            # U2 indices are (i,j) where i,j in 0,1.
            # U4 indices are (I,J) where I,J in 0..3.
            # I = 2*i + i', J = 2*j + j'.
            # U4[I, J] corresponds to U2[i, j] * U2[i', j']
            # So for a fixed c0, c1, we need to fill U4[:, c]
            # U4[I*4+J, c] = U2[i*2+j, c0] * U2[i'*2+j', c1]
            
            for i in range(2):
                for j in range(2):
                    for ip in range(2):
                        for jp in range(2):
                            I = 2*i + ip
                            J = 2*j + jp
                            U4[I*4+J, c] = U2[i*2+j, c0] * U2[ip*2+jp, c1]
                            
            # Similarly for V and W
            for j in range(2):
                for k in range(2):
                    for jp in range(2):
                        for kp in range(2):
                            J = 2*j + jp
                            K = 2*k + kp
                            V4[J*4+K, c] = V2[j*2+k, c0] * V2[jp*2+kp, c1]
                            
            for k in range(2):
                for i in range(2):
                    for kp in range(2):
                        for ip in range(2):
                            K = 2*k + kp
                            I = 2*i + ip
                            W4[K*4+I, c] = W2[k*2+i, c0] * W2[kp*2+ip, c1]
                            
    return U4, V4, W4

def _build_4x4x8_shared():
    """
    Build a rank-98 decomposition for 4x4 * 4x8.
    """
    U4, V4, W4 = _build_4x4x4_tensors()
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: k in 0..3
    U[:, 0:49] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = k_local
            V[j*8 + k_global, 0:49] = V4[j*4 + k_local, :]
    for k_local in range(4):
        k_global = k_local
        for i in range(4):
            W[k_global*4 + i, 0:49] = W4[k_local*4 + i, :]
            
    # Block 1: k in 4..7
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local
            V[j*8 + k_global, 49:98] = V4[j*4 + k_local, :]
    for k_local in range(4):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global*4 + i, 49:98] = W4[k_local*4 + i, :]
            
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8):
    """Verify exact decomposition using vectorized operations."""
    rank = U.shape[1]
    # Reshape to 3D tensors
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a contraction over r.
    # Result T[i,j,k]
    # We can do this via einsum or matmul
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Use np.einsum
    # U is (n, m, rank)
    # V is (m, p, rank)
    # W is (p, n, rank)
    # We want (n, m, p)
    
    # Note: W index is (k, i). So W_3d[k, i, r]
    # The term is U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    try:
        T = np.einsum('ijr,jkr,kir->ijp', U_3d, V_3d, W_3d)
        # Check if T is all ones
        if not np.all(T == 1):
            return False
        return True
    except Exception:
        return False

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    try:
        U, V, W = _build_4x4x8_shared()
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook rank-128
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int16)
    V = np.zeros((m*p, rank), dtype=np.int16)
    W = np.zeros((p*n, rank), dtype=np.int16)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
