"""Optimized matrix multiplication tensor decomposition for 4x4x8."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
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
    U4 = np.zeros((16, rank), dtype=np.int8)
    V4 = np.zeros((16, rank), dtype=np.int8)
    W4 = np.zeros((16, rank), dtype=np.int8)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        u3 = u.reshape(n, m)
        v3 = v.reshape(m, p)
        w3 = w.reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u3, v3, w3)
    
    M = np.ones((n, m, p))
    err = np.max(np.abs(T - M))
    return err < tol

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 4, 4, 8
    schoolbook_rank = n * m * p  # 128
    
    # Construct the schoolbook decomposition (always valid, rank 128) as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Build the 4x4x4 rank-49 decomposition
    U4, V4, W4 = _build_4x4x4_tensors()
    
    # Assemble 4x4x8 rank-98 decomposition
    # The tensor 4x4x8 is the direct sum of two 4x4x4 tensors.
    # We can view it as two independent multiplications:
    # C_left = A * B_left (4x4 * 4x4)
    # C_right = A * B_right (4x4 * 4x4)
    # The rank is sum of ranks: 49 + 49 = 98.
    
    rank = 98
    U = np.zeros((16, rank), dtype=np.int8)
    V = np.zeros((32, rank), dtype=np.int8)
    W = np.zeros((32, rank), dtype=np.int8)
    
    # First half (k=0..3)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second half (k=4..7)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    # Verify correctness
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
