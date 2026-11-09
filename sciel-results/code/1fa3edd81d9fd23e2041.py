import time
import numpy as np

DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _build_rank_40():
    n, m, p = 2, 4, 5
    rank = 40
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _build_rank_36():
    n, m, p = 2, 4, 5
    # Strassen decomposition for 2x2x2 blocks
    # Each entry is a 2x2 matrix flattened as [a,b,c,d] = [[a,b],[c,d]]
    strassen_U = [
        [1,0,0,1],   # M1: (A11+A22)(B11+B22) -> U contribution
        [0,0,1,1],   # M2: (A21+A22)B11
        [1,0,0,0],   # M3: A11(B21-B22)
        [0,0,0,1],   # M4: A22(B11-B12)
        [1,1,0,0],   # M5: (A11+A12)B22
        [-1,0,1,0],  # M6: (A21-A11)(B11+B12)
        [0,1,0,-1],  # M7: (A12-A22)(B21+B22)
    ]
    strassen_V = [
        [1,0,0,1],   # M1
        [1,0,0,0],   # M2
        [0,1,0,-1],  # M3
        [-1,0,1,0],  # M4
        [0,0,0,1],   # M5
        [1,1,0,0],   # M6
        [0,0,1,1],   # M7
    ]
    strassen_W = [
        [1,0,0,1],   # M1
        [0,1,0,-1],  # M2
        [0,0,1,1],   # M3
        [1,1,0,0],   # M4
        [-1,0,1,0],  # M5
        [0,0,0,1],   # M6
        [1,0,0,0],   # M7
    ]
    
    # 4 blocks of 2x2x2 (rank 7 each) + 8 terms for the 5th column
    rank = 4*7 + 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Handle j in {0,1} and {2,3}
    for j0 in (0, 2):
        # Handle k in {0,1} and {2,3}
        for k0 in (0, 2):
            for r in range(7):
                # U part: 2x2 block (rows i=0,1; cols j=j0,j0+1)
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                # V part: 2x2 block (rows j=j0,j0+1; cols k=k0,k0+1)
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                # W part: 2x2 block (rows k=k0,k0+1; cols i=0,1)
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                col += 1
        
    # Handle k=4 (last column of B)
    # This requires 8 terms: A[i,j] * B[j,4]
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W):
    n, m, p = 2, 4, 5
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w).reshape(n*m, m*p, p*n)
        
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[i*m+j, j*p+k, k*n+i] = 1.0
                
    return np.allclose(T, Texp, atol=1e-9)

def construct():
    n, m, p = 2, 4, 5
    
    # Try Rank 36 first
    try:
        U, V, W = _build_rank_36()
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # Fallback to Rank 40
    U, V, W = _build_rank_40()
    return {"U": U, "V": V, "W": W}
