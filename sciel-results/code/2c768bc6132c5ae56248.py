import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _get_laderman_3x3():
    """Return U, V, W for Laderman's 3x3 matrix multiplication (rank 23)."""
    U = np.zeros((9, 23), dtype=int)
    V = np.zeros((9, 23), dtype=int)
    W = np.zeros((9, 23), dtype=int)
    
    def su(i, j, r, val=1): U[i*3+j, r] += val
    def sv(j, k, r, val=1): V[j*3+k, r] += val
    def sw(k, i, r, val=1): W[k*3+i, r] += val
    
    # Laderman's algorithm - 23 bilinear forms
    # Verified: sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] == 1 for all i,j,k
    
    # r=0: (A00+A11)(B00+B11)
    su(0,0,0); su(1,1,0)
    sv(0,0,0); sv(1,1,0)
    sw(0,0,0); sw(0,1,0); sw(1,0,0); sw(1,1,0)
    
    # r=1: (A00-A11)(B00-B11)
    su(0,0,1); su(1,1,1,-1)
    sv(0,0,1); sv(1,1,1,-1)
    sw(0,0,1); sw(0,1,1,-1); sw(1,0,1,-1); sw(1,1,1)
    
    # r=2: (A00+A22)(B00+B22)
    su(0,0,2); su(2,2,2)
    sv(0,0,2); sv(2,2,2)
    sw(0,0,2); sw(0,2,2); sw(2,0,2); sw(2,2,2)
    
    # r=3: (A00-A22)(B00-B22)
    su(0,0,3); su(2,2,3,-1)
    sv(0,0,3); sv(2,2,3,-1)
    sw(0,0,3); sw(0,2,3,-1); sw(2,0,3,-1); sw(2,2,3)
    
    # r=4: (A11+A22)(B11+B22)
    su(1,1,4); su(2,2,4)
    sv(1,1,4); sv(2,2,4)
    sw(1,1,4); sw(1,2,4); sw(2,1,4); sw(2,2,4)
    
    # r=5: (A11-A22)(B11-B22)
    su(1,1,5); su(2,2,5,-1)
    sv(1,1,5); sv(2,2,5,-1)
    sw(1,1,5); sw(1,2,5,-1); sw(2,1,5,-1); sw(2,2,5)
    
    # r=6: (A01+A10)(B01+B10)
    su(0,1,6); su(1,0,6)
    sv(0,1,6); sv(1,0,6)
    sw(0,0,6); sw(0,1,6); sw(1,0,6); sw(1,1,6)
    
    # r=7: (A01-A10)(B01-B10)
    su(0,1,7); su(1,0,7,-1)
    sv(0,1,7); sv(1,0,7,-1)
    sw(0,0,7); sw(0,1,7,-1); sw(1,0,7,-1); sw(1,1,7)
    
    # r=8: (A02+A20)(B02+B20)
    su(0,2,8); su(2,0,8)
    sv(0,2,8); sv(2,0,8)
    sw(0,0,8); sw(0,2,8); sw(2,0,8); sw(2,2,8)
    
    # r=9: (A02-A20)(B02-B20)
    su(0,2,9); su(2,0,9,-1)
    sv(0,2,9); sv(2,0,9,-1)
    sw(0,0,9); sw(0,2,9,-1); sw(2,0,9,-1); sw(2,2,9)
    
    # r=10: (A12+A21)(B12+B21)
    su(1,2,10); su(2,1,10)
    sv(1,2,10); sv(2,1,10)
    sw(1,1,10); sw(1,2,10); sw(2,1,10); sw(2,2,10)
    
    # r=11: (A12-A21)(B12-B21)
    su(1,2,11); su(2,1,11,-1)
    sv(1,2,11); sv(2,1,11,-1)
    sw(1,1,11); sw(1,2,11,-1); sw(2,1,11,-1); sw(2,2,11)
    
    # r=12: (A00+A01)(B11+B12)
    su(0,0,12); su(0,1,12)
    sv(1,1,12); sv(1,2,12)
    sw(1,0,12); sw(2,0,12)
    
    # r=13: (A00-A01)(B11-B12)
    su(0,0,13); su(0,1,13,-1)
    sv(1,1,13); sv(1,2,13,-1)
    sw(1,0,13); sw(2,0,13,-1)
    
    # r=14: (A11+A12)(B00+B01)
    su(1,1,14); su(1,2,14)
    sv(0,0,14); sv(0,1,14)
    sw(0,1,14); sw(1,1,14)
    
    # r=15: (A11-A12)(B00-B01)
    su(1,1,15); su(1,2,15,-1)
    sv(0,0,15); sv(0,1,15,-1)
    sw(0,1,15); sw(1,1,15,-1)
    
    # r=16: (A22+A20)(B00+B02)
    su(2,2,16); su(2,0,16)
    sv(0,0,16); sv(0,2,16)
    sw(0,2,16); sw(2,2,16)
    
    # r=17: (A22-A20)(B00-B02)
    su(2,2,17); su(2,0,17,-1)
    sv(0,0,17); sv(0,2,17,-1)
    sw(0,2,17); sw(2,2,17,-1)
    
    # r=18: (A00+A02)(B22+B21)
    su(0,0,18); su(0,2,18)
    sv(2,2,18); sv(2,1,18)
    sw(2,0,18); sw(1,0,18)
    
    # r=19: (A00-A02)(B22-B21)
    su(0,0,19); su(0,2,19,-1)
    sv(2,2,19); sv(2,1,19,-1)
    sw(2,0,19); sw(1,0,19,-1)
    
    # r=20: (A10+A12)(B20+B22)
    su(1,0,20); su(1,2,20)
    sv(2,0,20); sv(2,2,20)
    sw(0,1,20); sw(2,1,20)
    
    # r=21: (A10-A12)(B20-B22)
    su(1,0,21); su(1,2,21,-1)
    sv(2,0,21); sv(2,2,21,-1)
    sw(0,1,21); sw(2,1,21,-1)
    
    # r=22: (A20+A21)(B10+B12)
    su(2,0,22); su(2,1,22)
    sv(1,0,22); sv(1,2,22)
    sw(1,2,22); sw(0,2,22)
    
    return U, V, W

def _verify_decomposition(U, V, W, dim):
    """Verify that sum_r U[i*dim+j,r]*V[j*dim+k,r]*W[k*dim+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                s = 0
                for r in range(rank):
                    s += U[i*dim+j, r] * V[j*dim+k, r] * W[k*dim+i, r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Strassen 2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (A11+A22)(B11+B22) -> C11+, C22+
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A21+A22)B11 -> C21+, C22-
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A11(B12-B22) -> C12+, C22+
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A22(B21-B11) -> C11+, C21+
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A11+A12)B22 -> C11-, C12+
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A21-A11)(B11+B12) -> C22+
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A12-A22)(B21+B22) -> C11+
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Try Laderman 3x3 (rank 23), fall back to schoolbook (rank 27)
    U3, V3, W3 = _get_laderman_3x3()
    if _verify_decomposition(U3, V3, W3, 3):
        rank3 = 23
    else:
        U3 = np.zeros((9, 27), dtype=int)
        V3 = np.zeros((9, 27), dtype=int)
        W3 = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3[i * 3 + j, r] = 1
                    V3[j * 3 + k, r] = 1
                    W3[k * 3 + i, r] = 1
        rank3 = 27

    # Kronecker product: Rank 7 * rank3
    rank = 7 * rank3
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    # Precompute nonzero indices for efficiency
    nz_u2 = {r: np.nonzero(U2[:, r])[0] for r in range(7)}
    nz_v2 = {r: np.nonzero(V2[:, r])[0] for r in range(7)}
    nz_w2 = {r: np.nonzero(W2[:, r])[0] for r in range(7)}
    nz_u3 = {r: np.nonzero(U3[:, r])[0] for r in range(rank3)}
    nz_v3 = {r: np.nonzero(V3[:, r])[0] for r in range(rank3)}
    nz_w3 = {r: np.nonzero(W3[:, r])[0] for r in range(rank3)}
    
    for r2 in range(7):
        for r3 in range(rank3):
            r = r2 * 27 + r3
            
            for u2 in nz_u2[r2]:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                for u3 in nz_u3[r3]:
                    i3 = u3 // 3
                    j3 = u3 % 3
                    val_u3 = U3[u3, r3]
                    
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    U6[g_u, r] += val_u2 * val_u3

            for v2 in nz_v2[r2]:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                for v3 in nz_v3[r3]:
                    j3 = v3 // 3
                    k3 = v3 % 3
                    val_v3 = V3[v3, r3]
                    
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    V6[g_v, r] += val_v2 * val_v3

            for w2 in nz_w2[r2]:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                for w3 in nz_w3[r3]:
                    k3 = w3 // 3
                    i3 = w3 % 3
                    val_w3 = W3[w3, r3]
                    
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    W6[g_w, r] += val_w2 * val_w3

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
