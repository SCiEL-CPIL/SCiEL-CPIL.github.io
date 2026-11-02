import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    # A indices: 00, 11 -> U[0,0]=1, U[3,0]=1
    # B indices: 00, 11 -> V[0,0]=1, V[3,0]=1
    # C indices: 00, 11 -> W[0,0]=1, W[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A indices: 10, 11 -> U[2,1]=1, U[3,1]=1
    # B indices: 00 -> V[0,1]=1
    # C indices: 10, 11 -> W[2,1]=1, W[3,1]=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    # A indices: 00 -> U[0,2]=1
    # B indices: 01, 11 -> V[1,2]=1, V[3,2]=-1
    # C indices: 01, 11 -> W[1,2]=1, W[3,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    # A indices: 11 -> U[3,3]=1
    # B indices: 10, 00 -> V[2,3]=1, V[0,3]=-1
    # C indices: 00, 10 -> W[0,3]=1, W[2,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    # A indices: 00, 01 -> U[0,4]=1, U[1,4]=1
    # B indices: 11 -> V[3,4]=1
    # C indices: 00, 01 -> W[0,4]=-1, W[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A indices: 10, 00 -> U[2,5]=1, U[0,5]=-1
    # B indices: 00, 01 -> V[0,5]=1, V[1,5]=1
    # C indices: 10 -> W[2,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A indices: 01, 11 -> U[1,6]=1, U[3,6]=-1
    # B indices: 10, 11 -> V[2,6]=1, V[3,6]=1
    # C indices: 00, 01 -> W[0,6]=-1, W[1,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W

def _kronecker_4x4x4():
    """
    Construct rank-49 decomposition for 4x4 * 4x4.
    T_{4,4,4} = T_{2,2,2} (outer) x T_{2,2,2} (inner)
    """
    U2, V2, W2 = _strassen_2x2()
    n1, m1, p1, r1 = 2, 2, 2, 7
    n2, m2, p2, r2 = 2, 2, 2, 7
    n, m, p = n1*n2, m1*m2, p1*p2
    rank = r1 * r2
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # U[i*m+j, c1*r2+c2] = U2[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    # i = n1*i1 + i2, j = m1*j1 + j2
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u1_val = U2[i1*m1+j1, c1]
                if u1_val == 0: continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            u2_val = U2[i2*m2+j2, c2]
                            if u2_val == 0: continue
                            i = n1*i1 + i2
                            j = m1*j1 + j2
                            c = c1*r2 + c2
                            U[i*m+j, c] += u1_val * u2_val
                            
    # V[j*p+k, c1*r2+c2] = V2[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
    # j = m1*j1 + j2, k = p1*k1 + k2
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                v1_val = V2[j1*p1+k1, c1]
                if v1_val == 0: continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            v2_val = V2[j2*p2+k2, c2]
                            if v2_val == 0: continue
                            j = m1*j1 + j2
                            k = p1*k1 + k2
                            c = c1*r2 + c2
                            V[j*p+k, c] += v1_val * v2_val
                            
    # W[k*n+i, c1*r2+c2] = W2[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    # k = p1*k1 + k2, i = n1*i1 + i2
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                w1_val = W2[k1*n1+i1, c1]
                if w1_val == 0: continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            w2_val = W2[k2*n2+i2, c2]
                            if w2_val == 0: continue
                            k = p1*k1 + k2
                            i = n1*i1 + i2
                            c = c1*r2 + c2
                            W[k*n+i, c] += w1_val * w2_val
                            
    return U, V, W

def _verify_4x4x8(U, V, W, rank):
    """Verify the 4x4*4x8 decomposition."""
    n, m, p = 4, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140
    
    n, m, p = DIMENSIONS
    
    # 1. Get 4x4x4 decomposition (Rank 49)
    U4, V4, W4 = _kronecker_4x4x4()
    rank_4 = 49
    
    # 2. Assemble 4x4x8 (Rank 98)
    rank_total = 98
    U_total = np.zeros((n*m, rank_total), dtype=float)
    V_total = np.zeros((m*p, rank_total), dtype=float)
    W_total = np.zeros((p*n, rank_total), dtype=float)
    
    # U is the same for both halves
    U_total[:, 0:rank_4] = U4
    U_total[:, rank_4:rank_total] = U4
    
    # V
    for j in range(m):
        for k in range(p):
            if k < 4:
                # Left half (cols 0-3 of B)
                k_prime = k
                V_total[j*p+k, 0:rank_4] = V4[j*4+k_prime, :]
            else:
                # Right half (cols 4-7 of B)
                k_prime = k - 4
                V_total[j*p+k, rank_4:rank_total] = V4[j*4+k_prime, :]
                
    # W
    for k in range(p):
        for i in range(n):
            if k < 4:
                # Left half (cols 0-3 of C)
                k_prime = k
                W_total[k*n+i, 0:rank_4] = W4[k_prime*n+i, :]
            else:
                # Right half (cols 4-7 of C)
                k_prime = k - 4
                W_total[k*n+i, rank_4:rank_total] = W4[k_prime*n+i, :]
                
    # Verify
    if time.time() < DEADLINE:
        if _verify_4x4x8(U_total, V_total, W_total, rank_total):
            return {
                "U": U_total.tolist(),
                "V": V_total.tolist(),
                "W": W_total.tolist()
            }
    
    # Fallback: Schoolbook (Rank 128)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print("Verification:", _verify_4x4x8(U, V, W, rank))
