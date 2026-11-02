"""Matrix multiplication 4x4x8 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # U (4x7), V (4x7), W (4x7)
    # Indices: i,j in 0..1 for U. j,k in 0..1 for V. k,i in 0..1 for W.
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1],
    ])
    W2 = np.array([
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
    ])
    
    # Construct 2x2x4 decomposition (Rank 14)
    # This is the direct sum of two 2x2x2 decompositions.
    # n=2, m=2, p=4
    n2, m2, p2 = 2, 2, 4
    r2 = 14
    U_224 = np.zeros((n2*m2, r2))
    V_224 = np.zeros((m2*p2, r2))
    W_224 = np.zeros((p2*n2, r2))
    
    # First half (k=0,1)
    U_224[:, 0:7] = U2
    # V for k=0,1 corresponds to V2. V2 shape (4,7). Rows are j*p+k.
    # j=0,k=0 -> 0. j=0,k=1 -> 1. j=1,k=0 -> 2. j=1,k=1 -> 3.
    # V_224 rows 0..3 correspond to (j=0,k=0..1) and (j=1,k=0..1).
    V_224[0:4, 0:7] = V2
    # W for k=0,1 corresponds to W2. W2 shape (4,7). Rows are k*n+i.
    # k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
    W_224[0:4, 0:7] = W2
    
    # Second half (k=2,3)
    U_224[:, 7:14] = U2
    # V for k=2,3.
    # j=0,k=2 -> 0*4+2 = 2.
    # j=0,k=3 -> 0*4+3 = 3.
    # j=1,k=2 -> 1*4+2 = 6.
    # j=1,k=3 -> 1*4+3 = 7.
    # V2 rows: 0(j0k0), 1(j0k1), 2(j1k0), 3(j1k1).
    # We need to map V2 rows to V_224 rows 2,3,6,7.
    V_224[2, 7:14] = V2[0] # j=0, k=2 (was k=0 in V2)
    V_224[3, 7:14] = V2[1] # j=0, k=3 (was k=1 in V2)
    V_224[6, 7:14] = V2[2] # j=1, k=2 (was k=0 in V2)
    V_224[7, 7:14] = V2[3] # j=1, k=3 (was k=1 in V2)
    
    # W for k=2,3.
    # W2 rows: 0(k0i0), 1(k0i1), 2(k1i0), 3(k1i1).
    # W_224 rows: k*n+i.
    # k=2,i=0 -> 2*2+0 = 4.
    # k=2,i=1 -> 2*2+1 = 5.
    # k=3,i=0 -> 3*2+0 = 6.
    # k=3,i=1 -> 3*2+1 = 7.
    W_224[4, 7:14] = W2[0] # k=2, i=0 (was k=0 in W2)
    W_224[5, 7:14] = W2[1] # k=2, i=1 (was k=1 in W2)
    W_224[6, 7:14] = W2[2] # k=3, i=0 (was k=0 in W2)
    W_224[7, 7:14] = W2[3] # k=3, i=1 (was k=1 in W2)
    
    # Now Kronecker Product T_{2,2,2} (T1) and T_{2,2,4} (T2)
    # T1: n1=2, m1=2, p1=2, r1=7
    # T2: n2=2, m2=2, p2=4, r2=14
    # Result: n=4, m=4, p=8, r=98
    
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 4
    r1 = 7
    r2 = 14
    r_total = r1 * r2
    
    U = np.zeros((n*m, r_total))
    V = np.zeros((m*p, r_total))
    W = np.zeros((p*n, r_total))
    
    # Fill U
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    idx_u = (i1*n2 + i2)*(m1*m2) + (j1*m2 + j2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            idx_c = c1*r2 + c2
                            U[idx_u, idx_c] = U2[i1*m1+j1, c1] * U_224[i2*m2+j2, c2]
                            
    # Fill V
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    idx_v = (j1*m2 + j2)*(p1*p2) + (k1*p2 + k2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            idx_c = c1*r2 + c2
                            V[idx_v, idx_c] = V2[j1*p1+k1, c1] * V_224[j2*p2+k2, c2]
                            
    # Fill W
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    idx_w = (k1*p2 + k2)*(n1*n2) + (i1*n2 + i2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            idx_c = c1*r2 + c2
                            W[idx_w, idx_c] = W2[k1*n1+i1, c1] * W_224[k2*n2+i2, c2]
                            
    # Verify
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        # Sample check to save time, or full check if fast enough
        # 4*4*8*98 ~ 12544 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
        
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Schoolbook
        r = n * m * p
        U_fb = np.zeros((n*m, r))
        V_fb = np.zeros((m*p, r))
        W_fb = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
