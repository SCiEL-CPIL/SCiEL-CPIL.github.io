import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    start_time = time.time()
    deadline = start_time + 140

    # Build rank-55 solution using Strassen on 2x2 blocks for j=1,2
    # Structure: 5 blocks of 2 k-values each
    # Block 0 (k=0,1): j=0 handled directly (4 terms per k, so 4 terms)
    # Actually let me restructure:
    # For each pair of k values (k0, k1), we compute contributions to all 3 rows of B
    # A is 2x3, B is 3x2 (for one k-pair), C is 2x2
    
    # For j=0: c[i,k] += a[i,0]*b[0,k] -> 2*2 = 4 terms per k-pair
    # For j=1,2: use Strassen on the 2x2 subproblem (a[i,1..2] * b[1..2,k])
    #   7 terms per k-pair
    
    # Total per k-pair: 4 + 7 = 11
    # 5 k-pairs: 55 total
    
    rank = 55
    U = np.zeros((n * m, rank))  # 6 x 55
    V = np.zeros((m * p, rank))  # 30 x 55
    W = np.zeros((p * n, rank))  # 20 x 55
    
    c = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Part A: j=0 direct (4 terms)
        for i in range(n):
            for kk in [k0, k1]:
                U[i * m + 0, c] = 1
                V[0 * p + kk, c] = 1
                W[kk * n + i, c] = 1
                c += 1
        
        # Part B: j=1,2 Strassen (7 terms)
        # A submatrix: rows i=0,1; cols j=1,2 -> a[i,j]
        # B submatrix: rows j=1,2; cols k=k0,k1 -> b[j,k]
        # C submatrix: rows i=0,1; cols k=k0,k1 -> c[i,k]
        
        u00 = 0 * m + 1  # a[0,1]
        u01 = 0 * m + 2  # a[0,2]
        u10 = 1 * m + 1  # a[1,1]
        u11 = 1 * m + 2  # a[1,2]
        
        v00 = 1 * p + k0  # b[1,k0]
        v01 = 1 * p + k1  # b[1,k1]
        v10 = 2 * p + k0  # b[2,k0]
        v11 = 2 * p + k1  # b[2,k1]
        
        w00 = k0 * n + 0  # c[0,k0]
        w01 = k1 * n + 0  # c[0,k1]
        w10 = k0 * n + 1  # c[1,k0]
        w11 = k1 * n + 1  # c[1,k1]
        
        # M1 = (a00+a11)(b00+b11) where a00=a[0,1], a11=a[1,2], b00=b[1,k0], b11=b[2,k1]
        # Wait, let me use standard Strassen notation for 2x2:
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # Here a00=u00=a[0,1], a01=u01=a[0,2], a10=u10=a[1,1], a11=u11=a[1,2]
        # b00=v00=b[1,k0], b01=v01=b[1,k1], b10=v10=b[2,k0], b11=v11=b[2,k1]
        # C = [[c00, c01], [c10, c11]] where c00=w00, c01=w01, c10=w10, c11=w11
        
        # M1 = (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11
        U[u00, c] += 1; U[u11, c] += 1
        V[v00, c] += 1; V[v11, c] += 1
        W[w00, c] += 1; W[w01, c] += 1; W[w10, c] += 1; W[w11, c] += 1
        c += 1
        
        # M2 = (a10+a11)(b00) -> contributes to c10, c11
        U[u10, c] += 1; U[u11, c] += 1
        V[v00, c] += 1
        W[w10, c] += 1; W[w11, c] += -1
        c += 1
        
        # M3 = a00(b01-b11) -> contributes to c00, c01
        U[u00, c] += 1
        V[v01, c] += 1; V[v11, c] += -1
        W[w00, c] += 1; W[w01, c] += 1
        c += 1
        
        # M4 = a11(b10-b00) -> contributes to c10, c11
        U[u11, c] += 1
        V[v10, c] += 1; V[v00, c] += -1
        W[w10, c] += 1; W[w11, c] += 1
        c += 1
        
        # M5 = (a00+a01)(b11) -> contributes to c00, c01
        U[u00, c] += 1; U[u01, c] += 1
        V[v11, c] += 1
        W[w00, c] += -1; W[w01, c] += 1
        c += 1
        
        # M6 = (a10-a00)(b00+b01) -> contributes to c10, c11
        U[u10, c] += 1; U[u00, c] += -1
        V[v00, c] += 1; V[v01, c] += 1
        W[w11, c] += 1
        c += 1
        
        # M7 = (a01-a11)(b10+b11) -> contributes to c00, c01
        U[u01, c] += 1; U[u11, c] += -1
        V[v10, c] += 1; V[v11, c] += 1
        W[w00, c] += 1
        c += 1
    
    # Verify
    def verify(U, V, W):
        nm = n * m
        mp = m * p
        pn = p * n
        r = U.shape[1]
        # Full verification
        for a in range(nm):
            for b in range(mp):
                for cc in range(pn):
                    val = np.sum(U[a, :] * V[b, :] * W[cc, :])
                    # Expected: T[a,b,cc] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, cc=k*n+i
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = cc // n
                    i2 = cc % n
                    expected = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
                    if abs(val - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to rank-60
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
