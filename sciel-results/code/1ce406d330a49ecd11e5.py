import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Rank-55 solution:
    # Part A: j=0 block (M_{2,1,10}) has rank 20
    # Part B: j=1,2 with 5 Strassen blocks of 2x2, each rank 7, total 35
    # Total rank = 55
    
    rank = 55
    U = np.zeros((n*m, rank))  # (6, 55)
    V = np.zeros((m*p, rank))  # (30, 55)
    W = np.zeros((p*n, rank))  # (20, 55)
    
    c = 0
    
    # Part A: j=0 term: A[i,0] * B[0,k] for all i in {0,1}, k in {0..9}
    # This gives C[i,k] += A[i,0]*B[0,k]
    # Each term is rank-1: U[i*m+0]=1, V[0*p+k]=1, W[k*n+i]=1
    for i in range(n):
        for k in range(p):
            U[i*m + 0, c] = 1
            V[0*p + k, c] = 1
            W[k*n + i, c] = 1
            c += 1
    # c is now 20
    
    # Part B: j=1,2 terms using Strassen on 2x2 blocks
    # For each pair of k values (k0, k1), compute the 2x2 output block
    # using Strassen on the 2x2 submatrix (rows 0,1 of A for cols 1,2)
    # and 2x2 submatrix (rows 1,2 of B for cols k0,k1)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # A indices: row i, col j -> i*m + j
        u11 = 0 * m + 1  # A[0,1]
        u12 = 0 * m + 2  # A[0,2]
        u21 = 1 * m + 1  # A[1,1]
        u22 = 1 * m + 2  # A[1,2]
        
        # B indices: row j, col k -> j*p + k
        v11 = 1 * p + k0  # B[1,k0]
        v12 = 1 * p + k1  # B[1,k1]
        v21 = 2 * p + k0  # B[2,k0]
        v22 = 2 * p + k1  # B[2,k1]
        
        # W indices: W[k*n + i, r]
        w00 = k0 * n + 0  # W[k0, 0] -> C[0, k0]
        w01 = k1 * n + 0  # W[k1, 0] -> C[0, k1]
        w10 = k0 * n + 1  # W[k0, 1] -> C[1, k0]
        w11 = k1 * n + 1  # W[k1, 1] -> C[1, k1]
        
        # Strassen's 7 multiplications for 2x2 matrix multiply:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # M1 = (a11 + a22) * (b11 + b22)
        U[u11, c] += 1; U[u22, c] += 1
        V[v11, c] += 1; V[v22, c] += 1
        W[w00, c] += 1; W[w11, c] += 1
        c += 1
        
        # M2 = (a21 + a22) * b11
        U[u21, c] += 1; U[u22, c] += 1
        V[v11, c] += 1
        W[w10, c] += 1; W[w11, c] += -1
        c += 1
        
        # M3 = a11 * (b12 - b22)
        U[u11, c] += 1
        V[v12, c] += 1; V[v22, c] += -1
        W[w01, c] += 1; W[w11, c] += 1
        c += 1
        
        # M4 = a22 * (b21 - b11)
        U[u22, c] += 1
        V[v21, c] += 1; V[v11, c] += -1
        W[w00, c] += 1; W[w10, c] += 1
        c += 1
        
        # M5 = (a11 + a12) * b22
        U[u11, c] += 1; U[u12, c] += 1
        V[v22, c] += 1
        W[w00, c] += -1; W[w01, c] += 1
        c += 1
        
        # M6 = (a21 - a11) * (b11 + b12)
        U[u21, c] += 1; U[u11, c] += -1
        V[v11, c] += 1; V[v12, c] += 1
        W[w11, c] += 1
        c += 1
        
        # M7 = (a12 - a22) * (b21 + b22)
        U[u12, c] += 1; U[u22, c] += -1
        V[v21, c] += 1; V[v22, c] += 1
        W[w00, c] += 1
        c += 1
        
        if time.time() > deadline:
            break

    # Convert to lists of floats for the interface
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
