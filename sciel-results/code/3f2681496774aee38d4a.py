import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the rank-55 solution
    # Strategy: 
    # 1. Handle j=0 (first column of A, first row of B) separately.
    #    This is M_{2,1,10}. Rank is 20.
    # 2. Handle j=1,2 (remaining 2 columns of A, remaining 2 rows of B) as 5 blocks of M_{2,2,2}.
    #    Each M_{2,2,2} has rank 7 (Strassen).
    #    Total rank = 20 + 5 * 7 = 55.
    
    rank = 55
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    c = 0
    # Part A: j=0
    # A is 2x3, B is 3x10.
    # j=0 term: A[i,0] * B[0,k].
    # We need 20 terms to handle the sparsity correctly for M_{2,1,10}.
    # Actually, M_{2,1,10} rank is 20.
    # Construction:
    # For each i in {0,1} and k in {0..9}, we need a term?
    # No, we can group them.
    # Let's use the standard decomposition for M_{2,1,p}.
    # It turns out rank(M_{2,1,p}) = 2p.
    # So we need 20 terms.
    for i in range(n):
        for k in range(p):
            U[i*m+0, c] = 1
            V[0*p+k, c] = 1
            W[k*n+i, c] = 1
            c += 1
            
    # Part B: j=1,2 with Strassen
    # We have 5 blocks of k: (0,1), (2,3), (4,5), (6,7), (8,9).
    for b in range(5):
        k0 = 2*b
        k1 = 2*b+1
        
        # Indices for A (rows 0,1; cols 1,2)
        u11, u12 = 0*m+1, 0*m+2
        u21, u22 = 1*m+1, 1*m+2
        
        # Indices for B (rows 1,2; cols k0, k1)
        v11, v12 = 1*p+k0, 1*p+k1
        v21, v22 = 2*p+k0, 2*p+k1
        
        # Indices for W (cols k0, k1; rows 0,1)
        w00, w01 = k0*n+0, k1*n+0
        w10, w11 = k0*n+1, k1*n+1
        
        # Strassen for 2x2 matrix multiplication
        # M1 = (a11+a22)(b11+b22)
        U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
        # M2 = (a21+a22)b11
        U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
        # M3 = a11(b12-b22)
        U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
        # M4 = a22(b21-b11)
        U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
        # M5 = (a11+a12)b22
        U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
        # M6 = (a21-a11)(b11+b12)
        U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
        # M7 = (a12-a22)(b21+b22)
        U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
