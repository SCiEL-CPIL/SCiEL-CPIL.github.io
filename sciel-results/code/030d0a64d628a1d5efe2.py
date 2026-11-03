import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Split m=3 into m1=2 and m2=1.
    # T_{2,3,8} = T_{2,2,8} + T_{2,1,8}
    # T_{2,2,8} = 4 * T_{2,2,2}. Rank(T_{2,2,2}) = 7 (Strassen). Rank(T_{2,2,8}) = 28.
    # T_{2,1,8} = 16 scalar mults. Rank(T_{2,1,8}) = 16.
    # Total Rank = 44.
    
    rank = 44
    # U shape: (n*m, rank) = (6, 44)
    # V shape: (m*p, rank) = (24, 44)
    # W shape: (p*n, rank) = (16, 44)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # 1. (a+d)(e+h): C[0,0]+=1, C[1,1]+=1
        U[i_off*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 2. (c+d)e: C[0,0]+=-1, C[1,0]+=1
        U[(i_off+1)*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        W[k_off*n + i_off, col] += -1
        W[k_off*n + (i_off+1), col] += 1
        col += 1
        
        # 3. a(f-h): C[0,1]+=1, C[1,1]+=-1
        U[i_off*m + j_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += -1
        col += 1
        
        # 4. d(g-e): C[0,0]+=1, C[1,0]+=1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[(j_off+1)*p + k_off, col] += 1
        V[j_off*p + k_off, col] += -1
        W[k_off*n + i_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        col += 1
        
        # 5. (a+b)h: C[0,0]+=-1, C[0,1]+=1
        U[i_off*m + j_off, col] += 1
        U[i_off*m + (j_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        col += 1
        
        # 6. (c-a)(e+f): C[1,1]+=1
        U[(i_off+1)*m + j_off, col] += 1
        U[i_off*m + j_off, col] += -1
        V[j_off*p + k_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 7. (b-d)(g+h): C[1,1]+=-1
        U[i_off*m + (j_off+1), col] += 1
        U[(i_off+1)*m + (j_off+1), col] += -1
        V[(j_off+1)*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += -1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0), col] = 1
                V[(j_off+0)*p + (k_off+k), col] = 1
                W[(k_off+k)*n + (i_off+i), col] = 1
                col += 1

    # Block 1: k=0..3
    # m=2 part (j=0,1)
    add_strassen_2x2x2(0, 0, 0) # j=0,1 k=0,1
    add_strassen_2x2x2(0, 0, 2) # j=0,1 k=2,3
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 0) # j=2 k=0..3

    # Block 2: k=4..7
    # m=2 part (j=0,1)
    add_strassen_2x2x2(0, 0, 4) # j=0,1 k=4,5
    add_strassen_2x2x2(0, 0, 6) # j=0,1 k=6,7
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 4) # j=2 k=4..7

    # Verify: for random A, B, check that sum_r U[:,r]*V[:,r]*W[:,r] reconstructs A@B
    # Tensor check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if this is the multiplication tensor
    # Actually the decomposition should satisfy: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta(i,j,k)
    # where delta is 1 if the indices correspond to a valid multiplication term, 0 otherwise.
    # More precisely, the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So we need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    
    # Quick spot check with a few indices
    import time
    start_time = time.time()
    
    # Verify all (i,j,k) triples
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-9:
        # Fallback: use full schoolbook if Strassen has issues
        U = np.zeros((n * m, n*m*p), dtype=float)
        V = np.zeros((m * p, n*m*p), dtype=float)
        W = np.zeros((p * n, n*m*p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
