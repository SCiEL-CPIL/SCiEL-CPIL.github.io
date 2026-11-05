import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 78 decomposition
    rank = 78
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # 2x2x2 Strassen
        # U indices: i0..i0+1, j0..j0+1
        # V indices: j0..j0+1, k0..k0+1
        # W indices: k0..k0+1, i0..i0+1
        
        # M1
        U[i0*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M2
        U[(i0+1)*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1
        
        # M3
        U[i0*m + j0, col] = 1
        V[(j0+1)*p + k0, col] = -1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        # M4
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = -1
        V[j0*p + (k0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M5
        U[i0*m + j0, col] = 1
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+1)*n + i0, col] = -1
        col += 1
        
        # M6
        U[(i0+1)*m + j0, col] = 1
        U[i0*m + j0, col] = -1
        V[j0*p + k0, col] = 1
        V[j0*p + (k0+1), col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1
        
        # M7
        U[i0*m + (j0+1), col] = 1
        U[(i0+1)*m + (j0+1), col] = -1
        V[(j0+1)*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1

    def add_schoolbook(i0, j0, k0, ni, mj, pk):
        nonlocal col
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    U[(i0+i)*m + (j0+j), col] = 1
                    V[(j0+j)*p + (k0+k), col] = 1
                    W[(k0+k)*n + (i0+i), col] = 1
                    col += 1

    # m parts: 0-1, 2-3
    for j_part in range(2):
        j0 = j_part * 2
        # p parts: 0-1, 2-3, 4-6
        # p=2
        add_strassen(0, j0, 0) # n=0,1, j=j0..j0+1, k=0,1
        add_schoolbook(2, j0, 0, 1, 2, 2) # n=2, j=j0..j0+1, k=0,1
        
        add_strassen(0, j0, 2) # n=0,1, j=j0..j0+1, k=2,3
        add_schoolbook(2, j0, 2, 1, 2, 2) # n=2, j=j0..j0+1, k=2,3
        
        # p=3 (k=4,5,6)
        # k=4,5 (p=2)
        add_strassen(0, j0, 4) # n=0,1, j=j0..j0+1, k=4,5
        add_schoolbook(2, j0, 4, 1, 2, 2) # n=2, j=j0..j0+1, k=4,5
        
        # k=6 (p=1)
        add_schoolbook(0, j0, 6, 3, 2, 1) # n=0..2, j=j0..j0+1, k=6

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
