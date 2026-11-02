import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # We will use a hybrid decomposition.
    # Identify disjoint 2x2x2 blocks.
    # Blocks start at (i, j, k) where i is even, j is even, k is even?
    # We can just pick specific blocks.
    # Max blocks: floor(3/2)=1, floor(5/2)=2, floor(7/2)=3.
    # Let's pick blocks at (0,0,0), (0,2,0), (0,4,0), (0,0,2), (0,2,2), (0,4,2).
    # Note: i only has 1 block (0,1). j has (0,1) and (2,3). k has (0,1), (2,3), (4,5).
    
    blocks = []
    for i in range(0, n, 2):
        for j in range(0, m, 2):
            for k in range(0, p, 2):
                if i+1 < n and j+1 < m and k+1 < p:
                    blocks.append((i, j, k))
                    
    # Rank calculation
    # Total schoolbook terms: n*m*p
    # Each block removes 8 terms and adds 7.
    # Net change per block: -1.
    rank = n * m * p - len(blocks)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Map (i,j,k) to column index if schoolbook
    # We need to handle the blocks first or integrate.
    # Let's iterate through all (i,j,k).
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    # L(A) and R(B) coefficients
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We define the 7 terms as (u_vec, v_vec, w_vec)
    # u_vec is in A space (size 4 for the block), v_vec in B space (size 4), w_vec in C space (size 4)?
    # No, u, v, w are global vectors.
    
    # For each block, we add 7 columns.
    # We need to know which (i,j,k) are covered by blocks.
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
                    
    # Process blocks
    for (i0, j0, k0) in blocks:
        # Define the 7 Strassen terms
        # Indices in A: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # Indices in B: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # Indices in C: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
        
        # We will create 7 new columns
        # Term 1: M1
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        # W coefficients for M1: C00 += 1, C01 += 0, C10 += 0, C11 += -1?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So for M1:
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # Term 2: M2
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[(k0+1)*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = -1
        col += 1
        
        # Term 3: M3
        U[i0*m + j0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # Term 4: M4
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # Term 5: M5
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = -1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # Term 6: M6
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # Term 7: M7
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
        
    # Process schoolbook terms for non-covered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
