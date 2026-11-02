"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # Strategy:
    # Split A (2x3) into A1 (2x1) and A2 (2x2)
    # Split B (3x10) into B1 (1x10) and B2 (2x10)
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 (2x1) * B1 (1x10) -> 2x10
    # Rank = 2 * 10 = 20
    # Part 2: A2 (2x2) * B2 (2x10) -> 2x10
    # Split B2 into 5 blocks of 2x2
    # Each 2x2 * 2x2 uses Strassen rank 7
    # Rank = 5 * 7 = 35
    # Total Rank = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)] # 6x55
    V = [[0] * rank for _ in range(m * p)] # 30x55
    W = [[0] * rank for _ in range(p * n)] # 20x55
    
    col = 0
    
    # Part 1: A1 B1
    # A1 is col 0 of A. Indices i in {0,1}, k=0.
    # B1 is row 0 of B. Indices k=0, l in {0..9}.
    # C is 2x10. Indices i in {0,1}, l in {0..9}.
    # Term: A[i,0] * B[0,l] * C[i,l]
    for i in range(n):
        for l in range(p):
            # U index: i*m + 0
            U[i * m + 0][col] = 1
            # V index: 0*p + l
            V[0 * p + l][col] = 1
            # W index: l*n + i
            W[l * n + i][col] = 1
            col += 1
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Part 2: A2 B2
    # A2 is cols 1,2 of A. Indices i in {0,1}, k in {1,2}.
    # B2 is rows 1,2 of B. Indices k in {1,2}, l in {0..9}.
    # Split B2 into 5 blocks of 2x2 columns: l in {0,1}, {2,3}, {4,5}, {6,7}, {8,9}
    # For each block, use Strassen on 2x2 * 2x2
    
    # Strassen 2x2 * 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # We need to map these to the global indices.
    # For block b (0..4), columns are 2b, 2b+1.
    # A2 local indices: 
    #   a = A[0,1], b = A[0,2], c = A[1,1], d = A[1,2]
    # B2 local indices for block b:
    #   e = B[1,2b], f = B[1,2b+1], g = B[2,2b], h = B[2,2b+1]
    # C local indices for block b:
    #   C[0,2b] = m1+m4-m5+m7
    #   C[0,2b+1] = m3+m5
    #   C[1,2b] = m2+m4
    #   C[1,2b+1] = m1-m2+m3+m6
    
    for block in range(5):
        l0 = 2 * block
        l1 = 2 * block + 1
        
        # m1 = (A[0,1] + A[1,1]) * (B[1,l0] + B[2,l0])
        # This is not a single rank 1 term, it's a product of sums.
        # We need to expand into rank 1 terms in the tensor decomposition.
        # Wait, Strassen terms are multiplications of linear forms.
        # In the tensor decomposition, each "multiplication" corresponds to a column in U, V, W.
        # The entries in U, V, W are the coefficients of the linear forms.
        
        # Term 1: m1
        # U: A[0,1] + A[1,1] -> U[0*m+1]=1, U[1*m+1]=1
        # V: B[1,l0] + B[2,l0] -> V[1*p+l0]=1, V[2*p+l0]=1
        # W: C[0,l0] += 1, C[1,l1] += 1
        # Wait, m1 contributes to C[0,l0] and C[1,l1].
        # So W[0*n+l0]=1, W[1*n+l1]=1
        
        U[0*m+1][col] = 1
        U[1*m+1][col] = 1
        V[1*p+l0][col] = 1
        V[2*p+l0][col] = 1
        W[0*n+l0][col] = 1
        W[1*n+l1][col] = 1
        col += 1
        
        # Term 2: m2 = (A[1,1] + A[1,2]) * B[1,l0]
        # U: A[1,1]+A[1,2] -> U[1*m+1]=1, U[1*m+2]=1
        # V: B[1,l0] -> V[1*p+l0]=1
        # W: C[1,l0] += 1, C[1,l1] -= 1
        U[1*m+1][col] = 1
        U[1*m+2][col] = 1
        V[1*p+l0][col] = 1
        W[1*n+l0][col] = 1
        W[1*n+l1][col] = -1
        col += 1
        
        # Term 3: m3 = A[0,1] * (B[1,l1] - B[2,l1])
        # U: A[0,1] -> U[0*m+1]=1
        # V: B[1,l1]-B[2,l1] -> V[1*p+l1]=1, V[2*p+l1]=-1
        # W: C[0,l1] += 1, C[1,l1] += 1
        U[0*m+1][col] = 1
        V[1*p+l1][col] = 1
        V[2*p+l1][col] = -1
        W[0*n+l1][col] = 1
        W[1*n+l1][col] = 1
        col += 1
        
        # Term 4: m4 = A[1,2] * (B[2,l0] - B[1,l0])
        # U: A[1,2] -> U[1*m+2]=1
        # V: B[2,l0]-B[1,l0] -> V[2*p+l0]=1, V[1*p+l0]=-1
        # W: C[0,l0] += 1, C[1,l0] += 1
        U[1*m+2][col] = 1
        V[2*p+l0][col] = 1
        V[1*p+l0][col] = -1
        W[0*n+l0][col] = 1
        W[1*n+l0][col] = 1
        col += 1
        
        # Term 5: m5 = (A[0,1] + A[0,2]) * B[2,l1]
        # U: A[0,1]+A[0,2] -> U[0*m+1]=1, U[0*m+2]=1
        # V: B[2,l1] -> V[2*p+l1]=1
        # W: C[0,l0] -= 1, C[0,l1] += 1
        U[0*m+1][col] = 1
        U[0*m+2][col] = 1
        V[2*p+l1][col] = 1
        W[0*n+l0][col] = -1
        W[0*n+l1][col] = 1
        col += 1
        
        # Term 6: m6 = (A[1,1] - A[0,1]) * (B[1,l0] + B[1,l1])
        # U: A[1,1]-A[0,1] -> U[1*m+1]=1, U[0*m+1]=-1
        # V: B[1,l0]+B[1,l1] -> V[1*p+l0]=1, V[1*p+l1]=1
        # W: C[1,l1] += 1
        U[1*m+1][col] = 1
        U[0*m+1][col] = -1
        V[1*p+l0][col] = 1
        V[1*p+l1][col] = 1
        W[1*n+l1][col] = 1
        col += 1
        
        # Term 7: m7 = (A[0,2] - A[1,2]) * (B[2,l0] + B[2,l1])
        # U: A[0,2]-A[1,2] -> U[0*m+2]=1, U[1*m+2]=-1
        # V: B[2,l0]+B[2,l1] -> V[2*p+l0]=1, V[2*p+l1]=1
        # W: C[0,l0] += 1
        U[0*m+2][col] = 1
        U[1*m+2][col] = -1
        V[2*p+l0][col] = 1
        V[2*p+l1][col] = 1
        W[0*n+l0][col] = 1
        col += 1
        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
