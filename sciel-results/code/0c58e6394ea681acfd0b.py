"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Kronecker decomposition parameters
    # n = 2*1, m = 2*2, p = 2*3
    n1, n2 = 2, 1
    m1, m2 = 2, 2
    p1, p2 = 2, 3
    
    # Rank of sub-decompositions
    R1 = 7  # Strassen for 2x2
    R2 = 6  # Schoolbook for 1x2 by 2x3
    R = R1 * R2
    
    # Initialize matrices
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen factors for 2x2 x 2x2
    # Indices: U[2*i+j], V[2*j+k], W[2*k+i]
    # r=0: P1=(a+d)(e+h) -> W: c11, c22
    # r=1: P2=(c+d)e -> W: c21
    # r=2: P3=a(f-h) -> W: c12, c21
    # r=3: P4=d(g-e) -> W: c12, c22
    # r=4: P5=(a+b)(e+f) -> W: -c11, c12
    # r=5: P6=(c-b)(g+h) -> W: c21, c22
    # r=6: P7=(a-d)(h+f) -> W: c11, c22
    
    U1 = [
        [1, 0, 0, 1, 0, 0, 0], # r0: a, d
        [0, 0, 1, 1, 0, 0, 0], # r1: c, d
        [1, 0, 0, 0, 0, 0, 0], # r2: a
        [0, 0, 0, 1, 0, 0, 0], # r3: d
        [1, 1, 0, 0, 0, 0, 0], # r4: a, b
        [0, -1, 1, 0, 0, 0, 0],# r5: c, -b
        [1, 0, 0, -1, 0, 0, 0] # r6: a, -d
    ]
    # U1 indexed by [row, r1]. Row = 2*i1 + j1.
    
    V1 = [
        [1, 0, 0, 1, 0, 0, 0], # r0: e, h
        [1, 0, 0, 0, 0, 0, 0], # r1: e
        [0, 1, 0, -1, 0, 0, 0],# r2: f, -h
        [-1, 0, 1, 0, 0, 0, 0],# r3: -e, g
        [1, 1, 0, 0, 0, 0, 0], # r4: e, f
        [0, 0, 1, 1, 0, 0, 0], # r5: g, h
        [0, 1, 0, 1, 0, 0, 0]  # r6: f, h
    ]
    # V1 indexed by [row, r1]. Row = 2*j1 + k1.
    
    W1 = [
        [1, 0, 0, 1, 0, 0, 0], # r0: c11, c22
        [0, 0, 1, 0, 0, 0, 0], # r1: c21
        [0, 1, 1, 0, 0, 0, 0], # r2: c12, c21
        [0, 1, 0, 1, 0, 0, 0], # r3: c12, c22
        [-1, 1, 0, 0, 0, 0, 0],# r4: -c11, c12
        [0, 0, 1, 1, 0, 0, 0], # r5: c21, c22
        [1, 0, 0, 1, 0, 0, 0]  # r6: c11, c22
    ]
    # W1 indexed by [row, r1]. Row = 2*k1 + i1.

    # Schoolbook factors for 1x2 by 2x3
    # n2=1, m2=2, p2=3
    # U2 size 1*2 = 2. Row = i2*m2 + j2 = j2.
    # V2 size 2*3 = 6. Row = j2*p2 + k2.
    # W2 size 3*1 = 3. Row = k2*n2 + i2 = k2.
    
    # R2 = 6 terms. r2 corresponds to (j2, k2).
    # r2 = j2*3 + k2.
    
    # U2[j2, r2] = 1 if j2 matches.
    U2 = [
        [1, 1, 1, 0, 0, 0], # j2=0: r2 in 0,1,2
        [0, 0, 0, 1, 1, 1]  # j2=1: r2 in 3,4,5
    ]
    
    # V2[j2*p2+k2, r2] = 1 if matches. Identity.
    V2 = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ]
    
    # W2[k2, r2] = 1 if k2 matches.
    W2 = [
        [1, 0, 0, 0, 0, 0], # k2=0: r2 in 0,3
        [0, 1, 0, 0, 0, 0], # k2=1: r2 in 1,4
        [0, 0, 1, 0, 0, 0]  # k2=2: r2 in 2,5
    ]
    # Wait, W2[k2, r2].
    # r2=0 (j2=0, k2=0) -> W2[0,0]=1
    # r2=1 (j2=0, k2=1) -> W2[1,1]=1
    # r2=2 (j2=0, k2=2) -> W2[2,2]=1
    # r2=3 (j2=1, k2=0) -> W2[0,3]=1
    # r2=4 (j2=1, k2=1) -> W2[1,4]=1
    # r2=5 (j2=1, k2=2) -> W2[2,5]=1
    
    W2_correct = [
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1]
    ]
    W2 = W2_correct

    # Construct full U, V, W
    for i in range(n):
        i1 = i // n2 # n2=1, so i1=i
        i2 = i % n2  # i2=0
        
        for j in range(m):
            j1 = j // m2
            j2 = j % m2
            
            # U[i, r]
            for r1 in range(R1):
                val_u1 = U1[i1*m1 + j1][r1]
                if val_u1 == 0: continue
                for r2 in range(R2):
                    val_u2 = U2[j2][r2] # U2 row is j2
                    if val_u2 == 0: continue
                    r = r1 * R2 + r2
                    U[i*m + j][r] = val_u1 * val_u2

    for j in range(m):
        j1 = j // m2
        j2 = j % m2
        
        for k in range(p):
            k1 = k // p2
            k2 = k % p2
            
            # V[j, r]
            for r1 in range(R1):
                val_v1 = V1[j1*m1 + k1][r1]
                if val_v1 == 0: continue
                for r2 in range(R2):
                    val_v2 = V2[j2*p2 + k2][r2]
                    if val_v2 == 0: continue
                    r = r1 * R2 + r2
                    V[j*p + k][r] = val_v1 * val_v2

    for k in range(p):
        k1 = k // p2
        k2 = k % p2
        
        for i in range(n):
            i1 = i // n2
            i2 = i % n2
            
            # W[k, r]
            for r1 in range(R1):
                val_w1 = W1[k1*p1 + i1][r1]
                if val_w1 == 0: continue
                for r2 in range(R2):
                    val_w2 = W2[k2][r2]
                    if val_w2 == 0: continue
                    r = r1 * R2 + r2
                    W[k*n + i][r] = val_w1 * val_w2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
