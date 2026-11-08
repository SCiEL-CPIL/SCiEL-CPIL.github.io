import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Define Strassen decomposition for 2x2 x 2x2 (Rank 7)
    # U1: (4, 7), V1: (4, 7), W1: (4, 7)
    # n=2, m=2, p=2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+ch], [ce+dg, cf+dh]]
    # W index: k*n + i. C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
    
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # p1 = (a+d)(e+g)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[2, 0] = 1
    W1[0, 0] = 1; W1[1, 0] = 1; W1[2, 0] = 1; W1[3, 0] = 1
    
    # p2 = (c+d)e
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    
    # p3 = a(f-g)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[2, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    
    # p4 = d(h-e)
    U1[3, 3] = 1
    V1[3, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = -1; W1[1, 3] = 1
    
    # p5 = (a+b)g
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[2, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # p6 = (c+a)(h+f)
    U1[2, 5] = 1; U1[0, 5] = 1
    V1[3, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = -1
    
    # p7 = (b+d)(g-e)
    U1[1, 6] = 1; U1[3, 6] = 1
    V1[2, 6] = 1; V1[0, 6] = -1
    W1[0, 6] = 1

    # 2. Define decomposition for 2x2 x 2x3 (Rank 11)
    # U2: (4, 11), V2: (6, 11), W2: (6, 11)
    # n=2, m=2, p=3
    # A = [[a, b], [c, d]]
    # B = [[e, f, g], [h, i, j]]
    # V indices: e=0, f=1, g=2, h=3, i=4, j=5
    # W indices: C is 2x3. k*n+i.
    # C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5
    
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float)
    W2 = np.zeros((6, 11), dtype=float)
    
    # First 7 columns: Strassen for first 2 cols of B (e, f, h, i)
    # Note: In 2x2x2 Strassen, B was [[e,f],[g,h]].
    # Here B cols 0,1 are [[e,f],[h,i]]. So g->h, h->i.
    # We map V1 indices to V2 indices:
    # V1[0] (e) -> V2[0] (e)
    # V1[1] (f) -> V2[1] (f)
    # V1[2] (g) -> V2[3] (h)
    # V1[3] (h) -> V2[4] (i)
    
    # Copy U1 to U2 (first 7 cols)
    U2[:, 0:7] = U1
    
    # Map V1 to V2
    V2[0, 0:7] = V1[0, 0:7] # e
    V2[1, 0:7] = V1[1, 0:7] # f
    V2[3, 0:7] = V1[2, 0:7] # h (was g)
    V2[4, 0:7] = V1[3, 0:7] # i (was h)
    
    # Map W1 to W2
    # W1 indices: 0(C00), 1(C10), 2(C01), 3(C11)
    # W2 indices: 0(C00), 1(C10), 2(C01), 3(C11), 4(C02), 5(C12)
    W2[0, 0:7] = W1[0, 0:7]
    W2[1, 0:7] = W1[1, 0:7]
    W2[2, 0:7] = W1[2, 0:7]
    W2[3, 0:7] = W1[3, 0:7]
    
    # Last 4 columns: Schoolbook for 3rd col (g, j)
    # ag
    U2[0, 7] = 1
    V2[2, 7] = 1 # g
    W2[4, 7] = 1 # C[0,2]
    
    # bj
    U2[1, 8] = 1
    V2[5, 8] = 1 # j
    W2[4, 8] = 1 # C[0,2]
    
    # cg
    U2[2, 9] = 1
    V2[2, 9] = 1 # g
    W2[5, 9] = 1 # C[1,2]
    
    # dj
    U2[3, 10] = 1
    V2[5, 10] = 1 # j
    W2[5, 10] = 1 # C[1,2]

    # 3. Kronecker Product
    # U_total = kron(U1, U2)
    # V_total = kron(V1, V2)
    # W_total = kron(W1, W2)
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook if we are too slow (unlikely)
        return _schoolbook()

    U_total = np.kron(U1, U2)
    V_total = np.kron(V1, V2)
    W_total = np.kron(W1, W2)
    
    # Convert to list of lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
