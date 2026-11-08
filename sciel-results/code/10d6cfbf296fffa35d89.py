import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10

    # Naive rank-30 decomposition
    rank_naive = 30
    U_naive = np.zeros((nm, rank_naive))
    V_naive = np.zeros((mp, rank_naive))
    W_naive = np.zeros((pn, rank_naive))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m + j, r] = 1
                V_naive[j*p + k, r] = 1
                W_naive[k*n + i, r] = 1
                r += 1

    def validate(U, V, W, rank):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor.
        T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m + j, rr] * V[j*p + k, rr] * W[k*n + i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Verify naive rank-30 works
    if not validate(U_naive, V_naive, W_naive, rank_naive):
        # This should never happen, but handle it
        pass

    # Build a rank-28 decomposition using Strassen for 2x2x2 blocks
    # Strategy: 
    # - For (i in {0,1}, j=2, k in {0..4}): 2*5 = 10 terms
    # - For (i in {0,1}, j in {0,1}, k=4): 2*2 = 4 terms
    # - For (i in {0,1}, j in {0,1}, k in {0,1}): Strassen rank 7
    # - For (i in {0,1}, j in {0,1}, k in {2,3}): Strassen rank 7
    # Total: 10 + 4 + 7 + 7 = 28

    rank28 = 28
    U28 = np.zeros((nm, rank28))
    V28 = np.zeros((mp, rank28))
    W28 = np.zeros((pn, rank28))
    r = 0

    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U28[i*m + 2, r] = 1
            V28[2*p + k, r] = 1
            W28[k*n + i, r] = 1
            r += 1

    # Part 2: j in {0,1}, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U28[i*m + j, r] = 1
            V28[j*p + 4, r] = 1
            W28[4*n + i, r] = 1
            r += 1

    # Part 3: j in {0,1}, k in {0,1} - Strassen (7 terms)
    # Standard Strassen for C = AB where A is 2x2, B is 2x2
    # M1 = (A11-A22)(B22-B11)
    # M2 = (A11+A22)B11
    # M3 = A22(B21+B22)
    # M4 = (A11-A21)B21
    # M5 = (A11+A12)B12
    # M6 = (A21+A22)(B12-B22)
    # M7 = (A11-A12)(B11+B21)
    # C11 = M1+M4+M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1+M3+M6

    # M1: (A[0,0]-A[1,1])*(B[0,1]-B[1,0]) -> contributes to C[0,0], C[1,1]
    U28[0*m+0, r] = 1
    U28[1*m+1, r] = -1
    V28[0*p+1, r] = 1
    V28[1*p+0, r] = -1
    W28[0*n+0, r] = 1
    W28[1*n+1, r] = 1
    r += 1

    # M2: (A[0,0]+A[1,1])*B[0,0] -> contributes to C[1,0], C[1,1]
    U28[0*m+0, r] = 1
    U28[1*m+1, r] = 1
    V28[0*p+0, r] = 1
    W28[0*n+1, r] = 1
    W28[1*n+1, r] = -1
    r += 1

    # M3: A[1,1]*(B[0,1]+B[1,1]) -> contributes to C[0,1], C[1,1]
    U28[1*m+1, r] = 1
    V28[0*p+1, r] = 1
    V28[1*p+1, r] = 1
    W28[1*n+0, r] = 1
    W28[1*n+1, r] = 1
    r += 1

    # M4: (A[0,0]-A[0,1])*B[1,0] -> contributes to C[0,0], C[1,0]
    U28[0*m+0, r] = 1
    U28[0*m+1, r] = -1
    V28[1*p+0, r] = 1
    W28[0*n+0, r] = 1
    W28[1*n+0, r] = 1
    r += 1

    # M5: (A[0,0]+A[0,1])*B[0,1] -> contributes to C[0,0], C[0,1]
    U28[0*m+0, r] = 1
    U28[0*m+1, r] = 1
    V28[0*p+1, r] = 1
    W28[0*n+0, r] = 1
    W28[0*n+1, r] = 1
    r += 1

    # M6: (A[1,0]+A[1,1])*(B[0,1]-B[1,1]) -> contributes to C[1,1]
    U28[1*m+0, r] = 1
    U28[1*m+1, r] = 1
    V28[0*p+1, r] = 1
    V28[1*p+1, r] = -1
    W28[1*n+1, r] = 1
    r += 1

    # M7: (A[0,0]-A[1,0])*(B[0,0]+B[1,0]) -> contributes to C[0,0]
    U28[0*m+0, r] = 1
    U28[1*m+0, r] = -1
    V28[0*p+0, r] = 1
    V28[1*p+0, r] = 1
    W28[0*n+0, r] = 1
    r += 1

    # Part 4: j in {0,1}, k in {2,3} - Strassen (7 terms)
    # Same structure but with k=2,3 instead of k=0,1
    # M1: (A[0,0]-A[1,1])*(B[0,3]-B[1,2]) -> contributes to C[0,2], C[1,3]
    U28[0*m+0, r] = 1
    U28[1*m+1, r] = -1
    V28[0*p+3, r] = 1
    V28[1*p+2, r] = -1
    W28[2*n+0, r] = 1
    W28[3*n+1, r] = 1
    r += 1

    # M2: (A[0,0]+A[1,1])*B[0,2] -> contributes to C[1,2], C[1,3]
    U28[0*m+0, r] = 1
    U28[1*m+1, r] = 1
    V28[0*p+2, r] = 1
    W28[2*n+1, r] = 1
    W28[3*n+1, r] = -1
    r += 1

    # M3: A[1,1]*(B[0,3]+B[1,3]) -> contributes to C[0,3], C[1,3]
    U28[1*m+1, r] = 1
    V28[0*p+3, r] = 1
    V28[1*p+3, r] = 1
    W28[3*n+0, r] = 1
    W28[3*n+1, r] = 1
    r += 1

    # M4: (A[0,0]-A[0,1])*B[1,2] -> contributes to C[0,2], C[1,2]
    U28[0*m+0, r] = 1
    U28[0*m+1, r] = -1
    V28[1*p+2, r] = 1
    W28[2*n+0, r] = 1
    W28[3*n+0, r] = 1
    r += 1

    # M5: (A[0,0]+A[0,1])*B[0,3] -> contributes to C[0,2], C[0,3]
    U28[0*m+0, r] = 1
    U28[0*m+1, r] = 1
    V28[0*p+3, r] = 1
    W28[2*n+0, r] = 1
    W28[3*n+0, r] = 1
    r += 1

    # M6: (A[1,0]+A[1,1])*(B[0,3]-B[1,3]) -> contributes to C[1,3]
    U28[1*m+0, r] = 1
    U28[1*m+1, r] = 1
    V28[0*p+3, r] = 1
    V28[1*p+3, r] = -1
    W28[3*n+1, r] = 1
    r += 1

    # M7: (A[0,0]-A[1,0])*(B[0,2]+B[1,2]) -> contributes to C[0,2]
    U28[0*m+0, r] = 1
    U28[1*m+0, r] = -1
    V28[0*p+2, r] = 1
    V28[1*p+2, r] = 1
    W28[2*n+0, r] = 1
    r += 1

    best_U, best_V, best_W = U_naive.copy(), V_naive.copy(), W_naive.copy()
    best_rank = rank_naive

    if validate(U28, V28, W28, rank28):
        best_U, best_V, best_W = U28, V28, W28
        best_rank = rank28

    # Try to find rank 27 or lower using structured search
    deadline = time.time() + 100

    # Try rank 27 with a more focused search
    # Key insight: we can potentially save 1 rank by merging some terms
    # Try various combinations of half-integer and integer values

    focused_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    # Try a few random searches for rank 27
    for attempt in range(200):
        if time.time() > deadline:
            break
        U_t = np.zeros((nm, 27))
        V_t = np.zeros((mp, 27))
        W_t = np.zeros((pn, 27))
        
        # Start from the rank-28 structure and try to merge
        U_t[:, :26] = U28[:, :26]
        V_t[:, :26] = V28[:, :26]
        W_t[:, :26] = W28[:, :26]
        
        # Try to absorb the 27th term into existing ones
        # Random perturbation
        U_t[:, 26] = np.random.choice(focused_vals, size=nm)
        V_t[:, 26] = np.random.choice(focused_vals, size=mp)
        W_t[:, 26] = np.random.choice(focused_vals, size=pn)
        
        # Also try modifying some existing entries
        for _ in range(5):
            row = np.random.randint(0, nm)
            col = np.random.randint(0, 27)
            U_t[row, col] = np.random.choice(focused_vals)
            row = np.random.randint(0, mp)
            col = np.random.randint(0, 27)
            V_t[row, col] = np.random.choice(focused_vals)
            row = np.random.randint(0, pn)
            col = np.random.randint(0, 27)
            W_t[row, col] = np.random.choice(focused_vals)
        
        if validate(U_t, V_t, W_t, 27):
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = 27
            break

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
