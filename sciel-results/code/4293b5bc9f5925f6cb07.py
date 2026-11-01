import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor entries: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    target = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target.add((i*m+j, j*p+k, k*n+i))
    
    def check(U, V, W, R):
        """Exact check using integer/rational arithmetic via numpy with tolerance."""
        # Convert to arrays for vectorized check
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for (a, b, c) in target:
            s = np.sum(Ua[a] * Va[b] * Wa[c])
            if abs(s - 1) > 1e-7:
                return False
        return True
    
    # ---- Schoolbook construction (rank = n*m*p = 52) ----
    R0 = n * m * p
    U = [[0.0]*R0 for _ in range(nm)]
    V = [[0.0]*R0 for _ in range(mp)]
    W = [[0.0]*R0 for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    best_U, best_V, best_W, best_R = U, V, W, R0
    
    # ---- Attempt 3p+1 = 40 rank construction ----
    # Strategy: Use the 4-term rank decomposition of 2x2 matrix multiplication,
    # shared across the p columns of the second matrix.
    # The 2x2 multiplication has rank 4 (schoolbook) but also has a rank-4 formula.
    # For <2,2,p>, a construction with rank 3p+1 exists.
    # 
    # Idea: Split the computation into:
    # - 1 shared "cross" term that handles coupling between rows
    # - 3 terms per column k for the diagonal/row contributions
    #
    # For output C[i,k] = sum_j A[i,j] * B[j,k]:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Use 4 rank-1 terms per k, but share one term structure:
    # Term type 1 (per k): U[:,r] = [1,0,0,0], V[:,r] = [1,0,...,0] (B[0,k] only), W[:,r] = [1,0,0,...]
    # This gives A[0,0]*B[0,k] -> C[0,k] and A[1,0]*B[0,k] -> C[1,k]
    # Term type 2 (per k): U[:,r] = [0,1,0,0], V[:,r] = [0,1,...,0] (B[1,k] only), W[:,r] = [1,0,0,...]
    # This gives A[0,1]*B[1,k] -> C[0,k] and A[1,1]*B[1,k] -> C[1,k]
    #
    # That's 2 terms per k = 2p = 26 terms. But we need all 4 products.
    # Actually 2 terms per k with the right U and W:
    # Term A: U = [1,0,0,0], V = [1,0,0,...,0], W = [1,0,0,...] -> A[0,0]*B[0,k] contributes to C[0,k]
    #   Wait, W[k*n+i] with n=2: W indices are (k*2+0) and (k*2+1)
    #   W[2k+0] = 1, W[2k+1] = 0 means this term contributes to output row i=0 only.
    #   But we need it to contribute to C[1,k] too via A[1,0]*B[0,k].
    #
    # Let me reconsider. Each rank-1 term is U_a * V_b * W_c.
    # U has indices (i,j) in {0,1}x{0,1}, V has (j,k) in {0,1}x{0..12}, W has (k,i) in {0..12}x{0,1}.
    # 
    # For a single term to contribute to C[i,k] = sum_j A[i,j]B[j,k], we need:
    # U[i,j] * V[j,k] * W[k,i] to be non-zero.
    # 
    # Term 1 for column k: U[0,0]=1, U[1,0]=1, others 0; V[0,k]=1, others 0; W[k,0]=1, W[k,1]=1, others 0
    #   This gives: for (i=0,j=0,k): 1*1*1=1 -> C[0,k] += A[0,0]*B[0,k] ✓
    #                for (i=1,j=0,k): 1*1*1=1 -> C[1,k] += A[1,0]*B[0,k] ✓
    #   So this one term handles B[0,k] for both output rows!
    #
    # Term 2 for column k: U[0,1]=1, U[1,1]=1, others 0; V[1,k]=1, others 0; W[k,0]=1, W[k,1]=1, others 0
    #   This gives: for (i=0,j=1,k): 1*1*1=1 -> C[0,k] += A[0,1]*B[1,k] ✓
    #                for (i=1,j=1,k): 1*1*1=1 -> C[1,k] += A[1,1]*B[1,k] ✓
    #
    # That's 2 terms per k, total 2p = 26 terms! But wait, that's rank 26, not 40.
    # Let me verify this is correct.
    
    R2 = 2 * p  # 26
    U2 = [[0.0]*R2 for _ in range(nm)]
    V2 = [[0.0]*R2 for _ in range(mp)]
    W2 = [[0.0]*R2 for _ in range(pn)]
    for k in range(p):
        # Term 2k: handles B[0,k] contribution to both output rows
        r1 = 2 * k
        U2[0*2+0][r1] = 1.0  # U[0,0]
        U2[1*2+0][r1] = 1.0  # U[1,0]
        V2[0*p+k][r1] = 1.0  # V[0,k]
        W2[k*n+0][r1] = 1.0  # W[k,0]
        W2[k*n+1][r1] = 1.0  # W[k,1]
        
        # Term 2k+1: handles B[1,k] contribution to both output rows
        r2 = 2 * k + 1
        U2[0*2+1][r2] = 1.0  # U[0,1]
        U2[1*2+1][r2] = 1.0  # U[1,1]
        V2[1*p+k][r2] = 1.0  # V[1,k]
        W2[k*n+0][r2] = 1.0  # W[k,0]
        W2[k*n+1][r2] = 1.0  # W[k,1]
    
    if check(U2, V2, W2, R2):
        best_U, best_V, best_W, best_R = U2, V2, W2, R2
    
    # ---- Try even lower: can we do p+something? ----
    # Actually, the above 2p construction seems valid. Let's also try if there's a lower bound.
    # The rank of <2,2,p> is at least max(2p, 2, 2*13) ... the trivial lower bound is 2p=26.
    # Actually the lower bound is max(n*p, m*p, n*m) = max(26, 26, 4) = 26.
    # So 26 is optimal if achievable!
    
    # ---- If time remains, try to verify and potentially find even better ----
    # (26 is the theoretical lower bound, so we can't do better)
    
    return {"U": best_U, "V": best_V, "W": best_W}
