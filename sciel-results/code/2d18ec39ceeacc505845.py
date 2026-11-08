"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Rank-20 construction for 2x3 x 3x9 matrix multiplication
    # beta(2, 3, 9) = 2*9 + 3 - 1 = 20
    # 
    # Tensor convention: T[i,j,k] = delta(i,j,k) for the (i,j,k)-th entry
    # where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Known construction for n=2, general m,p with p >= m:
    # For k in 0..p-1:
    #   r = 2*k:     U[0*m+0, r]=1, V[0*p+k, r]=1, W[k*n+0, r]=1
    #   r = 2*k+1:   U[1*m+0, r]=1, V[0*p+k, r]=1, W[k*n+1, r]=1
    # For j in 1..m-1:
    #   r = 2*p + j - 1:
    #     U[0*m+j, r] = 1, U[1*m+j, r] = 1
    #     For k in 0..p-1: V[j*p+k, r] = 1
    #     For k in 0..p-1: W[k*n+0, r] = -1, W[k*n+1, r] = 1
    #
    # Wait, let me verify this more carefully.
    # We need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta(i,i')*delta(j,j')*delta(k,k')
    #
    # Actually the correct known construction is:
    # For each k in 0..p-1, i in 0..n-1:
    #   r = i*p + k
    #   U[i*m + 0, r] = 1
    #   V[0*p + k, r] = 1
    #   W[k*n + i, r] = 1
    # For each j in 1..m-1:
    #   r = n*p + j - 1
    #   U[i*m + j, r] = 1 for i in 0..n-1
    #   V[j*p + k, r] = 1 for k in 0..p-1
    #   W[k*n + i, r] = 0 ... this doesn't close properly
    #
    # Let me use the standard Winograd-style construction for 2xm x mxp.
    # The correct rank-(2p+m-1) construction:
    #
    # For k = 0..p-1:
    #   For i = 0..1:
    #     r = 2*k + i
    #     U[i*m + 0, r] = 1
    #     V[0*p + k, r] = 1
    #     W[k*n + i, r] = 1
    # For j = 1..m-1:
    #   r = 2*p + j - 1
    #   For i = 0..1:
    #     U[i*m + j, r] = 1
    #   For k = 0..p-1:
    #     V[j*p + k, r] = 1
    #     W[k*n + 0, r] = 0
    #     W[k*n + 1, r] = 0
    #   # But we need W entries to make this work...
    #
    # Let me just verify computationally with a known-good approach.
    # The simplest correct construction I can verify:
    
    rank = 2 * p + m - 1  # 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2p terms handling j=0
    for k in range(p):
        for i in range(n):
            r = i * p + k
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
    
    # Part 2: m-1 terms handling j=1..m-1
    # For each j in 1..m-1, use one rank-1 term
    # U[i*m+j, r] = 1 for all i
    # V[j*p+k, r] = 1 for all k  
    # W[k'*n+i', r] must be 0 unless... 
    # This doesn't work directly because we need W to select specific (k',i')
    #
    # Alternative: use the fact that for j>0, we can write:
    # A[i,j] * B[j,k] = A[i,j] * B[j,k]
    # We need separate rank-1 terms or a clever combination.
    #
    # Actually, the correct construction uses:
    # For j in 1..m-1:
    #   r = 2*p + j - 1
    #   U[0*m+j, r] = 1
    #   U[1*m+j, r] = 1
    #   V[j*p+k, r] = 1 for k=0..p-1
    #   W[k*n+0, r] = 0 for all k
    #   W[k*n+1, r] = 0 for all k
    # This gives zero contribution, which is wrong.
    #
    # Let me try a different known construction. For 2x3 x 3x9:
    # Use the formula from Strassen-type results.
    # 
    # Actually, let me just verify the simple construction works:
    # For j=0: 2p terms as above.
    # For j>0: we need to handle A[i,j]*B[j,k] for i=0,1 and k=0..p-1
    # That's 2*p*(m-1) = 2*9*2 = 36 more terms if done naively.
    # But the bound is 2p+m-1 = 20, so we need to be smarter.
    #
    # The key insight: for j>0, we can share the k-index.
    # For a fixed j>0:
    #   A[0,j]*B[j,k] and A[1,j]*B[j,k] for k=0..p-1
    # We can use p terms if we could factor, but we can't directly.
    #
    # Let me try: for j in 1..m-1, use p terms where:
    #   r = 2*p + (j-1)*p + k for k=0..p-1
    #   U[0*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+0, r] = 1
    #   U[1*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+1, r] = 1
    # But that's 2p terms per j, giving 2p + 2p*(m-1) = 2p*m = 54.
    #
    # I think the rank-20 bound requires a more sophisticated construction.
    # Let me just use the schoolbook for now and try to improve.
    
    # Fallback: schoolbook rank 54
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
