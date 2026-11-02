"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # We will attempt to find a rank-minimal decomposition.
    # Known results:
    # Rank of M_{2,2,2} is 7 (Strassen).
    # Rank of M_{2,2,1} is 4 (Schoolbook).
    # Simple block decomposition of 15 columns into 7 pairs + 1 single:
    # Rank = 7*7 + 4 = 53.
    
    # Can we do better?
    # The tensor is M_{2,2,15}.
    # Lower bound: max(2*2, 2*15, 15*2) = 30.
    # Upper bound: 53.
    # 
    # Let's try to construct a decomposition of rank R.
    # The tensor T has indices (i, j, k) with i in {0,1}, j in {0,1}, k in {0..14}.
    # T[i,j,k] = 1.
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    #
    # Strategy:
    # 1. Start with the rank-53 block decomposition as a baseline.
    # 2. Try to reduce the rank by finding linear dependencies or using a smarter construction.
    # 3. Use a local search / optimization to see if we can reduce the rank.
    #
    # However, finding a rank reduction for a specific tensor instance is hard.
    # Let's first generate the rank-53 solution.
    
    # Strassen coefficients for M_{2,2,2}
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook coefficients for M_{2,2,1}
    U_school = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_school = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_school = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    def build_block_decomposition():
        rank = 7 * 7 + 4
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col_offset = 0
        for block in range(7):
            k0 = 2 * block
            k1 = 2 * block + 1
            
            for r in range(7):
                c = col_offset + r
                # U
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, c] = U_strassen[i * 2 + j, r]
                # V
                for j in range(2):
                    V[j * p + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
                # W
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk * n + i, c] = W_strassen[k * 2 + i, r]
            col_offset += 7
        
        # Last block
        k_last = 14
        for r in range(4):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j, c] = U_school[i * 2 + j, r]
            for j in range(2):
                V[j * p + k_last, c] = V_school[j, r]
            for i in range(2):
                W[k_last * n + i, c] = W_school[i, r]
                
        return U, V, W

    U_base, V_base, W_base = build_block_decomposition()
    best_rank = U_base.shape[1]
    best_U, best_V, best_W = U_base, V_base, W_base

    # Verification function
    def verify(U, V, W):
        # T[i,j,k] = 1
        # Check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook if something went wrong
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank_schoolbook

    # Attempt to reduce rank
    # We can try to remove columns that are linearly dependent or redundant.
    # However, tensor rank reduction is not just matrix rank reduction.
    # A column c is redundant if T - sum_{c' != c} term_{c'} can be represented with rank 0? No.
    # A column c is redundant if there exist U', V', W' of size (rows, R-1) such that
    # sum_{c'=0}^{R-2} U'[:,c'] V'[:,c'] W'[:,c'] = T.
    
    # Heuristic: Try to set one column to zero and see if we can solve for the rest?
    # This is a nonlinear feasibility problem.
    
    # Alternative: Use the fact that M_{2,2,15} might have a lower rank.
    # According to literature, rank of M_{2,2,p} is 2p+2 for p>=2?
    # If so, rank(2,2,15) = 32.
    # Let's try to find a rank-32 decomposition.
    
    # Construction for rank 2p+2:
    # This is a known construction for rectangular matrices.
    # See "On the rank of the matrix multiplication tensor" or similar.
    # For 2x2xp, the rank is indeed 2p+2 for p >= 2.
    # Let's try to construct it.
    
    # Reference: 
    # The rank of M_{2,2,p} is 2p+2.
    # Construction:
    # Let R = 2p+2.
    # We need to fill U (4xR), V (2pxR), W (2pxR).
    
    # One such construction uses the following pattern:
    # For k in 0..p-1:
    #   We have 2 terms associated with k.
    #   Plus 2 extra terms at the end.
    
    # Let's try to derive or find the specific coefficients.
    # This might be complex to derive from scratch in the time limit.
    
    # Let's try a simpler heuristic:
    # Start with the rank-53 solution.
    # Try to remove columns one by one and solve for the new coefficients using a local optimizer?
    # Too slow.
    
    # Let's try to construct the rank-32 solution directly if we can find the pattern.
    # Pattern for 2x2xp with rank 2p+2:
    # 
    # U[0, c] = 1 if c in 0..p-1 (for i=0,j=0)
    # ...
    
    # Actually, let's look at the structure of M_{2,2,2} (Strassen).
    # It uses 7 terms. 2*2+2 = 6. Wait, 2p+2 for p=2 is 6. But Strassen is 7.
    # So the formula 2p+2 is not correct for p=2.
    # The rank of M_{2,2,2} is 7.
    # The rank of M_{2,2,3} is ?
    # 
    # Let's check the lower bound again.
    # For M_{2,2,p}, the slice rank lower bound is max(2*2, 2*p, p*2) = 2p.
    # Strassen gives 7 for p=2. 2p=4.
    # 
    # There is a result that rank(M_{2,2,p}) = 2p+2 for p >= 3?
    # If p=3, rank = 8?
    # If p=15, rank = 32?
    
    # Let's try to verify if a rank-32 solution exists by attempting to construct one.
    # If we can't construct it quickly, we return the rank-53 solution.
    
    # Let's try a random search for a rank-32 solution with a time limit.
    # This is unlikely to succeed with random search due to the non-convexity and high dimension.
    
    # Instead, let's try to improve the rank-53 solution by checking for linear dependencies
    # in the "columns" of the tensor decomposition in a specific way.
    
    # For now, we return the rank-53 solution.
    # It is a significant improvement over the schoolbook rank of 60.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
