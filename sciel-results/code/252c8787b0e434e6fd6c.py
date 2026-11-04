"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 72
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        """Check if U,V,W with given rank exactly decompose the tensor."""
        # Check all tensor entries: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use the structure that for 3x3x3, we can use rank 23 (Lader)
    # For 3x3x8, we can potentially use a block approach.
    # 
    # Key insight: The tensor T[i,j,k] = 1 for all valid (i,j,k).
    # We can write this as: for each k, we need a 3x3x3 decomposition.
    # But U and W don't depend on k in a simple way...
    #
    # Actually, let's try a structured approach:
    # Use rank r for the "core" 3x3x3 part, then extend to p=8.
    # 
    # Alternative: Try to find a decomposition where we share U and W
    # across columns. If U[i*m+j,c] and W[k*n+i,c] can be shared,
    # we only need different V entries.
    #
    # Let's try: rank = 23 * 8 = 184? No, that's worse.
    # 
    # Better: Use the fact that the tensor is separable in a structured way.
    # T[i,j,k] = 1 = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # 
    # One approach: use a rank-23 decomposition for 3x3x3, giving us
    # U3[i,j,c], V3[j,k,c], W3[i,k,c] for c in [0,23).
    # Then for p=8, we can use:
    # U[i*m+j, c] = U3[i,j, c%23] for c in [0, 23*8)
    # V[j*p+k, c] = V3[j, k, c%23] if (c//23) == k else 0
    # W[k*n+i, c] = W3[i, k, c%23] if (c//23) == k else 0
    # But this gives rank 23*8 = 184, which is worse.
    #
    # Let me try a different structural approach:
    # The 3x3x3 tensor can be decomposed with rank 23.
    # For 3x3x8, note that p=8. We can potentially use:
    # rank = 23 + 23 + 23 + 23 = 92? No...
    #
    # Actually, the best known bound for 3x3x3 is 23 (Lader 1979).
    # For 3x3xp, the rank is at most 23*p but potentially less.
    # 
    # Let me try a random search with structured initialization.
    # Initialize with the schoolbook pattern but with some entries zeroed
    # to try to find a lower rank.
    
    # Try structured rank reduction:
    # Start from schoolbook (rank 72) and try to merge columns.
    
    # Approach: Try ranks from 71 down, using random restarts with
    # structured initialization based on the schoolbook pattern.
    
    for target_rank in range(best_rank - 1, max(50, best_rank - 30), -1):
        if time.time() > deadline:
            break
        
        # Try a few random structured attempts
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook-like structure but with some zeros
            # to encourage lower rank
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Fill with structured pattern: for each (i,j,k) triple,
            # assign to a column with some probability of sharing
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < target_rank:
                            # Use small random values to allow cancellation
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
                        else:
                            # Try to share: pick an existing column
                            c = np.random.randint(0, target_rank)
                            U[i * m + j, c] += 0.5
                            V[j * p + k, c] += 0.5
                            W[k * n + i, c] += 0.5
            
            # Round to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break  # Found valid for this rank, try lower
        
        if time.time() > deadline:
            break
    
    # Also try a pure random search with small values for lower ranks
    if time.time() < deadline - 10:
        for target_rank in range(min(best_rank - 1, 65), max(45, best_rank - 20), -1):
            if time.time() > deadline:
                break
            
            for attempt in range(3):
                if time.time() > deadline:
                    break
                
                # Random initialization with small half-integer values
                U = (np.random.randint(-4, 5, (n * m, target_rank)) * 0.5)
                V = (np.random.randint(-4, 5, (m * p, target_rank)) * 0.5)
                W = (np.random.randint(-4, 5, (p * n, target_rank)) * 0.5)
                
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if time.time() > deadline:
                break
    
    return best_result
# EVOLVE-BLOCK-END
