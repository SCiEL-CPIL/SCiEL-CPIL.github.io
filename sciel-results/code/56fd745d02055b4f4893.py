"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Matrix multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) with T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Naive rank-60 decomposition
    rank_naive = n * m * p  # 60
    U_naive = np.zeros((n*m, rank_naive))
    V_naive = np.zeros((m*p, rank_naive))
    W_naive = np.zeros((p*n, rank_naive))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m + j, r] = 1
                V_naive[j*p + k, r] = 1
                W_naive[k*n + i, r] = 1
                r += 1
    
    best_rank = rank_naive
    best_decomp = (U_naive.copy(), V_naive.copy(), W_naive.copy())
    
    def verify_fast(U, V, W, tol=1e-9):
        """Fast vectorized verification."""
        n, m, p = DIMENSIONS
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_all(U, V, W, tol=1e-9):
        """Verify all (i,j,k) triples."""
        n, m, p = DIMENSIONS
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Use tensor train / HOOI-inspired rank reduction
    # Start from naive decomposition and iteratively try to reduce rank
    
    # First, try the Strassen-based block approach for structured reduction
    # Split j into pairs and use 7-product Strassen-like schemes
    
    def try_strassen_blocks():
        """Try Strassen-based block decomposition."""
        n, m, p = DIMENSIONS
        
        # Split j = {0,1,2,3,4} into pairs: (0,1), (2,3), and singleton {4}
        # Split k = {0,1,2,3,4,5} into pairs: (0,1), (2,3), (4,5)
        
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        # For each (j_pair, k_pair), use 7 Strassen products for the 2x2 block
        # Plus handle the singleton j=4 with naive 2*6=12 terms
        
        # Total: 2 * 3 * 7 + 12 = 42 + 12 = 54
        target_rank = 2 * len(j_pairs) * len(k_pairs) * 7 + n * p  # 42 + 12 = 54
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen products for 2x2 * 2x2 block
                # We're computing C[i, k0..k1] for i in {0,1} using A[:, j0..j1] * B[j0..j1, :]
                
                # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
                if col < target_rank:
                    U[0*m+j0, col] += 1
                    U[1*m+j1, col] += 1
                    V[j0*p+k0, col] += 1
                    V[j1*p+k1, col] += 1
                    W[k0*n+0, col] += 1
                    W[k1*n+1, col] += 1
                    col += 1
                
                # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
                if col < target_rank:
                    U[1*m+j0, col] += 1
                    U[1*m+j1, col] += 1
                    V[j0*p+k0, col] += 1
                    W[k0*n+1, col] += 1
                    W[k1*n+1, col] -= 1
                    col += 1
                
                # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
                if col < target_rank:
                    U[0*m+j0, col] += 1
                    V[j0*p+k1, col] += 1
                    V[j1*p+k1, col] -= 1
                    W[k1*n+0, col] += 1
                    W[k1*n+1, col] += 1
                    col += 1
                
                # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
                if col < target_rank:
                    U[1*m+j1, col] += 1
                    V[j1*p+k0, col] += 1
                    V[j0*p+k0, col] -= 1
                    W[k0*n+0, col] += 1
                    W[k0*n+1, col] += 1
                    col += 1
                
                # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
                if col < target_rank:
                    U[0*m+j0, col] += 1
                    U[0*m+j1, col] += 1
                    V[j1*p+k1, col] += 1
                    W[k0*n+0, col] -= 1
                    W[k1*n+0, col] += 1
                    col += 1
                
                # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
                if col < target_rank:
                    U[1*m+j0, col] += 1
                    U[0*m+j0, col] -= 1
                    V[j0*p+k0, col] += 1
                    V[j0*p+k1, col] += 1
                    W[k1*n+1, col] += 1
                    col += 1
                
                # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
                if col < target_rank:
                    U[0*m+j1, col] += 1
                    U[1*m+j1, col] -= 1
                    V[j1*p+k0, col] += 1
                    V[j1*p+k1, col] += 1
                    W[k0*n+0, col] += 1
                    col += 1
        
        # Handle j=4 (singleton) with naive terms
        for k in range(p):
            if col < target_rank:
                U[0*m+4, col] += 1
                V[4*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
            if col < target_rank:
                U[1*m+4, col] += 1
                V[4*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        if col == target_rank and verify_fast(U, V, W):
            return target_rank, U.copy(), V.copy(), W.copy()
        return None
    
    # Try the Strassen block approach
    result = try_strassen_blocks()
    if result is not None:
        best_rank, U_best, V_best, W_best = result
        best_decomp = (U_best, V_best, W_best)
    
    # Direct sum decomposition: split j into {0,1} and {2,3,4}
    # Use Strassen (7 terms) for 2x2x2 blocks and naive for 2x3x2 blocks
    # Then combine with k-pairing
    
    def try_improved_strassen():
        """Try improved Strassen with different block structures."""
        n, m, p = DIMENSIONS
        
        # Option 1: Split j into (0,1) and (2,3,4), k into pairs
        # For (j_pair, k_pair): 7 Strassen terms
        # For (j_triple, k_pair): 3*2=6 naive terms per k-pair, so 6 terms
        # Total: 2*3*7 + 3*3*2 = 42 + 18 = 60. Not better.
        
        # Option 2: Use a 5-term scheme for 2x3x1 if it exists
        # Rank of [2x3x1] is 6 (lower bound = max(6,3,2)=6), so no 5-term scheme.
        
        # Option 3: Try to share terms across k-pairs
        # This is complex and may not work.
        
        # For now, return None and rely on the search below
        return None
    
    result2 = try_improved_strassen()
    if result2 is not None:
        r2, U2, V2, W2 = result2
        if r2 < best_rank:
            best_rank = r2
            best_decomp = (U2, V2, W2)
    
    # Systematic search for lower ranks using random linear combinations
    rng = np.random.default_rng(123)
    
    if best_rank <= 54 and time.time() < DEADLINE - 30:
        U_curr, V_curr, W_curr = best_decomp
        
        # Try to find linear dependencies by checking if removing a term
        # and adjusting others works
        for target_rank in range(best_rank - 1, max(29, best_rank - 10), -1):
            if time.time() > DEADLINE - 25:
                break
            
            max_attempts = 50
            
            for attempt in range(max_attempts):
                if time.time() > DEADLINE - 20:
                    break
                
                # Randomly select which terms to keep
                keep_mask = np.ones(best_rank, dtype=bool)
                num_remove = best_rank - target_rank
                remove_indices = rng.choice(best_rank, size=num_remove, replace=False)
                keep_mask[remove_indices] = False
                
                U_try = U_curr[:, keep_mask].copy()
                V_try = V_curr[:, keep_mask].copy()
                W_try = W_curr[:, keep_mask].copy()
                
                # Try to absorb removed terms into kept terms with random weights
                for r_rem in remove_indices:
                    r_target = rng.integers(0, target_rank)
                    weight = rng.choice([-1, 1, 0.5, -0.5])
                    U_try[:, r_target] += weight * U_curr[:, r_rem]
                    V_try[:, r_target] += weight * V_curr[:, r_rem]
                    W_try[:, r_target] += weight * W_curr[:, r_rem]
                
                if verify_fast(U_try, V_try, W_try):
                    best_rank = target_rank
                    best_decomp = (U_try.copy(), V_try.copy(), W_try.copy())
                    break
    
    # Use the best decomposition found
    U_best, V_best, W_best = best_decomp
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U_best]
    V_list = [[float(x) for x in row] for row in V_best]
    W_list = [[float(x) for x in row] for row in W_best]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
