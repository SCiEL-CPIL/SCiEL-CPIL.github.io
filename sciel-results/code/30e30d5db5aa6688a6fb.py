"""Deterministic rank-reduction for 4x5 * 5x6 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the schoolbook decomposition (rank = n*m*p = 120)
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=np.int64)
    V = np.zeros((m * p, rank_school), dtype=np.int64)
    W = np.zeros((p * n, rank_school), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    def verify(U_arr, V_arr, W_arr, r):
        """Verify decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
        if time.time() > deadline - 5:
            return False
        U_r = U_arr.reshape(n, m, r)
        V_r = V_arr.reshape(m, p, r)
        W_r = W_arr.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Verify schoolbook
    if not verify(U, V, W, rank_school):
        # Fallback: return schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strategy: Try to reduce rank by finding linear dependencies
    # The tensor equation is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #
    # For the schoolbook, each term c corresponds to (i_c, j_c, k_c) with:
    #   U[i_c*m+j_c, c] = 1, V[j_c*p+k_c, c] = 1, W[k_c*n+i_c, c] = 1
    #
    # We can try to merge terms where the (i,j), (j,k), (k,i) indices overlap
    # in a way that allows algebraic cancellation.
    #
    # Better approach: Use the fact that for fixed (i,j), the sum over k of
    # V[j,k,c]*W[k,i,c] must equal 1 for the term where U[i,j,c]=1.
    #
    # Let's try a direct rank-reduction via structured elimination:
    # Group terms by (i,j) pairs. For each (i,j), we have p terms (one per k).
    # The constraint is: sum_k V[j,k,c_k]*W[k,i,c_k] = 1 for the active terms.
    #
    # Alternative: Try to find a rank-r decomposition by solving the system.
    # For small r, we can use the following constructive approach:
    #
    # Partition the problem: Write A (4x5) and B (5x6).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can use the identity: for any partition of j into two groups,
    # we can apply Strassen-like tricks. But for 4x5x6, let's try:
    #
    # Direct approach: Try ranks from 119 down, and for each rank,
    # attempt to solve the system using the structure.

    best_U, best_V, best_W = U, V, W
    best_rank = rank_school

    # Try to reduce rank by 1 at a time using structured search
    # For each candidate rank r < 120, try to construct a valid decomposition
    #
    # Key idea: If we can find a set of r rank-1 terms that span the tensor,
    # we're done. Start from schoolbook and try to eliminate one term at a time.
    
    # For rank 119: try to merge two schoolbook terms into one
    # Term c1 = (i1,j1,k1), term c2 = (i2,j2,k2)
    # They can be merged if they share structure.
    
    # Actually, let's use a more systematic approach:
    # The multiplication tensor M_{4,5,6} has a known structure.
    # Let's try to use the "border rank" or "tensor rank" reduction.
    
    # Practical approach: Try random integer decompositions for lower ranks
    # with verification, using numpy for speed.
    
    np.random.seed(42)
    
    # Try a few target ranks with random search
    for target_rank in [119, 118, 117, 116, 115]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize with schoolbook and perturb
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Try to zero out one column and adjust others
            # Pick a random column to eliminate
            col_to_elim = np.random.randint(0, rank_school)
            
            # Get the (i,j,k) indices for this column
            # col = i*m*p + j*p + k in schoolbook ordering
            i_idx = col_to_elim // (m * p)
            j_idx = (col_to_elim // p) % m
            k_idx = col_to_elim % p
            
            # We need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # If we remove column col_to_elim, we need other columns to compensate.
            # The contribution of col_to_elim to (i_idx,j_idx,k_idx) is 1*1*1 = 1.
            # We need another term or combination to provide this.
            
            # Find a column that shares i_idx and j_idx (same U pattern)
            # or shares j_idx and k_idx (same V pattern)
            
            # Try adding the contribution to a nearby term
            # Find a term with same (i,j) but different k
            target_col = None
            for k2 in range(p):
                if k2 != k_idx:
                    c2 = i_idx * m * p + j_idx * p + k2
                    # Check if this column exists and can absorb
                    # We'd need to modify W[k2*n+i_idx, c2] and W[k_idx*n+i_idx, c2]
                    # This gets complex. Let's try a different approach.
                    target_col = c2
                    break
            
            if target_col is None:
                continue
            
            # Try: set U[i,j,col_to_elim] = 0 and adjust another term
            # This is heuristic. Let's just try random small modifications.
            
            # Random approach: generate random integer matrices and verify
            U_rand = np.random.randint(-2, 3, size=(n * m, target_rank))
            V_rand = np.random.randint(-2, 3, size=(m * p, target_rank))
            W_rand = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            # Quick check with a subset
            if verify(U_rand, V_rand, W_rand, target_rank):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Also try a structured approach: use the fact that we can combine
    # terms with the same (i,j) by using the linearity in k.
    #
    # For fixed (i,j): C[i,k] = sum_j A[i,j]*B[j,k]
    # The j-th contribution is A[i,j] * B[j, :]
    # We can write B[j, :] as a sum of rank-1 terms if we allow
    # more scalar multiplications.
    
    # Let's try a specific construction for rank < 120:
    # Use the identity: for 2x2 matrices, rank 7 < 8 (Strassen)
    # For 4x5x6, we can embed Strassen-like blocks.
    
    # Partition A into 2x2 blocks of size 2x2 (covering 4x4 of the 4x5)
    # and B into 2x2 blocks of size 2x2 (covering 4x4 of the 5x6)
    # This is getting complex. Let's stick with the random search + schoolbook fallback.
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
