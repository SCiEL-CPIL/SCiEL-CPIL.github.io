<<<<<<< SEARCH
def _search_rank(r, max_c=2, time_budget=10.0):
    """
    Search for a rank-r decomposition with entries in [-max_c, max_c].
    Uses a structured approach: fix U to be sparse, then solve for V and W.
    """
    start_time = time.time()
    
    # The identity is: for all (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # There are n*m*p = 56 constraints.
    # We have r*(NM + MP + PN) = r*(8+28+14) = 50*r unknowns.
    
    # For r=28: 50*28 = 1400 unknowns, 56 constraints. Very underdetermined.
    # For r=30: 50*30 = 1500 unknowns, 56 constraints.
    
    # Strategy: Use a greedy/iterative approach.
    # 1. Initialize U, V, W as zero matrices of shape (NM, r), (MP, r), (PN, r)
    # 2. For each constraint (i,j,k), try to satisfy it by adjusting one term t.
    # 3. If we can't satisfy all, try different initializations.
    
    # Actually, let's try a more systematic approach:
    # Fix the structure of U and V to be sparse, then solve for W.
    
    # For each column t (0 to r-1), we have vectors u_t (length NM), v_t (length MP), w_t (length PN).
    # The constraint is: for all (i,j,k), sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1
    
    # Let's fix u_t and v_t to have exactly 1 non-zero entry each (value 1).
    # Then u_t[i*m+j] * v_t[j*p+k] is 1 only if i*m+j = a_t and j*p+k = b_t for some (a_t, b_t).
    # This means j is determined by both: i*m+j = a_t and j*p+k = b_t.
    # So j = a_t - i*m = (b_t - k)/p. This only works for specific (i,j,k).
    
    # This is getting complex. Let me try a different approach.
    
    # Approach: Random initialization + local search.
    # 1. Randomly initialize U, V, W with small integers.
    # 2. Compute the error for each constraint.
    # 3. Try to reduce the error by changing one entry at a time.
    
    # But we need exact solutions. Let me try to construct a solution directly.
    
    # Key insight: The matrix multiplication tensor has a specific structure.
    # For 2x4 * 4x7, we can potentially use a block structure.
    
    # Let me try: rank = 28 (the lower bound).
    # Use 28 terms. For term t, let:
    #   u_t has a 1 at position (i_t * m + j_t)
    #   v_t has a 1 at position (j_t * p + k_t)
    #   w_t has a 1 at position (k_t * n + i_t)
    # This is the schoolbook approach, but with rank 56.
    
    # To get rank 28, we need to share terms. 
    # Idea: For each j (0 to 3), use 7 terms (one for each k).
    # For term (j, k):
    #   u_{j,k} has 1s at positions (0, j) and (1, j) [i.e., i*m+j for i=0,1]
    #   v_{j,k} has a 1 at position (j*p + k)
    #   w_{j,k} has 1s at positions (k*n + 0) and (k*n + 1)
    # 
    # Check: For (i,j',k'), sum_{j,k} u_{j,k}[i*m+j'] * v_{j,k}[j'*p+k'] * w_{j,k}[k'*n+i]
    # = sum_{j,k} [j==j'] * [k==k'] * [i==0 or i==1]
    # = 1 if j==j' and k==k', else 0.
    # This gives the identity matrix in the (j,k) space, not what we want.
    
    # We want: sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1 for all (i,j,k).
    
    # Let me try: For each (j,k), define a term where:
    #   u_{j,k}[i*m + j] = 1 for all i
    #   v_{j,k}[j*p + k] = 1
    #   w_{j,k}[k*n + i] = 1 for all i
    # Then: sum_{j,k} u_{j,k}[i*m+j'] * v_{j,k}[j'*p+k'] * w_{j,k}[k'*n+i]
    # = sum_{j,k} [j==j'] * [k==k'] * 1
    # = 1. This works! And rank = m*p = 28.
    
    # Wait, let me re-check. u_{j,k} has 1s at positions i*m+j for i=0,1.
    # So u_{j,k}[i*m+j'] = 1 if j==j', else 0.
    # v_{j,k}[j'*p+k'] = 1 if j==j' and k==k', else 0.
    # w_{j,k}[k'*n+i] = 1 if k==k', else 0.
    # Product: 1 if j==j' and k==k', else 0.
    # Sum over j,k: 1 if (j,k)==(j',k'), else 0.
    # But we want the sum to be 1 for all (i,j,k), not just when (j,k)==(j',k').
    
    # I made an error. Let me redo.
    
    # We want: for all (i,j,k), sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1.
    
    # If I use the construction above, for a fixed (i,j,k):
    # sum_{j',k'} u_{j',k'}[i*m+j] * v_{j',k'}[j*p+k] * w_{j',k'}[k*n+i]
    # = sum_{j',k'} [j==j'] * [j==j' and k==k'] * [k==k']
    # = sum_{j',k'} [j==j'] * [k==k']
    # = 1 (only when j'=j and k'=k).
    
    # So the sum is 1 for all (i,j,k). This works! Rank = m*p = 28.
    
    # Let me construct this.
    if r >= m * p:
        U = [[0] * r for _ in range(NM)]
        V = [[0] * r for _ in range(MP)]
        W = [[0] * r for _ in range(PN)]
        col = 0
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                for i in range(n):
                    W[k * n + i][col] = 1
                col += 1
        if col <= r:
            # Pad with zeros if needed
            for i in range(NM):
                for c in range(col, r):
                    U[i][c] = 0
            for i in range(MP):
                for c in range(col, r):
                    V[i][c] = 0
            for i in range(PN):
                for c in range(col, r):
                    W[i][c] = 0
            return {"U": U, "V": V, "W": W}
    
    # Try other constructions for lower rank.
    # For rank < m*p, we need a more clever decomposition.
    
    # Lower bound: max(NM, MP, PN) = max(8, 28, 14) = 28.
    # So rank 28 is the theoretical minimum. If we can't find rank < 28, return the rank-28 solution.
    
    if r < m * p:
        return None
    
    return None
=======
def _find_decomposition_rank(r, time_budget=5.0):
    """
    Try to find a rank-r decomposition using random search with integer coefficients.
    """
    start_time = time.time()
    
    # For r >= 28, use the block decomposition
    if r >= m * p:
        U = [[0] * r for _ in range(NM)]
        V = [[0] * r for _ in range(MP)]
        W = [[0] * r for _ in range(PN)]
        col = 0
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                for i in range(n):
                    W[k * n + i][col] = 1
                col += 1
        # Pad with zeros if r > m*p
        for i in range(NM):
            for c in range(col, r):
                U[i][c] = 0
        for i in range(MP):
            for c in range(col, r):
                V[i][c] = 0
        for i in range(PN):
            for c in range(col, r):
                W[i][c] = 0
        return {"U": U, "V": V, "W": W}
    
    # For r < 28, try random search
    # This is very hard, so we'll do a limited search
    import random
    random.seed(42)
    
    max_iters = 1000
    for iter in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization with small integers
        U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(NM)]
        V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(MP)]
        W = [[random.randint(-1, 1) for _ in range(r)] for _ in range(PN)]
        
        # Check if this works (unlikely for r < 28)
        if _validate_numeric(U, V, W, tol=1e-9):
            if _validate(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None
>>>>>>> REPLACE

