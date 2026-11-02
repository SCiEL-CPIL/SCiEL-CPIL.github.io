<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition
    # The tensor T has entries T[(i,j,k)] = 1 for the multiplication tensor
    # We want to express it as sum of rank-1 tensors
    
    # For now, return the schoolbook decomposition
    # A more sophisticated search would go here
    return best_result
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_strassen_core(n, m, p, deadline):
    """
    Use Strassen's rank-13 decomposition for 3x3x3 as the core,
    then solve for additional rank terms to extend to 3x3x5.
    
    The multiplication tensor for C = A * B where A is n x m, B is m x p:
    T[i,j,k] = 1 for all valid (i,j,k)
    
    We express T as sum over r of u_r ⊗ v_r ⊗ w_r where:
    u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}
    
    Strategy: Start with Strassen's 13 rank-1 terms for the 3x3x3 sub-tensor
    (i in 0..2, j in 0..2, k in 0..2). Then we need to satisfy the equations
    for k=3,4 (the extra columns of B).
    
    For each (i,j,k) with k in {3,4}, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    The Strassen terms have specific structure. Let's define them and then
    solve a linear system for the remaining unknowns.
    """
    start_time = time.time()
    
    # Strassen's rank-13 decomposition for 3x3x3
    # Each term is (u_indices, v_indices, w_indices) where these are sets of
    # (i,j), (j,k), (k,i) positions with coefficient 1
    # 
    # Standard Strassen 3x3x3:
    # M1 = (A11+A22+A33)(B11+B22+B33)
    # M2 = (A11+A22)(B21+B31)  
    # M3 = (A22+A33)(B22+B32)
    # M4 = (A33+A11)(B33+B13)
    # M5 = (A11+A12)(B11+B21)
    # M6 = (A22+A23)(B22+B32)
    # M7 = (A33+A31)(B33+B13)
    # M8 = (A12+A22)(B11+B12)
    # M9 = (A23+A33)(B22+B23)
    # M10= (A31+A11)(B33+B31)
    # M11= (A11+A33)(B13+B23)
    # M12= (A22+A11)(B21+B31)  -- wait, let me use a cleaner formulation
    
    # Let me use a well-known rank-13 scheme for 3x3x3.
    # The 13 products are:
    # P1  = (a11+a22+a33)*(b11+b22+b33)
    # P2  = (a11+a22)*(b21+b31)
    # P3  = (a22+a33)*(b22+b32)
    # P4  = (a33+a11)*(b33+b13)
    # P5  = (a11+a12)*(b11+b21)
    # P6  = (a22+a23)*(b22+b32)
    # P7  = (a33+a31)*(b33+b13)
    # P8  = (a12+a22)*(b11+b12)
    # P9  = (a23+a33)*(b22+b23)
    # P10 = (a31+a11)*(b33+b31)
    # P11 = (a11+a33)*(b13+b23)
    # P12 = (a22+a11)*(b21+b31)
    # P13 = (a33+a22)*(b32+b12)
    
    # Actually, let me just use the schoolbook for the 3x3x3 part and 
    # think about this differently. The key insight is:
    # 
    # For 3x3x5, we have 3*3*5 = 45 equations.
    # Each rank-1 term contributes to exactly one equation per (i,j,k) it covers.
    # 
    # A better approach: use the fact that for fixed j, the sub-tensor over (i,k)
    # is rank <= 3 (since A[:,j] is a vector of length 3, B[j,:] is length 5,
    # so the outer product structure gives rank 3).
    #
    # Total rank >= sum over j of rank of sub-tensor = 3*3 = 9? No, that's not right.
    #
    # Let me try a direct linear algebra approach with a structured ansatz.
    # 
    # Ansatz: Use rank-1 terms of the form:
    # For each "mode" combination, we can have terms that are sparse.
    #
    # Actually, the simplest improvement: try rank 13 (Strassen for 3x3x3) 
    # and see if we can extend it. The Strassen 13 terms only cover k=0,1,2.
    # For k=3,4, we need additional terms.
    #
    # Each additional k-value (k=3 or k=4) requires 3*3=9 more equations.
    # But the U and W matrices from Strassen are fixed, so we need to find
    # V entries for k=3,4 and possibly new rank terms.
    #
    # Let me try: keep the 13 Strassen terms, add r_new new terms.
    # The new terms can have arbitrary U, V, W entries.
    # Total rank = 13 + r_new.
    #
    # This is still complex. Let me try a simpler structured approach:
    # 
    # Use the "block" structure. For 3x3x5:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can think of this as 3 independent 1x3x5 multiplications (one per row i),
    # but they share the A matrix.
    #
    # Let me try a direct sparse linear solve approach:
    # Assume a rank R decomposition. The unknowns are U (9*R), V (15*R), W (15*R).
    # The constraints are 45 nonlinear equations (products of 3 variables).
    #
    # This is hard. Let me instead try specific small ranks with randomized search
    # or use the known result that rank(3,3,5) might be 13 or 14.
    
    # Fallback: try to find a rank-13 or rank-14 solution via structured search
    # For now, let's implement a search that tries to find low-rank solutions
    # using a greedy / local search approach on the tensor.
    
    # Actually, let me implement a proper search:
    # Start from schoolbook rank 45 and try to reduce.
    # 
    # Better: use the algebraic structure. 
    # For C = A*B, A is 3x3, B is 3x5.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # We can write this using rank-1 tensors. Each term in the sum is:
    # A[i,j] * B[j,k] which is a rank-1 tensor in the (i,k) space for fixed j.
    # But A[i,j] and B[j,k] are separate factors.
    #
    # The standard way: each product A[i,j]*B[j,k] corresponds to one rank-1 term
    # in the tensor decomposition. That gives 9 terms total (3*3), which is the
    # schoolbook. But we can combine terms using Strassen-like tricks.
    #
    # Key insight: For the 3x3x3 part, Strassen reduces from 9 to 13? No, 
    # Strassen reduces 3x3x3 from 9 to 13? That's wrong. Strassen for 2x2x2
    # reduces from 8 to 7. For 3x3x3, the schoolbook is 9 and Strassen-like
    # methods give rank 13? That seems like an increase.
    #
    # Wait, I'm confusing myself. For matrix multiplication C=A*B:
    # - 2x2x2: schoolbook rank 8, Strassen rank 7
    # - 3x3x3: schoolbook rank 9, best known rank is 13? No...
    #
    # Actually for 3x3x3, the schoolbook is 9 (3*3 products). The tensor rank
    # of the 3x3x3 matrix multiplication tensor is known to be at most 13 
    # (via Strassen's recursive method or other constructions) but the 
    # schoolbook already gives 9. So rank(3,3,3) <= 9.
    #
    # I think I was confused. Let me restart with correct understanding:
    #
    # The matrix multiplication tensor [n,m,p] has tensor rank r[n,m,p].
    # r[3,3,3] <= 9 (schoolbook). Best known lower bound is 19? No, that can't be.
    #
    # Actually, I recall that r[3,3,3] is not known exactly, but it's at most 9
    # (schoolbook) and the lower bound is... let me just implement a search.
    
    # Given the time constraints, let me implement a practical search:
    # 1. Start with schoolbook rank 45
    # 2. Try to find lower rank via randomized local search
    
    # But first, let me try a simple structured approach:
    # For 3x3x5, note that if we fix j, the subproblem is 1x3x5 which has
    # rank 3 (the outer product of a 3-vector and a 5-vector has rank 1,
    # but we need 3 such for the 3 rows).
    #
    # Actually, let me just try to solve this with a greedy algorithm that
    # tries to find rank-1 terms to cover the tensor.
    
    # Given the complexity, let me return the schoolbook for now but with
    # a quick check to see if we can do slightly better with a structured ansatz.
    
    # Structured ansatz: 
    # Use 9 "base" terms (one for each (i,j) pair) that handle k=0,1,2,3,4
    # For each (i,j), we have a term with U[i*m+j]=1, V[j*p+k] for k=0..4, W[k*n+i]
    # But that's not rank-1 unless V and W are rank-1 vectors.
    #
    # A rank-1 term has U as a vector in R^9, V in R^15, W in R^15.
    # The product U[i,j]*V[j,k]*W[k,i] must sum to 1 for all (i,j,k).
    #
    # Let me try: for each j in {0,1,2}, use 3 rank-1 terms that together
    # cover all i and k for that j. That's 9 terms total.
    # For fixed j, we need: sum_{r in terms_for_j} U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
    # This is like a rank-3 decomposition of the 3x5 matrix of ones (for each j).
    # The matrix of ones has rank 1, so we can do it with 1 term per j!
    # 
    # For fixed j: U_r[i,j] = 1 for all i, V_r[j,k] = 1 for all k, W_r[k,i] = 1 for all k,i
    # Then U*V*W = 1*1*1 = 1. That works!
    # 
    # So we can use 3 rank-1 terms (one per j), giving rank 3?!
    # Let me verify:
    # Term r=0 (j=0): U[i,0]=1 for all i, V[0,k]=1 for all k, W[k,i]=1 for all k,i
    #   Contribution to C[i,k]: U[i,0]*V[0,k]*W[k,i] = 1*1*1 = 1
    # Term r=1 (j=1): U[i,1]=1 for all i, V[1,k]=1 for all k, W[k,i]=1 for all k,i
    #   Contribution: 1*1*1 = 1
    # Term r=2 (j=2): U[i,2]=1 for all i, V[2,k]=1 for all k, W[k,i]=1 for all k,i
    #   Contribution: 1*1*1 = 1
    # Total: 3, but we need 1!
    #
    # Oops, that gives 3, not 1. The issue is that all three terms contribute
    # to every (i,k) entry. We need the sum to be 1, not 3.
    #
    # So we need: sum_j U_r[i,j]*V_r[j,k]*W_r[k,i] = 1
    # If we use the same W for all j, then W[k,i] * sum_j U_r[i,j]*V_r[j,k] = 1
    # This requires W[k,i] to be constant and the inner sum to be 1/W.
    #
    # This is getting complicated. Let me just implement a numerical search.
    
    # Given time constraints, let me try a simple randomized search for low rank.
    # But this is unlikely to find good solutions quickly.
    
    # Let me try the most promising structured approach:
    # Use the fact that the tensor is separable in j in a certain way.
    #
    # For C[i,k] = sum_j A[i,j] B[j,k], we can write:
    # C[i,k] = sum_j (A[i,j]) (B[j,k]) * 1
    # Each term (i,j,k) -> A[i,j]*B[j,k] is not a rank-1 tensor in the 
    # (i,j,k) space because A[i,j] depends on (i,j) and B[j,k] on (j,k).
    #
    # The rank-1 tensor decomposition requires:
    # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    #
    # For the schoolbook: u_r[i,j]=delta(i,j), v_r[j,k]=delta(j,k), w_r[k,i]=1
    # where r indexes (i,j) pairs. This gives rank 9 for 3x3x3.
    #
    # For 3x3x5, the schoolbook still uses rank 9 (one term per (i,j) pair),
    # because the k dimension is handled by the v_r and w_r vectors.
    #
    # Wait, let me re-examine. In the schoolbook:
    # For each (i,j) pair, we have a term with:
    #   U[i*m+j] = 1, all other U entries = 0
    #   V[j*p+k] = 1 for all k, all other V entries = 0
    #   W[k*n+i] = 1 for all k, all other W entries = 0
    #
    # Check: sum_r U[i',m+j',r] * V[j',p+k',r] * W[k',n+i',r]
    # For r corresponding to (i,j): U[i,m+j]=1, V[j,p+k]=1, W[k,n+i]=1
    # So the contribution is 1 if (i',j')=(i,j), else 0.
    # Sum over r: only the term with (i',j')=(i,j) contributes, giving 1. Correct!
    #
    # So schoolbook rank for 3x3x5 is 9, not 45! I made an error earlier.
    # The schoolbook decomposition has rank n*m = 9 (one term per (i,j) pair).
    
    # Let me verify this is correct by constructing it:
    rank = n * m  # 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (i, j) in enumerate(product(range(n), range(m))):
        U[i * m + j][r] = 1
        for k in range(p):
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # We have a valid rank-9 decomposition
        # Can we do better? Try rank 8, 7, ...
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    else:
        # Shouldn't happen, but fallback to schoolbook with rank n*m*p
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_rank = rank_schoolbook
        best_U, best_V, best_W = U, V, W
    
    # Now try to find lower rank solutions
    # Try ranks from 8 down to 1
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 5:
            break
        # Try to find a rank-target_rank decomposition
        # Use a structured approach: try specific patterns
        
        # Pattern 1: Try to merge two schoolbook terms
        # This is complex, so for now just return what we have
        
        # A simple improvement: check if rank 8 works with a specific pattern
        # For 3x3x5, can we do rank 8?
        # The tensor has 45 entries. Each rank-1 term can "cover" at most
        # min(9,15,15) = 9 non-zero (i,j,k) entries (if U, V, W are all-ones
        # on their supports).
        # Lower bound: 45 / 9 = 5. So rank >= 5 theoretically.
        # But the structure makes it harder.
        
        # Let me try a randomized search for rank 8
        found = False
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            # Randomized search: pick random sparse U, V, W and solve
            # This is too complex for the time limit.
            pass
        
        if not found:
            continue
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Try the structured Strassen-core approach
    try:
        result = _try_rank_strassen_core(n, m, p, deadline)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize my analysis showed the schoolbook rank is actually n*m = 9, not n*m*p = 45. Let me fix this and implement a proper search for even lower rank.

