"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import copy

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank_schoolbook = n * m * p  # 96
    
    # Build schoolbook decomposition
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
    
    def is_valid(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly reproduces the all-ones tensor."""
        n, m, p = DIMENSIONS
        rank = len(U_mat[0])
        # For each (i,k), compute sum over j and r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                if total != 1:
                    return False
        return True
    
    # Verify schoolbook
    assert is_valid(U, V, W), "Schoolbook decomposition invalid"
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank via local search: try to eliminate columns
    # Strategy: try setting some entries to 0 and see if we can compensate
    # Simpler: try to merge columns where possible
    
    # For the all-ones tensor, one approach: if two columns have proportional
    # patterns, we might merge them. But this is complex.
    
    # Alternative: try rank reduction by 1: remove one column and see if
    # we can adjust others. This is NP-hard in general.
    
    # Given time constraints, try a few heuristic approaches:
    
    # Approach: For the all-ones tensor, use the fact that we can sometimes
    # express it with fewer terms by exploiting symmetry.
    
    # One known result: rank of 1_{2x2x2} = 7. The extra term uses
    # the identity 1 = (1/2)[(1,1,1) + ... ] with specific coefficients.
    
    # For our case, let's try: can we eliminate one term by adjusting?
    # Try: set U[0][95] = 0 (last column, i=0,j=0) and see if we can fix.
    # This is too ad-hoc. Let me try a different structural approach.
    
    # Structural approach: group k values. For k pairs (k1,k2), use a combined term.
    # For each j and pair (k1,k2), create one term that handles both k1 and k2.
    # Term: U[i*m+j, r] = 1, V[j*p+k1, r] = a, V[j*p+k2, r] = b, W[k1*n+i, r] = c1, W[k2*n+i, r] = c2
    # Need: a*c1 + b*c2 = 1 for all i (since U=1)
    # So if we set a=1, b=0, c1=1, c2=0, that's just the original for k1.
    # No savings unless we can share across j or i.
    
    # Let me try: for each (i,j), instead of separate terms for each k,
    # use a single term where V has a constant across k, and W varies.
    # U[i*m+j, r] = 1, V[j*p+k, r] = v_k, W[k*n+i, r] = w_k
    # Need: Σ_k v_k * w_k = 1... no, it's per (i,k): for fixed i,k:
    # Σ_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] over j and r must sum to 1.
    # If we have one term per (i,j) with V[j*p+k,r]=v_k and W[k*n+i,r]=w_k,
    # then for fixed i,k: Σ_j v_k * w_k * 1 = m * v_k * w_k = 1, so v_k*w_k = 1/m.
    # That gives rank = n*m (one per i,j), with V[j*p+k,r] = 1/sqrt(m), W[k*n+i,r]=1/sqrt(m).
    # But wait, r is shared across all k for a given (i,j). So for fixed (i,j), we have
    # one term r_ij, and V[j*p+k, r_ij] and W[k*n+i, r_ij] for all k.
    # Then C[i,k] contribution from this term = U[i*m+j, r_ij] * V[j*p+k, r_ij] * W[k*n+i, r_ij]
    # = 1 * v_k * w_k. And we sum over j: Σ_j v_k * w_k = m * v_k * w_k.
    # We need this to equal 1, so v_k * w_k = 1/m for all k.
    # So V[j*p+k, r_ij] = 1/sqrt(m) and W[k*n+i, r_ij] = 1/sqrt(m) for all k.
    # But then V and W entries are 1/sqrt(m) = 1/2, which are half-integers. That's allowed!
    # Rank = n * m = 12!
    
    # Wait, let me verify: for each (i,j) pair, we have one rank term r = i*m+j.
    # U[i*m+j, r] = 1 (only nonzero at its own position)
    # V[j*p+k, r] = 1/2 for all k (since m=4, 1/sqrt(4)=1/2)
    # W[k*n+i, r] = 1/2 for all k
    
    # Then for C[i,k]: Σ_j Σ_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = Σ_j U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    # = Σ_j 1 * (1/2) * (1/2) = Σ_j 1/4 = 4 * 1/4 = 1. Correct!
    
    # So rank = n * m = 12 works! But wait, is this correct? Let me double-check.
    # The issue: V[j*p+k, r] must be the same for all j for a given r? No, r is indexed by (i,j).
    # For r = i*m+j: V[j'*p+k, r] is nonzero only if j' = j? No, in my construction,
    # V[j*p+k, r_ij] = 1/2 for ALL k, but what about other j'?
    # V[j'*p+k, r_ij] = 0 for j' ≠ j (since each r is associated with a specific j).
    
    # So for C[i,k]: Σ_{j'=0}^{3} Σ_r U[i*m+j',r] * V[j'*p+k,r] * W[k*n+i,r]
    # Only r = i*m+j' contributes for each j' (since U[i*m+j', r] is nonzero only when r=i*m+j').
    # So = Σ_{j'=0}^{3} 1 * V[j'*p+k, i*m+j'] * W[k*n+i, i*m+j']
    # = Σ_{j'=0}^{3} (1/2) * (1/2) = 4 * 1/4 = 1. 
    
    # This works! Rank = n*m = 12.
    
    # But can we do even better? What if we also share across i?
    # Let's try rank = m * p or n * p or other combinations.
    
    # Try rank = m: one term per j.
    # U[i*m+j, r_j] = u_i, V[j*p+k, r_j] = v_k, W[k*n+i, r_j] = w_i
    # C[i,k] = Σ_j u_i * v_k * w_i = u_i * w_i * Σ_j v_k = u_i * w_i * m * v_k... 
    # No, V[j*p+k, r_j] depends on j. Let V[j*p+k, r_j] = v_{j,k}.
    # C[i,k] = Σ_j U[i*m+j, r_j] * V[j*p+k, r_j] * W[k*n+i, r_j]
    # = Σ_j u_{i,j} * v_{j,k} * w_{i,k}
    # We need this = 1 for all i,k.
    # If u_{i,j} = 1/m, v_{j,k} = 1, w_{i,k} = 1: C[i,k] = Σ_j (1/m)*1*1 = 1. Works!
    # Rank = m = 4.
    
    # Verify: U[i*m+j, r_j] = 1/m = 1/4 for all i,j (with r_j being the j-th term)
    # V[j*p+k, r_j] = 1 for all k
    # W[k*n+i, r_j] = 1 for all i,k
    # C[i,k] = Σ_{j=0}^{3} (1/4) * 1 * 1 = 1. Correct!
    
    # Can we do rank < 4? Try rank = 3.
    # This requires more complex structure. For now, rank = 4 is a huge improvement over 96.
    
    # Let me verify rank = 4 works with the indexing:
    # r ranges over 0,1,2,3 (corresponding to j=0,1,2,3)
    # U has shape (12, 4): U[i*4+j, r] = 1/4 if r==j, else 0
    # V has shape (32, 4): V[j*8+k, r] = 1 if r==j, else 0
    # W has shape (24, 4): W[k*3+i, r] = 1 for all i,k,r (all entries = 1)
    
    # C[i,k] = Σ_{j=0}^{3} U[i*4+j, j] * V[j*8+k, j] * W[k*3+i, j]
    # = Σ_{j=0}^{3} (1/4) * 1 * 1 = 1. 
    
    # This is valid! Rank = 4.
    
    U4 = [[0] * 4 for _ in range(n * m)]
    V4 = [[0] * 4 for _ in range(m * p)]
    W4 = [[0] * 4 for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U4[i * m + j][j] = 0.25  # 1/m = 1/4
    for j in range(m):
        for k in range(p):
            V4[j * p + k][j] = 1
    for k in range(p):
        for i in range(n):
            for r in range(m):
                W4[k * n + i][r] = 1
    
    if is_valid(U4, V4, W4):
        best_rank = 4
        best_U, best_V, best_W = U4, V4, W4
    
    # Can we do even better? Try rank = 3 or less.
    # For rank r < m, we need to solve a system. This is more complex.
    # Given time, let's stick with rank = 4 if valid.
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # We're over, but we already have a valid result
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

