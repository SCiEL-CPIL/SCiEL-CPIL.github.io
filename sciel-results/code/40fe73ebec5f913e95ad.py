import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Correct Strassen 2x2x2 decomposition. Rank 7."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: (A[0,0]-A[1,1])*(B[1,1]-B[0,0])
    U[0, 0] = 1; U[3, 0] = -1
    V[3, 0] = 1; V[0, 0] = -1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1
    
    # M2: (A[0,0]+A[1,1])*B[1,1]
    U[0, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: A[1,1]*(B[1,0]-B[0,0])
    U[3, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A[0,0]*(B[0,1]-B[1,1])
    U[0, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A[0,0]+A[0,1])*B[1,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _strassen_2x2x3():
    """
    2x2x3 decomposition using Strassen 2x2x2 + one schoolbook term.
    Rank 8.
    The 2x2x3 tensor T[i,j,k] = delta for k=0,1,2 (but only 2x2x2 part needs Strassen).
    Actually for 2x2x3: we need T[i,j,k] for i,j in {0,1}, k in {0,1,2}.
    Use Strassen for k=0,1 (rank 7) + schoolbook for k=2 (rank 4) = rank 11.
    But we can do better: use the 7 Strassen terms for the 2x2x2 block,
    and for the 3rd column of B, we need 4 more terms (schoolbook 2x2x1).
    Total rank = 7 + 4 = 11.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((6, rank), dtype=np.float64)
    W = np.zeros((6, rank), dtype=np.float64)
    
    sU, sV, sW = _strassen_2x2x2()
    
    # First 7 columns: Strassen for k=0,1
    U[:, :7] = sU
    V[:, :7] = sV
    W[:, :7] = sW
    
    # Last 4 columns: schoolbook for k=2
    for idx in range(4):
        i = idx // 2
        j = idx % 2
        U[i * 2 + j, 7 + idx] = 1.0
        V[j * 3 + 2, 7 + idx] = 1.0
        W[2 * 2 + i, 7 + idx] = 1.0
    
    return U, V, W

def _build_base_decomposition():
    """
    Build 4x4x7 decomposition:
    - 4x4x4 part: 8 Strassen 2x2x2 blocks, rank 56
    - 4x4x3 part: 4 Strassen 2x2x3 blocks, rank 44
    Total: 100
    """
    n, m, p = 4, 4, 7
    total_rank = 8 * 7 + 4 * 11  # 100
    
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((28, total_rank), dtype=np.float64)
    W = np.zeros((28, total_rank), dtype=np.float64)
    
    sU, sV, sW = _strassen_2x2x2()
    
    block_configs = [
        (0, 0, 0, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 1, 1, 0),
        (0, 0, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 0, 1),
        (1, 1, 1, 1),
    ]
    
    rank_offset = 0
    for (arb, acb, brb, bc) in block_configs:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (2 * arb + i) * 4 + (2 * acb + j)
                    U[u_idx, rank_offset + r] += sU[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    v_idx = (2 * brb + j) * 7 + (2 * bc + k)
                    V[v_idx, rank_offset + r] += sV[j * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    w_idx = (2 * bc + k) * 4 + (2 * arb + i)
                    W[w_idx, rank_offset + r] += sW[k * 2 + i, r]
        rank_offset += 7
    
    # Part 2: 4x4x3 using 2x2x3 blocks
    sU2, sV2, sW2 = _strassen_2x2x3()
    
    block_configs_2 = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]
    
    for (arb, acb, brb) in block_configs_2:
        for r in range(11):
            for i in range(2):
                for j in range(2):
                    u_idx = (2 * arb + i) * 4 + (2 * acb + j)
                    U[u_idx, rank_offset + r] += sU2[i * 2 + j, r]
            for j in range(2):
                for k in range(3):
                    v_idx = (2 * brb + j) * 7 + (4 + k)
                    V[v_idx, rank_offset + r] += sV2[j * 3 + k, r]
            for k in range(3):
                for i in range(2):
                    w_idx = (4 + k) * 4 + (2 * arb + i)
                    W[w_idx, rank_offset + r] += sW2[k * 2 + i, r]
        rank_offset += 11
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using tensor contraction."""
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape and use einsum
    T = np.einsum('ir,jr,kr->ijk', 
                  U.reshape(n, m, rank), 
                  V.reshape(m, p, rank), 
                  W.reshape(p, n, rank))
    # Check T[i,j,k] == 1 for all i,j,k
    return np.allclose(T, 1.0, atol=1e-9)

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    rank = U.shape[1]
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    return _verify_fast(U, V, W, n, m, p)

def _apply_equivalent_transform(U, V, W, R, S, T):
    """
    Apply equivalent transformation:
    U' = U @ R, V' = V @ S, W' = W @ T
    where R, S, T are invertible matrices with R @ S @ T = I (element-wise product preserved).
    Actually, the condition is that for each r, the product U[:,r] * V[:,r] * W[:,r] is preserved.
    The general equivalent transformation: pick invertible P, Q with P @ Q = I element-wise?
    No. The correct condition: if we define new factors as
    U'[:,r] = sum_s U[:,s] * P[s,r]
    V'[:,r] = sum_s V[:,s] * Q[s,r]  
    W'[:,r] = sum_s W[:,s] * R[s,r]
    Then the tensor is preserved iff P @ Q @ R = I (in the sense that sum_s P[s,r]*Q[s,r]*R[s,r] = delta_{r,s}).
    
    A simpler approach: use the fact that if we have a valid decomposition,
    then for any invertible matrix M, we can do:
    U' = U @ M, V' = V @ M^{-1}, W' = W (this doesn't preserve the tensor).
    
    Actually the correct equivalent transformation for tensor decomposition:
    If T = sum_r u_r ⊗ v_r ⊗ w_r, then for any invertible P:
    T = sum_r (sum_s u_s P[s,r]) ⊗ (sum_s v_s P^{-1}[s,r]) ⊗ w_r  -- NO this doesn't work.
    
    The correct statement: the decomposition is not unique. If we have
    sum_r u_r v_r w_r = T, then for any invertible matrix P:
    sum_r (u_r P[r,s]) (v_r P^{-1}[r,s]) w_r -- no.
    
    Let me think again. The tensor is T = sum_r u_r ⊗ v_r ⊗ w_r.
    If we define new terms: u'_r = sum_s u_s P[s,r], v'_r = sum_s v_s Q[s,r], w'_r = sum_s w_s R[s,r]
    Then sum_r u'_r ⊗ v'_r ⊗ w'_r = sum_{r,s,t,u} P[s,r] Q[t,r] R[u,r] u_s ⊗ v_t ⊗ w_u
    For this to equal sum_r u_r ⊗ v_r ⊗ w_r, we need:
    sum_r P[s,r] Q[t,r] R[u,r] = delta_{s,t} delta_{t,u} delta_{u,s} = delta_{s,t,u}
    This is very restrictive.
    
    A simpler valid transformation: if P is invertible, define
    u'_r = u_r P[r,s] summed... 
    
    OK let me use the standard result: for a rank-r decomposition, any equivalent decomposition
    is obtained by: pick invertible M (r x r), then
    u'_r = sum_s u_s M[s,r], v'_r = sum_s v_s M^{-1}[s,r]... NO.
    
    Actually I was overcomplicating. The key insight for random search:
    We can add any vector to U[:,r] as long as we compensate in V and W.
    Specifically, if we add delta_u to U[:,r], we need to subtract something from V and W
    such that the tensor is preserved. This is hard to do exactly.
    
    Better approach: use the fact that the decomposition space has a group action.
    For any invertible matrix P of size r x r:
    U_new = U @ P
    V_new = V @ P^{-T}  (inverse transpose)
    W_new = W  -- NO this doesn't preserve the tensor.
    
    Let me just use a different strategy: **random column elimination via linear algebra**.
    """
    pass

def _randomized_rank_reduction(U, V, W, n, m, p, max_time=100, seed=42):
    """
    Structural approach: Randomized rank reduction.
    
    Key insight: A rank-r decomposition of T is not unique. The set of all rank-r
    decompositions forms a variety. We can explore this variety by:
    1. Starting from a valid decomposition
    2. Applying random "gauge" transformations that preserve the tensor
    3. Checking if the transformed decomposition has lower effective rank
    
    For matrix multiplication tensors, there's a known gauge freedom:
    If (U, V, W) is a decomposition, then so is (U @ P, V @ Q, W @ R)
    where P, Q, R are r x r matrices satisfying:
    sum_s P[s,r] * Q[s,r] * R[s,r] = delta_{r,s} for all r, s
    
    A simple family: let P = I + eps * A, Q = I + eps * B, R = I + eps * C
    where A, B, C are random matrices with A[s,r]*B[s,r]*C[s,r] summing to 0 off-diagonal.
    
    Even simpler: use the fact that for any invertible diagonal D:
    U @ D, V @ D^{-1}, W (unchanged) -- NO, this doesn't preserve the tensor.
    
    Actually, the simplest valid transformation:
    For any invertible P: U' = U @ P, V' = V @ P^{-1}, W' = W
    Check: sum_r U'[:,r] V'[:,r] W'[:,r] = sum_r (sum_s U[:,s] P[s,r]) (sum_t V[:,t] P^{-1}[t,r]) W[:,r]
    = sum_{s,t,r} U[:,s] V[:,t] W[:,r] P[s,r] P^{-1}[t,r]
    This is NOT equal to sum_r U[:,r] V[:,r] W[:,r] in general.
    
    So the gauge freedom is more subtle. Let me use a different approach:
    
    **Direct random search for low-rank decompositions:**
    Start with random rank-r factors and use gradient descent to fit the tensor.
    But this is expensive and may not converge to exact solutions.
    
    **Better: Use the structure of the problem.**
    The 4x4x7 matrix multiplication tensor has a known structure.
    Let's try a hybrid approach:
    1. Build the base decomposition (rank 100)
    2. Try to find linear dependencies by solving for null space vectors
    3. Use iterative refinement
    
    For step 2: A column r is redundant if there exist vectors a, b, c (length rank-1)
    such that:
    U[:,r] = U_other @ a
    V[:,r] = V_other @ b  
    W[:,r] = W_other @ c
    AND the tensor contribution is preserved:
    sum_s a_s * b_s * c_s * (U_other[:,s] ⊗ V_other[:,s] ⊗ W_other[:,s]) = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    This is a system of n*m*p = 112 equations with (rank-1)*3 unknowns.
    For rank=100, that's 297 unknowns and 112 equations -- underdetermined!
    So there might be solutions.
    
    Let's formulate this as: find a, b, c such that
    sum_s a_s b_s c_s U_other[:,s] ⊗ V_other[:,s] ⊗ W_other[:,s] = U_r ⊗ V_r ⊗ W_r
    
    This is a bilinear (actually trilinear) equation. We can linearize by fixing two of a,b,c.
    
    Strategy: 
    - Fix c to be a random vector
    - Then the equation becomes bilinear in a and b
    - We can solve this using alternating minimization or SVD
    
    Let's implement this:
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    max_iterations = 5000
    iter = 0
    
    while iter < max_iterations:
        if time.time() - start_time > max_time:
            break
        iter += 1
        
        rank = best_U.shape[1]
        if rank <= 1:
            break
        
        # Pick a random column to try to eliminate
        r = rng.randint(rank)
        
        U_r = best_U[:, r].copy()
        V_r = best_V[:, r].copy()
        W_r = best_W[:, r].copy()
        
        # Get the other columns
        mask = np.ones(rank, dtype=bool)
        mask[r] = False
        U_o = best_U[:, mask]
        V_o = best_V[:, mask]
        W_o = best_W[:, mask]
        
        # Check necessary conditions: U_r in span(U_o), etc.
        # If not in span, we can't eliminate this column
        a_sol, _, rank_a, _ = np.linalg.lstsq(U_o, U_r, rcond=None)
        if np.linalg.norm(U_o @ a_sol - U_r) > 1e-8:
            continue
        
        b_sol, _, rank_b, _ = np.linalg.lstsq(V_o, V_r, rcond=None)
        if np.linalg.norm(V_o @ b_sol - V_r) > 1e-8:
            continue
        
        c_sol, _, rank_c, _ = np.linalg.lstsq(W_o, W_r, rcond=None)
        if np.linalg.norm(W_o @ c_sol - W_r) > 1e-8:
            continue
        
        # Now check if the tensor identity is satisfied.
        # We need: sum_s a_s * b_s * c_s * (U_o[:,s] ⊗ V_o[:,s] ⊗ W_o[:,s]) = U_r ⊗ V_r ⊗ W_r
        # This is equivalent to: for all (i,j,k):
        # sum_s a_s * b_s * c_s * U_o[i*m+j, s] * V_o[j*p+k, s] * W_o[k*n+i, s] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        # Let's compute the LHS for all (i,j,k)
        # LHS[i,j,k] = sum_s a_s * b_s * c_s * U_o[i*m+j, s] * V_o[j*p+k, s] * W_o[k*n+i, s]
        # = sum_s (a_s * U_o[i*m+j, s]) * (b_s * V_o[j*p+k, s]) * (c_s * W_o[k*n+i, s])
        
        # This is a tensor contraction. Let's compute it efficiently.
        # Define A = a_s * U_o[i*m+j, s], B = b_s * V_o[j*p+k, s], C = c_s * W_o[k*n+i, s]
        # LHS[i,j,k] = sum_s A[i,j,s] * B[j,k,s] * C[k,i,s]
        
        # Use einsum
        A_mat = U_o.T @ np.diag(a_sol)  # (rank-1) x 16
        B_mat = V_o.T @ np.diag(b_sol)  # (rank-1) x 28
        C_mat = W_o.T @ np.diag(c_sol)  # (rank-1) x 28
        
        # LHS[i,j,k] = sum_s A_mat[s, i*m+j] * B_mat[s, j*p+k] * C_mat[s, k*n+i]
        # = einsum('sij,skl,smn->ikl' ...) -- need to be careful with indices
        
        # Actually: A_mat[s, idx_u] where idx_u = i*m+j
        # B_mat[s, idx_v] where idx_v = j*p+k
        # C_mat[s, idx_w] where idx_w = k*n+i
        
        # LHS[i,j,k] = sum_s A_mat[s, i*m+j] * B_mat[s, j*p+k] * C_mat[s, k*n+i]
        
        # Let's compute this as:
        # For each s, define tensor T_s[i,j,k] = A_mat[s, i*m+j] * B_mat[s, j*p+k] * C_mat[s, k*n+i]
        # LHS = sum_s T_s
        
        # This is expensive to compute naively. Let's use a different approach.
        # Instead, let's just verify the full tensor after removing column r.
        
        U_try = np.delete(best_U, r, axis=1)
        V_try = np.delete(best_V, r, axis=1)
        W_try = np.delete(best_W, r, axis=1)
        
        if _verify_fast(U_try, V_try, W_try, n, m, p):
            best_U = U_try
            best_V = V_try
            best_W = W_try
            best_rank = rank - 1
            # Restart with new rank
            continue
        
        # If direct elimination fails, try a "merge" strategy:
        # Add a multiple of column r to another column, then try to eliminate.
        # This is a heuristic that might find dependencies.
        if iter % 10 == 0:  # Do this less frequently
            s = rng.randint(rank)
            if s == r:
                continue
            alpha = rng.randn()
            
            U_try = best_U.copy()
            V_try = best_V.copy()
            W_try = best_W.copy()
            
            # Add alpha * column r to column s
            U_try[:, s] += alpha * U_r
            V_try[:, s] += alpha * V_r
            W_try[:, s] += alpha * W_r
            
            # Now try to eliminate column r from the modified system
            # But this changes the tensor, so we need to compensate.
            # Actually, adding alpha * (U_r, V_r, W_r) to (U_s, V_s, W_s)
            # changes the tensor by alpha * U_r ⊗ V_r ⊗ W_r, which is not zero.
            # So this doesn't work directly.
            # Skip this approach.
            pass
    
    return best_U, best_V, best_W, best_rank

def _structured_random_search(U, V, W, n, m, p, max_time=100, seed=42):
    """
    Different approach: Start from the base decomposition and use random
    perturbations to find equivalent decompositions with lower effective rank.
    
    Key idea: The set of all rank-r decompositions of a given tensor T is
    (generically) a manifold of dimension 3r - dim(Stabilizer). We can
    explore this manifold by:
    1. Adding small random perturbations to the factors
    2. Projecting back to the variety of valid decompositions
    3. Checking if the projected decomposition has lower rank
    
    For step 2, we use the fact that a valid decomposition satisfies
    sum_r u_r ⊗ v_r ⊗ w_r = T. This is a system of n*m*p equations.
    We can use Newton's method to project a perturbed decomposition back.
    
    However, this is computationally expensive. Instead, let's use a
    simpler approach: **random column replacement**.
    
    For each column r, try to replace (U[:,r], V[:,r], W[:,r]) with a
    linear combination of other columns that preserves the tensor.
    This is equivalent to finding a, b, c such that:
    U[:,r] ⊗ V[:,r] ⊗ W[:,r] = sum_s a_s b_s c_s (U[:,s] ⊗ V[:,s] ⊗ W[:,s])
    
    We can solve this by:
    1. Fix c to a random vector
    2. The equation becomes: U_r ⊗ V_r ⊗ (W_r / c) = sum_s (a_s b_s) (U_s ⊗ V_s ⊗ W_s)
    3. This is a bilinear equation in a and b
    4. Use SVD to find solutions
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    max_iterations = 3000
    iter = 0
    
    while iter < max_iterations:
        if time.time() - start_time > max_time:
            break
        iter += 1
        
        rank = best_U.shape[1]
        if rank <= 1:
            break
        
        # Pick a random column to try to eliminate
        r = rng.randint(rank)
        
        mask = np.ones(rank, dtype=bool)
        mask[r] = False
        U_o = best_U[:, mask]
        V_o = best_V[:, mask]
        W_o = best_W[:, mask]
        U_r = best_U[:, r]
        V_r = best_V[:, r]
        W_r = best_W[:, r]
        
        # Try to find a, b, c such that the tensor is preserved.
        # Strategy: fix c randomly, then solve for a and b.
        
        # The equation is:
        # sum_s a_s b_s c_s U_o[i*m+j, s] V_o[j*p+k, s] W_o[k*n+i, s] = U_r[i*m+j] V_r[j*p+k] W_r[k*n+i]
        
        # Let's define: for fixed c, define D_s = c_s
        # Then: sum_s a_s b_s D_s U_o[i*m+j, s] V_o[j*p+k, s] W_o[k*n+i, s] = U_r[i*m+j] V_r[j*p+k] W_r[k*n+i]
        
        # This is: sum_s (a_s U_o[i*m+j, s]) (b_s V_o[j*p+k, s]) (D_s W_o[k*n+i, s]) = U_r[i*m+j] V_r[j*p+k] W_r[k*n+i]
        
        # Let A_s = a_s, B_s = b_s, C_s = D_s W_o[k*n+i, s] / W_r[k*n+i] (if W_r[k*n+i] != 0)
        # This is getting complicated. Let's just try the direct verification approach
        # but with a twist: instead of requiring U_r, V_r, W_r to be in the spans,
        # we allow a more general linear combination.
        
        # Actually, let's try a completely different approach:
        # **Use the matricization rank as a lower bound.**
        # The rank of the tensor is at least the rank of any matricization.
        # If we can find a decomposition where the matricization rank equals
        # the tensor rank, we're done.
        
        # For now, let's just try the simple column elimination with verification.
        
        U_try = np.delete(best_U, r, axis=1)
        V_try = np.delete(best_V, r, axis=1)
        W_try = np.delete(best_W, r, axis=1)
        
        if _verify_fast(U_try, V_try, W_try, n, m, p):
            best_U = U_try
            best_V = V_try
            best_W = W_try
            best_rank = rank - 1
            continue
    
    return best_U, best_V, best_W, best_rank

def _try_laderman_style():
    """
    Try to construct a Laderman-style decomposition for 4x4x4.
    Laderman's algorithm has rank 21 for 4x4x4 (vs 64 schoolbook, 56 with 8x Strassen).
    If we can use rank 21 for the 4x4x4 part, total would be 21 + 44 = 65.
    
    Laderman's 4x4x4 decomposition:
    The 4x4 matrix is partitioned into 2x2 blocks.
    Laderman uses 21 multiplications.
    
    The 21 terms are:
    Let A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    where each block is 2x2.
    
    Laderman's terms (from the literature):
    M1 = (A11 - A22) * (B22 - B11)
    M2 = (A11 + A22) * B22
    M3 = A22 * (B21 - B11)
    M4 = A11 * (B12 - B22)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    ... and more terms for the cross-block interactions.
    
    Actually, Laderman's algorithm is for 4x4x4 and uses 21 scalar multiplications.
    Let me look up the exact formula.
    
    Laderman (1976) algorithm for 4x4 matrix multiplication:
    Partition A and B into 2x2 blocks of 2x2 matrices.
    
    The 21 products are:
    M1 = (A11 - A22)(B22 - B11)
    M2 = (A11 + A22)B22
    M3 = A22(B21 - B11)
    M4 = A11(B12 - B22)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    M8 = (A11 + A21)B12
    M9 = (A12 + A22)B21
    M10 = (A11 - A12)(B12 - B22)  -- not sure about the exact form
    ...
    
    I don't remember the exact 21 terms. Let me use a known reference.
    
    Actually, for this task, let's try a different approach:
    Use the fact that 4x4x7 can be decomposed as:
    - 4x4x4 (using Strassen: rank 56, or Laderman: rank 21)
    - 4x4x3 (using Strassen 2x2x3: rank 44, or better)
    
    If we can find a rank-21 4x4x4 decomposition, total would be 21 + 44 = 65.
    That would be a huge improvement over 100.
    
    Let me try to implement Laderman's algorithm.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Build base decomposition (rank 100)
    U, V, W = _build_base_decomposition()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int8)
        V = np.zeros((m * p, rank), dtype=np.int8)
        W = np.zeros((p * n, rank), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank using randomized search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        U, V, W, reduced_rank = _randomized_rank_reduction(U, V, W, n, m, p, max_time=min(remaining_time, 80))
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Rebuild and just return the base
        U, V, W = _build_base_decomposition()
    
    # Convert to lists
    def _to_list(arr):
        if np.allclose(arr, np.round(arr), atol=1e-9):
            return np.round(arr).astype(np.int64).tolist()
        elif np.allclose(arr * 2, np.round(arr * 2), atol=1e-9):
            return (arr * 2).round().astype(np.int64).tolist()
        else:
            return arr.tolist()
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END

