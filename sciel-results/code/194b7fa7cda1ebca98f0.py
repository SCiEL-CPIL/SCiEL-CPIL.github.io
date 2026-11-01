"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    # Build the target tensor
    T = np.zeros((A_dim, B_dim, C_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Start with schoolbook decomposition (rank = n*m*p = 64)
    U = np.zeros((A_dim, n * m * p))
    V = np.zeros((B_dim, n * m * p))
    W = np.zeros((C_dim, n * m * p))
    rank = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, rank] = 1.0
                V[b_idx, rank] = 1.0
                W[c_idx, rank] = 1.0
                rank += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy: Try to eliminate terms by finding linear dependencies
    # A term r can be eliminated if its contribution U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # can be expressed as a linear combination of other terms.
    # 
    # This is equivalent to checking if the vector (U[:,r] ⊗ V[:,r] ⊗ W[:,r])
    # is in the span of the other term vectors.
    #
    # More practically: we can try pairwise cancellations.
    # If two terms have the same support pattern (same non-zero positions in U, V, W),
    # we can combine them.
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition is correct."""
        if time.time() > deadline:
            return False
        T_rec = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.allclose(T, T_rec, atol=1e-10)
    
    def try_eliminate_term(U_mat, V_mat, W_mat, rank, term_to_remove):
        """Try to eliminate term_to_remove by expressing it as combo of others."""
        if time.time() > deadline:
            return None
        
        U_mat = U_mat.copy()
        V_mat = V_mat.copy()
        W_mat = W_mat.copy()
        
        # Check if term can be eliminated
        # The term contributes: U[:,t] * V[:,t] * W[:,t]
        # We need to find coefficients c_r such that:
        # sum_{r != t} c_r * U[:,r] * V[:,r] * W[:,r] = U[:,t] * V[:,t] * W[:,t]
        
        # This is a nonlinear constraint in general, but for our structured problem,
        # we can try a simpler approach: check if the "pattern" of the term is redundant.
        
        # Alternative approach: use the fact that if two terms have proportional
        # U, V, or W vectors, we can combine them.
        
        # For now, use a greedy approach: try to find if term t's pattern is
        # already covered by other terms.
        
        # Simpler heuristic: if U[:,t], V[:,t], W[:,t] are each linear combinations
        # of the corresponding vectors of other terms, AND the product structure
        # is preserved, we can eliminate.
        
        # Actually, let's use a different structural approach:
        # Group terms by their "signature" - which indices are non-zero.
        
        return None  # Placeholder
    
    # Better structural approach: exploit the block structure of matrix multiplication.
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # 
    # We can use the fact that for fixed i,k, the sum over j is a dot product.
    # 
    # Key insight: We can use "sharing" of products across different (i,k) pairs
    # that share the same j index.
    
    # Let's try a different decomposition strategy:
    # Split A into two halves: A_left (2x2), A_right (2x2)
    # Split B into 4 quarters: B_0 (4x2), B_1 (4x2), B_2 (4x2), B_3 (4x2)
    # 
    # C[:, 0:2] = A_left * B_0 + A_right * B_0  -- NO, this isn't right.
    # 
    # Correct: C[k, i] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # For C[:, 0:2] (k=0,1): C[k,i] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # 
    # We can write: C[:, 0:2] = A[:,0:2] * B[0:2, 0:2] + A[:,2:4] * B[2:4, 0:2]
    # where A[:,0:2] is 2x2, B[0:2,0:2] is 2x2, etc.
    # 
    # Each 2x2 * 2x2 multiplication can be done with 7 multiplications (Strassen).
    # We have 4 such blocks (for k=0:2, 2:4, 4:6, 6:8), each with 2 sub-blocks.
    # Total: 4 * 2 * 7 = 56 multiplications.
    
    # But we can do better by sharing across blocks!
    # 
    # Notice that A[:, 0:2] is used in ALL 4 blocks for the top half of B.
    # Similarly, A[:, 2:4] is used in ALL 4 blocks for the bottom half of B.
    # 
    # So we can precompute Strassen factors for A_left and A_right,
    # and then use them with each B block.
    
    # Let's construct a decomposition with sharing:
    # 
    # Precompute for A_left = A[:, 0:2]:
    #   s1 = a00 + a11, s2 = a10 + a11, s3 = a00, s4 = a11, s5 = a00+a01, s6 = a10-a00, s7 = a01-a11
    # 
    # Precompute for A_right = A[:, 2:4]:
    #   t1 = a20 + a31, t2 = a30 + a31, t3 = a20, t4 = a31, t5 = a20+a21, t6 = a30-a20, t7 = a21-a31
    # 
    # For each B block B_g (4x2) where g=0,1,2,3:
    #   B_g_top = B[0:2, 2g:2g+2] (2x2)
    #   B_g_bot = B[2:4, 2g:2g+2] (2x2)
    # 
    #   C_block = A_left * B_g_top + A_right * B_g_bot
    # 
    #   Using Strassen:
    #   M1 = s1 * (b00+b11), M2 = s2 * b00, M3 = s3 * (b10-b11), M4 = s4 * (b01-b10)
    #   M5 = s5 * b11, M6 = s6 * (b00+b10), M7 = s7 * (b01+b11)
    # 
    #   N1 = t1 * (b20+b31), N2 = t2 * b20, N3 = t3 * (b30-b31), N4 = t4 * (b21-b30)
    #   N5 = t5 * b31, N6 = t6 * (b20+b30), N7 = t7 * (b21+b31)
    # 
    #   c00 = M1 - M5 + M7 + M4 + N1 - N5 + N7 + N4
    #   c01 = M3 + M5 + N3 + N5
    #   c10 = M2 + M4 + N2 + N4
    #   c11 = M1 - M2 + M3 + M6 + N1 - N2 + N3 + N6
    
    # Total multiplications: 4 * (7 + 7) = 56. Same as before, no sharing benefit yet.
    
    # To get sharing, we need to find common subexpressions.
    # 
    # Key observation: The Strassen factors for A are FIXED across all B blocks.
    # So we can "pre-multiply" them. But in the tensor decomposition framework,
    # each multiplication is a separate term, so we can't directly share.
    
    # However, we CAN share if we restructure. Let's think differently.
    
    # Alternative: Use the "laser method" or "substitution" approach.
    # 
    # For now, let's try a random search approach:
    # Start with rank 56 (Strassen-based) and try to reduce by finding
    # linear dependencies.
    
    # Build Strassen-based decomposition first
    U_strassen = np.zeros((A_dim, 56))
    V_strassen = np.zeros((B_dim, 56))
    W_strassen = np.zeros((C_dim, 56))
    
    u_strassen = np.array([
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1]   # M7: a01-a11
    ], dtype=float)
    
    v_strassen = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b11
        [0, 1, -1, 0],  # M4: b01-b10
        [0, 0, 0, 1],   # M5: b11
        [1, 0, 1, 0],   # M6: b00+b10
        [0, 1, 0, 1]    # M7: b01+b11
    ], dtype=float)
    
    w_strassen = np.array([
        [1, 0, 0, 1],   # M1: c00+c11
        [0, 0, 1, -1],  # M2: c10-c11
        [0, 1, 0, 1],   # M3: c01+c11
        [1, 0, 1, 0],   # M4: c00+c10
        [-1, 1, 0, 0],  # M5: -c00+c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ], dtype=float)
    
    col_offset = 0
    for g in range(4):
        a_l_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
        a_r_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
        b_top_idx = [0*8+(2*g), 0*8+(2*g+1), 1*8+(2*g), 1*8+(2*g+1)]
        b_bot_idx = [2*8+(2*g), 2*8+(2*g+1), 3*8+(2*g), 3*8+(2*g+1)]
        c_block_idx = [(2*g)*2 + 0, (2*g)*2 + 1, (2*g+1)*2 + 0, (2*g+1)*2 + 1]
        
        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_l_idx):
                U_strassen[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_top_idx):
                V_strassen[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_block_idx):
                W_strassen[global_k, col] = w_strassen[s, local_k]
        col_offset += 7
        
        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_r_idx):
                U_strassen[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_bot_idx):
                V_strassen[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_block_idx):
                W_strassen[global_k, col] = w_strassen[s, local_k]
        col_offset += 7
    
    # Verify Strassen decomposition
    if verify(U_strassen, V_strassen, W_strassen, 56):
        best_rank = 56
        best_U, best_V, best_W = U_strassen.copy(), V_strassen.copy(), W_strassen.copy()
    
    # Now try to reduce the rank by finding linear dependencies
    # A term r is redundant if there exist coefficients c_s (s != r) such that:
    # U[:,r] ⊗ V[:,r] ⊗ W[:,r] = sum_{s != r} c_s * (U[:,s] ⊗ V[:,s] ⊗ W[:,s])
    
    # This is hard to check directly. Instead, use a heuristic:
    # Check if removing term r and adjusting other terms still gives the correct tensor.
    
    # For each term, try to eliminate it by solving for adjustments to other terms.
    # This is a system of linear equations in the adjustments.
    
    def try_eliminate(U_mat, V_mat, W_mat, rank, term_idx):
        """Try to eliminate term_idx. Returns (U', V', W', new_rank) or None."""
        if time.time() > deadline:
            return None
        
        # The contribution of term_idx is:
        # delta_T[a,b,c] = U[a,term_idx] * V[b,term_idx] * W[c,term_idx]
        
        # We need to find adjustments dU[a,s], dV[b,s], dW[c,s] for s != term_idx
        # such that the total change cancels delta_T.
        
        # This is nonlinear. Instead, use a simpler approach:
        # Check if the term's "fingerprint" is already present in other terms.
        
        # Fingerprint: the set of (a,b,c) where U[a,term]*V[b,term]*W[c,term] != 0
        # If this set is a subset of the union of fingerprints of other terms,
        # AND the values can be matched, we can eliminate.
        
        # For our structured problem, let's try a different tactic:
        # Use the fact that many terms have the same U or V or W patterns.
        
        # Group terms by their U column pattern
        U_col = U_mat[:, term_idx]
        V_col = V_mat[:, term_idx]
        W_col = W_mat[:, term_idx]
        
        # Find other terms with proportional U, V, or W
        # If U[:,s] = alpha * U[:,term], V[:,s] = beta * V[:,term], W[:,s] = gamma * W[:,term]
        # and alpha*beta*gamma = 1, then term s can replace term term.
        
        # More generally, if we can find a linear combination of other terms
        # that matches the product pattern.
        
        # For now, return None to indicate we can't easily eliminate this term.
        return None
    
    # Try a random perturbation approach:
    # Randomly modify the decomposition and check if rank can be reduced.
    
    # Actually, let's try a completely different structural decomposition.
    # 
    # Use the "four Russians" method or a more clever blocking.
    # 
    # Key idea: Instead of splitting B into 4 independent 2-column blocks,
    # use overlapping blocks to share computations.
    
    # For example, compute:
    # P1 = A[:, 0:2] * B[0:2, 0:4]  (2x4 result)
    # P2 = A[:, 2:4] * B[2:4, 0:4]  (2x4 result)
    # Then C[:, 0:4] = P1 + P2
    # 
    # P1 can be computed as: [A_left * B_top_left, A_left * B_top_right]
    # where B_top_left = B[0:2, 0:2], B_top_right = B[0:2, 2:4]
    # 
    # Using Strassen for each 2x2*2x2, we get 7+7=14 multiplications for P1.
    # Similarly 14 for P2. Total 28 for C[:, 0:4].
    # 
    # For C[:, 4:8], another 28. Total 56. Same as before.
    
    # To beat 56, we need true sharing. Let's try:
    # 
    # Compute shared Strassen factors for A_left:
    # s1 = a00+a11, s2 = a10+a11, s3 = a00, s4 = a11, s5 = a00+a01, s6 = a10-a00, s7 = a01-a11
    # 
    # For B[0:2, 0:4], define:
    # b1 = B[0,0]+B[1,3], b2 = B[0,0], b3 = B[1,0]-B[1,3], b4 = B[0,1]-B[1,0]
    # b5 = B[1,3], b6 = B[0,0]+B[1,0], b7 = B[0,1]+B[1,3]
    # 
    # Wait, this doesn't work because Strassen requires specific combinations.
    
    # I think 56 might be close to optimal for this structure. Let's try to verify
    # if we can do better with a different approach.
    
    # Lower bound: The tensor rank of matrix multiplication is at least
    # max(n*m, m*p, p*n) = max(8, 32, 16) = 32.
    # 
    # Known results: For 2x4x8, the rank is between 32 and 56.
    # 
    # Let's try to find a decomposition with rank < 56 using a search.
    
    # Use a randomized approach: start with rank 50 and try to find a valid decomposition.
    
    def random_decomposition(rank, seed=42):
        """Generate a random decomposition and try to optimize it."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        U_rand = np.random.randn(A_dim, rank)
        V_rand = np.random.randn(B_dim, rank)
        W_rand = np.random.randn(C_dim, rank)
        
        # Normalize
        for r in range(rank):
            U_norm = np.linalg.norm(U_rand[:, r])
            V_norm = np.linalg.norm(V_rand[:, r])
            W_norm = np.linalg.norm(W_rand[:, r])
            if U_norm > 1e-10 and V_norm > 1e-10 and W_norm > 1e-10:
                U_rand[:, r] /= U_norm
                V_rand[:, r] /= V_norm
                W_rand[:, r] /= (U_norm * V_norm * W_norm)
        
        return U_rand, V_rand, W_rand
    
    # Try to find a decomposition with rank < 56
    # This is a hard problem, so we'll use a heuristic.
    
    # For now, return the best we have (Strassen with rank 56)
    if best_rank <= 56:
        return {"U": best_U[:, :best_rank].tolist(), "V": best_V[:, :best_rank].tolist(), "W": best_W[:, :best_rank].tolist()}
    
    # Fallback to schoolbook
    return {"U": U[:, :rank].tolist(), "V": V[:, :rank].tolist(), "W": W[:, :rank].tolist()}
# EVOLVE-BLOCK-END
