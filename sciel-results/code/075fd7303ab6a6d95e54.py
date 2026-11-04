import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank-56 Strassen decomposition for 2x4 * 4x8
    # This uses 8 blocks of 2x2 * 2x2 Strassen (rank 7 each)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard 2x2 Strassen decomposition
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi][cr] += val
    
    # 8 blocks: split 4x8 into 4 blocks of 2x8 (each split into 2 blocks of 2x2)
    # Actually: 2x4 * 4x8 = sum over j_blocks of 2x2 * 2x8
    # Split the 4 columns of A into 2 blocks of 2, and 8 columns of B into 4 blocks of 2
    # So 2 * 4 = 8 blocks of 2x2 * 2x2
    
    # Block (i_off, j_off, k_off) where i_off in {0}, j_off in {0,1}, k_off in {0,1,2,3}
    for j_off in range(2):  # 2 blocks of width 2 in the middle dimension
        for k_off in range(4):  # 4 blocks of width 2 in the output dimension
            base_rank = (j_off * 4 + k_off) * 7
            add_block(base_rank, 0, j_off * 2, k_off * 2)
    
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def compute_tensor(U_arr, V_arr, W_arr, rank_val):
        """Compute the tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]"""
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :rank_val]
                for k in range(p):
                    v_row = V_arr[j*p+k, :rank_val]
                    w_row = W_arr[k*n+i, :rank_val]
                    tensor[i,j,k] = np.dot(u_row, v_row * w_row)
        return tensor
    
    def is_valid(U_arr, V_arr, W_arr, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        tensor = compute_tensor(U_arr, V_arr, W_arr, rank_val)
        expected = np.ones((n, m, p))
        return np.allclose(tensor, expected, atol=1e-10)
    
    def find_nullspace_basis(U_arr, V_arr, W_arr, R):
        """
        Find a null-space vector c in R^R such that for all i,j,k:
        sum_r c_r * U[i,j,r] * V[j,k,r] * W[k,i,r] = 0
        This is a bilinear condition. We use the approach:
        For each pair (a,b) of rank indices, check if there exists c with c_a != 0, c_b != 0
        that makes the tensor vanish.
        
        Better approach: Build the "flattened" constraint. The condition is:
        For all (i,j,k): sum_r c_r * (U[i,j,r] * V[j,k,r] * W[k,i,r]) = 0
        
        We can think of this as: let f_r(i,j,k) = U[i,j,r]*V[j,k,r]*W[k,i,r]
        Then we need sum_r c_r * f_r = 0 as a tensor.
        
        This means the vectors f_1, ..., f_R (flattened to R^64) are linearly dependent
        with coefficients c. But that's not quite right because the constraint is
        element-wise product, not linear combination of the factor vectors.
        
        Actually, the correct approach: The tensor T = sum_r c_r * (u_r ⊗ v_r ⊗ w_r)
        must be zero. This is equivalent to saying that the rank-1 tensors
        u_r ⊗ v_r ⊗ w_r are linearly dependent with coefficients c.
        
        So we flatten each rank-1 tensor and find null space of the matrix whose
        columns are these flattened tensors.
        """
        # Flatten each rank-1 tensor u_r ⊗ v_r ⊗ w_r into a vector of size n*m*p = 64
        # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        flat_tensors = np.zeros((n*m*p, R))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        flat_tensors[idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        
        # Find null space of flat_tensors (64 x R matrix)
        # We want c such that flat_tensors @ c = 0
        # Use SVD
        try:
            U_svd, s, Vt = np.linalg.svd(flat_tensors, full_matrices=True)
        except np.linalg.LinAlgError:
            return None
        
        # The null space dimension is R - rank(flat_tensors)
        # s has length min(64, R) = 64 (since R=56 < 64)
        # Null space vectors are in the last R-64 columns of Vt if R > 64, else in the last (R-rank) columns
        tol = 1e-10
        rank_flat = np.sum(s > tol)
        null_dim = R - rank_flat
        
        if null_dim <= 0:
            return None
        
        # Null space basis: last null_dim rows of Vt
        null_basis = Vt[R-rank_flat:R, :]  # shape (null_dim, R)
        
        return null_basis
    
    def apply_nullspace_reduction(U_arr, V_arr, W_arr, R, null_basis):
        """
        Given a null-space basis, find a change of basis that zeros out some columns.
        We want to find an invertible matrix P (R x R) such that
        new_U = U @ P, new_V = V @ P, new_W = W @ P
        has some zero columns.
        
        Actually, the decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        If we apply P: T = sum_s (sum_r U[:,r] P[r,s]) ⊗ (sum_r V[:,r] P[r,s]) ⊗ (sum_r W[:,r] P[r,s])
        Wait, that's not right. The tensor is:
        T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        If we change basis with P: new_U[i,j,s] = sum_r U[i,j,r] P[r,s]
        Then T[i,j,k] = sum_s new_U[i,j,s] * V[j,k,s] * W[k,i,s]  -- NO!
        
        The correct transformation: T = sum_r (U[:,r] ⊗ V[:,r] ⊗ W[:,r])
        This is NOT a simple linear combination. Each term is a product of three vectors.
        
        The null space approach: if c is in the null space, then
        sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
        This means we can replace the set of rank-1 tensors by a linearly independent subset.
        
        To actually reduce rank: if the rank-1 tensors span a space of dimension d < R,
        then we can express them in terms of d independent ones. But the new "rank-1 tensors"
        are linear combinations, which are NOT rank-1 in general.
        
        The correct way: We need to find a different decomposition with fewer terms.
        The null space tells us the original R terms are linearly dependent as tensors,
        but that doesn't directly give a lower-rank decomposition.
        
        Alternative approach: Iteratively try to find pairs (a,b) and a scalar α such that
        replacing term a with (term a + α * term b) and term b with (term b) allows
        one of them to be removed. This is more complex.
        
        Let me use a different strategy: greedy elimination with proper basis changes.
        """
        return None  # Placeholder - will use simpler approach below
    
    # Verify initial decomposition
    if not is_valid(U_np, V_np, W_np, rank):
        # Fallback: return the original Strassen decomposition
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Use the tensor's linear structure to find reducible decompositions
    # Key insight: The Strassen decomposition has shared sub-structures across blocks.
    # Specifically, the "M1 = (A11+A22)(B11+B22)" terms across different blocks
    # may have linear dependencies.
    
    # Better approach: Build the full tensor and use algebraic geometry / linear algebra
    # to find the minimal rank. For small tensors, we can use the "border rank" approach
    # or direct enumeration.
    
    # For now, use a more effective greedy approach:
    # 1. Try all pairs of columns (a, b) and all scalars α
    # 2. Check if replacing column a with (a + α*b) allows column b to be zeroed
    #    (or vice versa)
    
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    best_rank = rank
    best_U = current_U.copy()
    best_V = current_V.copy()
    best_W = current_W.copy()
    
    # Precompute tensor for fast validation
    def tensor_diff(U_arr, V_arr, W_arr, R):
        """Compute T - I (the error tensor)"""
        T = compute_tensor(U_arr, V_arr, W_arr, R)
        return T - 1.0
    
    # First pass: try simple column zeroing (fast)
    improved = True
    while improved:
        improved = False
        for r in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            test_U = current_U.copy()
            test_V = current_V.copy()
            test_W = current_W.copy()
            test_U[:, r] = 0
            test_V[:, r] = 0
            test_W[:, r] = 0
            
            if is_valid(test_U, test_V, test_W, current_rank):
                mask = [x for x in range(current_rank) if x != r]
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                improved = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                break
        
        if time.time() > deadline - 10:
            break
    
    # Second pass: try pairwise linear combinations
    # For each pair (a, b), try to find α such that we can eliminate one term
    # This is done by checking if the "combined" tensor can be represented
    # with one fewer term.
    
    # More efficient: use the null space of the flattened tensor matrix
    # to identify which linear combinations vanish.
    
    # Build flattened tensor matrix
    def build_flat_matrix(U_arr, V_arr, W_arr, R):
        """Build matrix where column r is the flattened rank-1 tensor u_r ⊗ v_r ⊗ w_r"""
        flat = np.zeros((n*m*p, R))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        flat[idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        return flat
    
    if time.time() < deadline - 20:
        flat_mat = build_flat_matrix(current_U, current_V, current_W, current_rank)
        
        # Find null space
        try:
            U_svd, s, Vt = np.linalg.svd(flat_mat, full_matrices=True)
            tol = 1e-8
            rank_flat = np.sum(s > tol)
            null_dim = current_rank - rank_flat
            
            if null_dim > 0:
                null_basis = Vt[current_rank-rank_flat:current_rank, :]  # (null_dim, current_rank)
                
                # For each null space vector, try to use it to reduce rank
                # A null vector c means: sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
                # This means the set {u_r ⊗ v_r ⊗ w_r} is linearly dependent.
                # We can replace one term with a linear combination.
                
                # Strategy: For each null vector c, find the index r with max |c_r|
                # and try to express term r as a combination of others.
                # But this doesn't directly reduce rank unless we can make one term zero.
                
                # Alternative: Use the null space to find a change of basis that
                # creates zero columns.
                
                # Let's try: for each null vector c, create a new set of terms where
                # one term is replaced by (term_r + c_r/c_s * term_s) for some s.
                # This is getting complex. Let's use a simpler heuristic.
                
                # Simple heuristic: if null_dim > 0, there are dependencies.
                # Try to find a column that is "close to" a linear combination of others
                # and eliminate it.
                
                # For each column r, check if it's in the span of the others
                # by checking if the null space of the matrix without column r
                # still has dimension >= 1.
                
                for r in range(current_rank):
                    if time.time() > deadline - 15:
                        break
                    
                    # Remove column r and check if the remaining columns still span the tensor
                    mask = [x for x in range(current_rank) if x != r]
                    if len(mask) < current_rank - 1:
                        continue
                    
                    # Check if column r is in the span of the others
                    # by checking if the null space of flat_mat[:, mask] 
                    # contains a vector that when extended with the r-th component
                    # gives a valid representation
                    flat_without_r = flat_mat[:, mask]
                    
                    # The tensor T = sum_s flat_mat[:, s] (as a vector)
                    # We need: sum_{s in mask} α_s * flat_mat[:, s] = flat_mat[:, r]
                    # This is a linear system: flat_without_r @ α = flat_mat[:, r]
                    
                    try:
                        alpha, residuals, rank_sys, sv = np.linalg.lstsq(flat_without_r, flat_mat[:, r], rcond=tol)
                    except np.linalg.LinAlgError:
                        continue
                    
                    # Check if the solution is accurate
                    reconstructed = flat_without_r @ alpha
                    if np.allclose(reconstructed, flat_mat[:, r], atol=1e-8):
                        # Column r is in the span of the others!
                        # But we need to verify this actually gives a valid decomposition.
                        # The issue: even if the flattened tensors span, the actual
                        # rank-1 structure may not be preserved.
                        
                        # Let's verify by actually constructing the new decomposition
                        new_U = current_U[:, mask].copy()
                        new_V = current_V[:, mask].copy()
                        new_W = current_W[:, mask].copy()
                        
                        # The new tensor is sum_{s in mask} (new_U[:,s] ⊗ new_V[:,s] ⊗ new_W[:,s])
                        # This equals the original tensor minus the r-th term plus the
                        # linear combination. But the linear combination of rank-1 tensors
                        # is NOT a rank-1 tensor in general!
                        
                        # So this approach only works if the "linear combination" can be
                        # expressed as a single rank-1 tensor, which is generally not true.
                        
                        # For now, let's just try the naive zeroing and see if it works.
                        test_U = current_U.copy()
                        test_V = current_V.copy()
                        test_W = current_W.copy()
                        test_U[:, r] = 0
                        test_V[:, r] = 0
                        test_W[:, r] = 0
                        
                        if is_valid(test_U, test_V, test_W, current_rank):
                            current_U = current_U[:, mask]
                            current_V = current_V[:, mask]
                            current_W = current_W[:, mask]
                            current_rank -= 1
                            if current_rank < best_rank:
                                best_rank = current_rank
                                best_U = current_U.copy()
                                best_V = current_V.copy()
                                best_W = current_W.copy()
                            break
        except Exception:
            pass
    
    # Third pass: try structured reductions based on the block structure
    # The 8 Strassen blocks share the same pattern. Look for cross-block dependencies.
    
    if time.time() < deadline - 10 and current_rank > 48:
        # Try to find if any two blocks can be merged or if there are
        # cross-block linear dependencies
        # For each pair of blocks (block_a, block_b), check if a linear
        # combination of their 7 terms each can reduce the total.
        
        # This is a more expensive search. Limit it.
        block_size = 7
        num_blocks = current_rank // block_size
        
        if num_blocks >= 2:
            # Try eliminating one term from each pair of blocks
            for block_a in range(num_blocks):
                if time.time() > deadline - 10:
                    break
                for block_b in range(block_a + 1, num_blocks):
                    # Try all pairs of terms (one from each block)
                    for ra in range(block_size):
                        for rb in range(block_size):
                            if time.time() > deadline - 10:
                                break
                            
                            idx_a = block_a * block_size + ra
                            idx_b = block_b * block_size + rb
                            
                            # Try zeroing idx_a
                            test_U = current_U.copy()
                            test_V = current_V.copy()
                            test_W = current_W.copy()
                            test_U[:, idx_a] = 0
                            test_V[:, idx_a] = 0
                            test_W[:, idx_a] = 0
                            
                            if is_valid(test_U, test_V, test_W, current_rank):
                                mask = [x for x in range(current_rank) if x != idx_a]
                                current_U = current_U[:, mask]
                                current_V = current_V[:, mask]
                                current_W = current_W[:, mask]
                                current_rank -= 1
                                if current_rank < best_rank:
                                    best_rank = current_rank
                                    best_U = current_U.copy()
                                    best_V = current_V.copy()
                                    best_W = current_W.copy()
                                break
                            # Try zeroing idx_b
                            test_U = current_U.copy()
                            test_V = current_V.copy()
                            test_W = current_W.copy()
                            test_U[:, idx_b] = 0
                            test_V[:, idx_b] = 0
                            test_W[:, idx_b] = 0
                            
                            if is_valid(test_U, test_V, test_W, current_rank):
                                mask = [x for x in range(current_rank) if x != idx_b]
                                current_U = current_U[:, mask]
                                current_V = current_V[:, mask]
                                current_W = current_W[:, mask]
                                current_rank -= 1
                                if current_rank < best_rank:
                                    best_rank = current_rank
                                    best_U = current_U.copy()
                                    best_V = current_V.copy()
                                    best_W = current_W.copy()
                                break
    
    # Convert back to lists
    if best_rank < rank:
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
        
        U_final = [row[:best_rank] for row in U_final]
        V_final = [row[:best_rank] for row in V_final]
        W_final = [row[:best_rank] for row in W_final]
        
        # Round to nearest half-integer if close
        for mat in [U_final, V_final, W_final]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-10:
                        mat[i][j] = rounded
        
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
