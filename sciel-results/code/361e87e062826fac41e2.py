import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28

    # Build the true tensor T[a, b, c]
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1

    # Strassen factors for 2x2x2 (rank 7)
    U_strassen = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0]
    ], dtype=np.int64)
    V_strassen = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, -1, 0, 1],
        [1, 0, 0, 1, 0, -1, -1]
    ], dtype=np.int64)
    W_strassen = np.array([
        [1, -1, 1, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]
    ], dtype=np.int64)

    # Build the Strassen block decomposition (rank 49)
    rank = 7 * (p // 2)
    U = np.zeros((size_a, rank), dtype=np.int64)
    V = np.zeros((size_b, rank), dtype=np.int64)
    W = np.zeros((size_c, rank), dtype=np.int64)

    for b in range(p // 2):
        # B indices for this 2x2 block: columns 2b and 2b+1 of B (which is 2x14)
        # B is 2x14, so row j, col k -> index j*14+k
        # For block b (0-indexed), we use k = 2b and k = 2b+1
        # B indices: j=0 -> 0*14+2b = 2b, j=0 -> 0*14+2b+1 = 2b+1
        #             j=1 -> 1*14+2b = 14+2b, j=1 -> 1*14+2b+1 = 14+2b+1
        b_idx_map = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # C indices: C is 14x2, so row k, col i -> index k*2+i
        # For block b, k = 2b and k = 2b+1
        # C indices: k=2b, i=0 -> 2b*2+0 = 4b
        #            k=2b, i=1 -> 2b*2+1 = 4b+1
        #            k=2b+1, i=0 -> (2b+1)*2+0 = 4b+2
        #            k=2b+1, i=1 -> (2b+1)*2+1 = 4b+3
        c_idx_map = [4*b, 4*b+1, 4*b+2, 4*b+3]
        
        for r in range(7):
            global_col = b * 7 + r
            for i_local in range(4):
                U[i_local, global_col] = U_strassen[i_local, r]
            for j_local in range(4):
                V[b_idx_map[j_local], global_col] = V_strassen[j_local, r]
            for k_local in range(4):
                W[c_idx_map[k_local], global_col] = W_strassen[k_local, r]

    # Verify full decomposition
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

    if not np.array_equal(T_recon, T_true):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank by finding linear dependencies
    # The tensor has a block structure. We can try to find a smaller decomposition.
    
    # Approach: Use the fact that the problem is linear.
    # We have 49 rank-1 tensors. We want to find if any subset is linearly dependent.
    # This is equivalent to finding the rank of the 49x49 matrix M where
    # M[r1, r2] = <T_{r1}, T_{r2}> (inner product of the two rank-1 tensors)
    # But that's not quite right. We need to check if sum alpha_r T_r = 0.
    
    # Better approach: Flatten each rank-1 tensor T_r into a vector of length 4*28*28 = 3136
    # Then we have 49 vectors in R^3136. The rank of the matrix formed by these vectors
    # is the tensor rank (if the decomposition is minimal).
    
    # But 3136 * 49 is manageable. Let's try it.
    
    # Actually, let's think more carefully. The tensor T is in the space of 4x28x28 tensors.
    # Each rank-1 component T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] is a vector in this space.
    # The span of these 49 vectors has dimension at most 49.
    # If the dimension is less than 49, we have a linear dependency and can reduce rank.
    
    # Let's compute the rank of the 3136x49 matrix.
    # But 3136x49 is fine for SVD.
    
    # Flatten the tensors
    # T_r as a vector: for each (a,b,c), the value is U[a,r]*V[b,r]*W[c,r]
    # We can compute this as an outer product.
    
    # Create a matrix M of shape (3136, 49)
    M = np.zeros((size_a * size_b * size_c, rank), dtype=np.int64)
    
    for r in range(rank):
        # T_r[a,b,c] = U[a,r] * V[b,r] * W[c,r]
        # Flatten: index = (a * size_b + b) * size_c + c
        Tr = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        M[:, r] = Tr.flatten()
    
    # Compute the rank of M
    # Use SVD
    try:
        if time.time() > deadline - 10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Use float for SVD
        M_float = M.astype(np.float64)
        U_svd, s, Vt = np.linalg.svd(M_float, full_matrices=False)
        
        # Determine numerical rank
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # We have linear dependencies. Try to find a minimal decomposition.
            # The rank of the tensor is num_rank.
            # We need to find a new decomposition of rank num_rank.
            
            # This is more complex. For now, let's just note that the rank is at most num_rank.
            # But we need to actually construct the decomposition.
            
            # For now, return the current decomposition.
            # In a future iteration, we could try to find the actual minimal decomposition.
            pass
    except Exception:
        pass

    # Alternative approach: Try to find a smaller decomposition using the block structure.
    # The 2x14 matrix B can be seen as 7 blocks of 2x2.
    # The Strassen algorithm applies to each 2x2 block independently.
    # But maybe we can share some of the intermediate computations.
    
    # Actually, for 2x2 * 2x14, the tensor rank is known to be 49 (from 7 Strassen blocks).
    # Let's check if we can do better.
    
    # Known result: For 2x2 * 2xN, the rank is 7*N/2 for even N.
    # So for N=14, rank = 7*7 = 49.
    
    # But wait, is this optimal? Let's check the literature.
    # Actually, I recall that for 2x2 * 2xN, the rank might be less than 7*N/2 for some N.
    
    # Let's try a different approach: Use the fact that the A matrix is 2x2 and shared.
    # The Strassen algorithm for 2x2x2 uses 7 multiplications.
    # For 2x2 * 2x14, we have 7 such blocks.
    # Total: 7*7 = 49 multiplications.
    
    # Can we do better? Let's try to find a rank-48 decomposition.
    # This would require finding a linear dependency among the 49 rank-1 tensors.
    
    # Let's try to solve the linear system directly.
    # We want to find alpha_1, ..., alpha_49 such that sum alpha_r T_r = 0.
    # This is a homogeneous system with 3136 equations and 49 unknowns.
    # We can solve it using SVD.
    
    try:
        if time.time() > deadline - 10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Compute the SVD of M
        M_float = M.astype(np.float64)
        U_svd, s, Vt = np.linalg.svd(M_float, full_matrices=False)
        
        # The null space is spanned by the right singular vectors corresponding to zero singular values.
        # Find the null space basis.
        tol = 1e-10
        null_space_idx = np.where(s <= tol)[0]
        
        if len(null_space_idx) > 0:
            # We have a non-trivial null space.
            # The null space basis vectors are the rows of Vt corresponding to null_space_idx.
            null_space = Vt[null_space_idx, :]
            
            # For each null space vector, we have a linear dependency.
            # We can use this to reduce the rank.
            
            # For now, let's just return the current decomposition.
            # In a future iteration, we could try to construct a smaller decomposition.
            pass
    except Exception:
        pass

    # Let's try a completely different approach:
    # Use the fact that 14 = 7*2 and try to find a decomposition that exploits this structure.
    
    # Actually, let's try to use the "laser method" or "subrank" approach.
    # But this is complex and might not yield a smaller rank.
    
    # For now, let's return the Strassen decomposition with rank 49.
    # This is a valid decomposition and is likely close to optimal.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
