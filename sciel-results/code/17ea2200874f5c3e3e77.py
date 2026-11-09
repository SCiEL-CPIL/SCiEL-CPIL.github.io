import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    t0 = time.time()
    deadline = 120.0  # safety margin
    
    # We'll use a rank-32 decomposition based on the following insight:
    # The matrix multiplication tensor M_{n,m,p} can be decomposed using
    # the structure that for each output position (i,k), we sum over j.
    # 
    # Key idea: Use the fact that we can group the computation by exploiting
    # the rectangular structure. For 2x4 * 4x8, we can use a decomposition
    # that achieves rank 32.
    #
    # The decomposition works as follows:
    # For each of the 4 "column groups" of B (groups of 2 columns each),
    # and for each of the 2 "row groups" of A (groups of 2 rows each),
    # we use a 2x2x2 tensor decomposition (rank 7 via Strassen-like)
    # but adapted to our rectangular case.
    #
    # Actually, let me use a cleaner approach: 
    # The rank of M_{2,4,8} is at most 32. Here's a construction:
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Group j into pairs: j = 2*b + jp where b in {0,1}, jp in {0,1}
    # Group k into groups of 2: k = 2*g + kp where g in {0,1,2,3}, kp in {0,1}
    #
    # For each (b, g) pair, we have a 2x2x2 sub-problem:
    # C_sub[i,kp] = sum_{jp=0}^{1} A_sub[i,jp] * B_sub[jp,kp]
    # where A_sub is 2x2 (rows i=0,1; cols jp=0,1 from A's row i, cols 2b+jp)
    # B_sub is 2x2 (rows jp=0,1; cols kp=0,1 from B's rows 2b+jp, cols 2g+kp)
    #
    # Each 2x2x2 matrix multiplication has rank 7 (Strassen).
    # We have 2 (b values) * 4 (g values) = 8 such sub-problems.
    # Total rank = 8 * 7 = 56.
    #
    # But we can do better! Let's try a different grouping.
    #
    # Alternative: Use the fact that n=2 is small.
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    #
    # For each k, this is a linear combination of the rows of B weighted by A[0,:] and A[1,:].
    #
    # Let me try a direct rank-32 construction:
    # For each j in {0,1,2,3} and each k in {0,...,7}, we have terms A[i,j]*B[j,k]
    # The naive decomposition has rank n*m*p = 64.
    #
    # To reduce rank, we can use the following trick:
    # For fixed j, the contribution to C is A[:,j] * B[j,:] (outer product)
    # This is a rank-1 update to C for each j.
    # But A[:,j] is 2x1 and B[j,:] is 1x8, so the outer product is 2x8.
    #
    # The tensor for fixed j: T_j[i,k] = A[i,j] * B[j,k]
    # This can be written as: sum over r=1}^{2} u_r * v_r where u_r is in R^2, v_r is in R^8
    # Specifically, T_j = A[:,j] * B[j,:] = (A[0,j], A[1,j])^T * (B[j,0], ..., B[j,7])
    # This is already rank 1 as a matrix (outer product of two vectors).
    #
    # So the full tensor is: M = sum_{j=0}^{3} sum_{k=0}^{7} e_i * e_{i,j} * e_{j,k} * e_{i,k}
    # Wait, I'm getting confused again.
    #
    # Let me just use the rank-56 Strassen decomposition which is known to work,
    # and see if I can verify it and potentially improve it.
    
    # Actually, let me try a rank-32 decomposition using a different approach.
    # 
    # Insight: For 2x4 * 4x8, we can view this as 4 independent 2x1 * 1x8 products
    # summed together. Each 2x1 * 1x8 product has rank 1 (it's an outer product).
    # But the tensor structure couples them.
    #
    # Let me try: rank = m * max(n, p/n) = 4 * max(2, 4) = 16? Probably not achievable.
    #
    # Known bounds: rank(M_{n,m,p}) >= max(n*p, m*n, m*p) / something... 
    # Actually, the asymptotic rank is about (np)^{log_2 7} for square matrices.
    # For 2x4*4x8, a reasonable rank is around 32.
    #
    # Let me construct a rank-32 decomposition explicitly.
    #
    # Approach: Use the "border rank" or "approximate rank" ideas, but we need exact.
    #
    # Here's a concrete rank-32 construction:
    # For each j in {0,1,2,3}, define 8 terms (one for each k):
    # Term (j,k): U has A[:,j] in positions (0, j) and (1, j)
    #              V has B[j,:] in positions (j, 0) through (j, 7)
    #              W has 1 in position (k, i) for i=0,1... 
    # This gives rank 4*8 = 32!
    #
    # Let me verify: 
    # C[i,k] = sum_{j=0}^{3} sum_{k'=0}^{7} U[i,j] * V[j,k'] * W[k',i]
    # We want C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # If U[i,j] = A[i,j], V[j,k] = B[j,k], W[k',i] = delta_{k',k}... 
    # But W can't depend on k in that way for a fixed decomposition.
    #
    # Actually, the decomposition is:
    # M = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    # where u_r in R^{nm}, v_r in R^{mp}, w_r in R^{pn}
    #
    # And the multiplication is: C = sum_r (U[:,r] * A_flat) * (V[:,r] * B_flat) * W[:,r]
    # Wait, that's not right either. The bilinear form is:
    # C = sum_r (u_r^T A_flat) * (v_r^T B_flat) * w_r
    #
    # So for each r, we compute scalar_r = (u_r^T A_flat) * (v_r^T B_flat)
    # and then C = sum_r scalar_r * w_r
    #
    # For the naive decomposition with rank nmp:
    # For each (i,j,k): u has 1 at position (i,j), v has 1 at position (j,k), w has 1 at position (k,i) [or (i,k) depending on convention]
    # Then scalar = A[i,j] * B[j,k], and C[k,i] += A[i,j]*B[j,k] (or C[i,k] += ...)
    #
    # With the W[k*n+i, r] convention: W index is k*n + i.
    #
    # So for naive rank 64:
    # For each (i,j,k): 
    #   U[i*m + j, r] = 1
    #   V[j*p + k, r] = 1  
    #   W[k*n + i, r] = 1
    #
    # Now, can we reduce this to rank 32?
    #
    # Idea: For each j, group the 2*8=16 terms (i in {0,1}, k in {0,...,7}).
    # For fixed j, we have terms: (i,j,k) for i=0,1 and k=0,...,7.
    # U part: U[i*m+j, r] = 1 for i=0,1
    # V part: V[j*p+k, r] = 1 for k=0,...,7
    # W part: W[k*n+i, r] = 1
    #
    # For fixed j, the U vectors are: e_{0,m+j} and e_{1,m+j} (two vectors in R^{nm})
    # The V vectors are: e_{j*p+0}, ..., e_{j*p+7} (eight vectors in R^{mp})
    # The W vectors are: e_{k*n+i} for k=0..7, i=0,1 (16 vectors in R^{pn})
    #
    # The sub-tensor for fixed j is: sum_{i=0}^{1} sum_{k=0}^{7} e_{i,m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # This is a tensor of size 2 x 8 x 16. What's its rank?
    # It can be written as: sum_{i,k} u_i ⊗ v_k ⊗ w_{i,k}
    # where u_i = e_{i,m+j}, v_k = e_{j*p+k}, w_{i,k} = e_{k*n+i}
    #
    # Note that w_{i,k} depends on both i and k. The 16 W vectors are all distinct basis vectors.
    # The U vectors are only 2 distinct ones, V vectors are 8 distinct ones.
    #
    # This sub-tensor has rank at most 2*8 = 16 (trivially), but can we do better?
    # The W vectors span a 16-dimensional space (they're all distinct basis vectors),
    # so the rank is at least 16... unless there's linear dependence.
    #
    # Actually, the rank of the sub-tensor for fixed j is exactly 16 (since the W components
    # are 16 linearly independent vectors, and the tensor is "diagonal" in some sense).
    #
    # So total rank = 4 * 16 = 64. No improvement.
    #
    # Let me try a different grouping. What if we group by k instead?
    # For fixed k, we have terms (i,j,k) for i=0,1 and j=0,1,2,3.
    # U vectors: e_{i*m+j} for i=0,1, j=0,1,2,3 (8 vectors)
    # V vectors: e_{j*p+k} for j=0,1,2,3 (4 vectors)
    # W vectors: e_{k*n+i} for i=0,1 (2 vectors)
    #
    # Sub-tensor: sum_{i=0}^{1} sum_{j=0}^{3} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # The W vectors are only 2: e_{k*n+0} and e_{k*n+1}.
    # Can we write this as a sum of 2 rank-1 tensors?
    # 
    # For i=0: sum_j e_{j} ⊗ e_{j*p+k} ⊗ e_{k*n+0}
    # For i=1: sum_j e_{m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+1}
    #
    # The i=0 part: sum_j e_j ⊗ e_{j*p+k} ⊗ e_{k*n+0}
    # This is a tensor in R^8 x R^4 x R^2. The W component is always e_{k*n+0} (constant).
    # So this is: (sum_j e_j ⊗ e_{j*p+k}) ⊗ e_{k*n+0}
    # The first factor is a matrix in R^{8 x 4} with entries [i,j] = delta_{i,j} for i=0..3, j=0..3, and 0 for i=4..7.
    # This matrix has rank 4. So the i=0 part has rank 4.
    #
    # Similarly, the i=1 part has rank 4.
    #
    # Total for fixed k: rank 8.
    # Total for all k: 8 * 8 = 64. Still no improvement.
    #
    # I think the rank is actually 64 for this specific case, or close to it.
    # Let me just use the rank-56 Strassen decomposition.

    num_groups = 4  # k-blocks of size 2
    num_ablocks = 2  # j-blocks of size 2
    num_strassen = 7

    rank = num_groups * num_ablocks * num_strassen  # 56

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    X_forms = np.array([
        [1, 0, 0, 1],    # m1: a + d
        [0, 0, 1, 1],    # m2: c + d
        [1, 0, 0, 0],    # m3: a
        [0, 0, 0, 1],    # m4: d
        [1, 1, 0, 0],    # m5: a + b
        [-1, 0, 1, 0],   # m6: c - a
        [0, 1, 0, -1],   # m7: b - d
    ], dtype=np.int64)

    Y_forms = np.array([
        [1, 0, 0, 1],    # m1: e + h
        [1, 0, 0, 0],    # m2: e
        [0, 1, 0, -1],   # m3: f - h
        [-1, 0, 1, 0],   # m4: g - e
        [0, 0, 0, 1],    # m5: h
        [1, 1, 0, 0],    # m6: e + f
        [0, 0, 1, 1],    # m7: g + h
    ], dtype=np.int64)

    Z_combos = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # c00 = m1 + m4 - m5 + m7
        [0, 0, 1, 0, 1, 0, 0],    # c01 = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # c10 = m2 + m4
        [1, -1, 1, 0, 0, 1, 0],   # c11 = m1 - m2 + m3 + m6
    ], dtype=np.int64)

    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx, r] = X_forms[s, i * 2 + jp]

                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx, r] = Y_forms[s, jp * 2 + kp]

                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx, r] = Z_combos[i * 2 + kp, s]

                r += 1

    # Verify within time budget
    if time.time() - t0 < deadline - 10:
        # Quick sanity check
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Test with random matrices
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        # Compute C from decomposition
        C_computed = np.zeros(n * p)
        for r_idx in range(rank):
            scalar = np.dot(U_np[:, r_idx], A_flat) * np.dot(V_np[:, r_idx], B_flat)
            C_computed += scalar * W_np[:, r_idx]
        
        C_computed = C_computed.reshape(n, p)
        
        # Check if close
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to naive rank-64
            rank64 = n * m * p
            U64 = np.zeros((n * m, rank64), dtype=np.int64)
            V64 = np.zeros((m * p, rank64), dtype=np.int64)
            W64 = np.zeros((p * n, rank64), dtype=np.int64)
            
            r_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U64[i * m + j, r_idx] = 1
                        V64[j * p + k, r_idx] = 1
                        W64[k * n + i, r_idx] = 1
                        r_idx += 1
            
            return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
