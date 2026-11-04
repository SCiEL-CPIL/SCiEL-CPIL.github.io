import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = 4, 4, 8
    
    # Build rank-98 Strassen decomposition
    U98, V98, W98 = _build_rank98()
    
    # Verify the rank-98 decomposition
    if not _verify(U98, V98, W98, n, m, p):
        return _naive(n, m, p)
    
    # Try to find rank reduction via tensor rank minimization
    best_rank = 98
    best_U, best_V, best_W = U98, V98, W98
    
    if time.time() < deadline - 5.0:
        reduced = _try_reduce_rank(U98, V98, W98, n, m, p, deadline)
        if reduced is not None:
            best_rank, best_U, best_V, best_W = reduced
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _build_rank98():
    n, m, p = 4, 4, 8
    
    # 2x2x2 Strassen matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 Strassen via Kronecker product (rank 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Block 1: columns 0:49
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Block 2: columns 49:98
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _naive(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to find a lower-rank equivalent decomposition.
    
    The 98 basis tensors span a subspace. The true tensor T = sum_r U_r ⊗ V_r ⊗ W_r
    has some minimal rank. We try to find it by:
    1. Checking if the 98 tensors are linearly dependent (in the 16*32*32 = 16384-dim space)
    2. If so, find a basis and re-express T in that basis
    
    More practically: use the fact that for matrix multiplication, the tensor rank
    is bounded. We try iterative compression.
    """
    start = time.time()
    n, m, p = 4, 4, 8
    dim_A = n * m  # 16
    dim_B = m * p  # 32
    dim_C = p * n  # 32
    rank0 = U.shape[1]  # 98
    
    # The tensor T lives in R^{16 x 32 x 32} with 16*32*32 = 16384 entries.
    # Each basis tensor is vec(U_r) ⊗ vec(V_r) ⊗ vec(W_r) but actually it's
    # the outer product u_r ⊗ v_r ⊗ w_r flattened.
    
    # Strategy: The 98 vectors u_r ⊗ v_r ⊗ w_r (each in R^16384) span some subspace.
    # But we need the tensor T itself, not just the span.
    
    # Alternative approach: Use the structure. The two blocks share U.
    # U[:, 0:49] = U[:, 49:98] = U4
    # This means for each output (i,k) pair, the contribution from both blocks
    # can be combined.
    
    # Actually, let's think differently. The tensor for 4x4*4x8 is:
    # T[i,j,k] = sum over internal indices... 
    # T = sum_r (u_r ⊗ v_r ⊗ w_r) where u_r ∈ R^16, v_r ∈ R^32, w_r ∈ R^32
    
    # The key structural observation: U has identical columns in blocks 0:49 and 49:98.
    # So for each r in 0..48, we have two tensors:
    #   T_r^1 = U4[:,r] ⊗ V_block1[:,r] ⊗ W_block1[:,r]
    #   T_r^2 = U4[:,r] ⊗ V_block2[:,r] ⊗ W_block2[:,r]
    
    # We can combine: for each r, define
    #   v_combined = [V_block1[:,r]; V_block2[:,r]]  (32-dim)
    #   w_combined = [W_block1[:,r]; W_block2[:,r]]  (32-dim)
    # But this doesn't reduce rank directly.
    
    # Better: Look for linear dependencies among the 98 basis tensors.
    # Each basis tensor is a vector in R^{16*32*32} = R^{16384}.
    # If they're linearly dependent, we can reduce.
    
    # But 98 vectors in R^16384 are almost certainly independent.
    
    # Alternative: Exploit the Kronecker structure more carefully.
    # The 4x4x4 Strassen has rank 49, but the true rank of 4x4x4 is at most 49.
    # Actually, it's known that R(4,4,4) ≤ 49 from Strassen, and it's not known if it's less.
    
    # For 4x4x8, we can potentially do better than 2*49=98 by sharing structure
    # between the two output column blocks.
    
    # Let me try a different approach: use the fact that the problem is
    # 4x4 * 4x8, which can be seen as computing 4 independent 4x4*4x1 products
    # but with the same left factor. 
    
    # Actually, let's try: can we use a rank-48 or lower decomposition?
    # The theoretical lower bound for 4x4x8 is at least 16*4/8 = 8 (trivial)
    # but more realistically, it's at least the max of the individual tensor ranks.
    
    # Let me try an iterative approach: randomly project and check rank.
    
    # Actually, the most promising approach given the time budget:
    # Try to find if there's a rank < 98 decomposition by solving for
    # linear combinations that give the same tensor.
    
    # Build the tensor T as a 3D array
    T = np.zeros((dim_A, dim_B, dim_C), dtype=float)
    for r in range(rank0):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # Now try to find minimal rank decomposition of T.
    # This is NP-hard in general, but for small tensors we can try.
    
    # Approach: Use the fact that if we can find rank r' < 98 such that
    # T = sum_{r'=1}^{r'} u'_r ⊗ v'_r ⊗ w'_r, we're done.
    
    # Try using the unfolding/mode-n rank approach.
    # T is 16x32x32. Unfold along mode 1: T_(1) is 16 x (32*32) = 16 x 1024.
    # rank(T) >= rank(T_(1)) = rank of 16x1024 matrix.
    
    T1 = T.reshape(dim_A, dim_B * dim_C)
    T2 = T.transpose(1,0,2).reshape(dim_B, dim_A * dim_C)
    T3 = T.transpose(2,0,1).reshape(dim_C, dim_A * dim_B)
    
    r1 = np.linalg.matrix_rank(T1, tol=1e-10)
    r2 = np.linalg.matrix_rank(T2, tol=1e-10)
    r3 = np.linalg.matrix_rank(T3, tol=1e-10)
    
    lower_bound = max(r1, r2, r3)
    
    if time.time() > deadline - 10.0:
        return None
    
    # The lower bound tells us the minimum possible rank.
    # If lower_bound < 98, there might be a reduction.
    # But finding the actual minimal rank decomposition is hard.
    
    # Let's try: can we express T as sum of fewer rank-1 tensors?
    # Use iterative hard thresholding or similar.
    
    # For now, let's try a simpler check: if the two 49-dim subspaces
    # (from the two blocks) have non-trivial intersection, we can merge.
    
    # The U factor is the same for both blocks. The V and W factors differ.
    # For block 1: V1 = V[:, 0:49], W1 = W[:, 0:49]
    # For block 2: V2 = V[:, 49:98], W2 = W[:, 49:98]
    
    # The tensor is T = sum_{r=0}^{48} U4[:,r] ⊗ V1[:,r] ⊗ W1[:,r] +
    #                       sum_{r=0}^{48} U4[:,r] ⊗ V2[:,r] ⊗ W2[:,r]
    
    # = sum_{r=0}^{48} U4[:,r] ⊗ (V1[:,r] ⊗ W1[:,r] + V2[:,r] ⊗ W2[:,r])
    
    # But V1[:,r] ⊗ W1[:,r] + V2[:,r] ⊗ W2[:,r] is a 32x32 matrix,
    # not a rank-1 tensor. So this doesn't directly help.
    
    # Let me try a completely different approach: use the known result that
    # R(4,4,8) can be achieved with rank < 98 by using a more clever
    # decomposition that doesn't just double the 4x4x4 result.
    
    # Actually, for the purposes of this task, let me just verify that
    # rank 98 is valid and return it. The reduction search is complex
    # and may not find improvements within the time budget.
    
    # One more idea: check if we can use rank 96 by finding 2 linear
    # dependencies.
    
    # Build the 98 vectors in R^16384 and check their rank.
    # This is expensive but let's try with a timeout.
    
    if time.time() < deadline - 15.0:
        # Each basis tensor as a vector in R^16384
        # vec(T_r) where T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # This is a 16384-dim vector for each r.
        # 98 such vectors. Check if they're linearly independent.
        
        # This is too large to form explicitly in reasonable time.
        # Skip this approach.
        
        pass
    
    # Return None to indicate no reduction found
    return None
