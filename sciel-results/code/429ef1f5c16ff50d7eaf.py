import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Target: Rank 42 via Kronecker product structure (Strassen 2x2x2)
    # The multiplication tensor for 2x4 @ 4x6 can be viewed as:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # 
    # By decomposing j into j1 in {0,1} (block) and j2 in {0,1} (offset within block)?
    # Actually, the standard block decomposition is:
    # A is 2x4, B is 4x6.
    # Split m=4 into two blocks of size 2: j = 2*j_block + j_local
    # Split p=6 into three blocks of size 2: k = 2*k_block + k_local
    # 
    # The tensor is a sum over j_block, k_block of:
    #   T_{i, 2*jb+jl, 2*kb+kl} = sum_{r_local} U^{Strassen}_{i, jl, r_local} * V^{Strassen}_{jl, kl, r_local} * W^{Strassen}_{kl, i, r_local}
    #
    # This is exactly the Kronecker product of the identity on the block indices
    # and the Strassen tensor on the local indices.
    # Rank = (number of j_blocks) * (number of k_blocks) * (rank of Strassen)
    # Rank = 2 * 3 * 7 = 42.
    #
    # Is there a way to do better?
    # The rank of the 2x2x2 tensor is 7.
    # The "block" structure is essentially a 2x3 grid of 2x2x2 tensors.
    # The total rank is additive across the blocks because the supports in the j and k indices are disjoint for different blocks.
    # Wait, the j indices in U and V overlap?
    # U_{i, j, r}. For j_block=0, j is {0,1}. For j_block=1, j is {2,3}. Disjoint.
    # V_{j, k, r}. For j_block=0, j is {0,1}. For j_block=1, j is {2,3}. Disjoint.
    # So the terms for different j_blocks do not interfere in the sum over j.
    # Similarly for k_blocks, W_{k, i, r} has disjoint k supports.
    # Therefore, the rank is indeed the sum of the ranks of the blocks.
    # Rank = 6 * 7 = 42.
    
    # Can we go lower?
    # Lower bound for 2x4x6:
    # Rank(2x2x2) = 7.
    # Rank(1x2x2) = 4 (since it's just 2x2 matrix mult, rank 4? No, 1x2x2 tensor T_{1,j,k} = A_{1j}B_{jk}. This is rank 4? No, it's a bilinear map R^2 x R^2 -> R^2. Rank is 4. Wait, 1x2x2 is just 2x2 matrix multiplication? No.
    # A is 1x2, B is 2x2. C is 1x2.
    # C_0 = A_0 B_00 + A_1 B_10
    # C_1 = A_0 B_01 + A_1 B_11
    # This is 4 multiplications. Can it be done in 3?
    # It's equivalent to multiplying a 2x2 matrix by a 1x2 vector? Or 1x2 by 2x2.
    # Rank of 1xn x nxm tensor is nm? No.
    # The rank of the 1x2x2 tensor is 4.
    # So Rank(2x4x6) >= Rank(1x2x2) * something?
    # Actually, we can embed a 1x2x2 tensor into 2x4x6.
    # If we set A to be 2x4 with only row 0 non-zero in cols 0,1?
    # Then we are effectively doing 1x2 @ 2x2.
    # So Rank(2x4x6) >= Rank(1x2x2) = 4.
    # Also Rank(2x4x6) >= Rank(2x2x1) = 4.
    # Also Rank(2x4x6) >= Rank(2x2x2) = 7.
    
    # There is a known result: Rank(2x2xn) = ?
    # For n=1, rank 4.
    # For n=2, rank 7.
    # For n=3, rank ?
    # 
    # Let's stick to the constructive rank 42. It is a solid improvement over 48 (schoolbook).
    # The inspirations confirm rank 42 is achievable.
    
    rank = 42
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    # Strassen 2x2 @ 2x2 decomposition
    # Local indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_U = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: 1},       # P2
        {0: 1},             # P3
        {3: 1},             # P4
        {0: 1, 1: 1},       # P5
        {0: -1, 2: 1},      # P6
        {1: 1, 3: -1},      # P7
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1
        {0: 1},             # P2
        {1: 1, 3: -1},      # P3
        {2: 1, 0: -1},      # P4
        {3: 1},             # P5
        {0: 1, 1: 1},       # P6
        {2: 1, 3: 1},       # P7
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: -1},      # P2
        {1: 1, 3: 1},       # P3
        {0: 1, 2: 1},       # P4
        {0: -1, 1: 1},      # P5
        {3: 1},             # P6
        {0: 1},             # P7
    ]
    
    col = 0
    for j_block in range(2):  # j = 0,1 and j = 2,3
        for k_block in range(3):  # k = 0,1 and k = 2,3 and k = 4,5
            for prod_idx in range(7):
                r = col + prod_idx
                
                # Fill U
                # U index: i * m + j
                # i in {0,1}, j in {j_block*2, j_block*2+1}
                for local_idx, sign in strassen_U[prod_idx].items():
                    gi = local_idx // 2
                    gj = local_idx % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign
                
                # Fill V
                # V index: j * p + k
                # j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
                for local_idx, sign in strassen_V[prod_idx].items():
                    gj = local_idx // 2
                    gk = local_idx % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                
                # Fill W
                # W index: k * n + i
                # k in {k_block*2, k_block*2+1}, i in {0,1}
                for local_idx, sign in strassen_W[prod_idx].items():
                    gi = local_idx // 2
                    gk = local_idx % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign
            
            col += 7
            
    # Verification
    def _verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        result = np.zeros((n, p))
        for r in range(rank):
            M = U_3d[:, :, r] @ V_3d[:, :, r]
            W_r = W_3d[:, :, r]
            result += M * W_r.T
        
        target = np.ones((n, p))
        return np.allclose(result, target, atol=1e-9)

    if not _verify(U, V, W, n, m, p):
        # Fallback to naive if verification fails (should not happen)
        rank_naive = n * m * p
        U_naive = np.zeros((n*m, rank_naive))
        V_naive = np.zeros((m*p, rank_naive))
        W_naive = np.zeros((p*n, rank_naive))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j, idx] = 1
                    V_naive[j * p + k, idx] = 1
                    W_naive[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
        
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
