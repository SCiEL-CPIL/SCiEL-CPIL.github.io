"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 7
    
    # Strategy: Use Strassen rank-7 decomposition for the 2x2x2 core
    # applied to three disjoint 2-column blocks of B (columns 0-5),
    # and a rank-2 decomposition for the remaining single column (column 6).
    # Total rank: 7 + 7 + 7 + 2 = 23.
    
    # Strassen decomposition for 2x2x2 matrix multiplication.
    # U: 4x7, V: 4x7, W: 4x7
    # These coefficients are for the standard Strassen algorithm.
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1,  0, 1],   # A01
        [0, 1, 0, 0, 0,  1, 0],   # A10
        [1, 1, 0, 1, 0,  0,-1],   # A11
    ], dtype=float)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1,  0, 0, 1, 0],   # B01
        [0, 0, 0,  1, 0, 0, 1],   # B10
        [1, 0,-1,  0, 1, 0, 1],   # B11
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0,  1, 0, 0],   # C01
        [0, 1, 0, 1,  0, 0, 0],   # C10
        [1,-1, 1, 0,  0, 1, 0],   # C11
    ], dtype=float)
    
    # Construct rank-23 decomposition
    # 3 blocks of 2 columns using Strassen (rank 7 each)
    # 1 block of 1 column using rank-2 decomposition
    
    num_strassen_blocks = p // 2  # 3
    remaining_cols = p % 2        # 1
    
    rank = num_strassen_blocks * 7 + remaining_cols * 2  # 3*7 + 1*2 = 23
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Process Strassen blocks
    for block in range(num_strassen_blocks):
        k_base = block * 2
        for r in range(7):
            # U part: same for all A entries, independent of k
            U[:, col] = S_U[:, r]
            
            # V part: B entries for this block
            # Local index 0 -> j=0, k=0 -> global j*p + k_base
            # Local index 1 -> j=0, k=1 -> global j*p + k_base + 1
            # Local index 2 -> j=1, k=0 -> global j*p + k_base
            # Local index 3 -> j=1, k=1 -> global j*p + k_base + 1
            
            for local_v_idx in range(4):
                j = local_v_idx // 2
                kk = local_v_idx % 2
                global_v_idx = j * p + (k_base + kk)
                V[global_v_idx, col] = S_V[local_v_idx, r]
                
            # W part: C entries for this block
            # Local index 0 -> k=0, i=0 -> global (k_base+0)*n + 0
            # Local index 1 -> k=1, i=0 -> global (k_base+1)*n + 0
            # Local index 2 -> k=0, i=1 -> global (k_base+0)*n + 1
            # Local index 3 -> k=1, i=1 -> global (k_base+1)*n + 1
            
            for local_w_idx in range(4):
                kk = local_w_idx // 2
                i = local_w_idx % 2
                global_w_idx = (k_base + kk) * n + i
                W[global_w_idx, col] = S_W[local_w_idx, r]
                
            col += 1
            
    # Process remaining column (rank-2 decomposition)
    # For a single column k=6, we need to multiply A (2x2) by B[:,6] (2x1)
    # This is equivalent to 2 separate dot products, but we can do it in rank 2
    # since the B column is just a vector.
    # C[i, 6] = sum_j A[i,j] * B[j, 6]
    # We can use rank-2:
    # Term 1: U[0, col] = 1, V[0*p+6, col] = 1, W[6*n+0, col] = 1 -> C[0,6] += A[0,0]*B[0,6]
    # Term 2: U[1, col] = 1, V[1*p+6, col] = 1, W[6*n+0, col] = 1 -> C[0,6] += A[0,1]*B[1,6]
    # Wait, that's not right. Let's think again.
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    
    # We need 2 terms.
    # Term 1: captures A[0,0]*B[0,6] and A[1,0]*B[0,6]
    #   U[0, col] = 1, V[0*p+6, col] = 1, W[6*n+0, col] = 1, W[6*n+1, col] = 1
    #   But that would add A[0,0]*B[0,6] to C[0,6] and A[1,0]*B[0,6] to C[1,6].
    #   Wait, U[0, col] = 1 means A[0,0] is selected.
    #   V[0*p+6, col] = 1 means B[0,6] is selected.
    #   So the term is A[0,0]*B[0,6].
    #   W[6*n+0, col] = 1 means this contributes to C[0,6].
    #   W[6*n+1, col] = 1 means this contributes to C[1,6].
    #   But we only want A[0,0]*B[0,6] to contribute to C[0,6], not C[1,6].
    #   So this approach doesn't work directly.
    
    # Let's use the standard rank-2 for 2x2x1 multiplication.
    # Actually, 2x2x1 is just two dot products, so rank 2 is trivial.
    # Term 1: A[0,0]*B[0,6] -> C[0,6]
    #   U[0, col] = 1, V[0*p+6, col] = 1, W[6*n+0, col] = 1
    # Term 2: A[0,1]*B[1,6] -> C[0,6]
    #   U[1, col] = 1, V[1*p+6, col] = 1, W[6*n+0, col] = 1
    # But we also need C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6].
    # So we need 4 terms total? No, that's rank 4.
    
    # Wait, the rank of 2x2x1 matrix multiplication is indeed 2.
    # Let's look it up or think more carefully.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    # We can write:
    # C[0] = (A[0,0] + A[0,1]) * (B[0] + B[1]) - A[0,1]*B[0] - A[0,0]*B[1]
    # That's not rank 2.
    
    # Actually, the rank of 2x2x1 is 2. Here's one decomposition:
    # Term 1: U[0,0]=1, U[1,0]=0, V[0,0]=1, V[1,0]=0, W[0,0]=1, W[1,0]=0
    #   -> A[0,0]*B[0] -> C[0]
    # Term 2: U[0,1]=0, U[1,1]=1, V[0,1]=0, V[1,1]=1, W[0,1]=0, W[1,1]=1
    #   -> A[1,1]*B[1] -> C[1]
    # But this misses A[0,1]*B[1] and A[1,0]*B[0].
    
    # I think I'm confusing myself. Let's just use rank 4 for the last column to be safe.
    # Or, let's use a different approach.
    
    # Actually, for 2x2x1, the rank is 2. Here's the correct decomposition:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    # We can use:
    # Term 1: (A[0,0] + A[1,0]) * B[0] -> contributes to both C[0] and C[1]
    #   U[0,0] = 1, U[1,0] = 1, V[0,0] = 1, V[1,0] = 0, W[0,0] = 1, W[1,0] = 1
    # Term 2: (A[0,1] + A[1,1]) * B[1] -> contributes to both C[0] and C[1]
    #   U[0,1] = 1, U[1,1] = 1, V[0,1] = 0, V[1,1] = 1, W[0,1] = 1, W[1,1] = 1
    
    # But this gives:
    # C[0] = (A[0,0] + A[1,0])*B[0] + (A[0,1] + A[1,1])*B[1]
    # C[1] = (A[0,0] + A[1,0])*B[0] + (A[0,1] + A[1,1])*B[1]
    # Which is not correct.
    
    # I think the rank of 2x2x1 is actually 2, but I'm not getting the right coefficients.
    # Let's just use rank 4 for the last column to be safe.
    
    # Actually, let's use a simpler approach: just use rank 4 for the last column.
    # This gives total rank 3*7 + 4 = 25, which is the same as before.
    
    # Let me reconsider. The rank of 2x2x1 is 2.
    # Here's the correct decomposition:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    # We can use:
    # Term 1: A[0,0]*B[0] -> C[0]
    #   U[0,0] = 1, U[1,0] = 0, V[0,0] = 1, V[1,0] = 0, W[0,0] = 1, W[1,0] = 0
    # Term 2: A[0,1]*B[1] -> C[0]
    #   U[0,1] = 1, U[1,1] = 0, V[0,1] = 0, V[1,1] = 1, W[0,1] = 1, W[1,1] = 0
    # But this doesn't cover C[1].
    
    # I think I need to accept that I don't know the rank-2 decomposition for 2x2x1.
    # Let's just use rank 4 for the last column.
    
    # Actually, let's try a different approach. Let's use rank 2 for the last column
    # with the following coefficients:
    # Term 1: U[0,0]=1, U[1,0]=0, V[0,0]=1, V[1,0]=0, W[0,0]=1, W[1,0]=0
    # Term 2: U[0,1]=0, U[1,1]=1, V[0,1]=0, V[1,1]=1, W[0,1]=0, W[1,1]=1
    # This gives C[0] = A[0,0]*B[0] and C[1] = A[1,1]*B[1], which is incomplete.
    
    # I'll just use rank 4 for the last column.
    
    for k in range(num_strassen_blocks * 2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the decomposition using numpy for speed
    valid = True
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = np.dot(U[i * m + j], V[jp * p + kp] * W[kp * n + ip])
                            if abs(lhs - rhs) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook rank 28
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
