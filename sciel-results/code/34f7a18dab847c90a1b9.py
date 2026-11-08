import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank 56 solution using block Strassen
    # Split P=8 into 4 blocks of size 2
    # Split M=4 into 2 blocks of size 2
    # Each (g, b) pair is a 2x2x2 multiplication using Strassen (rank 7)
    # Total rank = 4 * 2 * 7 = 56

    X_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    Y_forms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]

    num_groups = 4  # p blocks
    num_ablocks = 2  # m blocks
    num_strassen = 7

    rank = num_groups * num_ablocks * num_strassen  # 56

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # U: A elements
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][idx] = X_forms[s][i * 2 + jp]

                # V: B elements
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][idx] = Y_forms[s][jp * 2 + kp]

                # W: C elements
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][idx] = Z_combos[i * 2 + kp][s]

                idx += 1

    # Try to improve via local search / rank reduction
    # Convert to numpy arrays for efficient computation
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Verify correctness first
    def verify(U_m, V_m, W_m, n, m, p, tol=1e-9):
        """Check if the tensor decomposition is exact."""
        # The multiplication tensor T[i,j,k] = 1 if j=k else 0
        # Decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
        # Actually for matrix mult: C = A * B
        # T[i, j, k] where i indexes A row, j indexes A col/B row, k indexes B col
        # T[i,j,k] = delta(j,k)
        # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        
        # Build the target tensor
        T = np.zeros((n, m, p))
        for j in range(m):
            for k in range(p):
                if j == k:
                    for i in range(n):
                        T[i, j, k] = 1.0
        
        # Compute decomposed tensor
        # U has shape (n*m, rank) -> reshape to (n, m, rank)
        # V has shape (m*p, rank) -> reshape to (m, p, rank)  
        # W has shape (p*n, rank) -> reshape to (p, n, rank)
        
        rank_val = U_m.shape[1]
        U_r = U_m.reshape(n, m, rank_val)
        V_r = V_m.reshape(m, p, rank_val)
        W_r = W_m.reshape(p, n, rank_val)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, let me re-check the indexing convention.
        # The problem says W[k*n+i, r] so W is indexed by (k,i) where k is output col, i is output row
        # U is indexed by (i, j) where i is A row, j is A col
        # V is indexed by (j, k) where j is B row, k is B col
        
        # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        T_decomp = np.zeros((n, m, p))
        for r in range(rank_val):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
        
        return np.max(np.abs(T - T_decomp)) < tol
    
    # Quick verification
    if not verify(U_arr, V_arr, W_arr, n, m, p):
        # Something is wrong, return as is
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank solution
    # The theoretical lower bound for 2x4 * 4x8 is at least max(2*4, 4*8, 2*8) = 32
    # But Strassen gives 56. Can we do better?
    
    # Let's try a different blocking strategy or use known results
    # For 2x4 * 4x8, we can think of it as 4 independent 2x1 * 1x8 multiplications
    # No, that's not right. Let me think more carefully.
    
    # Actually, 2x4 * 4x8 can be seen as computing 16 outputs, each being a dot product of length 4.
    # The tensor rank is at least the max of the three dimensions' products... 
    # Actually, the lower bound is max(n*m, m*p, n*p) = max(8, 32, 16) = 32
    
    # Let me try to construct a rank 32 solution if possible, or at least try to reduce from 56.
    
    # One approach: use the fact that we can combine some Strassen terms.
    # Alternatively, try a direct optimization approach.
    
    # For now, let's try a simple local search to see if we can reduce rank.
    # We'll try to zero out some columns of U, V, W and see if the decomposition still works.
    
    # Try rank 48, 40, 32 by removing columns
    for target_rank in [48, 40, 32]:
        if time.time() > deadline - 10:
            break
            
        U_sub = U_arr[:, :target_rank]
        V_sub = V_arr[:, :target_rank]
        W_sub = W_arr[:, :target_rank]
        
        if verify(U_sub, V_sub, W_sub, n, m, p):
            # Found a valid lower rank!
            # Convert back to lists
            U_list = U_sub.tolist()
            V_list = V_sub.tolist()
            W_list = W_sub.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try a more sophisticated approach: use random initialization and gradient descent
    # to find a lower rank decomposition
    
    # For now, return the rank 56 solution
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    # Try to use a smarter decomposition
    # The key insight: 2x4 * 4x8 can be viewed as 2 independent 1x4 * 4x8 multiplications
    # Each 1x4 * 4x8 has rank at least 8, so total rank at least 16? No, that's not right either.
    
    # Let me try a different blocking: split into 2x2 * 2x4 blocks
    # 2x2 * 2x4 using Strassen-like: rank 7 for each
    # We have 2 (for the 4x8 part) * 2 (for the 2x4 part) = 4 such blocks? 
    # No, this is getting complicated. Let me just return the current solution.
    
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "The current program already achieves rank 56 via block Strassen. The additional verification and simple rank-reduction attempts (checking if subsets of columns still work) are unlikely to find a valid lower rank because the Strassen decomposition is tight for this blocking structure. The program will likely return the same rank 56 solution.", q: 0.05, delta: [0, 0]}
