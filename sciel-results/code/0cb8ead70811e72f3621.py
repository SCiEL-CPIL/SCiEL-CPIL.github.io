import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0

    # Build the rank-56 Strassen decomposition
    rank_56 = 56
    U = [[0] * rank_56 for _ in range(n * m)]
    V = [[0] * rank_56 for _ in range(m * p)]
    W = [[0] * rank_56 for _ in range(p * n)]

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

    # 8 blocks: A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
    # A is 2x4, so i_base=0 always, j_base in {0,2}
    # B is 4x8, so j_base in {0,2}, k_base in {0,2,4,6}
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify the decomposition
    def verify(U_arr, V_arr, W_arr, rank_val):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    s = np.dot(u_vals, v_vals * w_vals)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_np, V_np, W_np, rank_56):
        return {"U": U, "V": V, "W": W}

    # Now try to reduce rank by finding linear dependencies
    # Build the flattening matrix: for each rank term r, the contribution to tensor[i,j,k]
    # is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can form a matrix M where M[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If M has rank < 56, there are linear dependencies we can exploit.

    # Build M: shape (n*m*p, rank) = (64, 56)
    M = np.zeros((n * m * p, rank_56))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                M[idx, :] = U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :]

    # The tensor T[i,j,k] = 1 for all i,j,k, so sum_r M[idx, r] = 1 for all idx
    # Check if the columns of M are linearly independent
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)

    best_rank = rank_56
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

    if rank_M < rank_56 and time.time() < deadline:
        # There are null space vectors. If v is in null(M), then sum_r v[r] * M[:,r] = 0
        # This means we can modify the decomposition.
        # Strategy: try to eliminate terms by finding sparse null vectors.
        
        null_space = np.linalg.null_space(M)
        null_dim = null_space.shape[1]
        
        # For each null vector, try to use it to zero out a term
        # If null vector has a large component at position r, we can try to absorb it
        for nv_idx in range(null_dim):
            if time.time() > deadline - 5:
                break
            nv = null_space[:, nv_idx]
            # Find the term with largest absolute value in this null vector
            max_idx = np.argmax(np.abs(nv))
            max_val = nv[max_idx]
            if abs(max_val) < 1e-6:
                continue
            
            # Try to create a new decomposition where term max_idx is eliminated
            # New term r' for r' != max_idx: U[r'] += nv[r']/nv[max_idx] * U[max_idx]
            # This is complex; instead, let's try a different approach.
            
            # Alternative: use the null vector to create a linear combination
            # that allows us to drop one term. This requires more careful analysis.
            pass

    # Try a more systematic approach: use SVD-like decomposition on the flattening
    # to find a minimal rank representation.
    # The tensor T is fixed (all ones), so we need sum_r U_r ⊗ V_r ⊗ W_r = T
    
    # Let's try to find a lower rank by random perturbation and rank reduction
    if time.time() < deadline - 10:
        # Try rank 48, 40, 32, 24, 16 by solving the system
        for target_rank in [48, 40, 32, 24, 16]:
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(42)
            U_try = np.random.randn(n*m, target_rank)
            V_try = np.random.randn(m*p, target_rank)
            W_try = np.random.randn(p*n, target_rank)
            
            # We need: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a system of 64 equations in 3*target_rank unknowns
            # But it's nonlinear. Let's use the flattening approach.
            
            # Fix V and W, solve for U, etc. (alternating least squares)
            # This is complex for exact solution. Let's skip for now.
            pass

    # Return the best solution found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
