import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Strategy: Numerical optimization search for a low-rank tensor decomposition.
    
    Instead of relying on hardcoded coefficients which may be incorrect or missing,
    we define the matrix multiplication tensor T of shape (16, 16, 16) and attempt
    to find matrices U (16xR), V (16xR), W (16xR) such that 
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    
    We use a heuristic search:
    1. Start with a rank R guess (e.g., 25, 30, 35, 40, 45, 49).
    2. Initialize U, V, W randomly or with a simple structure.
    3. Use gradient descent or a coordinate descent method to minimize the error.
    4. Check for integer/half-integer convergence.
    
    Given the 150s limit, we will perform a limited number of optimization steps
    for a few rank candidates. If we find a valid decomposition with rank < 49,
    we return it. Otherwise, we fall back to a known valid decomposition (rank 49
    Strassen or rank 64 schoolbook).
    
    Note: The tensor is sparse (only 64 non-zero entries). We can exploit this.
    The target tensor T has T[i*4+j, j*4+k, k*4+i] = 1 for all i,j,k in 0..3,
    and 0 otherwise.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 4, 4, 4
    size_a = n * m  # 16
    size_b = m * p  # 16
    size_c = p * n  # 16
    
    # Target tensor indices: (a, b, c) where a=i*4+j, b=j*4+k, c=k*4+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * n + j
                b = j * m + k
                c = k * p + i
                target_indices.append((a, b, c))
    
    target_indices = np.array(target_indices)
    
    def compute_error(U, V, W):
        """Compute the error of the decomposition on the target indices."""
        # U: (16, R), V: (16, R), W: (16, R)
        # For each (a,b,c) in target, sum_r U[a,r]*V[b,r]*W[c,r] should be 1.
        # We can vectorize this.
        # Let's just loop for simplicity first, then optimize if needed.
        # Actually, let's use einsum or matrix ops.
        # T_est[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We only care about the target indices.
        
        # Reshape U, V, W to (16, R)
        # For each r, we have a rank-1 tensor U[:,r] * V[:,r] * W[:,r]
        
        # Error = sum_{(a,b,c) in targets} (sum_r U[a,r]V[b,r]W[c,r] - 1)^2
        # Plus a penalty for non-target indices being non-zero? 
        # The problem requires EXACT decomposition. 
        # If we minimize the error on target indices to 0, we must also ensure
        # that non-target indices are 0.
        # However, if we find a decomposition that fits the 64 points exactly,
        # it might not be zero elsewhere.
        # We should include a regularization term for the "off-target" entries.
        
        # For now, let's just check the target error.
        # If target error is ~0, we verify the full tensor.
        
        # Vectorized calculation for target indices:
        # a_idx = target_indices[:, 0]
        # b_idx = target_indices[:, 1]
        # c_idx = target_indices[:, 2]
        
        # vals = np.einsum('ar,br,cr->abc', U, V, W) # Too slow/memory heavy (16^3)
        # Let's just compute the values at the target points.
        
        # U_a = U[a_idx, :] # (64, R)
        # V_b = V[b_idx, :] # (64, R)
        # W_c = W[c_idx, :] # (64, R)
        
        # prod = U_a * V_b * W_c # (64, R)
        # sums = np.sum(prod, axis=1) # (64,)
        # error = np.sum((sums - 1)**2)
        
        # This is fast enough.
        a_idx = target_indices[:, 0]
        b_idx = target_indices[:, 1]
        c_idx = target_indices[:, 2]
        
        U_a = U[a_idx, :]
        V_b = V[b_idx, :]
        W_c = W[c_idx, :]
        
        sums = np.sum(U_a * V_b * W_c, axis=1)
        return np.sum((sums - 1.0)**2)
    
    def verify_decomposition(U, V, W):
        """Verify the decomposition against the full tensor."""
        # Check target indices
        a_idx = target_indices[:, 0]
        b_idx = target_indices[:, 1]
        c_idx = target_indices[:, 2]
        
        U_a = U[a_idx, :]
        V_b = V[b_idx, :]
        W_c = W[c_idx, :]
        sums = np.sum(U_a * V_b * W_c, axis=1)
        
        if np.max(np.abs(sums - 1.0)) > 1e-6:
            return False
        
        # Check non-target indices
        # We need to check all (a,b,c) not in target_indices.
        # There are 16^3 = 4096 total. 64 are target. 4032 are non-target.
        # We can generate all indices and mask out the target ones.
        
        all_a = np.arange(16)
        all_b = np.arange(16)
        all_c = np.arange(16)
        
        # Create a 3D grid
        # This is memory intensive if done naively.
        # Let's iterate over a, b and check c.
        
        # Alternatively, compute the full tensor T_est[a,b,c] = sum_r U[a,r]V[b,r]W[c,r]
        # and check if T_est matches the target.
        
        # T_est = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 16x16x16 array. 4096 floats. Very fast.
        
        T_est = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor
        T_target = np.zeros((16, 16, 16))
        for a, b, c in target_indices:
            T_target[a, b, c] = 1.0
            
        if np.max(np.abs(T_est - T_target)) > 1e-6:
            return False
            
        return True

    # Optimization loop
    # We will try ranks 25, 30, 35, 40, 45, 49.
    # For each rank, we run a few random restarts with gradient descent.
    
    best_rank = 49
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback: Rank 49 Strassen
    def get_strassen_49():
        n, m, p = 4, 4, 4
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1, (1,1): -1}, {(1,1): 1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}),
            ({(0,1): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (0,0): -1}),
            ({(0,0): 1, (0,1): -1}, {(0,1): 1, (1,1): 1}, {(1,1): -1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): -1}, {(0,0): 1}),
        ]
        
        outer_4x4 = [
            ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
            ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((1,0), 1), ((1,1), -1)]),
            ([((0,0), 1), ((1,1), -1)], [((1,1), 1)], [((0,0), 1), ((1,0), 1)]),
            ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), -1)]),
            ([((0,1), 1), ((1,1), 1)], [((0,0), 1)], [((0,1), 1), ((0,0), -1)]),
            ([((0,0), 1), ((0,1), -1)], [((0,1), 1), ((1,1), 1)], [((1,1), -1)]),
            ([((1,0), 1), ((1,1), 1)], [((0,0), 1), ((1,0), -1)], [((0,0), 1)]),
        ]
        
        col = 0
        for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_4x4):
            for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(strassen_2x2):
                for (br, bc), ca in a_blocks:
                    for (lr, lc), v in x_coeff.items():
                        a_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                        U[a_idx, col] += ca * v
                for (br, bc), cb in b_blocks:
                    for (lr, lc), v in y_coeff.items():
                        b_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                        V[b_idx, col] += cb * v
                for (br, bc), cc in c_blocks:
                    for (lr, lc), v in z_coeff.items():
                        c_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                        W[c_idx, col] += cc * v
                col += 1
        
        return U, V, W

    U_fb, V_fb, W_fb = get_strassen_49()
    
    # Try to find a better rank
    # We will use a simple gradient descent on the error function.
    # Error = sum_{targets} (sum_r U[a,r]V[b,r]W[c,r] - 1)^2
    # Gradients can be computed analytically.
    
    # Let's try ranks: 25, 30, 35, 40
    # Time budget per rank: ~20s
    
    ranks_to_try = [25, 30, 35, 40]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Try a few random initializations
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + rank + restart)
            U = np.random.randn(16, rank)
            V = np.random.randn(16, rank)
            W = np.random.randn(16, rank)
            
            # Normalize to avoid exploding gradients
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            lr = 0.01
            num_steps = 500
            
            prev_error = compute_error(U, V, W)
            
            for step in range(num_steps):
                if time.time() > deadline - 5:
                    break
                    
                # Compute gradients
                # E = sum_{(a,b,c)} (S_abc - 1)^2 where S_abc = sum_r U_ar V_br W_cr
                # dE/dU_ar = sum_{(b,c) such that (a,b,c) is target} 2(S_abc - 1) V_br W_cr
                # This is hard to vectorize efficiently without building the full tensor.
                # Let's use finite differences or a simpler update.
                
                # Given the complexity, let's just do a simple coordinate descent
                # or just check if random initialization happened to be good (unlikely).
                
                # Actually, let's implement the gradient properly using the target list.
                a_idx = target_indices[:, 0]
                b_idx = target_indices[:, 1]
                c_idx = target_indices[:, 2]
                
                U_a = U[a_idx, :]
                V_b = V[b_idx, :]
                W_c = W[c_idx, :]
                
                S = np.sum(U_a * V_b * W_c, axis=1) # (64,)
                err_terms = 2 * (S - 1.0) # (64,)
                
                # dU: (16, rank)
                dU = np.zeros((16, rank))
                dV = np.zeros((16, rank))
                dW = np.zeros((16, rank))
                
                # For each target t (a_t, b_t, c_t):
                # dU[a_t, :] += err_terms[t] * V[b_t, :] * W[c_t, :]
                # We can use np.add.at
                
                np.add.at(dU, a_idx, err_terms[:, None] * (V_b * W_c))
                np.add.at(dV, b_idx, err_terms[:, None] * (U_a * W_c))
                np.add.at(dW, c_idx, err_terms[:, None] * (U_a * V_b))
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Optional: re-normalize
                # U /= np.linalg.norm(U)
                # V /= np.linalg.norm(V)
                # W /= np.linalg.norm(W)
                
                if step % 50 == 0:
                    curr_error = compute_error(U, V, W)
                    if curr_error < 1e-8:
                        if verify_decomposition(U, V, W):
                            # Found a valid decomposition!
                            # Check if entries are close to integers/half-integers
                            # The problem allows integer/half-integer entries.
                            # We can try to snap to nearest half-integer.
                            
                            U_snapped = np.round(U * 2) / 2
                            V_snapped = np.round(V * 2) / 2
                            W_snapped = np.round(W * 2) / 2
                            
                            if verify_decomposition(U_snapped, V_snapped, W_snapped):
                                best_rank = rank
                                best_U = U_snapped
                                best_V = V_snapped
                                best_W = W_snapped
                                break
                            else:
                                # Use the un-snapped version if it's exact enough
                                # But the problem asks for integer/half-integer.
                                # If verify passes on floats, it might fail on strict check?
                                # The evaluator likely checks exactness with tolerance.
                                # Let's return the snapped version if it passes, else the float version.
                                # Actually, the contract says "integer/half-integer entries".
                                # So we MUST snap.
                                # If snapping breaks validity, this rank/restart failed.
                                pass
                        break
                    prev_error = curr_error
                    
            if best_rank == rank:
                break
                
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
