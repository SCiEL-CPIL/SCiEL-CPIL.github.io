"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Schoolbook decomposition (Rank 24) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1
                V_school[j * p + k, idx] = 1
                W_school[k * n + i, idx] = 1
                idx += 1
    
    def verify(U, V, W):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Vectorized verification"""
        rank = U.shape[1]
        # Build the full tensor contraction
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    # Fixed rank-22 decomposition using Strassen on 2x2 blocks
    # A is 2x3, B is 3x4
    # Split: A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    #        B = [B1; B2] where B1 is 2x4, B2 is 1x4
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x4 = two 2x2 multiplications (rank 7 each = 14)
    # A2*B2: 2x1 * 1x4 (rank 8)
    # Total: 22
    
    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))
    
    def add_u(i, j, c, val):
        U22[i * m + j, c] += val
    def add_v(j, k, c, val):
        V22[j * p + k, c] += val
    def add_w(k, i, c, val):
        W22[k * n + i, c] += val
    
    # Strassen for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)(b11)
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b10+b11)
    # C00 = m1 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    def add_strassen_block(ai, aj, bi, kj, offset):
        """
        Add Strassen decomposition for A[ai:ai+2, aj:aj+2] * B[bi:bi+2, kj:kj+2] -> C[ai:ai+2, kj:kj+2]
        """
        a00 = (ai, aj)
        a01 = (ai, aj+1)
        a10 = (ai+1, aj)
        a11 = (ai+1, aj+1)
        
        b00 = (bi, kj)
        b01 = (bi, kj+1)
        b10 = (bi+1, kj)
        b11 = (bi+1, kj+1)
        
        # W index: k*n + i, so C[ai, kj] -> W[kj*n + ai]
        c00 = (kj, ai)      # C[ai, kj]
        c01 = (kj+1, ai)    # C[ai, kj+1]
        c10 = (kj, ai+1)    # C[ai+1, kj]
        c11 = (kj+1, ai+1)  # C[ai+1, kj+1]
        
        # m1 = (a00+a11)(b00+b11) -> contributes to C00 and C11
        add_u(*a00, offset + 0, 1)
        add_u(*a11, offset + 0, 1)
        add_v(*b00, offset + 0, 1)
        add_v(*b11, offset + 0, 1)
        add_w(*c00, offset + 0, 1)
        add_w(*c11, offset + 0, 1)
        
        # m2 = (a10+a11)(b00) -> contributes to C10 and C11
        add_u(*a10, offset + 1, 1)
        add_u(*a11, offset + 1, 1)
        add_v(*b00, offset + 1, 1)
        add_w(*c10, offset + 1, 1)
        add_w(*c11, offset + 1, -1)
        
        # m3 = a00(b01-b11) -> contributes to C01 and C11
        add_u(*a00, offset + 2, 1)
        add_v(*b01, offset + 2, 1)
        add_v(*b11, offset + 2, -1)
        add_w(*c01, offset + 2, 1)
        add_w(*c11, offset + 2, 1)
        
        # m4 = a11(b10-b00) -> contributes to C10 only
        add_u(*a11, offset + 3, 1)
        add_v(*b10, offset + 3, 1)
        add_v(*b00, offset + 3, -1)
        add_w(*c10, offset + 3, 1)
        
        # m5 = (a00+a01)(b11) -> contributes to C00 and C01
        add_u(*a00, offset + 4, 1)
        add_u(*a01, offset + 4, 1)
        add_v(*b11, offset + 4, 1)
        add_w(*c00, offset + 4, -1)
        add_w(*c01, offset + 4, 1)
        
        # m6 = (a10-a00)(b00+b10) -> contributes to C11 only
        add_u(*a10, offset + 5, 1)
        add_u(*a00, offset + 5, -1)
        add_v(*b00, offset + 5, 1)
        add_v(*b10, offset + 5, 1)
        add_w(*c11, offset + 5, 1)
        
        # m7 = (a01-a11)(b10+b11) -> contributes to C00 only
        add_u(*a01, offset + 6, 1)
        add_u(*a11, offset + 6, -1)
        add_v(*b10, offset + 6, 1)
        add_v(*b11, offset + 6, 1)
        add_w(*c00, offset + 6, 1)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 7)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # This is 2x1 * 1x4, rank 8
    offset = 14
    for i in range(2):
        for k in range(4):
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
    
    # Verify the rank-22 decomposition
    if verify_fast(U22, V22, W22):
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
    
    # Check time before attempting search
    if time.time() - t0 > 130:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find a rank-21 decomposition using local search
    # Strategy: start from rank-22 and try to merge/eliminate one rank term
    # by finding a linear dependency in the tensor space
    
    def try_rank_reduction(U_init, V_init, W_init, target_rank, max_iters=5000):
        """Try to reduce rank by finding linear dependencies"""
        n_r = U_init.shape[1]
        if n_r <= target_rank:
            return U_init, V_init, W_init, True
        
        # Try to find a column that can be expressed as combination of others
        # For each column r, check if U[:,r], V[:,r], W[:,r] are linearly dependent
        # with the remaining columns in a way that preserves the tensor
        
        # Alternative: use the fact that if we have a rank-r decomposition,
        # we can try random linear combinations to see if rank drops
        
        best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
        
        for iteration in range(max_iters):
            if time.time() - t0 > 140:
                break
            
            # Pick two random columns and try to eliminate one
            r1 = random.randint(0, n_r - 1)
            r2 = random.randint(0, n_r - 1)
            if r1 == r2:
                continue
            
            # Try U[:,r1] = alpha * U[:,r2] + beta * U[:,other]
            # This is complex; instead try a simpler approach
            
            # Random perturbation approach: try to find if removing column r
            # and adjusting others still works
            for r_remove in range(n_r):
                # Remove column r_remove
                mask = [x for x in range(n_r) if x != r_remove]
                U_sub = U_init[:, mask]
                V_sub = V_init[:, mask]
                W_sub = W_init[:, mask]
                
                # Check if the remaining columns still span the tensor
                # by solving a least-squares problem
                # This is expensive, so only do it occasionally
                if iteration % 10 == 0:
                    # Quick check: is the removed column a linear combination?
                    # For U: check if U[:,r_remove] is in span of U[:,mask]
                    pass
        
        return best_U, best_V, best_W, False
    
    # Try a more systematic approach: search for rank-21 by modifying the rank-22
    # Use the known structure and try different block decompositions
    
    # Alternative: try a 2x2 Strassen on different column splits
    # A is 2x3, we can split columns as [1, 2] or [2, 1]
    
    # Try split: A[:, 0:1] * B[0:1, :] + A[:, 1:3] * B[1:3, :]
    # = 2x1 * 1x4 (rank 8) + 2x2 * 2x4 (rank 14) = 22
    
    # Try split: A[:, 0:2] * B[0:2, :] + A[:, 2:3] * B[2:3, :]
    # = 2x2 * 2x4 (rank 14) + 2x1 * 1x4 (rank 8) = 22 (same as before)
    
    # Try: A[:, 1:3] * B[1:3, :] + A[:, 0:1] * B[0:1, :]
    # = 2x2 * 2x4 (rank 14) + 2x1 * 1x4 (rank 8) = 22
    
    # All splits give 22 with this approach. Need a different strategy.
    
    # Try using a rank-7 decomposition for 2x2*2x4 that shares structure
    # 2x2 * 2x4 can be seen as two 2x2*2x2 blocks, but they share the A matrix
    # If we can share some rank terms between the two blocks, we might save rank
    
    # Actually, let's try a different approach: use the fact that the tensor
    # for 2x3*3x4 has a known minimal rank. Let's try rank 21 via search.
    
    # Random search for rank-21 decomposition
    def random_search_rank(target_rank, n_trials=10000):
        """Random search for a lower-rank decomposition"""
        for trial in range(n_trials):
            if time.time() - t0 > 140:
                break
            
            # Start with random initialization
            U_rand = np.random.randn(n * m, target_rank) * 0.5
            V_rand = np.random.randn(m * p, target_rank) * 0.5
            W_rand = np.random.randn(p * n, target_rank) * 0.5
            
            # Optimize using gradient descent on the loss
            # Loss = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - delta_{i,j,k})^2
            lr = 0.01
            for step in range(200):
                if time.time() - t0 > 140:
                    break
                
                # Compute loss and gradients
                loss = 0.0
                grad_U = np.zeros_like(U_rand)
                grad_V = np.zeros_like(V_rand)
                grad_W = np.zeros_like(W_rand)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_rand[i*m+j, :]
                            v_val = V_rand[j*p+k, :]
                            w_val = W_rand[k*n+i, :]
                            pred = np.dot(u_val, v_val * w_val)
                            err = pred - 1.0
                            loss += err * err
                            
                            # d(loss)/d(U[i,j,r]) = 2*err * V[j,k,r] * W[k,i,r]
                            grad_U[i*m+j, :] += 2 * err * v_val * w_val
                            grad_V[j*p+k, :] += 2 * err * u_val * w_val
                            grad_W[k*n+i, :] += 2 * err * u_val * v_val
                
                # Update
                U_rand -= lr * grad_U
                V_rand -= lr * grad_V
                W_rand -= lr * grad_W
                
                if loss < 1e-10:
                    return U_rand, V_rand, W_rand, True
        
        return None, None, None, False
    
    # Try random search for rank 21
    if time.time() - t0 < 135:
        U21, V21, W21, success = random_search_rank(21, n_trials=500)
        if success and verify_fast(U21, V21, W21):
            return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
    
    # Return schoolbook as final fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
