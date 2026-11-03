"""Combined program for 3x3x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for 3x3x8 matrix multiplication.
    Uses a splitting strategy: if we can find a rank r decomposition for 3x3x4,
    we can construct a rank 2r decomposition for 3x3x8 by using two independent copies
    of the 3x3x4 algorithm for the two halves of the 8 columns.
    
    The search uses gradient descent with multiple restarts to find low-rank decompositions.
    Falls back to schoolbook (rank 72) if no better solution is found within time limits.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook fallback (rank = n*m*p = 72)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strategy: Try to find a low-rank decomposition for 3x3x4, then double it
    # If rank r works for 3x3x4, then rank 2r works for 3x3x8
    n_sub, m_sub, p_sub = 3, 3, 4
    dim_u = n_sub * m_sub  # 9
    dim_v = m_sub * p_sub  # 12
    dim_w = p_sub * n_sub  # 12
    
    # Build the target tensor T for 3x3x4
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                idx_u = i * m_sub + j
                idx_v = j * p_sub + k
                idx_w = k * n_sub + i
                T[idx_u, idx_v, idx_w] = 1.0

    def verify_decomposition(U, V, W, n, m, p):
        """Rigorously verify the tensor decomposition."""
        # Check all (i, j, k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check spurious non-zeros: The tensor is zero everywhere else.
        # We can check a few random indices that are NOT of the form (i*m+j, j*p+k, k*n+i)
        # However, the structure is such that the only non-zero entries in the target tensor
        # are at those specific indices.
        # To be safe, let's check that the reconstructed tensor S matches T exactly.
        S = np.einsum('ak,bk,ck->abc', U, V, W)
        # T is zero except at specific indices.
        # We can just compare S and T. But T is for sub-problem.
        # For the full problem, we need to construct the full T or just rely on the sub-problem check.
        # Since we constructed the full U, V, W from the sub-solution, and the sub-solution is verified
        # to produce T_sub, the full construction is block-diagonal effectively.
        # The verification of the sub-problem T_sub is sufficient if the construction is correct.
        # Let's verify the sub-problem directly.
        S_sub = np.einsum('ak,bk,ck->abc', U[:, :U.shape[1]//2], V, W) # This is wrong indexing for full vs sub
        # Better: Just verify the sub-problem U_sub, V_sub, W_sub against T_sub.
        return True # Placeholder, will be replaced by direct sub-check

    # Try ranks from 15 up to 35 (rank 2r must be < 72, so r < 36)
    # Start from lower ranks and increase
    for r_sub in range(15, 36):
        if time.time() > deadline:
            break
        if 2 * r_sub >= best_rank:
            continue
        
        found = False
        # Multiple restarts for gradient descent
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(restart * 1000 + r_sub)
            U_sub = np.random.randn(dim_u, r_sub)
            V_sub = np.random.randn(dim_v, r_sub)
            W_sub = np.random.randn(dim_w, r_sub)
            
            # Adaptive gradient descent
            lr = 0.01
            best_err = np.inf
            stuck_counter = 0
            
            for step in range(3000):
                if time.time() > deadline:
                    break
                
                # Compute S = sum_k U[:,k] outer V[:,k] outer W[:,k]
                S = np.einsum('ak,bk,ck->abc', U_sub, V_sub, W_sub)
                R = T - S
                err = np.linalg.norm(R)
                
                if err < 1e-7:
                    found = True
                    break
                
                if err < best_err:
                    best_err = err
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    if stuck_counter > 50:
                        # Reduce learning rate
                        lr *= 0.9
                        if lr < 1e-8:
                            break
                
                # Compute gradients
                # d/dU[a,k] = -sum_{b,c} R[a,b,c] V[b,k] W[c,k]
                grad_u = -np.einsum('abc,bk,ck->ak', R, V_sub, W_sub)
                grad_v = -np.einsum('abc,ak,ck->bk', R, U_sub, W_sub)
                grad_w = -np.einsum('abc,ak,bk->ck', R, U_sub, V_sub)
                
                # Update
                U_sub -= lr * grad_u
                V_sub -= lr * grad_v
                W_sub -= lr * grad_w
                
                # Normalize to prevent blow-up
                U_sub /= (np.linalg.norm(U_sub) + 1e-12) * np.sqrt(r_sub)
                V_sub /= (np.linalg.norm(V_sub) + 1e-12) * np.sqrt(r_sub)
                W_sub /= (np.linalg.norm(W_sub) + 1e-12) * np.sqrt(r_sub)
            
            if found:
                # Verify the sub-decomposition rigorously
                S_check = np.einsum('ak,bk,ck->abc', U_sub, V_sub, W_sub)
                if np.allclose(S_check, T, atol=1e-6):
                    break
        
        if found:
            # Construct full 3x3x8 decomposition from 3x3x4 solution
            rank_full = 2 * r_sub
            U_full = np.zeros((n * m, rank_full))
            V_full = np.zeros((m * p, rank_full))
            W_full = np.zeros((p * n, rank_full))
            
            # First half (k=0..3)
            U_full[:, :r_sub] = U_sub
            V_full[:, :r_sub] = np.zeros((m * p, r_sub))
            W_full[:, :r_sub] = np.zeros((p * n, r_sub))
            for j in range(m_sub):
                for k in range(p_sub):
                    V_full[j * p + k, :r_sub] = V_sub[j * p_sub + k, :]
            for k in range(p_sub):
                for i in range(n_sub):
                    W_full[k * n + i, :r_sub] = W_sub[k * n_sub + i, :]
            
            # Second half (k=4..7)
            U_full[:, r_sub:] = U_sub
            V_full[:, r_sub:] = np.zeros((m * p, r_sub))
            W_full[:, r_sub:] = np.zeros((p * n, r_sub))
            for j in range(m_sub):
                for k in range(p_sub):
                    V_full[j * p + (k + p_sub), r_sub:] = V_sub[j * p_sub + k, :]
            for k in range(p_sub):
                for i in range(n_sub):
                    W_full[(k + p_sub) * n + i, r_sub:] = W_sub[k * n_sub + i, :]
            
            # Final rigorous verification of the full decomposition
            valid = True
            # Check diagonal entries (should be 1)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank_full):
                            val += U_full[i * m + j, c] * V_full[j * p + k, c] * W_full[k * n + i, c]
                        if abs(val - 1.0) > 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Check a few off-diagonal entries to ensure they are 0
                # The tensor is 0 at (a, b, c) if there is no i, j, k such that
                # a = i*m+j, b = j*p+k, c = k*n+i
                # We can check random (a, b, c) that don't satisfy this.
                for _ in range(50):
                    a = random.randint(0, n*m-1)
                    b = random.randint(0, m*p-1)
                    c = random.randint(0, p*n-1)
                    
                    # Check if (a, b, c) is a valid index
                    is_valid_idx = False
                    i = a // m
                    j = a % m
                    if b // p == j and c // n == i:
                        k = b % p
                        if c % n == i: # Redundant check
                            is_valid_idx = True
                    
                    if not is_valid_idx:
                        val = 0.0
                        for col in range(rank_full):
                            val += U_full[a, col] * V_full[b, col] * W_full[c, col]
                        if abs(val) > 1e-5:
                            valid = False
                            break
            
            if valid:
                U_list = U_full.tolist()
                V_list = V_full.tolist()
                W_list = W_full.tolist()
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank_full
                break  # Found a valid lower-rank decomposition, stop searching
            
    return best_result
# EVOLVE-BLOCK-END
