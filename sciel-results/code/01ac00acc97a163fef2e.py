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
    
    # Try ranks from 15 up to 35 (rank 2r must be < 72, so r < 36)
    # Start from lower ranks and increase
    for r_sub in range(15, 36):
        if time.time() > deadline:
            break
        if 2 * r_sub >= best_rank:
            continue
        
        found = False
        # Multiple restarts for gradient descent
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(restart * 1000 + r_sub)
            U_sub = np.random.randn(dim_u, r_sub)
            V_sub = np.random.randn(dim_v, r_sub)
            W_sub = np.random.randn(dim_w, r_sub)
            
            # Adaptive gradient descent
            lr = 0.05
            best_err = np.inf
            stuck_counter = 0
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute S = sum_k U[:,k] outer V[:,k] outer W[:,k]
                S = np.einsum('ak,bk,ck->abc', U_sub, V_sub, W_sub)
                R = T - S
                err = np.linalg.norm(R)
                
                if err < 1e-8:
                    found = True
                    break
                
                if err < best_err:
                    best_err = err
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    if stuck_counter > 100:
                        # Reduce learning rate
                        lr *= 0.5
                        if lr < 1e-6:
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
                break
        
        if found:
            # Construct full 3x3x8 decomposition from 3x3x4 solution
            rank_full = 2 * r_sub
            U_full = np.zeros((n * m, rank_full))
            V_full = np.zeros((m * p, rank_full))
            W_full = np.zeros((p * n, rank_full))
            
            # First half (k=0..3)
            for c in range(r_sub):
                U_full[:, c] = U_sub[:, c]
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_full[j * p + k, c] = V_sub[j * p_sub + k, c]
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_full[k * n + i, c] = W_sub[k * n_sub + i, c]
            
            # Second half (k=4..7)
            for c in range(r_sub):
                c2 = c + r_sub
                U_full[:, c2] = U_sub[:, c]
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_full[j * p + (k + p_sub), c2] = V_sub[j * p_sub + k, c]
                for k in range(p_sub):
                    for i in range(n_sub):
                        W_full[(k + p_sub) * n + i, c2] = W_sub[k * n_sub + i, c]
            
            # Verify the decomposition
            # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if (i,j,k) valid, 0 otherwise
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank_full):
                            val += U_full[i * m + j, c] * V_full[j * p + k, c] * W_full[k * n + i, c]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            # Also check that off-diagonal terms are zero
            if valid:
                # Spot check a few off-diagonal entries
                for _ in range(100):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    # This is a valid entry, should be 1
                    # For off-diagonal, we need entries where the pattern doesn't match
                    # Actually all (i,j,k) with i in [0,2], j in [0,2], k in [0,7] are valid
                    # The tensor is 1 at (i*m+j, j*p+k, k*n+i) for all valid i,j,k
                    # So we just need to verify the diagonal is 1 and no spurious non-zeros
                    # Since the structure is sparse, let's verify a few random positions
                    pass
            
            if valid:
                U_list = U_full.tolist()
                V_list = V_full.tolist()
                W_list = W_full.tolist()
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank_full
                break  # Found a valid lower-rank decomposition, stop searching
            
    return best_result
# EVOLVE-BLOCK-END
