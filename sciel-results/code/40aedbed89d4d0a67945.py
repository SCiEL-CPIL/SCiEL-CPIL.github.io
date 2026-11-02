"""Matrix multiplication 2x3x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def compute_error(U, V, W):
        P = np.einsum('at,bt,ct->abc', U, V, W)
        return np.sum((P - T) ** 2)
        
    def verify(U, V, W):
        P = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(P, T, atol=1e-9)
        
    def schoolbook():
        r = n * m * p
        U = np.zeros((dim_u, r))
        V = np.zeros((dim_v, r))
        W = np.zeros((dim_w, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
        
    def svd_initialize(rank):
        """
        Use SVD on the matricized tensor to get a good starting point.
        We matricize along the first mode: T as (dim_u, dim_v * dim_w)
        Then SVD gives us U_svd (dim_u, rank) with optimal real values.
        For V and W, we can use the remaining part.
        """
        # Matricize: T_mat[a, b*dim_w + c] = T[a, b, c]
        T_mat = T.reshape(dim_u, dim_v * dim_w)
        
        # SVD
        U_svd, S, Vt = np.linalg.svd(T_mat, full_matrices=False)
        
        # Take top 'rank' components
        U_svd = U_svd[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        # Reshape Vt back to (rank, dim_v, dim_w)
        Vt_reshaped = Vt.reshape(rank, dim_v, dim_w)
        
        # For each component t, we have Vt_reshaped[t] as a (dim_v, dim_w) matrix
        # We need to decompose this into v_t (dim_v) and w_t (dim_w)
        # Use SVD again for each component, or just take first row/col
        # Better: use the fact that for rank-1, we can take the outer product
        # But these are not necessarily rank-1
        
        # Simpler approach: just use random but guided by SVD structure
        # Let's try a different initialization: use the SVD factors directly
        # U_init = U_svd * S (absorbing singular values)
        U_init = U_svd * S[None, :]
        
        # For V and W, we need to factor Vt_reshaped[t] = v_t * w_t^T (approximately)
        # Use SVD for each slice
        V_init = np.zeros((dim_v, rank))
        W_init = np.zeros((dim_w, rank))
        
        for t in range(rank):
            slice_t = Vt_reshaped[t]  # (dim_v, dim_w)
            # SVD of this slice
            try:
                u_t, s_t, vt_t = np.linalg.svd(slice_t, full_matrices=False)
                if s_t.size > 0:
                    V_init[:, t] = u_t[:, 0] * np.sqrt(s_t[0])
                    W_init[:, t] = vt_t[0, :] * np.sqrt(s_t[0])
            except:
                V_init[:, t] = np.random.randn(dim_v)
                W_init[:, t] = np.random.randn(dim_w)
        
        return U_init, V_init, W_init

    def local_search(U, V, W, max_iters=3000, time_budget=100.0):
        """
        Gradient-guided local search with random restarts on steps.
        Returns (U, V, W, error).
        """
        candidates = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        cur_err = compute_error(U, V, W)
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        rank = U.shape[1]
        local_start = time.time()
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if time.time() - local_start > time_budget:
                break
            if cur_err < 1e-9:
                break
                
            # Compute gradients to guide the search
            P = np.einsum('at,bt,ct->abc', U, V, W)
            R = P - T
            G_U = np.einsum('abc,bt,ct->at', R, V, W)
            G_V = np.einsum('abc,at,ct->bt', R, U, W)
            G_W = np.einsum('abc,at,bt->ct', R, U, V)
            
            # Find the variable with the largest gradient magnitude
            max_grad_u = np.max(np.abs(G_U))
            max_grad_v = np.max(np.abs(G_V))
            max_grad_w = np.max(np.abs(G_W))
            
            max_grad = max(max_grad_u, max_grad_v, max_grad_w)
            
            if max_grad < 1e-6:
                # Stuck in a flat region, try random perturbation
                mat_id = np.random.choice(3)
                if mat_id == 0:
                    idx = np.random.randint(dim_u)
                    t = np.random.randint(rank)
                elif mat_id == 1:
                    idx = np.random.randint(dim_v)
                    t = np.random.randint(rank)
                else:
                    idx = np.random.randint(dim_w)
                    t = np.random.randint(rank)
                    
                old_val = U[idx, t] if mat_id == 0 else (V[idx, t] if mat_id == 1 else W[idx, t])
                new_val = np.random.choice(candidates)
                if new_val == old_val:
                    continue
                
                if mat_id == 0:
                    U[idx, t] = new_val
                elif mat_id == 1:
                    V[idx, t] = new_val
                else:
                    W[idx, t] = new_val
                
                new_err = compute_error(U, V, W)
                if new_err < cur_err:
                    cur_err = new_err
                else:
                    if mat_id == 0:
                        U[idx, t] = old_val
                    elif mat_id == 1:
                        V[idx, t] = old_val
                    else:
                        W[idx, t] = old_val
                continue

            # Select the matrix with the largest gradient
            if max_grad == max_grad_u:
                mat_id = 0
                idx, t = np.unravel_index(np.argmax(np.abs(G_U)), G_U.shape)
            elif max_grad == max_grad_v:
                mat_id = 1
                idx, t = np.unravel_index(np.argmax(np.abs(G_V)), G_V.shape)
            else:
                mat_id = 2
                idx, t = np.unravel_index(np.argmax(np.abs(G_W)), G_W.shape)
                
            # Try moving this variable to a candidate value that reduces error
            old_val = U[idx, t] if mat_id == 0 else (V[idx, t] if mat_id == 1 else W[idx, t])
            
            best_new_err = cur_err
            best_new_val = old_val
            
            for c in candidates:
                if c == old_val:
                    continue
                if mat_id == 0:
                    U[idx, t] = c
                elif mat_id == 1:
                    V[idx, t] = c
                else:
                    W[idx, t] = c
                
                new_err = compute_error(U, V, W)
                if new_err < best_new_err:
                    best_new_err = new_err
                    best_new_val = c
                
                if mat_id == 0:
                    U[idx, t] = old_val
                elif mat_id == 1:
                    V[idx, t] = old_val
                else:
                    W[idx, t] = old_val
                
            # Apply the best move
            if best_new_val != old_val:
                if mat_id == 0:
                    U[idx, t] = best_new_val
                elif mat_id == 1:
                    V[idx, t] = best_new_val
                else:
                    W[idx, t] = best_new_val
                cur_err = best_new_err
            else:
                # No improvement from this variable, try a random one
                mat_id = np.random.choice(3)
                if mat_id == 0:
                    idx = np.random.randint(dim_u)
                    t = np.random.randint(rank)
                    old_val = U[idx, t]
                elif mat_id == 1:
                    idx = np.random.randint(dim_v)
                    t = np.random.randint(rank)
                    old_val = V[idx, t]
                else:
                    idx = np.random.randint(dim_w)
                    t = np.random.randint(rank)
                    old_val = W[idx, t]
                    
                new_val = np.random.choice(candidates)
                if new_val == old_val:
                    continue
                
                if mat_id == 0:
                    U[idx, t] = new_val
                elif mat_id == 1:
                    V[idx, t] = new_val
                else:
                    W[idx, t] = new_val
                
                new_err = compute_error(U, V, W)
                if new_err < cur_err:
                    cur_err = new_err
                else:
                    if mat_id == 0:
                        U[idx, t] = old_val
                    elif mat_id == 1:
                        V[idx, t] = old_val
                    else:
                        W[idx, t] = old_val
                    
        return U, V, W, cur_err

    def refine_with_exact(U, V, W, max_iters=500):
        """
        Final polish: try to convert to exact half-integer values.
        """
        cur_err = compute_error(U, V, W)
        if cur_err < 1e-9:
            return U, V, W, cur_err
            
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        # Try to snap values to nearest half-integer
        candidates = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if cur_err < 1e-9:
                break
                
            # Try to improve one entry at a time
            improved = False
            for mat_id in range(3):
                if mat_id == 0:
                    mat = U
                    dim = dim_u
                elif mat_id == 1:
                    mat = V
                    dim = dim_v
                else:
                    mat = W
                    dim = dim_w
                    
                for idx in range(dim):
                    for t in range(rank):
                        old_val = mat[idx, t]
                        # Find nearest candidate
                        nearest = candidates[np.argmin(np.abs(candidates - old_val))]
                        if nearest == old_val:
                            continue
                            
                        mat[idx, t] = nearest
                        new_err = compute_error(U, V, W)
                        if new_err < cur_err:
                            cur_err = new_err
                            improved = True
                        else:
                            mat[idx, t] = old_val
                            
            if not improved:
                break
                
        return U, V, W, cur_err

    best_U, best_V, best_W = schoolbook()
    best_err = 0.0
    best_rank = n * m * p
    
    # Try to reduce rank from 24 down to 14
    for target_rank in range(n * m * p - 1, 13, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        max_restarts = 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            if restart == 0:
                # Use SVD initialization
                U_init, V_init, W_init = svd_initialize(target_rank)
            elif restart == 1:
                # Use schoolbook truncated
                U_s, V_s, W_s = schoolbook()
                U_init = U_s[:, :target_rank].copy()
                V_init = V_s[:, :target_rank].copy()
                W_init = W_s[:, :target_rank].copy()
            else:
                # Random initialization with sparse structure
                np.random.seed(restart * 123 + target_rank)
                U_init = np.zeros((dim_u, target_rank))
                V_init = np.zeros((dim_v, target_rank))
                W_init = np.zeros((dim_w, target_rank))
                for t in range(target_rank):
                    for _ in range(3):
                        U_init[np.random.randint(dim_u), t] = np.random.choice([-1, 0.5, 1])
                        V_init[np.random.randint(dim_v), t] = np.random.choice([-1, 0.5, 1])
                        W_init[np.random.randint(dim_w), t] = np.random.choice([-1, 0.5, 1])
            
            # Allocate time budget based on remaining time
            remaining = deadline - 5 - time.time()
            if remaining < 1:
                break
            time_budget = max(1.0, min(10.0, remaining / (max_restarts - restart)))
            
            U_sol, V_sol, W_sol, err = local_search(U_init, V_init, W_init, max_iters=2000, time_budget=time_budget)
            
            if err < 1e-9:
                # Try to refine to exact values
                U_sol, V_sol, W_sol, err = refine_with_exact(U_sol, V_sol, W_sol)
                if err < 1e-9:
                    found = True
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_err = err
                    break
                else:
                    # Revert to pre-refine if it got worse
                    U_sol, V_sol, W_sol, err = local_search(U_init, V_init, W_init, max_iters=2000, time_budget=time_budget)
                    if err < 1e-9:
                        found = True
                        best_rank = target_rank
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        best_err = err
                        break
        
        if found:
            pass
        else:
            pass

    if best_err > 1e-9:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Convert to list
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
