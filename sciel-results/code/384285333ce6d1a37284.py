import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _compute_error_and_C(E, V, W):
    """
    Compute C[u, r] = sum_{v,w} E[u,v,w] V[v,r] W[w,r]
    and norms of V_r, W_r
    """
    n_u, n_v, n_w = E.shape
    R = V.shape[1]
    C = np.zeros((n_u, R))
    v_norms = np.zeros(R)
    w_norms = np.zeros(R)
    
    for r in range(R):
        v_norms[r] = np.dot(V[:, r], V[:, r])
        w_norms[r] = np.dot(W[:, r], W[:, r])
        
    # C[u, r] = sum_v sum_w E[u,v,w] V[v,r] W[w,r]
    # This is like a tensor contraction
    for u in range(n_u):
        for r in range(R):
            # Sum over v, w
            s = 0.0
            for v in range(n_v):
                V_vr = V[v, r]
                if V_vr != 0:
                    for w in range(n_w):
                        s += E[u, v, w] * V_vr * W[w, r]
            C[u, r] = s
    return C, v_norms, w_norms

def _local_search(U, V, W, T, values, timeout=10.0):
    """
    Local search with incremental error update.
    """
    start_time = time.time()
    n_u, n_v, n_w = T.shape
    R = U.shape[1]
    
    # Initialize E = S - T
    # S = sum_r U_r V_r W_r
    E = np.zeros_like(T)
    for r in range(R):
        E += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    E -= T
    
    current_error = np.linalg.norm(E) ** 2
    if current_error == 0.0:
        return 0.0, U, V, W
    
    C, v_norms, w_norms = _compute_error_and_C(E, V, W)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < timeout:
        if current_error == 0.0:
            break
            
        best_move = None
        best_new_error = current_error
        
        # Try a batch of random moves
        batch_size = 20
        for _ in range(batch_size):
            # Pick a random matrix and index
            total_vars = U.size + V.size + W.size
            var_idx = random.randint(0, total_vars - 1)
            
            if var_idx < U.size:
                mat = U
                idx = var_idx
                # u, r = idx // R, idx % R
                u, r = idx // R, idx % R
                old_val = U[u, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                
                # Delta error = 2 * delta * C[u, r] + delta^2 * v_norms[r] * w_norms[r]
                delta_err = 2 * delta * C[u, r] + (delta * delta) * v_norms[r] * w_norms[r]
                new_error = current_error + delta_err
                
                if new_error < best_new_error:
                    best_new_error = new_error
                    best_move = ('U', u, r, old_val, new_val, delta)
                    
            elif var_idx < U.size + V.size:
                idx = var_idx - U.size
                mat = V
                v, r = idx // R, idx % R
                old_val = V[v, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                
                # For V, the gradient is not just C.
                # S = sum_r U_r V_r W_r
                # dS/dV[v, r] = U_r (at v) * W_r
                # We need to compute the gradient of error w.r.t V[v, r].
                # Error = ||E||^2
                # d(Error)/dV[v, r] = 2 * sum_{u, w} E[u, v, w] U[u, r] W[w, r]
                # Let's call this G_V[v, r]
                # We didn't precompute G_V. We need to.
                # G_V[v, r] = sum_{u, w} E[u, v, w] U[u, r] W[w, r]
                
                # To make this efficient, we should precompute gradients for U, V, W.
                # Let's switch strategy: Precompute full gradients.
                # G_U[u, r] = sum_{v, w} E[u, v, w] V[v, r] W[w, r]  (This is C)
                # G_V[v, r] = sum_{u, w} E[u, v, w] U[u, r] W[w, r]
                # G_W[w, r] = sum_{u, v} E[u, v, w] U[u, r] V[v, r]
                
                # We only computed C (G_U). Let's compute G_V and G_W too.
                # This will cost O(n_u n_v n_w R) which is 576 * 20 = 11520. Fast.
                
                # I will restructure the loop to compute all gradients.
                # For now, skip V and W moves in this iteration to keep code simple?
                # No, that's inefficient.
                
                # Let's break and restart with full gradient computation.
                pass 
            else:
                idx = var_idx - U.size - V.size
                mat = W
                w, r = idx // R, idx % R
                old_val = W[w, r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Similar to V, need G_W
                pass

        # The above logic is incomplete because I didn't precompute G_V and G_W.
        # Let's implement a version that computes all gradients.
        
        # Recompute gradients
        G_U = np.zeros_like(U)
        G_V = np.zeros_like(V)
        G_W = np.zeros_like(W)
        
        # G_U[u, r] = sum_{v,w} E[u,v,w] V[v,r] W[w,r]
        # G_V[v, r] = sum_{u,w} E[u,v,w] U[u,r] W[w,r]
        # G_W[w, r] = sum_{u,v} E[u,v,w] U[u,r] V[v,r]
        
        for r in range(R):
            # Compute contributions for this r
            # E is (n_u, n_v, n_w)
            # V_r is (n_v), W_r is (n_w), U_r is (n_u)
            
            # G_U[:, r]
            # G_U[u, r] = sum_{v,w} E[u,v,w] V_r[v] W_r[w]
            # This is dot product of E[u, :, :] with outer(V_r, W_r)
            # Outer(V_r, W_r) is (n_v, n_w)
            outer_VW = np.outer(V[:, r], W[:, r])
            G_U[:, r] = np.tensordot(E, outer_VW, axes=([1, 2], [0, 1]))
            
            # G_V[:, r]
            # G_V[v, r] = sum_{u,w} E[u,v,w] U_r[u] W_r[w]
            # This is sum over u, w.
            # Fix v. Sum_{u,w} E[u,v,w] U_r[u] W_r[w]
            # Can be computed as:
            # Temp[u, w] = E[u, :, w] ... no.
            # Let's use einsum or loops.
            # G_V[v, r] = sum_u sum_w E[u, v, w] U_r[u] W_r[w]
            # = sum_u U_r[u] (sum_w E[u, v, w] W_r[w])
            # Let M[u, v] = sum_w E[u, v, w] W_r[w]
            # Then G_V[v, r] = sum_u M[u, v] U_r[u]
            M = np.tensordot(E, W[:, r], axes=([2], [0])) # (n_u, n_v)
            G_V[:, r] = np.dot(M.T, U[:, r]) # (n_v,)
            
            # G_W[:, r]
            # G_W[w, r] = sum_{u,v} E[u,v,w] U_r[u] V_r[v]
            # Let N[u, w] = sum_v E[u, v, w] V_r[v]
            # Then G_W[w, r] = sum_u N[u, w] U_r[u]
            N = np.tensordot(E, V[:, r], axes=([1], [0])) # (n_u, n_w)
            G_W[:, r] = np.dot(N.T, U[:, r]) # (n_w,)
            
            # Update norms
            v_norms[r] = np.dot(V[:, r], V[:, r])
            w_norms[r] = np.dot(W[:, r], W[:, r])
            
        # Now search for best move
        best_move = None
        best_new_error = current_error
        
        # Check U moves
        for u in range(n_u):
            for r in range(R):
                old_val = U[u, r]
                for new_val in values:
                    if new_val == old_val:
                        continue
                    delta = new_val - old_val
                    # Delta E = delta * V_r W_r
                    # Delta Error = 2 * delta * G_U[u, r] + delta^2 * v_norms[r] * w_norms[r]
                    delta_err = 2 * delta * G_U[u, r] + (delta * delta) * v_norms[r] * w_norms[r]
                    new_error = current_error + delta_err
                    if new_error < best_new_error:
                        best_new_error = new_error
                        best_move = ('U', u, r, old_val, new_val, delta)
                        
        # Check V moves
        for v in range(n_v):
            for r in range(R):
                old_val = V[v, r]
                for new_val in values:
                    if new_val == old_val:
                        continue
                    delta = new_val - old_val
                    # Delta E = delta * U_r W_r
                    # Delta Error = 2 * delta * G_V[v, r] + delta^2 * u_norms[r] * w_norms[r]
                    # Need u_norms
                    u_norms = np.sum(U**2, axis=0)
                    delta_err = 2 * delta * G_V[v, r] + (delta * delta) * u_norms[r] * w_norms[r]
                    new_error = current_error + delta_err
                    if new_error < best_new_error:
                        best_new_error = new_error
                        best_move = ('V', v, r, old_val, new_val, delta)
                        
        # Check W moves
        for w in range(n_w):
            for r in range(R):
                old_val = W[w, r]
                for new_val in values:
                    if new_val == old_val:
                        continue
                    delta = new_val - old_val
                    # Delta E = delta * U_r V_r
                    # Delta Error = 2 * delta * G_W[w, r] + delta^2 * u_norms[r] * v_norms[r]
                    delta_err = 2 * delta * G_W[w, r] + (delta * delta) * u_norms[r] * v_norms[r]
                    new_error = current_error + delta_err
                    if new_error < best_new_error:
                        best_new_error = new_error
                        best_move = ('W', w, r, old_val, new_val, delta)
                        
        if best_move:
            mat_name, idx1, r, old_val, new_val, delta = best_move
            if mat_name == 'U':
                U[idx1, r] = new_val
                # Update E
                E += delta * np.einsum('v,w->uvw', V[:, r], W[:, r])
            elif mat_name == 'V':
                V[idx1, r] = new_val
                E += delta * np.einsum('u,w->uvw', U[:, r], W[:, r])
            else:
                W[idx1, r] = new_val
                E += delta * np.einsum('u,v->uvw', U[:, r], V[:, r])
                
            current_error = best_new_error
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
    return current_error, U, V, W

def _try_rank(R, n, m, p, T, values, time_limit):
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    num_restarts = 10
    time_per_restart = time_limit / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        init_values = [-1, 0, 1]
        U = np.random.choice(init_values, size=(n * m, R)).astype(float)
        V = np.random.choice(init_values, size=(m * p, R)).astype(float)
        W = np.random.choice(init_values, size=(p * n, R)).astype(float)
        
        remaining_time = time_per_restart - (time.time() - start_time)
        if remaining_time < 0.1:
            continue
            
        err, U, V, W = _local_search(U, V, W, T, values, timeout=remaining_time)
        
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error == 0.0:
            break
            
    if best_error == 0.0:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    start_time = time.time()
    deadline = 140.0
    
    # Try ranks lower than schoolbook
    for R in range(18, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / ranks_left
        time_budget = min(time_budget, 20.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify
            reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(reconstructed - T)
            if err < 1e-9:
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
                
    return best_result
