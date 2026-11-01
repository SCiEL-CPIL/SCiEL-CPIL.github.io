"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank = 18
    rank_sb = n * m * p
    U0 = np.zeros((n*m, rank_sb))
    V0 = np.zeros((m*p, rank_sb))
    W0 = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_sb
    
    def _verify_exact(U, V, W, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_approx - T)) > tol:
            return False
        for mat in [U, V, W]:
            for val in mat:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > 1e-6:
                    return False
        return True

    def _sa_search(U, V, W, max_time, rng):
        """
        Simulated Annealing on half-integer grid.
        Returns (U, V, W, error).
        """
        n_u, r = U.shape
        n_v, _ = V.shape
        n_w, _ = W.shape
        
        # Flatten for easier indexing? No, keep matrices.
        # Precompute S and D
        S = np.einsum('ar,br,cr->abc', U, V, W)
        D = T - S
        E = np.sum(D*D)
        
        if E < 1e-9:
            return U, V, W, E

        T_sim = 1.0
        cooling = 0.9999
        min_T = 1e-4
        
        # Values allowed: -2.0 to 2.0 step 0.5
        vals = np.arange(-2.0, 2.5, 0.5)
        
        start_time = time.time()
        
        while time.time() - start_time < max_time and E > 1e-9:
            if time.time() > _DEADLINE - 5:
                break
                
            # Pick random variable
            mat_idx = rng.integers(0, 3)
            if mat_idx == 0:
                i = rng.integers(0, n_u)
                j = rng.integers(0, r)
                # Change U[i, j]
                old_val = U[i, j]
                # Neighbor selection
                if rng.random() < 0.8:
                    # Small step
                    new_val = old_val + rng.choice([-0.5, 0.5])
                else:
                    # Jump
                    new_val = rng.choice(vals)
                
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                
                # Calculate Delta E
                # V[:, j] and W[:, j]
                v_col = V[:, j]
                w_col = W[:, j]
                outer = np.outer(v_col, w_col)
                
                # D[i, :, :] is the slice
                d_slice = D[i, :, :]
                
                # delta_E = sum( -2 * D * delta * Outer + delta^2 * Outer^2 )
                # Vectorized:
                term1 = -2.0 * delta * np.sum(d_slice * outer)
                term2 = (delta * delta) * np.sum(outer * outer)
                delta_E = term1 + term2
                
                # Acceptance
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T_sim):
                    # Update
                    U[i, j] = new_val
                    S[i, :, :] += delta * outer
                    D[i, :, :] -= delta * outer
                    E += delta_E
            elif mat_idx == 1:
                i = rng.integers(0, n_v)
                j = rng.integers(0, r)
                old_val = V[i, j]
                if rng.random() < 0.8:
                    new_val = old_val + rng.choice([-0.5, 0.5])
                else:
                    new_val = rng.choice(vals)
                
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                
                u_col = U[:, j]
                w_col = W[:, j]
                outer = np.outer(u_col, w_col)
                
                # D[:, i, :]
                d_slice = D[:, i, :]
                
                term1 = -2.0 * delta * np.sum(d_slice * outer)
                term2 = (delta * delta) * np.sum(outer * outer)
                delta_E = term1 + term2
                
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T_sim):
                    V[i, j] = new_val
                    S[:, i, :] += delta * outer
                    D[:, i, :] -= delta * outer
                    E += delta_E
            else:
                i = rng.integers(0, n_w)
                j = rng.integers(0, r)
                old_val = W[i, j]
                if rng.random() < 0.8:
                    new_val = old_val + rng.choice([-0.5, 0.5])
                else:
                    new_val = rng.choice(vals)
                
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                
                u_col = U[:, j]
                v_col = V[:, j]
                outer = np.outer(u_col, v_col)
                
                # D[:, :, i]
                d_slice = D[:, :, i]
                
                term1 = -2.0 * delta * np.sum(d_slice * outer)
                term2 = (delta * delta) * np.sum(outer * outer)
                delta_E = term1 + term2
                
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T_sim):
                    W[i, j] = new_val
                    S[:, :, i] += delta * outer
                    D[:, :, i] -= delta * outer
                    E += delta_E
            
            T_sim = max(T_sim * cooling, min_T)
            
        return U, V, W, E

    rng = np.random.default_rng(42)
    
    target_ranks = [14, 13]
    
    for rank in target_ranks:
        if _remaining() < 20:
            break
        
        found = False
        # Allow many short restarts
        max_restarts = 100
        
        for restart in range(max_restarts):
            if _remaining() < 10:
                break
            
            # Random half-integer initialization
            vals = np.arange(-2.0, 2.5, 0.5)
            U = rng.choice(vals, size=(n*m, rank))
            V = rng.choice(vals, size=(m*p, rank))
            W = rng.choice(vals, size=(p*n, rank))
            
            # Time budget per restart: 1-2 seconds
            time_per_restart = 1.0 if _remaining() > 50 else 0.5
            
            U_opt, V_opt, W_opt, err = _sa_search(U, V, W, time_per_restart, rng)
            
            if err < 1e-9:
                # Verify exactness
                if _verify_exact(U_opt, V_opt, W_opt, tol=1e-7):
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = rank
                    found = True
                    break
            
            # Optional: If we got very close, try to polish with tighter SA
            if err < 1e-4 and err > 1e-9:
                 # Try a short polish
                 U_pol, V_pol, W_pol, err_pol = _sa_search(U_opt, V_opt, W_opt, 0.5, rng)
                 if err_pol < 1e-9 and _verify_exact(U_pol, V_pol, W_pol, tol=1e-7):
                     best_U, best_V, best_W = U_pol, V_pol, W_pol
                     best_rank = rank
                     found = True
                     break

        if found:
            continue

    if best_rank < rank_sb:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        U_v = np.array(U_list)
        V_v = np.array(V_list)
        W_v = np.array(W_list)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-7:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
