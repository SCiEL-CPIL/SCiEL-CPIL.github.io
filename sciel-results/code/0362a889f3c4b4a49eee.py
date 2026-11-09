import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # 15-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def compute_error(U, V, W):
        """Compute max absolute error of reconstruction."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.max(np.abs(T_rec - T))
    
    def smooth_error(U, V, W, beta=10.0):
        """
        Smooth surrogate for max error using log-sum-exp.
        This is differentiable and helps with gradient-based refinement.
        """
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        diff = T_rec - T
        # Use soft-max as smooth approximation to max
        # Only consider entries where T is nonzero or where diff is large
        mask = (np.abs(T) > 0) | (np.abs(diff) > 0.1)
        if not np.any(mask):
            return 0.0
        d = diff[mask]
        # log-sum-exp for stability
        max_d = np.max(np.abs(d))
        if max_d < 1e-15:
            return 0.0
        scaled = beta * np.abs(d)
        shifted = scaled - np.max(scaled)
        lse = np.max(scaled) + np.log(np.sum(np.exp(shifted)))
        return lse / beta
    
    def gradient_step(U, V, W, lr=0.01, beta=5.0):
        """
        Perform one gradient descent step on the smooth error.
        Returns updated U, V, W.
        """
        r = U.shape[1]
        # Compute T_rec and error
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        diff = T_rec - T
        
        # Gradient with respect to each factor
        # d/dU[a, r] = sum_{b,c} V[b,r]*W[c,r]*sign(diff[a,b,c])
        # Use smooth sign: tanh(beta * diff)
        smooth_sign = np.tanh(beta * diff)
        
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for r_idx in range(r):
            # Gradient for U[:, r_idx]
            U_grad[:, r_idx] = np.einsum('abc,b,c->a', smooth_sign, V[:, r_idx], W[:, r_idx])
            # Gradient for V[:, r_idx]
            V_grad[:, r_idx] = np.einsum('abc,a,c->b', smooth_sign, U[:, r_idx], W[:, r_idx])
            # Gradient for W[:, r_idx]
            W_grad[:, r_idx] = np.einsum('abc,a,b->c', smooth_sign, U[:, r_idx], V[:, r_idx])
        
        U_new = U - lr * U_grad
        V_new = V - lr * V_grad
        W_new = W - lr * W_grad
        
        return U_new, V_new, W_new
    
    def polish(U, V, W, max_iter=200, tol=1e-10):
        """
        High-precision polish: use gradient descent on smooth error,
        then snap to nearest half-integers and verify.
        """
        U_cur = U.copy().astype(float)
        V_cur = V.copy().astype(float)
        W_cur = W.copy().astype(float)
        
        best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
        best_err = compute_error(best_U, best_V, best_W)
        
        lr = 0.01
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            U_cur, V_cur, W_cur = gradient_step(U_cur, V_cur, W_cur, lr=lr, beta=5.0)
            err = compute_error(U_cur, V_cur, W_cur)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
            
            if best_err < tol:
                break
            
            # Decay learning rate
            if it % 20 == 19:
                lr *= 0.5
        
        # Try snapping to half-integers
        for snap_factor in [1.0, 2.0]:
            U_snap = np.round(best_U * snap_factor) / snap_factor
            V_snap = np.round(best_V * snap_factor) / snap_factor
            W_snap = np.round(best_W * snap_factor) / snap_factor
            
            if compute_error(U_snap, V_snap, W_snap) < 1e-8:
                return U_snap, V_snap, W_snap
        
        # If not snapped, return best continuous solution
        if best_err < 1e-8:
            return best_U, best_V, best_W
        
        return None
    
    def try_structured_search(r, max_attempts=3000):
        """
        Structured search: exploit the tensor's (i,j,k) structure.
        Each rank-1 term should correspond to a pattern over (i,j,k).
        """
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_result = None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + attempt * 31)
            
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Strategy: assign each rank term a "role" based on (i,j,k) patterns
            for r_idx in range(r):
                # Choose which (i,j,k) combinations this term will cover
                # Use a structured pattern: pick a subset of i, j, k values
                
                # For U: nonzero at positions (i, j) for selected (i,j) pairs
                # For V: nonzero at positions (j, k) for selected (j,k) pairs
                # For W: nonzero at positions (k, i) for selected (k,i) pairs
                
                # Choose 1-3 i values, 1-3 j values, 1-3 k values
                n_i = np.random.randint(1, min(4, n) + 1)
                n_j = np.random.randint(1, min(4, m) + 1)
                n_k = np.random.randint(1, min(4, p) + 1)
                
                i_vals = np.random.choice(n, size=n_i, replace=False)
                j_vals = np.random.choice(m, size=n_j, replace=False)
                k_vals = np.random.choice(p, size=n_k, replace=False)
                
                # Assign values from {-1, 1} (or {-1, 0, 1})
                use_zero = np.random.random() < 0.3  # 30% chance to allow zeros
                
                for i in i_vals:
                    for j in j_vals:
                        val = np.random.choice([-1, 1])
                        if use_zero and np.random.random() < 0.2:
                            val = 0
                        if val != 0:
                            U[i * m + j, r_idx] = val
                
                for j in j_vals:
                    for k in k_vals:
                        val = np.random.choice([-1, 1])
                        if use_zero and np.random.random() < 0.2:
                            val = 0
                        if val != 0:
                            V[j * p + k, r_idx] = val
                
                for k in k_vals:
                    for i in i_vals:
                        val = np.random.choice([-1, 1])
                        if use_zero and np.random.random() < 0.2:
                            val = 0
                        if val != 0:
                            W[k * n + i, r_idx] = val
            
            err = compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-8:
                return (U, V, W)
            
            # If close, try polishing
            if err < 0.5 and attempt % 5 == 0:
                polished = polish(U, V, W, max_iter=100, tol=1e-9)
                if polished is not None:
                    return polished
        
        if best_result is not None:
            U, V, W = best_result
            if check_decomposition(U, V, W):
                return (U, V, W)
            # Try polish on best
            polished = polish(U, V, W, max_iter=200, tol=1e-10)
            if polished is not None:
                return polished
        
        return None
    
    def try_mixed_search(r, max_attempts=2000):
        """
        Mixed search: combine structured and random initializations,
        with gradient-based refinement.
        """
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_result = None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(99999 + attempt * 17)
            
            if attempt % 2 == 0:
                # Structured initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                
                for r_idx in range(r):
                    # Pick a random "center" (i0, j0, k0)
                    i0 = np.random.randint(0, n)
                    j0 = np.random.randint(0, m)
                    k0 = np.random.randint(0, p)
                    
                    # Cover a local neighborhood
                    i_range = range(max(0, i0-1), min(n, i0+2))
                    j_range = range(max(0, j0-1), min(m, j0+2))
                    k_range = range(max(0, k0-1), min(p, k0+2))
                    
                    for i in i_range:
                        for j in j_range:
                            U[i * m + j, r_idx] = np.random.choice([-1, 1])
                    
                    for j in j_range:
                        for k in k_range:
                            V[j * p + k, r_idx] = np.random.choice([-1, 1])
                    
                    for k in k_range:
                        for i in i_range:
                            W[k * n + i, r_idx] = np.random.choice([-1, 1])
            else:
                # Random sparse initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                
                for r_idx in range(r):
                    u_nnz = np.random.randint(2, 6)
                    v_nnz = np.random.randint(2, 6)
                    w_nnz = np.random.randint(2, 6)
                    
                    u_pos = np.random.choice(nm, size=u_nnz, replace=False)
                    for pos in u_pos:
                        U[pos, r_idx] = np.random.choice([-1, 1])
                    
                    v_pos = np.random.choice(mp, size=v_nnz, replace=False)
                    for pos in v_pos:
                        V[pos, r_idx] = np.random.choice([-1, 1])
                    
                    w_pos = np.random.choice(pn, size=w_nnz, replace=False)
                    for pos in w_pos:
                        W[pos, r_idx] = np.random.choice([-1, 1])
            
            err = compute_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-8:
                return (U, V, W)
            
            # Gradient refinement for promising candidates
            if err < 1.0:
                # Do a few gradient steps
                U_cur, V_cur, W_cur = U.astype(float), V.astype(float), W.astype(float)
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    U_cur, V_cur, W_cur = gradient_step(U_cur, V_cur, W_cur, lr=0.005, beta=3.0)
                    cur_err = compute_error(U_cur, V_cur, W_cur)
                    if cur_err < 1e-8:
                        return (U_cur, V_cur, W_cur)
                
                # Try snapping
                for sf in [1.0, 2.0]:
                    U_snap = np.round(U_cur * sf) / sf
                    V_snap = np.round(V_cur * sf) / sf
                    W_snap = np.round(W_cur * sf) / sf
                    if compute_error(U_snap, V_snap, W_snap) < 1e-8:
                        return (U_snap, V_snap, W_snap)
                
                if compute_error(U_cur, V_cur, W_cur) < best_err:
                    best_err = compute_error(U_cur, V_cur, W_cur)
                    best_result = (U_cur.copy(), V_cur.copy(), W_cur.copy())
        
        if best_result is not None:
            U, V, W = best_result
            if check_decomposition(U, V, W):
                return (U, V, W)
            polished = polish(U, V, W, max_iter=150, tol=1e-10)
            if polished is not None:
                return polished
        
        return None
    
    # Try to find the minimum rank
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks from a reasonable upper bound down to lower bound
    # Focus on promising ranks first
    ranks_to_try = [42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 21]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try structured search first (more likely to find solutions)
        result = try_structured_search(r, max_attempts=1500)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
        
        # Try mixed search
        result = try_mixed_search(r, max_attempts=1000)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
