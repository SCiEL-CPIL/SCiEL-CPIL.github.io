import numpy as np
import time
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor T[i,j,k] = 1 if consistent, else 0
    # For standard matrix multiplication, all indices are consistent.
    T = np.ones((n, m, p), dtype=np.float64)
    
    N, M, P = n, m, p
    
    def verify(U, V, W):
        """
        Check if U, V, W exactly decompose T.
        U: (N*M, r), V: (M*P, r), W: (P*N, r)
        """
        r = U.shape[1]
        U_reshaped = U.reshape(N, M, r)
        V_reshaped = V.reshape(M, P, r)
        W_reshaped = W.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check for exact match within tight tolerance
        return np.allclose(T_rec, T, atol=1e-9)
    
    def objective_and_grad(x, r):
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:]
        
        U = U_flat.reshape(N, M, r)
        V = V_flat.reshape(M, P, r)
        W = W_flat.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T_rec - T
        loss = np.sum(err**2)
        
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    def snap_to_rationals(U_opt, V_opt, W_opt, max_denom=6):
        """Try to snap to rational numbers with small denominators"""
        for denom in [1, 2, 3, 4, 5, 6]:
            U_snap = np.round(U_opt * denom) / denom
            V_snap = np.round(V_opt * denom) / denom
            W_snap = np.round(W_opt * denom) / denom
            if verify(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
        return None
    
    def try_rank_structured(r, seed=0):
        """Try rank r with structured initialization based on schoolbook"""
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        # Start with schoolbook and try to merge terms
        U_init = np.zeros((N, M, r))
        V_init = np.zeros((M, P, r))
        W_init = np.zeros((P, N, r))
        
        # Initialize first min(r, n*m*p) terms with schoolbook
        num_init = min(r, n * m * p)
        for idx in range(num_init):
            i = idx // (m * p)
            rem = idx % (m * p)
            j = rem // p
            k = rem % p
            U_init[i, j, idx] = 1.0
            V_init[j, k, idx] = 1.0
            W_init[k, i, idx] = 1.0
        
        # Add random perturbation for remaining terms
        if r > num_init:
            U_init[:, :, num_init:] = np.random.randn(N, M, r - num_init) * 0.1
            V_init[:, :, num_init:] = np.random.randn(M, P, r - num_init) * 0.1
            W_init[:, :, num_init:] = np.random.randn(P, N, r - num_init) * 0.1
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use L-BFGS-B
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
        except Exception:
            return None
            
        if time.time() > deadline:
            return None
            
        # Check if optimization converged to a low error
        if res.fun < 1e-6:
            U_opt = res.x[:size_U].reshape(N, M, r)
            V_opt = res.x[size_U:size_U+size_V].reshape(M, P, r)
            W_opt = res.x[size_U+size_V:].reshape(P, N, r)
            
            result = snap_to_rationals(U_opt, V_opt, W_opt)
            if result:
                U_snap, V_snap, W_snap = result
                return U_snap.flatten(axis=1), V_snap.flatten(axis=1), W_snap.flatten(axis=1)
                
            # Also try direct integer/half-integer snap
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            if verify(U_int, V_int, W_int):
                return U_int.flatten(axis=1), V_int.flatten(axis=1), W_int.flatten(axis=1)
                
            U_half = np.round(U_opt * 2) / 2.0
            V_half = np.round(V_opt * 2) / 2.0
            W_half = np.round(W_opt * 2) / 2.0
            if verify(U_half, V_half, W_half):
                return U_half.flatten(axis=1), V_half.flatten(axis=1), W_half.flatten(axis=1)

        return None
    
    def try_rank_random(r, seed=0):
        """Try rank r with random initialization"""
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        x0 = np.random.randn(total_size) * 0.3
        
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
        except Exception:
            return None
            
        if time.time() > deadline:
            return None
            
        if res.fun < 1e-6:
            U_opt = res.x[:size_U].reshape(N, M, r)
            V_opt = res.x[size_U:size_U+size_V].reshape(M, P, r)
            W_opt = res.x[size_U+size_V:].reshape(P, N, r)
            
            result = snap_to_rationals(U_opt, V_opt, W_opt)
            if result:
                U_snap, V_snap, W_snap = result
                return U_snap.flatten(axis=1), V_snap.flatten(axis=1), W_snap.flatten(axis=1)
                
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            if verify(U_int, V_int, W_int):
                return U_int.flatten(axis=1), V_int.flatten(axis=1), W_int.flatten(axis=1)
                
            U_half = np.round(U_opt * 2) / 2.0
            V_half = np.round(V_opt * 2) / 2.0
            W_half = np.round(W_opt * 2) / 2.0
            if verify(U_half, V_half, W_half):
                return U_half.flatten(axis=1), V_half.flatten(axis=1), W_half.flatten(axis=1)

        return None
    
    # Fallback: Schoolbook
    U_fb = np.zeros((N*M, rank_schoolbook))
    V_fb = np.zeros((M*P, rank_schoolbook))
    W_fb = np.zeros((P*N, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_schoolbook
    
    # Search for lower rank - expanded range
    ranks_to_try = list(range(rank_schoolbook - 1, max(rank_schoolbook - 8, 1), -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try structured initialization first (more likely to succeed)
        num_seeds_structured = 5
        for seed in range(num_seeds_structured):
            if time.time() > deadline:
                break
            result = try_rank_structured(r, seed)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = r
                break
        
        if best_rank < rank_schoolbook:
            continue  # Found valid, try lower rank
        
        # Try random initialization if structured didn't work
        num_seeds_random = 3
        for seed in range(num_seeds_random):
            if time.time() > deadline:
                break
            result = try_rank_random(r, seed)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = r
                break
    
    U_out, V_out, W_out = best_result
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
