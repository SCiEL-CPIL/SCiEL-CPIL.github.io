"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                target_indices.append((u_idx, v_idx, w_idx))
    
    def get_schoolbook():
        """Return schoolbook decomposition as dict of lists."""
        rank = n * m * p  # 50
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)
    
    def verify_and_clean(U, V, W, atol=1e-8):
        """Verify and try to clean values to half-integers."""
        if not verify_decomposition(U, V, W, atol):
            return None
        
        # Try snapping to half-integers
        u_snap = np.round(U * 2.0) / 2.0
        v_snap = np.round(V * 2.0) / 2.0
        w_snap = np.round(W * 2.0) / 2.0
        if verify_decomposition(u_snap, v_snap, w_snap, atol=1e-7):
            # Clean tiny values
            u_snap = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
            v_snap = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
            w_snap = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
            if verify_decomposition(u_snap, v_snap, w_snap, atol=1e-6):
                return (u_snap, v_snap, w_snap)
        
        # Try integer snap
        u_int = np.round(U)
        v_int = np.round(V)
        w_int = np.round(W)
        if verify_decomposition(u_int, v_int, w_int, atol=1e-6):
            return (u_int, v_int, w_int)
        
        # Return as-is if valid
        return (U, V, W)
    
    def solve_linear_system_for_patterns(patterns):
        """
        Given a list of patterns, solve for coefficients such that
        sum_r c_r * u_r ⊗ v_r ⊗ w_r = target.
        Returns (coeffs, U, V, W) if solution exists, else None.
        """
        rank = len(patterns)
        
        # Build the linear system
        all_positions = set()
        for r in range(rank):
            u_sup, v_sup, w_sup = patterns[r]
            for a in u_sup:
                for b in v_sup:
                    for c in w_sup:
                        all_positions.add((a, b, c))
        
        pos_list = list(all_positions)
        if not pos_list:
            return None
        
        A = np.zeros((len(pos_list), rank))
        b_vec = np.zeros(len(pos_list))
        
        for pos_idx, (a, b, c) in enumerate(pos_list):
            b_vec[pos_idx] = target[a, b, c]
            for r in range(rank):
                u_sup, v_sup, w_sup = patterns[r]
                if a in u_sup and b in v_sup and c in w_sup:
                    A[pos_idx, r] = 1.0
        
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
            
            # Check if the solution is exact
            T = np.zeros((dim_u, dim_v, dim_w))
            for r in range(rank):
                if abs(coeffs[r]) > 1e-10:
                    u_sup, v_sup, w_sup = patterns[r]
                    for a in u_sup:
                        for b in v_sup:
                            for c in w_sup:
                                T[a, b, c] += coeffs[r]
            
            if not np.allclose(T, target, atol=1e-8):
                return None
            
            # Build U, V, W
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            for r in range(rank):
                u_sup, v_sup, w_sup = patterns[r]
                for a in u_sup:
                    U[a, r] = coeffs[r]
                for b in v_sup:
                    V[b, r] = 1.0
                for c in w_sup:
                    W[c, r] = 1.0
            
            return (coeffs, U, V, W)
            
        except Exception:
            return None
    
    def generate_all_line_patterns():
        """
        Generate all 45 structured line patterns.
        Returns list of (u_support, v_support, w_support) tuples.
        """
        patterns = []
        
        # Type 1: Fix (i,j), vary k
        for i in range(n):
            for j in range(m):
                u_sup = {i * m + j}
                v_sup = {j * p + k for k in range(p)}
                w_sup = {k * n + i for k in range(p)}
                patterns.append((u_sup, v_sup, w_sup))
        
        # Type 2: Fix (j,k), vary i
        for j in range(m):
            for k in range(p):
                u_sup = {i * m + j for i in range(n)}
                v_sup = {j * p + k}
                w_sup = {k * n + i for i in range(n)}
                patterns.append((u_sup, v_sup, w_sup))
        
        # Type 3: Fix (i,k), vary j
        for i in range(n):
            for k in range(p):
                u_sup = {i * m + j for j in range(m)}
                v_sup = {j * p + k for j in range(m)}
                w_sup = {k * n + i}
                patterns.append((u_sup, v_sup, w_sup))
        
        return patterns
    
    def try_optimization(rank, max_restarts=10, max_iter=80):
        """
        Try to find a decomposition of given rank using optimization.
        Uses multiple random initializations.
        """
        if time.time() > deadline - 10.0:
            return None
        
        best_loss = float('inf')
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(2000 + restart * 31 + rank * 7)
            
            # Initialize with random sparse structure
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            for r in range(rank):
                # Each column gets a few non-zero entries
                num_u = np.random.randint(1, 4)
                num_v = np.random.randint(1, 3)
                num_w = np.random.randint(1, 4)
                
                u_indices = np.random.choice(dim_u, num_u, replace=False)
                v_indices = np.random.choice(dim_v, num_v, replace=False)
                w_indices = np.random.choice(dim_w, num_w, replace=False)
                
                for idx in u_indices:
                    U[idx, r] = np.random.randn() * 0.5
                for idx in v_indices:
                    V[idx, r] = np.random.randn() * 0.5
                for idx in w_indices:
                    W[idx, r] = np.random.randn() * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def fun(params):
                u_flat = params[:dim_u * rank]
                v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
                w_flat = params[dim_u * rank + dim_v * rank : ]
                
                U_m = u_flat.reshape(dim_u, rank)
                V_m = v_flat.reshape(dim_v, rank)
                W_m = w_flat.reshape(dim_w, rank)
                
                T = np.einsum('ar,br,cr->abc', U_m, V_m, W_m)
                diff = T - target
                loss = np.sum(diff * diff)
                
                grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V_m, W_m)
                grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U_m, W_m)
                grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U_m, V_m)
                
                grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
                return loss, grad
            
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True,
                               options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_params = res.x.copy()
                
                if res.fun < 1e-10:
                    u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    
                    result = verify_and_clean(u_final, v_final, w_final)
                    if result is not None:
                        return result
                        
            except Exception:
                continue
        
        # If we got close but not exact, try to polish
        if best_params is not None and best_loss < 1e-6:
            u_final = best_params[:dim_u * rank].reshape(dim_u, rank)
            v_final = best_params[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            w_final = best_params[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
            
            # Additional polish with tighter tolerance
            try:
                res2 = minimize(fun, best_params, method='L-BFGS-B', jac=True,
                                options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
                if res2.fun < 1e-10:
                    u_final = res2.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res2.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res2.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    result = verify_and_clean(u_final, v_final, w_final)
                    if result is not None:
                        return result
            except Exception:
                pass
        
        return None
    
    # Generate all 45 line patterns
    all_patterns = generate_all_line_patterns()
    
    best_result = None
    best_rank = 50  # schoolbook
    
    # Strategy 1: Try subsets of the 45 line patterns
    # First try rank 45 (all patterns)
    if time.time() < deadline - 30.0:
        sol = solve_linear_system_for_patterns(all_patterns)
        if sol is not None:
            coeffs, U, V, W = sol
            result = verify_and_clean(U, V, W)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 45
    
    # Strategy 2: Try to find smaller subsets
    # For ranks 40-44, try random subsets of the 45 patterns
    if best_rank > 40 and time.time() < deadline - 20.0:
        for target_rank in range(40, min(45, best_rank)):
            if time.time() > deadline - 10.0:
                break
            
            found = False
            num_trials = 50
            np.random.seed(42 + target_rank)
            
            for trial in range(num_trials):
                if time.time() > deadline - 5.0:
                    break
                
                subset_indices = np.random.choice(45, target_rank, replace=False)
                subset = [all_patterns[i] for i in subset_indices]
                
                sol = solve_linear_system_for_patterns(subset)
                if sol is not None:
                    coeffs, U, V, W = sol
                    result = verify_and_clean(U, V, W)
                    if result is not None:
                        U, V, W = result
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
    
    # Strategy 3: Use optimization for even lower ranks
    if best_rank > 35 and time.time() < deadline - 15.0:
        for target_rank in range(35, best_rank):
            if time.time() > deadline - 5.0:
                break
            
            result = try_optimization(target_rank, max_restarts=5, max_iter=60)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
