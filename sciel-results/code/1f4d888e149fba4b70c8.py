"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from itertools import product
from scipy.optimize import nnls

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
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
    
    def try_pattern_decomposition(rank, patterns, max_time=10.0):
        """
        Try to find a decomposition using given support patterns.
        Each pattern is a tuple (u_support, v_support, w_support) where each is a set of indices.
        We solve for coefficients c_r such that sum_r c_r * u_r ⊗ v_r ⊗ w_r = target.
        """
        if time.time() > deadline - 5.0:
            return None
        
        # Build the linear system: for each tensor position (a,b,c) that is 1 in target,
        # we need sum over r where a in u_support_r, b in v_support_r, c in w_support_r of c_r = 1
        # For positions that are 0 in target, the sum must be 0.
        
        # Collect all relevant tensor positions
        all_positions = set()
        for r in range(rank):
            u_sup, v_sup, w_sup = patterns[r]
            for a in u_sup:
                for b in v_sup:
                    for c in w_sup:
                        all_positions.add((a, b, c))
        
        # Build the linear system A @ c = b
        # A[pos_idx, r] = 1 if position pos is covered by pattern r
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
        
        # Solve the linear system
        try:
            # Use least squares
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
                    U[a, r] = 1.0
                for b in v_sup:
                    V[b, r] = 1.0
                for c in w_sup:
                    W[c, r] = 1.0
                # Scale by coefficient
                if abs(coeffs[r]) > 1e-10:
                    U[:, r] *= coeffs[r]
                    # Actually, we can put the coefficient in any one of U, V, W
                    # Let's put it in U
                    U[:, r] /= coeffs[r]  # undo
                    U[:, r] *= coeffs[r]  # redo - just keep it simple
            
            # Better: put coefficient in U
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
            
            result = verify_and_clean(U, V, W)
            if result is not None:
                return result
                
        except Exception:
            return None
        
        return None
    
    def generate_structured_patterns(rank):
        """
        Generate structured support patterns for rank-1 terms.
        Each term covers a "line" or "plane" in the tensor.
        
        Strategy: For each (i,j) pair, create terms that cover varying k,
        and for each (j,k) pair, create terms that cover varying i, etc.
        """
        patterns = []
        
        # Type 1: Fix (i,j), vary k -> covers (i*m+j, j*p+k, k*n+i) for all k
        # This gives a pattern with u_sup={i*m+j}, v_sup={j*p+k for k}, w_sup={k*n+i for k}
        # But this is one term per (i,j) which is n*m = 10 terms, each covering p=5 positions
        
        # Type 2: Fix (j,k), vary i -> covers (i*m+j, j*p+k, k*n+i) for all i
        # u_sup={i*m+j for i}, v_sup={j*p+k}, w_sup={k*n+i for i}
        # m*p = 25 terms, each covering n=2 positions
        
        # Type 3: Fix (i,k), vary j -> covers (i*m+j, j*p+k, k*n+i) for all j
        # u_sup={i*m+j for j}, v_sup={j*p+k for j}, w_sup={k*n+i}
        # n*p = 10 terms, each covering m=5 positions
        
        # Let's create a mix of these patterns
        
        # Start with some "line" patterns
        for i in range(n):
            for j in range(m):
                # Fix (i,j), vary k
                u_sup = {i * m + j}
                v_sup = {j * p + k for k in range(p)}
                w_sup = {k * n + i for k in range(p)}
                patterns.append((u_sup, v_sup, w_sup))
        
        for j in range(m):
            for k in range(p):
                # Fix (j,k), vary i
                u_sup = {i * m + j for i in range(n)}
                v_sup = {j * p + k}
                w_sup = {k * n + i for i in range(n)}
                patterns.append((u_sup, v_sup, w_sup))
        
        for i in range(n):
            for k in range(p):
                # Fix (i,k), vary j
                u_sup = {i * m + j for j in range(m)}
                v_sup = {j * p + k for j in range(m)}
                w_sup = {k * n + i}
                patterns.append((u_sup, v_sup, w_sup))
        
        # Shuffle and take first 'rank' patterns
        np.random.seed(42)
        indices = np.random.permutation(len(patterns))[:rank]
        selected = [patterns[i] for i in indices]
        
        return selected
    
    def try_optimization(rank, max_restarts=5, max_iter=50):
        """
        Try to find a decomposition of given rank using optimization.
        Uses a structured initialization.
        """
        if time.time() > deadline - 10.0:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                return None
            
            np.random.seed(1000 + restart * 17 + rank)
            
            # Initialize with structured patterns
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            # Add random small perturbations to structured base
            for r in range(rank):
                # Randomly assign some entries
                for a in range(dim_u):
                    if np.random.rand() < 0.3:
                        U[a, r] = np.random.randn() * 0.5
                for b in range(dim_v):
                    if np.random.rand() < 0.2:
                        V[b, r] = np.random.randn() * 0.5
                for c in range(dim_w):
                    if np.random.rand() < 0.3:
                        W[c, r] = np.random.randn() * 0.5
            
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
                from scipy.optimize import minimize
                res = minimize(fun, params, method='L-BFGS-B', jac=True,
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 15})
                
                if res.fun < 1e-8:
                    u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    
                    result = verify_and_clean(u_final, v_final, w_final)
                    if result is not None:
                        return result
                        
            except Exception:
                continue
        
        return None
    
    # Try various approaches
    best_result = None
    
    # First, try structured pattern-based approach for various ranks
    # The line patterns give us 10 + 25 + 10 = 45 base patterns
    # Try to find a subset of these that works
    
    # Try rank 45 first (all line patterns)
    if time.time() < deadline - 20.0:
        patterns_45 = generate_structured_patterns(45)
        result = try_pattern_decomposition(45, patterns_45)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try lower ranks with optimization
    if best_result is None:
        for target_rank in [40, 42, 44, 45, 46, 48]:
            if time.time() > deadline - 15.0:
                break
            
            result = try_optimization(target_rank, max_restarts=3, max_iter=40)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
