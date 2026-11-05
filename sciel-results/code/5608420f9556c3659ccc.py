"""Matrix multiplication 3,4,8 tensor decomposition - combined approach."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Combines structured search with gradient-based refinement.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    
    def schoolbook():
        """Return the standard rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build true tensor
        T_true = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m + j, j*p + k, k*n + i] = 1.0
        
        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        if not np.allclose(T_recon, T_true, atol=tol):
            return False
        return True
    
    def is_rational_halfint(x, tol=1e-8):
        """Check if x is an integer or half-integer within tolerance."""
        for scale in [1, 2]:
            val = x * scale
            if abs(val - round(val)) < tol:
                return True
        return False
    
    def try_snap_and_verify(U, V, W, scales=[1, 2]):
        """Try to snap to integer/half-integer and verify."""
        for scale in scales:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r, V_r, W_r
        return None
    
    def try_rational_reconstruct(U, V, W, max_denom=8):
        """Try to find exact rational entries with small denominators."""
        from fractions import Fraction
        best = None
        for denom in [1, 2, 4, 8]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r, V_r, W_r
        return None
    
    # Check time - if running low, return schoolbook
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return schoolbook()
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Build true tensor once
    T_true = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Lower bound: slice rank arguments
    # Fixing j: need at least n*p = 24
    # Fixing i: need at least m*p = 32  
    # Fixing k: need at least n*m = 12
    # So lower bound is max(24, 32, 12) = 32
    lb = max(n*p, m*p, n*m)  # 32
    
    found = False
    best_U = None
    best_V = None
    best_W = None
    
    # Strategy: Try structured decompositions first, then gradient-based search
    
    # Strategy 1: Block-structured decomposition
    # For 3x4 * 4x8, we can try to decompose using block structure
    # Think of the 4x8 matrix as having structure in the 8 dimension
    
    # Try ranks from lower bound up
    candidate_ranks = list(range(lb, best_rank + 1, 1))
    
    for r in candidate_ranks:
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        if r >= best_rank:
            break
        
        # Multiple initialization strategies
        init_strategies = []
        
        # Strategy A: Random initialization
        for seed in range(3):
            rng = np.random.RandomState(seed * 10000 + r)
            U_init = rng.randn(nm, r) * 0.5
            V_init = rng.randn(mp, r) * 0.5
            W_init = rng.randn(pn, r) * 0.5
            init_strategies.append((U_init, V_init, W_init))
        
        # Strategy B: Structured initialization based on partial schoolbook
        rng = np.random.RandomState(42 + r)
        U_struct = np.zeros((nm, r))
        V_struct = np.zeros((mp, r))
        W_struct = np.zeros((pn, r))
        
        # Fill with schoolbook pattern up to r columns
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U_struct[i*m + j, col] = 1
                        V_struct[j*p + k, col] = 1
                        W_struct[k*n + i, col] = 1
                        col += 1
                    else:
                        break
                if col >= r:
                    break
            if col >= r:
                break
        
        # Add small perturbation
        U_struct += rng.randn(nm, r) * 0.01
        V_struct += rng.randn(mp, r) * 0.01
        W_struct += rng.randn(pn, r) * 0.01
        init_strategies.append((U_struct, V_struct, W_struct))
        
        # Strategy C: Convolution-like structure
        # For the inner dimension m=4, try to use a 4-point DFT-like structure
        if r >= 16:
            U_conv = np.zeros((nm, r))
            V_conv = np.zeros((mp, r))
            W_conv = np.zeros((pn, r))
            
            # Use roots of unity for the m=4 dimension
            # w = exp(2*pi*i*j/4) for j=0,1,2,3
            roots = [1, 1j, -1, -1j]
            
            col = 0
            for i in range(n):
                for k in range(p):
                    if col + 4 <= r:
                        for j in range(m):
                            U_conv[i*m + j, col + j] = 1
                            V_conv[j*p + k, col + j] = roots[j]
                            W_conv[k*n + i, col + j] = 1
                        col += 4
                    else:
                        break
                if col >= r:
                    break
            if col >= r:
                break
            
            if col > 0:
                # Scale to make it real-valued
                U_conv += rng.randn(nm, r) * 0.01
                V_conv += rng.randn(mp, r) * 0.01
                W_conv += rng.randn(pn, r) * 0.01
                init_strategies.append((U_conv, V_conv, W_conv))
        
        # Try each initialization with gradient descent
        for U_init, V_init, W_init in init_strategies:
            if time.time() - START_TIME > TIME_LIMIT - 30:
                break
            
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            eta = 0.005
            best_err = np.inf
            patience = 0
            max_iter = 5000
            
            for it in range(max_iter):
                if time.time() - START_TIME > TIME_LIMIT - 30:
                    break
                    
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T_true - T_recon
                err = np.max(np.abs(R))
                
                if err < best_err:
                    best_err = err
                    patience = 0
                    if err < 1e-9:
                        # Try to snap to rational
                        result = try_rational_reconstruct(U, V, W)
                        if result is not None:
                            best_U, best_V, best_W = result
                            best_rank = r
                            found = True
                            break
                else:
                    patience += 1
                    if patience > 1000:
                        break
                
                # Compute gradients
                grad_U = np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = np.einsum('abc,ar,br->cr', R, U, V)
                
                # Adaptive learning rate
                if err < 1e-6:
                    eta = 0.001
                elif err < 1e-4:
                    eta = 0.003
                elif err < 1e-2:
                    eta = 0.01
                else:
                    eta = 0.02
                
                U += eta * grad_U
                V += eta * grad_V
                W += eta * grad_W
                
                # Occasional snap attempt
                if it % 500 == 0 and err < 1e-7:
                    result = try_rational_reconstruct(U, V, W)
                    if result is not None:
                        best_U, best_V, best_W = result
                        best_rank = r
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    if found:
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
        if verify(U_final, V_final, W_final):
            return {"U": U_final, "V": V_final, "W": W_final}
    
    return best_result
# EVOLVE-BLOCK-END
