import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    deadline = _t0 + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 26 solution
    def get_rank_26():
        rank = 26
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_term(u_idx, v_idx, w_idx, coeff):
            nonlocal col
            U[u_idx][col] += coeff
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

        # Strassen 2x2
        U[0][col] = 1; U[4][col] = 1; V[0][col] = 1; V[4][col] = 1; W[0][col] = 1; W[4][col] = 1; col += 1
        U[3][col] = 1; U[4][col] = 1; V[0][col] = 1; W[1][col] = 1; W[4][col] = -1; col += 1
        U[0][col] = 1; V[1][col] = 1; V[4][col] = -1; W[3][col] = 1; W[4][col] = 1; col += 1
        U[4][col] = 1; V[3][col] = 1; V[0][col] = -1; W[0][col] = 1; W[1][col] = 1; col += 1
        U[0][col] = 1; U[1][col] = 1; V[4][col] = 1; W[0][col] = -1; W[3][col] = 1; col += 1
        U[3][col] = 1; U[0][col] = -1; V[0][col] = 1; V[1][col] = 1; W[4][col] = 1; col += 1
        U[1][col] = 1; U[4][col] = -1; V[3][col] = 1; V[4][col] = 1; W[0][col] = 1; col += 1
        
        # Linear terms
        add_term(2, 6, 0, 1)
        add_term(2, 7, 3, 1)
        add_term(5, 6, 1, 1)
        add_term(5, 7, 4, 1)
        
        add_term(0, 2, 6, 1)
        add_term(1, 5, 6, 1)
        add_term(3, 2, 7, 1)
        add_term(4, 5, 7, 1)
        
        add_term(2, 8, 6, 1)
        add_term(5, 8, 7, 1)
        
        add_term(6, 0, 2, 1)
        add_term(7, 3, 2, 1)
        add_term(6, 1, 5, 1)
        add_term(7, 4, 5, 1)
        
        add_term(8, 6, 2, 1)
        add_term(8, 7, 5, 1)
        
        add_term(6, 2, 8, 1)
        add_term(7, 5, 8, 1)
        
        add_term(8, 8, 8, 1)
        
        return {"U": U, "V": V, "W": W}
    
    # Construct target tensor T
    T = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[c, a, b] = 1.0
    
    T_flat = T.flatten()
    
    def verify_solution(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        T_calc = np.einsum('ar,br,cr->cab', U, V, W)
        return np.allclose(T_calc, T, atol=atol)
    
    def try_rank(rank, max_time=100):
        """Try to find a rank-r decomposition."""
        start = time.time()
        
        def residuals(x):
            U = x[0:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:27*rank].reshape(9, rank)
            T_calc = np.einsum('ar,br,cr->cab', U, V, W)
            return T_calc.flatten() - T_flat
        
        best_result = None
        best_cost = float('inf')
        
        # Multiple random initializations with different strategies
        seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999]
        
        for seed in seeds:
            if time.time() - start > max_time:
                break
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            
            # Strategy 1: Small random initialization
            x0 = np.random.randn(27 * rank) * 0.3
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
            except Exception:
                continue
        
        # Strategy 2: Structured initialization based on Strassen-like patterns
        if time.time() - start < max_time - 5:
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                np.random.seed(10000 + attempt)
                
                # Start with Strassen-like structure and add small perturbations
                U0 = np.zeros((9, rank))
                V0 = np.zeros((9, rank))
                W0 = np.zeros((9, rank))
                
                # Fill first 7 columns with Strassen pattern
                U0[0,0] = 1; U0[4,0] = 1; V0[0,0] = 1; V0[4,0] = 1; W0[0,0] = 1; W0[4,0] = 1
                U0[3,1] = 1; U0[4,1] = 1; V0[0,1] = 1; W0[1,1] = 1; W0[4,1] = -1
                U0[0,2] = 1; V0[1,2] = 1; V0[4,2] = -1; W0[3,2] = 1; W0[4,2] = 1
                U0[4,3] = 1; V0[3,3] = 1; V0[0,3] = -1; W0[0,3] = 1; W0[1,3] = 1
                U0[0,4] = 1; U0[1,4] = 1; V0[4,4] = 1; W0[0,4] = -1; W0[3,4] = 1
                U0[3,5] = 1; U0[0,5] = -1; V0[0,5] = 1; V0[1,5] = 1; W0[4,5] = 1
                U0[1,6] = 1; U0[4,6] = -1; V0[3,6] = 1; V0[4,6] = 1; W0[0,6] = 1
                
                # Add random perturbations to remaining columns
                for r in range(7, rank):
                    U0[:, r] = np.random.randn(9) * 0.5
                    V0[:, r] = np.random.randn(9) * 0.5
                    W0[:, r] = np.random.randn(9) * 0.5
                
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                
                try:
                    result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                    if result.cost < best_cost:
                        best_cost = result.cost
                        best_result = result
                except Exception:
                    continue
        
        if best_result is not None and best_cost < 1e-8:
            U_opt = best_result.x[0:9*rank].reshape(9, rank)
            V_opt = best_result.x[9*rank:18*rank].reshape(9, rank)
            W_opt = best_result.x[18*rank:27*rank].reshape(9, rank)
            
            # Round to nearest half-integer
            U_rounded = np.round(U_opt * 2) / 2.0
            V_rounded = np.round(V_opt * 2) / 2.0
            W_rounded = np.round(W_opt * 2) / 2.0
            
            # Verify exactness
            if verify_solution(U_rounded, V_rounded, W_rounded, atol=1e-9):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Try rounding to integers if half-integers don't work
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            if verify_solution(U_int, V_int, W_int, atol=1e-9):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
        
        return None
    
    # Try rank 23 first (known optimal)
    result_23 = try_rank(23, max_time=60)
    if result_23 is not None:
        return result_23
    
    # Try rank 24 as intermediate
    if time.time() < deadline - 20:
        result_24 = try_rank(24, max_time=40)
        if result_24 is not None:
            return result_24
    
    # Fallback to rank 26
    return get_rank_26()
# EVOLVE-BLOCK-END
