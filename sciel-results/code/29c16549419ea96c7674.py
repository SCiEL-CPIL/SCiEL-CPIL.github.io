"""Improved program: analytical solution via Rayleigh quotient + Nelder-Mead fallback with high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin (evaluator kills at 210s)
    
    # Hermite polynomial values at 0 (physicists' convention)
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0

    # Variance (effective n) for each Hermite mode: <x^2> = n + 1/2
    n0 = 0.5
    n4 = 4.5
    n8 = 8.5
    n12 = 12.5

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        w0 = c0**2
        w4 = c4**2
        w8 = c8**2
        w12 = c12**2
        
        total_w = w0 + w4 + w8 + w12
        if total_w < 1e-15:
            return 1e15
            
        var_x = (w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12) / total_w
        return var_x * var_x

    # Seed coefficients
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_coeffs = seed_coeffs.copy()
    best_score = objective(best_coeffs)
    
    try:
        # Analytical approach: The objective is a Rayleigh quotient
        # We want to minimize (c^T N c) / (c^T c) where c = [c0, c4, c8, c12]
        # but c12 is linearly constrained: c12 = -(c0 + 12*c4 + 1680*c8) / 1039500
        # So we can substitute and get a 3D problem
        # The minimum of a Rayleigh quotient is the smallest eigenvalue
        
        # Build the effective 3x3 matrix after substituting c12
        # Let c = [c0, c4, c8], then c12 = A @ c where A = [-1/1039500, -12/1039500, -1680/1039500]
        # w = [c0^2, c4^2, c8^2, c12^2]
        # Objective numerator: c0^2*n0 + c4^2*n4 + c8^2*n8 + (A@c)^2*n12
        # Objective denominator: c0^2 + c4^2 + c8^2 + (A@c)^2
        
        A = np.array([-1.0/h12_0, -h4_0/h12_0, -h8_0/h12_0])
        
        # Numerator: c^T (diag([n0,n4,n8]) + n12 * outer(A,A)) c
        N_matrix = np.diag([n0, n4, n8]) + n12 * np.outer(A, A)
        
        # Denominator: c^T (I + outer(A,A)) c
        D_matrix = np.eye(3) + np.outer(A, A)
        
        # Solve generalized eigenvalue problem: N_matrix @ v = lambda * D_matrix @ v
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(N_matrix, D_matrix)
        
        # The minimum eigenvalue gives the minimum of the Rayleigh quotient
        # But we need to return the coefficients, not just the score
        # The eigenvector corresponding to the smallest eigenvalue gives the optimal direction
        min_idx = np.argmin(eigvals)
        optimal_dir = eigvecs[:, min_idx]
        
        # Scale to match the seed's scale (or any non-zero scale)
        # The objective is scale-invariant, so any non-zero scalar multiple works
        # Use a reasonable scale
        scale = 1.0
        analytical_coeffs = optimal_dir * scale
        
        analytical_score = objective(analytical_coeffs)
        if analytical_score < best_score:
            best_score = analytical_score
            best_coeffs = analytical_coeffs.copy()
        
        # Also try the negative direction
        analytical_coeffs_neg = -optimal_dir * scale
        analytical_score_neg = objective(analytical_coeffs_neg)
        if analytical_score_neg < best_score:
            best_score = analytical_score_neg
            best_coeffs = analytical_coeffs_neg.copy()
        
        # Now do a high-precision polish from the best found
        if time.time() < deadline - 10:
            from scipy.optimize import minimize
            
            # Polish with very tight tolerances
            polish_result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-17, 'adaptive': True}
            )
            
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_coeffs = polish_result.x.copy()
                
            # One more ultra-fine polish
            if time.time() < deadline - 5:
                ultra_result = minimize(
                    objective,
                    best_coeffs,
                    method='Nelder-Mead',
                    options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
                )
                
                if ultra_result.fun < best_score:
                    best_score = ultra_result.fun
                    best_coeffs = ultra_result.x.copy()
                    
    except Exception:
        # Fallback: use multi-start Nelder-Mead if analytical approach fails
        try:
            from scipy.optimize import minimize
            
            rng = np.random.RandomState(42)
            initial_points = [seed_coeffs.copy()]
            
            for scale in [0.05, 0.1, 0.2, 0.5]:
                noise = rng.normal(0, scale, size=3)
                initial_points.append(seed_coeffs + noise)
            
            structured_points = [
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([-0.5, 0.1, 0.01]),
                np.array([-1.2, 0.05, 0.001]),
            ]
            initial_points.extend(structured_points)
            
            rng.shuffle(initial_points)
            
            max_restarts = 8
            for i, x0 in enumerate(initial_points[:max_restarts]):
                if time.time() > deadline - 15:
                    break
                
                opts = {'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True}
                
                try:
                    result = minimize(objective, x0, method='Nelder-Mead', options=opts)
                    
                    if result.fun < best_score:
                        best_score = result.fun
                        best_coeffs = result.x.copy()
                        
                    if best_score <= 0.25000001:
                        break
                        
                except Exception:
                    continue
                    
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
