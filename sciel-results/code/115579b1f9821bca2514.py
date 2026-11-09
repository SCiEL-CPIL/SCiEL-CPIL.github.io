"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 210s kill

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

    # Precompute constants for the constraint: c12 = -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0
    a12_0 = h0_0 / h12_0
    a12_4 = h4_0 / h12_0
    a12_8 = h8_0 / h12_0

    def get_c12(c0, c4, c8):
        return -(c0 * a12_0 + c4 * a12_4 + c8 * a12_8)

    def objective_and_grad(coeffs):
        """Compute objective value and exact gradient."""
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Weights (squared coefficients)
        w0 = c0**2
        w4 = c4**2
        w8 = c8**2
        w12 = c12**2
        
        total_w = w0 + w4 + w8 + w12
        if total_w < 1e-15:
            return 1e15, np.zeros(3)
            
        # Variance (effective n)
        var_x = (w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12) / total_w
        obj = var_x * var_x

        # Exact gradient computation
        # Let S = w0 + w4 + w8 + w12, N = w0*n0 + w4*n4 + w8*n8 + w12*n12
        # f = (N/S)^2
        # df/dc_i = 2*(N/S) * (N_i/S - N*S_i/S^2) = 2*(N/S) * (N_i*S - N*S_i)/S^2
        # where S_i = dS/dc_i, N_i = dN/dc_i
        
        # dc12/dc0 = -a12_0, dc12/dc4 = -a12_4, dc12/dc8 = -a12_8
        dc12_dc = np.array([-a12_0, -a12_4, -a12_8])
        
        # dS/dc_i for i in {0,4,8}:
        # S = c0^2 + c4^2 + c8^2 + c12^2
        # dS/dc0 = 2*c0 + 2*c12*(-a12_0)
        # dS/dc4 = 2*c4 + 2*c12*(-a12_4)
        # dS/dc8 = 2*c8 + 2*c12*(-a12_8)
        dS_dc = np.array([
            2*c0 + 2*c12*(-a12_0),
            2*c4 + 2*c12*(-a12_4),
            2*c8 + 2*c12*(-a12_8)
        ])
        
        # dN/dc_i:
        # N = c0^2*n0 + c4^2*n4 + c8^2*n8 + c12^2*n12
        # dN/dc0 = 2*c0*n0 + 2*c12*n12*(-a12_0)
        dN_dc = np.array([
            2*c0*n0 + 2*c12*n12*(-a12_0),
            2*c4*n4 + 2*c12*n12*(-a12_4),
            2*c8*n8 + 2*c12*n12*(-a12_8)
        ])
        
        N = w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12
        S = total_w
        
        # grad = 2*(N/S) * (dN_dc*S - N*dS_dc) / S^2
        grad = 2.0 * (N / S) * (dN_dc * S - N * dS_dc) / (S * S)
        
        return obj, grad

    def objective_only(coeffs):
        val, _ = objective_and_grad(coeffs)
        return val

    # Seed coefficients
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_coeffs = seed_coeffs.copy()
    best_score = objective_only(best_coeffs)
    
    try:
        from scipy.optimize import minimize, minimize_scalar
        import numpy as np
        
        # Strategy: Use L-BFGS-B with exact gradients for efficiency
        # Multiple restarts with diverse initial points
        
        rng = np.random.RandomState(12345)
        
        # Build diverse initial points
        initial_points = [seed_coeffs.copy()]
        
        # Perturbed seeds at various scales
        for scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for _ in range(2):
                noise = rng.normal(0, scale, size=3)
                initial_points.append(seed_coeffs + noise)
        
        # Structured points emphasizing different modes
        structured_points = [
            np.array([0.0, 0.0, 0.0]),           # Pure H12 (c12=0)
            np.array([1.0, 0.0, 0.0]),           # Emphasize H0
            np.array([-1.0, 0.0, 0.0]),          # Negative H0
            np.array([0.0, 1.0, 0.0]),           # Emphasize H4
            np.array([0.0, 0.0, 1.0]),           # Emphasize H8
            np.array([-0.5, 0.1, 0.01]),
            np.array([-1.2, 0.05, 0.001]),
            np.array([-1.0, 0.04, 0.003]),
            np.array([-0.8, 0.06, 0.005]),
            np.array([-1.5, 0.02, 0.0005]),
            np.array([-0.3, 0.15, 0.02]),
        ]
        initial_points.extend(structured_points)
        
        # Deduplicate approximately
        unique_points = []
        for p in initial_points:
            is_dup = False
            for q in unique_points:
                if np.linalg.norm(p - q) < 1e-6:
                    is_dup = True
                    break
            if not is_dup:
                unique_points.append(p)
        
        initial_points = unique_points
        rng.shuffle(initial_points)
        
        max_restarts = min(20, len(initial_points))
        
        for i, x0 in enumerate(initial_points[:max_restarts]):
            if time.time() > deadline - 15:
                break
            
            # Use L-BFGS-B with exact gradient for first-order efficiency
            try:
                result = minimize(
                    objective_only,
                    x0,
                    jac=lambda c: objective_and_grad(c)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x.copy()
                    
                    # Early exit if very close to theoretical minimum
                    if best_score <= 0.250000001:
                        break
                        
            except Exception:
                # Fallback to Nelder-Mead if gradient method fails
                try:
                    result = minimize(
                        objective_only,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True}
                    )
                    if result.fun < best_score:
                        best_score = result.fun
                        best_coeffs = result.x.copy()
                        if best_score <= 0.250000001:
                            break
                except Exception:
                    continue
        
        # Final high-precision polish with tighter tolerances
        if time.time() < deadline - 8:
            try:
                polish_result = minimize(
                    objective_only,
                    best_coeffs,
                    jac=lambda c: objective_and_grad(c)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
                )
                if polish_result.fun < best_score:
                    best_score = polish_result.fun
                    best_coeffs = polish_result.x.copy()
            except Exception:
                pass
        
        # Ultra-fine final polish
        if time.time() < deadline - 3:
            try:
                ultra_result = minimize(
                    objective_only,
                    best_coeffs,
                    jac=lambda c: objective_and_grad(c)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 200}
                )
                if ultra_result.fun < best_score:
                    best_score = ultra_result.fun
                    best_coeffs = ultra_result.x.copy()
            except Exception:
                pass
        
    except Exception:
        pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
