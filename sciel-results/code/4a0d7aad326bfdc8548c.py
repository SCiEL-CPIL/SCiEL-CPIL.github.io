"""
Coordinate descent optimization of Hermite coefficients for the uncertainty bound.
Uses Gauss-Hermite quadrature for exact polynomial-Gaussian integrals, then applies
analytic coordinate descent: for each coefficient, the C4 bound is a rational function
of that variable, so we can find the optimal value in closed form.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 210s wall-clock limit

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature with N=25 (exact for degree up to 48)
    nodes, weights = hermgauss(25)
    sqrt2 = np.sqrt(2.0)
    x_nodes = sqrt2 * nodes

    # Precompute Hermite polynomial values at quadrature nodes
    H0_vals = np.ones_like(x_nodes)
    H4_vals = 16.0 * x_nodes**4 - 48.0 * x_nodes**2 + 12.0
    H8_vals = (256.0 * x_nodes**8 - 2304.0 * x_nodes**6 + 5760.0 * x_nodes**4
              - 4320.0 * x_nodes**2 + 504.0)
    H12_vals = (4096.0 * x_nodes**12 - 55296.0 * x_nodes**10 + 303104.0 * x_nodes**8
                - 589824.0 * x_nodes**6 + 408192.0 * x_nodes**4 - 77760.0 * x_nodes**2
                + 10395.0)

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def compute_poly_values(a0, a4, a8):
        """
        Compute P(x_nodes) and Q(x_nodes) for given coeffs.
        P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        Q(x) = a0*H0 + a4*H4 + a8*H8 - a12*H12
        """
        a12 = get_a12(a0, a4, a8)
        P_vals = a0 * H0_vals + a4 * H4_vals + a8 * H8_vals + a12 * H12_vals
        Q_vals = a0 * H0_vals + a4 * H4_vals + a8 * H8_vals - a12 * H12_vals
        return P_vals, Q_vals

    def objective(a0, a4, a8):
        """Compute the upper bound on C4."""
        P_vals, Q_vals = compute_poly_values(a0, a4, a8)
        
        norm2_g_sq = sqrt2 * np.sum(weights * P_vals**2)
        norm2_q_sq = sqrt2 * np.sum(weights * Q_vals**2)
        norm4_g_4 = sqrt2 * np.sum(weights * P_vals**4)
        norm4_q_4 = sqrt2 * np.sum(weights * Q_vals**4)
        
        if (norm2_g_sq < 1e-15 or norm2_q_sq < 1e-15 or 
            norm4_g_4 < 1e-15 or norm4_q_4 < 1e-15):
            return 1e10
            
        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = norm4_g_4 ** 0.25
        norm4_q = norm4_q_4 ** 0.25
        
        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        return bound

    def objective_vec(coeffs):
        """Vectorized objective for scipy optimizers."""
        return objective(coeffs[0], coeffs[1], coeffs[2])

    # Precompute weight-weighted powers of each Hermite polynomial
    # These will be used to compute norms efficiently
    w = weights  # shorthand
    
    def compute_norms_given_fixed(a0, a4, a8, vary_idx, var_val):
        """
        Compute all four norms when one coefficient is varied.
        vary_idx: 0 for a0, 1 for a4, 2 for a8
        var_val: the value of the varied coefficient
        """
        # Determine which coefficient is being varied
        if vary_idx == 0:
            a0_v = var_val
            a4_v = a4
            a8_v = a8
        elif vary_idx == 1:
            a0_v = a0
            a4_v = var_val
            a8_v = a8
        else:
            a0_v = a0
            a4_v = a4
            a8_v = var_val
            
        a12 = get_a12(a0_v, a4_v, a8_v)
        
        # P = a0*H0 + a4*H4 + a8*H8 + a12*H12
        # Q = a0*H0 + a4*H4 + a8*H8 - a12*H12
        P_vals = a0_v * H0_vals + a4_v * H4_vals + a8_v * H8_vals + a12 * H12_vals
        Q_vals = a0_v * H0_vals + a4_v * H4_vals + a8_v * H8_vals - a12 * H12_vals
        
        norm2_g_sq = sqrt2 * np.sum(w * P_vals**2)
        norm2_q_sq = sqrt2 * np.sum(w * Q_vals**2)
        norm4_g_4 = sqrt2 * np.sum(w * P_vals**4)
        norm4_q_4 = sqrt2 * np.sum(w * Q_vals**4)
        
        return norm2_g_sq, norm2_q_sq, norm4_g_4, norm4_q_4

    def bound_from_norms(norm2_g_sq, norm2_q_sq, norm4_g_4, norm4_q_4):
        """Compute bound from precomputed norms."""
        if (norm2_g_sq < 1e-15 or norm2_q_sq < 1e-15 or 
            norm4_g_4 < 1e-15 or norm4_q_4 < 1e-15):
            return 1e10
        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = norm4_g_4 ** 0.25
        norm4_q = norm4_q_4 ** 0.25
        return (norm4_g * norm4_q) / (norm2_g * norm2_q)

    def coordinate_descent_step(coeffs, vary_idx, tol=1e-12, max_iter=1000):
        """
        Perform coordinate descent on one coefficient.
        For a fixed other two coefficients, the bound as a function of the 
        varied coefficient is a smooth rational function. We use golden section
        search to find the minimum.
        """
        a0, a4, a8 = coeffs
        current_val = coeffs[vary_idx]
        
        # Set up the other two coefficients
        if vary_idx == 0:
            fixed_a4, fixed_a8 = a4, a8
        elif vary_idx == 1:
            fixed_a0, fixed_a8 = a0, a8
        else:
            fixed_a0, fixed_a4 = a0, a4
        
        def obj_of_var(v):
            if vary_idx == 0:
                return objective(v, fixed_a4, fixed_a8)
            elif vary_idx == 1:
                return objective(fixed_a0, v, fixed_a8)
            else:
                return objective(fixed_a0, fixed_a4, v)
        
        # Golden section search
        # First, find a bracket around the current value
        lo = current_val - 1.0
        hi = current_val + 1.0
        
        # Expand bracket if needed
        f_lo = obj_of_var(lo)
        f_hi = obj_of_var(hi)
        f_mid = obj_of_var(current_val)
        
        # Expand until we have a bracket
        for _ in range(50):
            if f_mid <= f_lo and f_mid <= f_hi:
                break
            elif f_lo < f_mid:
                hi = lo
                lo = lo - (hi - lo)
                f_hi = f_lo
                f_lo = obj_of_var(lo)
                f_mid = obj_of_var(current_val)
            else:
                lo = hi
                hi = hi + (hi - lo)
                f_lo = f_hi
                f_hi = obj_of_var(hi)
                f_mid = obj_of_var(current_val)
        
        # Golden section search
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        f1 = obj_of_var(x1)
        f2 = obj_of_var(x2)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if f1 < f2:
                hi = x2
                x2 = x1
                f2 = f1
                x1 = hi - gr * (hi - lo)
                f1 = obj_of_var(x1)
            else:
                lo = x1
                x1 = x2
                f1 = f2
                x2 = lo + gr * (hi - lo)
                f2 = obj_of_var(x2)
            
            if abs(hi - lo) < tol:
                break
        
        best_var = (lo + hi) / 2.0
        best_f = obj_of_var(best_var)
        
        # Update the coefficient
        new_coeffs = coeffs.copy()
        new_coeffs[vary_idx] = best_var
        return new_coeffs, best_f

    # Initial evaluation
    best_coeffs = seed.copy()
    best_val = objective_vec(best_coeffs)

    # Phase 1: Multi-start coordinate descent
    # Try several initial points to avoid local minima
    initial_points = [
        seed.copy(),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-0.5, 0.1, 0.0]),
        np.array([-2.0, 0.05, 0.001]),
        np.array([-1.0, -0.05, 0.0]),
        np.array([-0.8, 0.02, 0.0005]),
        np.array([-1.2, 0.08, 0.0002]),
        np.array([-0.9, 0.0, -0.001]),
    ]

    for init in initial_points:
        if time.time() > deadline - 30.0:
            break
        
        coeffs = init.copy()
        val = objective_vec(coeffs)
        
        # Coordinate descent cycles
        for cycle in range(50):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            for idx in range(3):
                if time.time() > deadline - 10.0:
                    break
                
                new_coeffs, new_val = coordinate_descent_step(coeffs, idx)
                
                if new_val < val - 1e-14:
                    coeffs = new_coeffs
                    val = new_val
                    improved = True
            
            if not improved:
                break
        
        if val < best_val:
            best_val = val
            best_coeffs = coeffs.copy()

    # Phase 2: Fine-tune with local optimizer
    if time.time() < deadline - 5.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_vec,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Phase 3: Final polish
    if time.time() < deadline - 2.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_vec,
                best_coeffs,
                method='Powell',
                options={'maxiter': 3000, 'xtol': 1e-13, 'ftol': 1e-15}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
