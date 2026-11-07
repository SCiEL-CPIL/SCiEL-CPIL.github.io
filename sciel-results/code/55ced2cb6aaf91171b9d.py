"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin below 150s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_grid = resource_grid
    n_m = len(m_grid)

    # Define the domain in log-space for Chebyshev approximation
    # m ranges from 0.001 to 40
    log_m_min = np.log(0.001)
    log_m_max = np.log(40)
    
    # Number of Chebyshev nodes
    n_cheb = 128
    
    # Chebyshev nodes in [-1, 1]
    # Avoid endpoints to prevent singularities in some derivative calculations if needed, 
    # but standard Chebyshev includes endpoints.
    # We map these to log_m domain.
    k = np.arange(n_cheb)
    # Standard Chebyshev points of first kind: cos(pi * (2*k + 1) / (2*n))
    # This avoids endpoints which can be good for stability if we have boundary issues, 
    # but for interpolation on a closed interval, endpoints are usually included.
    # Let's use standard Chebyshev points including endpoints: cos(pi * k / (n-1))
    # Actually, for spectral collocation/interpolation, nodes are usually cos(pi*k/(n-1)).
    # Let's stick to standard nodes.
    x_cheb = np.cos(np.pi * k / (n_cheb - 1))
    
    # Map to log_m domain: log_m = (log_m_min + log_m_max)/2 + (log_m_max - log_m_min)/2 * x
    log_m_cheb = (log_m_min + log_m_max) / 2 + (log_m_max - log_m_min) / 2 * x_cheb
    m_cheb = np.exp(log_m_cheb)
    
    # Sort m_cheb in ascending order for interpolation routines if needed, 
    # but Chebyshev coefficients are defined on the standard node ordering.
    # We will store values at these nodes.
    
    # Utility functions
    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def get_utility_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Value function representation:
    # V[t, z] will be represented by Chebyshev coefficients in log(m).
    # We need to evaluate V and its derivative V' at arbitrary m.
    
    # Precompute Chebyshev polynomial evaluation matrices?
    # Or just evaluate on the fly?
    # For Newton's method, we need V'(m_next).
    # V'(m) = dV/dlog_m * (1/m).
    
    # Let's store V values at the Chebyshev nodes for each t, z.
    # V_vals[t, z, j] corresponds to m_cheb[j].
    V_vals = np.zeros((horizon, 2, n_cheb))
    c_opt = np.zeros((horizon, 2, n_m))
    
    # Terminal period: c = m
    # V[horizon-1, z, m] = U(m)
    for z in range(2):
        V_vals[horizon - 1, z, :] = get_utility(m_cheb, gamma)
        
    # We need a function to evaluate V and dV/dm given coefficients or values at nodes.
    # Since we are doing backward induction, we have V_{t+1} values at nodes.
    # We can compute Chebyshev coefficients for V_{t+1} to evaluate derivatives?
    # Or use finite differences on the Chebyshev grid?
    # Differentiating Chebyshev series is efficient.
    
    # Let's compute Chebyshev coefficients for V_{t+1} at each step.
    # Coefficients a_k such that V(x) = sum a_k T_k(x).
    # We can use the discrete cosine transform (DCT) to get coefficients from values at Chebyshev nodes.
    # scipy.fft.dct can be used.
    # Type II DCT is suitable for Chebyshev nodes?
    # Actually, numpy doesn't have DCT built-in, but scipy.fft does.
    
    from scipy.fft import dct, idct
    
    # To get coefficients from values at Chebyshev nodes x_j = cos(pi*j/(n-1)):
    # The relationship is: a_k = (2/n) * sum_{j=0}^{n-1} ' f(x_j) T_k(x_j) with prime on j=0 and j=n-1.
    # This is essentially a DCT-I or similar.
    # Let's use a simpler approach: 
    # We don't strictly need the coefficients if we can interpolate.
    # But for Newton's method, we need V'(m).
    # We can approximate V'(m) by differentiating the Chebyshev interpolant.
    
    # Alternative: 
    # Use the fact that the problem is smooth.
    # We can compute the derivative of the Chebyshev interpolant analytically.
    
    # Let's precompute the Chebyshev coefficient matrix or just use the fact that 
    # V'(x) = sum k * a_k * U_{k-1}(x) ... this is getting complex to implement from scratch efficiently.
    
    # Simpler Structural Change:
    # Instead of full Chebyshev coefficients, use **High-Order Polynomial Interpolation (PCHIP is piecewise cubic, let's use global high-order?)**
    # No, the directive says Chebyshev spectral collocation.
    
    # Let's implement the Chebyshev derivative evaluation.
    # Given values f_j at Chebyshev nodes x_j, we can compute the derivative f'(x) at any x.
    # There is a direct formula for the derivative of the Chebyshev interpolant.
    # Or, we can compute the coefficients a_k.
    
    # Coefficients from values at Chebyshev nodes (including endpoints):
    # a_k = (2/n) * sum_{j=0}^{n-1} f_j T_k(x_j) * w_j
    # where w_0 = w_{n-1} = 1/2, w_j = 1 for j=1..n-2.
    # This is equivalent to a DCT-I.
    
    # Let's use scipy.fft.dct(type=1) ? 
    # scipy.fft.dct(x, type=1, n=None, norm=None)
    # The DCT-I is defined as: Y_k = X_0 + (-1)^k X_{N-1} + 2 * sum_{j=1}^{N-2} X_j cos(pi*k*j/(N-1))
    # The Chebyshev coefficients are related to DCT-I.
    # Specifically, if we have values f_j at x_j = cos(pi*j/(N-1)),
    # The Chebyshev series coefficients c_k (where f(x) = 1/2 c_0 + sum c_k T_k(x)) are:
    # c_k = (2/N) * Y_k ? 
    # Let's check scaling.
    # DCT-I output Y_k.
    # c_k = (2/N) * Y_k for k < N-1? And c_{N-1} = (1/N) * Y_{N-1}?
    # Actually, the standard relation is:
    # f_j = 1/2 a_0 + sum_{k=1}^{N-1} a_k T_k(x_j)
    # a_k = (2/N) sum_{j=0}^{N-1} ' f_j T_k(x_j)
    # The DCT-I computes Y_k = sum_{j=0}^{N-1} ' f_j cos(pi k j / (N-1))
    # Note T_k(cos theta) = cos(k theta). So T_k(x_j) = cos(pi k j / (N-1)).
    # So Y_k is exactly the sum in the definition of a_k, but without the 2/N factor and with the prime convention handled by DCT-I definition?
    # scipy.fft.dct type=1 uses:
    # Y_k = X_0 + (-1)^k X_{N-1} + 2 * sum_{j=1}^{N-2} X_j cos(pi*k*j/(N-1))
    # This matches the primed sum: sum' means 1/2 for endpoints.
    # So Y_k = sum' f_j T_k(x_j) * 2 ? No.
    # sum' f_j T_k(x_j) = 1/2 f_0 T_k(x_0) + sum_{j=1}^{N-2} f_j T_k(x_j) + 1/2 f_{N-1} T_k(x_{N-1})
    # Y_k = 2 * (1/2 f_0 ... + ... + 1/2 f_{N-1} ...) = 2 * sum' f_j T_k(x_j)
    # So sum' f_j T_k(x_j) = Y_k / 2.
    # Then a_k = (2/N) * (Y_k / 2) = Y_k / N.
    
    # So Chebyshev coefficients a_k = dct(type=1, f) / N.
    
    def cheb_coeffs_from_vals(vals, n):
        # vals shape (n,) at Chebyshev nodes
        y = dct(vals, type=1, n=n)
        return y / n

    def cheb_eval(coeffs, x):
        # Evaluate Chebyshev series at x in [-1, 1]
        # coeffs a_k such that f(x) = sum a_k T_k(x)
        # Use Clenshaw's algorithm or just sum?
        # For n=128, summing is O(n). Clenshaw is O(n) but faster.
        # Let's use a simple vectorized sum for now, or Clenshaw.
        # Clenshaw:
        b_kp2 = 0
        b_kp1 = 0
        for k in range(len(coeffs) - 1, -1, -1):
            b_k = 2 * x * b_kp1 - b_kp2 + coeffs[k]
            b_kp2 = b_kp1
            b_kp1 = b_k
        return b_kp1 + 0.5 * coeffs[0] # Wait, standard Clenshaw for T_k assumes a0/2 term?
        # If f(x) = a0/2 + sum_{k=1} a_k T_k(x), then Clenshaw returns that.
        # My coeffs a_k are defined such that f(x) = sum a_k T_k(x).
        # So a0 is the full coefficient for T_0=1.
        # Standard Clenshaw for f(x) = sum_{k=0}^{N} a_k T_k(x):
        # b_{N+1} = b_N = 0
        # b_k = 2x b_{k+1} - b_{k+2} + a_k
        # Return b_0 + 0.5 a_0? No.
        # If we use a0 as the coefficient of T_0, then the recurrence works if we treat a0 specially?
        # Actually, if f(x) = a0 T_0(x) + a1 T_1(x) + ...,
        # Clenshaw is usually derived for c0/2 + sum c_k T_k.
        # So if I have a_k, I should set c_0 = 2*a_0, c_k = a_k for k>0.
        
        # Let's just use numpy.polynomial.chebyshev.chebval
        # It expects coefficients [a0, a1, ...] where f(x) = sum a_k T_k(x).
        return np.polynomial.chebyshev.chebval(x, coeffs)

    def cheb_deriv_eval(coeffs, x):
        # Derivative of Chebyshev series
        # d/dx T_k(x) = U_{k-1}(x)
        # We can convert Chebyshev series to derivative series or evaluate directly.
        # Easier: use numpy.polynomial.chebyshev.chebder to get derivative coefficients, then chebval.
        d_coeffs = np.polynomial.chebyshev.chebder(coeffs)
        return np.polynomial.chebyshev.chebval(x, d_coeffs)

    # Backward Induction
    # V[t, z] is represented by Chebyshev coefficients in log(m).
    # Let's store the coefficients directly to avoid recomputing from values if we want to be efficient.
    # But we need V at specific m_next points for the expectation.
    # And we need V'(m_next) for the Euler equation? 
    # Actually, the Euler equation for the optimal c is:
    # U'(c) = beta * E[ R * V'(m') ]
    # We can solve this for c using Newton's method.
    
    # We need V_{t+1}'(m) for z'=0 and z'=1.
    # Let's store Chebyshev coefficients for V_{t+1, 0} and V_{t+1, 1}.
    
    V_coeffs = np.zeros((horizon, 2, n_cheb))
    
    # Terminal period
    for z in range(2):
        V_vals_term = get_utility(m_cheb, gamma)
        V_coeffs[horizon - 1, z, :] = cheb_coeffs_from_vals(V_vals_term, n_cheb)
        # Policy is c = m
        c_opt[horizon - 1, z, :] = m_grid
    
    # Precompute x_cheb in [-1, 1] for evaluation
    # We have x_cheb already.
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback
            for tt in range(t, -1, -1):
                frac = 1.0 / (horizon - tt)
                c_opt[tt, :, :] = m_grid[np.newaxis, :] * frac
            break
        
        for z in range(2):
            # We need to find optimal c for each m in m_grid.
            # U'(c) = beta * (transition[z,0] * R * V'_{t+1,0}(m') + transition[z,1] * R * V'_{t+1,1}(m'))
            # where m' = R(m-c) + income[z']
            
            # This is a root finding problem for c for each m.
            # Let's vectorize over m_grid.
            
            # Initial guess for c:
            # For log utility, c is roughly proportional to m.
            # Let's use c0 = m / (1 + beta) or similar heuristic.
            # Or use the previous period's policy if available?
            # Let's use a simple heuristic: c = m / (horizon - t)
            c_curr = m_grid.copy() / (horizon - t)
            
            # Newton Iteration
            max_iter = 20
            tol = 1e-8
            
            for _ in range(max_iter):
                # Compute m' for z'=0 and z'=1
                m_next_0 = R * (m_grid - c_curr) + income[0]
                m_next_1 = R * (m_grid - c_curr) + income[1]
                
                # Clip m_next to domain [0.001, 40]
                m_next_0 = np.clip(m_next_0, 0.001, 40)
                m_next_1 = np.clip(m_next_1, 0.001, 40)
                
                # Convert m_next to log space for Chebyshev evaluation
                log_m_next_0 = np.log(m_next_0)
                log_m_next_1 = np.log(m_next_1)
                
                # Map log_m to x in [-1, 1]
                x_next_0 = (2 * log_m_next_0 - (log_m_min + log_m_max)) / (log_m_max - log_m_min)
                x_next_1 = (2 * log_m_next_1 - (log_m_min + log_m_max)) / (log_m_max - log_m_min)
                
                # Evaluate V'_{t+1, 0} and V'_{t+1, 1}
                # V_coeffs[t+1, 0, :] and V_coeffs[t+1, 1, :]
                
                # cheb_deriv_eval expects coeffs and x.
                # We need to handle vector x.
                # np.polynomial.chebyshev.chebval supports vector x.
                
                V_prime_next_0 = cheb_deriv_eval(V_coeffs[t+1, 0, :], x_next_0)
                V_prime_next_1 = cheb_deriv_eval(V_coeffs[t+1, 1, :], x_next_1)
                
                # Wait, cheb_deriv_eval returns dV/dx.
                # We need dV/dm = dV/dx * dx/dlog_m * dlog_m/dm
                # x = (2 log_m - (min+max)) / (max-min)
                # dx/dlog_m = 2 / (max-min)
                # dlog_m/dm = 1/m
                # So dV/dm = V_prime_x * (2 / (log_m_max - log_m_min)) * (1/m)
                
                dV_dm_0 = V_prime_next_0 * (2.0 / (log_m_max - log_m_min)) / m_next_0
                dV_dm_1 = V_prime_next_1 * (2.0 / (log_m_max - log_m_min)) / m_next_1
                
                # Expected Marginal Value
                E_V_prime = transition[z, 0] * R * dV_dm_0 + transition[z, 1] * R * dV_dm_1
                
                # Euler Equation Residual:
                # U'(c) - beta * E_V_prime = 0
                # U'(c) = c^(-gamma)
                
                U_prime_c = c_curr ** (-gamma)
                
                Residual = U_prime_c - beta * E_V_prime
                
                # Derivative of Residual w.r.t c:
                # d/dc (c^(-gamma)) - 0 = -gamma * c^(-gamma - 1)
                # Note: E_V_prime depends on c via m_next.
                # m_next = R(m-c) + inc
                # dm_next/dc = -R
                # dE_V_prime/dc = -R * (trans0 * V''_0 + trans1 * V''_1)
                # We need V'' for the Newton step?
                # If we ignore the second derivative term (secant method or simplified Newton), 
                # we might converge slower.
                # Let's include V'' if possible.
                
                # V'' in x-space:
                V_double_prime_x_0 = cheb_deriv_eval(cheb_deriv_eval(V_coeffs[t+1, 0, :], x_next_0), x_next_0) # This is wrong, chebder gives coeffs.
                # Let's compute V'' coeffs once.
                
                # Actually, let's just use a robust Secant method or Bisection if Newton is unstable.
                # Given the smoothness, Newton with full derivative is best.
                
                # Let's compute V'' coeffs for t+1.
                # V_coeffs[t+1, 0] -> V_prime_coeffs -> V_double_prime_coeffs
                
                # This is getting heavy. Let's assume the dependence of E_V_prime on c is weak 
                # or use the simplified Newton (ignoring dE/dc) first, which is a fixed-point iteration 
                # if we rearrange? No, it's a nonlinear equation.
                
                # Let's use the full Newton.
                # We need V''(m_next).
                
                # Precompute V'' coeffs for current t+1 outside the z loop?
                # No, V_coeffs[t+1] is fixed.
                
                # Let's compute V'' coeffs here.
                V_prime_coeffs_0 = np.polynomial.chebyshev.chebder(V_coeffs[t+1, 0, :])
                V_double_prime_coeffs_0 = np.polynomial.chebyshev.chebder(V_prime_coeffs_0)
                V_prime_coeffs_1 = np.polynomial.chebyshev.chebder(V_coeffs[t+1, 1, :])
                V_double_prime_coeffs_1 = np.polynomial.chebyshev.chebder(V_prime_coeffs_1)
                
                V_double_prime_x_0 = np.polynomial.chebyshev.chebval(x_next_0, V_double_prime_coeffs_0)
                V_double_prime_x_1 = np.polynomial.chebyshev.chebval(x_next_1, V_double_prime_coeffs_1)
                
                # dV''/dm = V''_x * (2/(log_m_max - log_m_min)) * (1/m)
                d2V_dm2_0 = V_double_prime_x_0 * (2.0 / (log_m_max - log_m_min)) / m_next_0
                d2V_dm2_1 = V_double_prime_x_1 * (2.0 / (log_m_max - log_m_min)) / m_next_1
                
                # dE_V_prime/dc = -R * (transition[z,0] * d2V_dm2_0 + transition[z,1] * d2V_dm2_1)
                dE_dc = -R * (transition[z, 0] * d2V_dm2_0 + transition[z, 1] * d2V_dm2_1)
                
                # dResidual/dc = -gamma * c^(-gamma-1) - beta * dE_dc
                dRes_dc = -gamma * c_curr ** (-gamma - 1) - beta * dE_dc
                
                # Newton step
                step = Residual / dRes_dc
                c_new = c_curr - step
                
                # Ensure c is valid: 0 < c <= m
                c_new = np.clip(c_new, 1e-12, m_grid)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < tol:
                    c_curr = c_new
                    break
                c_curr = c_new
                
            c_opt[t, z, :] = c_curr
            
            # Update V_coeffs[t, z]
            # V[t, z, m] = U(c) + beta * E[V_{t+1}(m')]
            # We need to evaluate this at the Chebyshev nodes m_cheb.
            
            # Re-calculate m' at m_cheb
            m_next_0_cheb = R * (m_cheb - c_opt[t, z, :][np.interp(np.log(m_cheb), np.log(m_grid), np.log(c_opt[t, z, :]))]) + income[0]
            # Wait, c_opt is defined on m_grid. We need c at m_cheb.
            # Interpolate c_opt from m_grid to m_cheb.
            
            log_c_opt = np.log(c_opt[t, z, :])
            log_m_cheb_interp = np.log(m_cheb)
            log_c_cheb = np.interp(log_m_cheb_interp, np.log(m_grid), log_c_opt)
            c_cheb = np.exp(log_c_cheb)
            
            m_next_0_cheb = R * (m_cheb - c_cheb) + income[0]
            m_next_1_cheb = R * (m_cheb - c_cheb) + income[1]
            
            m_next_0_cheb = np.clip(m_next_0_cheb, 0.001, 40)
            m_next_1_cheb = np.clip(m_next_1_cheb, 0.001, 40)
            
            log_m_next_0_cheb = np.log(m_next_0_cheb)
            log_m_next_1_cheb = np.log(m_next_1_cheb)
            
            x_next_0_cheb = (2 * log_m_next_0_cheb - (log_m_min + log_m_max)) / (log_m_max - log_m_min)
            x_next_1_cheb = (2 * log_m_next_1_cheb - (log_m_min + log_m_max)) / (log_m_max - log_m_min)
            
            V_next_0_cheb = np.polynomial.chebyshev.chebval(x_next_0_cheb, V_coeffs[t+1, 0, :])
            V_next_1_cheb = np.polynomial.chebyshev.chebval(x_next_1_cheb, V_coeffs[t+1, 1, :])
            
            E_V_next_cheb = transition[z, 0] * V_next_0_cheb + transition[z, 1] * V_next_1_cheb
            
            U_cheb = get_utility(c_cheb, gamma)
            
            V_cheb = U_cheb + beta * E_V_next_cheb
            
            V_coeffs[t, z, :] = cheb_coeffs_from_vals(V_cheb, n_cheb)

    # Final validity check
    consumption = c_opt
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    consumption = np.where(np.isfinite(consumption), consumption, resource_grid[np.newaxis, np.newaxis, :])
    
    return consumption
# EVOLVE-BLOCK-END
