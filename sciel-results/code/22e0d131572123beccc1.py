"""Compact dark-energy growth approximation: six-coefficient gamma model."""
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    # Pre-compute reference data for a set of training cosmologies
    # This "new approach" uses numerical fitting on a representative subset
    # to find the optimal coefficients for the 6-parameter ansatz.
    
    cosmos = [
        (0.3, -1.0, 0.0),   # LambdaCDM
        (0.3, -1.2, 0.0),   # Phantom
        (0.3, -0.5, 0.0),   # Quintessence
        (0.3, -1.0, -1.0),  # CPL negative wa
        (0.3, -1.0, 1.0),   # CPL positive wa
        (0.25, -1.0, 0.0),  # Different Om
    ]
    
    x_grid = np.linspace(np.log(0.1), 0.0, 100)
    a_grid = np.exp(x_grid)
    
    ref_data = []
    
    def compute_reference(Om0, w0, wa):
        # RK4 integration of D'' + (3 + 0.5 dlnE2/dx) D' - 1.5 Om(a) D = 0
        N = len(x_grid)
        h = x_grid[1] - x_grid[0]
        D = np.zeros(N)
        Dp = np.zeros(N)
        D[0] = 1.0
        Dp[0] = 0.55 # Initial guess for f at a=0.1
        
        for i in range(N-1):
            x = x_grid[i]
            a = a_grid[i]
            u = 1.0 + w0 + wa * (1.0 - a)
            term_m = Om0 * np.exp(-3.0 * x)
            term_de = (1.0 - Om0) * np.exp(-3.0 * u * x)
            E2 = term_m + term_de
            dE2 = -3.0 * term_m - 3.0 * u * term_de
            coeff = 3.0 + 0.5 * (dE2 / E2)
            Om_a = term_m / E2
            
            def rhs(D, Dp):
                Dpp = -coeff * Dp + 1.5 * Om_a * D
                return [Dp, Dpp]
            
            k1 = rhs(D[i], Dp[i])
            k2 = rhs(D[i] + 0.5*h*k1[0], Dp[i] + 0.5*h*k1[1])
            k3 = rhs(D[i] + 0.5*h*k2[0], Dp[i] + 0.5*h*k2[1])
            k4 = rhs(D[i] + h*k3[0], Dp[i] + h*k3[1])
            
            D[i+1] = D[i] + h/6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            Dp[i+1] = Dp[i] + h/6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            
        # Normalize to D(1) = 1
        D = D / D[-1]
        Dp = Dp / D[-1]
        return D, Dp, a_grid, x_grid

    for Om0, w0, wa in cosmos:
        D_ref, f_ref, a_g, x_g = compute_reference(Om0, w0, wa)
        # Store parameters needed for approximation
        ref_data.append({
            'Om0': Om0, 'w0': w0, 'wa': wa,
            'a': a_g, 'x': x_g,
            'D_ref': D_ref, 'f_ref': f_ref
        })

    def cost_function(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for data in ref_data:
            Om0 = data['Om0']
            w0 = data['w0']
            wa = data['wa']
            a_g = data['a']
            x_g = data['x']
            D_ref = data['D_ref']
            f_ref = data['f_ref']
            
            # Calculate q and u
            u = 1.0 + w0 + wa * (1.0 - a_g)
            
            # Calculate Om(a) for q
            # We can re-calculate Om(a) from a_g
            term_m = Om0 * np.exp(-3.0 * x_g)
            term_de = (1.0 - Om0) * np.exp(-3.0 * u * x_g)
            E2 = term_m + term_de
            Om_a = term_m / E2
            q = 1.0 - Om_a
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_g + c5*u*q
            
            # Ensure gamma is in valid range for f calculation
            # If out of range, error is high
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            
            f_approx = Om_a ** gamma_clipped
            
            # Error in f
            err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
            max_err_f = np.max(err_f)
            
            # Error in D
            # D_approx(x) = exp(-int_x^0 f_approx dt)
            # We compute D_approx by integrating f_approx backwards from 0 to x
            # Since we have a grid, we can use cumulative trapezoidal integration
            # Note: x goes from log(0.1) to 0.
            # We want D(a)/D(1). D(1)=1.
            # D(x) = exp(-int_x^0 f(t) dt)
            # Let's compute integral from x to 0.
            # int_x^0 f(t) dt = int_{x_grid[i]}^{0} f(t) dt
            # We can compute cumulative sum of f * dx from the end backwards.
            
            dx = x_g[1] - x_g[0]
            # f_approx is on x_g
            # Int = cumsum from right to left
            # Using trapezoidal rule
            f_trap = 0.5 * (f_approx[:-1] + f_approx[1:]) * dx
            # int from x_i to 0 is sum of f_trap from i to end
            # Let's reverse
            f_trap_rev = f_trap[::-1]
            cum_int_rev = np.cumsum(f_trap_rev)
            # Add the last half step? 
            # Simpler: 
            # integral[i] = sum_{j=i}^{N-1} 0.5*(f[j]+f[j+1])*dx
            # Let's just use np.trapz on slices? Too slow.
            
            # Vectorized cumulative trapezoidal integration from right to left
            # int_i = 0.5*f[i]*dx + sum_{j=i+1}^{N-2} f[j]*dx + 0.5*f[N-1]*dx
            # This is roughly cumsum(f)*dx adjusted.
            
            # Let's use a simple cumulative sum of f * dx, assuming dx is constant
            # It's an approximation, but good enough for fitting.
            # Actually, let's just compute D_approx via exp(-cumulative_integral)
            
            # Reverse f and dx
            f_rev = f_approx[::-1]
            dx_rev = np.full_like(x_g, dx)
            
            # Trapezoidal cumulative
            # int_rev[0] = 0
            # int_rev[k] = int_rev[k-1] + 0.5*(f_rev[k-1]+f_rev[k])*dx
            int_rev = np.zeros_like(f_approx)
            for i in range(1, len(f_approx)):
                int_rev[i] = int_rev[i-1] + 0.5 * (f_rev[i-1] + f_rev[i]) * dx
                
            # int_rev[i] is integral from x_N-1 to x_N-1-i
            # We want integral from x_i to x_N-1 (which is 0)
            # So integral_from_x_i_to_0 = int_rev[N-1-i]
            int_approx = int_rev[::-1]
            
            D_approx = np.exp(-int_approx)
            
            err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
            max_err_D = np.max(err_D)
            
            local_max_err = max(max_err_f, max_err_D)
            if local_max_err > max_err:
                max_err = local_max_err
                
        return max_err

    # Initial guess
    x0 = [0.55, -0.05, 0.0, -0.02, 0.0, 0.0]
    
    # Optimize
    # Use Nelder-Mead
    res = minimize(cost_function, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    return res.x
# EVOLVE-BLOCK-END
