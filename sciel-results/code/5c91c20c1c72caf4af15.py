"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import time
from scipy.integrate import solve_ivp

def construct():
    """
    Fit six coefficients for the growth-index formula using iterative weighted 
    least-squares to approximate the min-max (Chebyshev) objective.
    """
    start_time = time.time()
    deadline = start_time + 78  # Safety margin before 90s kill

    # Dense grid for basis construction
    # We focus on a range of w0, wa, Om0 that covers typical validation scenarios
    w0_vals = np.linspace(-1.5, -0.5, 21)
    wa_vals = np.linspace(-1.0, 1.0, 21)
    Om0_vals = np.array([0.20, 0.25, 0.27, 0.30, 0.35])
    a_vals = np.linspace(0.05, 1.0, 150)

    # Generate reference data on dense grid
    all_data = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                try:
                    f_ref, D_ref = _solve_growth_ode_fast(Om0, w0, wa, a_vals)
                    if f_ref is None:
                        continue
                    for a, f_r, D_r in zip(a_vals, f_ref, D_ref):
                        Om_a = _compute_Om_a(Om0, w0, wa, a)
                        q = 1 - Om_a
                        u = 1 + w0 + wa * (1 - a)
                        if np.isfinite(f_r) and f_r > 0 and np.isfinite(D_r) and D_r > 0:
                            all_data.append((Om_a, q, u, wa, a, f_r, D_r))
                except Exception:
                    continue

    if len(all_data) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Convert to arrays
    Om_a_arr = np.array([d[0] for d in all_data])
    q_arr = np.array([d[1] for d in all_data])
    u_arr = np.array([d[2] for d in all_data])
    wa_arr = np.array([d[3] for d in all_data])
    a_arr = np.array([d[4] for d in all_data])
    f_ref_arr = np.array([d[5] for d in all_data])
    D_ref_arr = np.array([d[6] for d in all_data])

    # Downsample for optimization speed if necessary
    # The grid is 5 * 21 * 21 * 150 = 330,750 points. This is manageable for vectorized ops.
    # However, solving ODEs took the most time. The array operations are fast.
    # Let's keep all points for better generalization, but subsample if it gets too slow.
    n_total = len(Om_a_arr)
    
    # Build design matrix for the six ansatz terms
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    X = np.column_stack([
        np.ones_like(q_arr),
        q_arr,
        q_arr**2,
        u_arr,
        wa_arr * a_arr,
        u_arr * q_arr
    ])

    # Phase 1: Initial L2 Fit to get a good starting point
    # We want to minimize ||X @ c - gamma_true||_2
    # gamma_true = log(f_ref) / log(Om_a)
    valid_om = (Om_a_arr > 0.01) & (Om_a_arr < 0.99)
    gamma_true = np.zeros_like(f_ref_arr)
    gamma_true[valid_om] = np.log(f_ref_arr[valid_om]) / np.log(Om_a_arr[valid_om])
    gamma_true[~valid_om] = 0.55
    
    # Only use valid points for initial fit
    X_valid = X[valid_om]
    gamma_valid = gamma_true[valid_om]
    
    try:
        c_init, residuals, rank, s = np.linalg.lstsq(X_valid, gamma_valid, rcond=None)
        c_init = np.clip(c_init, -10, 10)
    except:
        c_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_params = c_init.copy()
    best_score = float('inf')

    # Helper to compute errors and max error
    def compute_errors_and_score(params):
        c0, c1, c2, c3, c4, c5 = params
        gamma = c0 + c1 * q_arr + c2 * q_arr**2 + c3 * u_arr + c4 * wa_arr * a_arr + c5 * u_arr * q_arr
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            # Penalize heavily
            penalty = np.sum(np.maximum(0.1 - gamma, 0)) + np.sum(np.maximum(gamma - 1.2, 0))
            return None, None, 10.0 + penalty
        
        f_model = Om_a_arr ** gamma
        
        # Relative error for f
        rel_err_f = np.abs(f_model - f_ref_arr) / np.maximum(f_ref_arr, 1e-10)
        
        # Compute D from model
        # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # We need to compute this efficiently.
        # Since a_arr is sorted, we can compute cumulative integrals.
        # x = ln(a). dx = da/a.
        # Integral from x_i to 0 of f(x) dx.
        # Let's precompute the grid structure once outside the loop if possible, 
        # but here we just compute it directly for the current model.
        
        # To speed up, we can approximate the integral using the discrete grid.
        # D_i = exp(- sum_{j=i}^{N-1} 0.5 * (f_j + f_{j+1}) * (x_{j+1} - x_j) )
        # Note: The integral is from ln(a_i) to 0. 
        # Our grid a_arr goes from 0.05 to 1.0.
        # x_arr = ln(a_arr).
        # We need the integral from x_i to x_max (which is 0).
        
        x_arr = np.log(a_arr)
        # Ensure a_arr is sorted ascending (it is)
        dx = np.diff(x_arr)
        # Average f values for trapezoidal rule
        f_avg = 0.5 * (f_model[:-1] + f_model[1:])
        integrands = f_avg * dx
        
        # Cumulative sum from the end
        # integral_i = sum_{j=i}^{N-2} integrands[j]
        # We need to be careful with the last point (a=1.0, x=0)
        # integrands has length N-1.
        # integral for i is sum(integrands[i:])
        
        # To do this efficiently, we can use cumsum in reverse.
        # rev_integrands = integrands[::-1]
        # cumsum_rev = np.cumsum(rev_integrands)
        # integral_i = cumsum_rev[N-1-i]
        
        rev_int = integrands[::-1]
        cumsum_rev = np.cumsum(rev_int)
        
        # For index i, the number of steps to the end is N-1-i
        # The sum is the first (N-1-i) elements of cumsum_rev? No.
        # cumsum_rev[k] = sum of first k+1 elements of rev_int.
        # We want sum of integrands[i : N-1].
        # This corresponds to rev_int[0 : N-1-i].
        # So we want cumsum_rev[N-2-i] if N-1-i > 0.
        
        N = len(a_arr)
        D_model = np.ones(N)
        # For i=0 to N-2:
        # indices in cumsum_rev: 0 to N-2-i
        # Let's just loop or use a vectorized trick.
        # Actually, since N is ~150, a simple loop or vectorized slicing is fine.
        # But we are doing this inside an optimizer, so speed matters.
        
        # Vectorized approach:
        # integral_i = cumsum_rev[N-2-i] for i < N-1
        # integral_{N-1} = 0
        
        # Create index array
        # i = 0 -> idx = N-2
        # i = N-2 -> idx = 0
        # i = N-1 -> 0
        
        idx_map = np.arange(N-2, -1, -1)
        # integrals for i=0..N-2
        integrals = cumsum_rev[idx_map]
        D_model[:N-1] = np.exp(-integrals)
        D_model[N-1] = 1.0
        
        rel_err_D = np.abs(D_model - D_ref_arr) / np.maximum(D_ref_arr, 1e-10)
        
        # Combined max error
        max_err = np.maximum(np.max(rel_err_f), np.max(rel_err_D))
        
        return rel_err_f, rel_err_D, max_err

    # Initial score
    _, _, best_score = compute_errors_and_score(best_params)

    # Phase 2: Iterative Weighted Least Squares (Chebyshev Approximation)
    # We minimize || W @ (X @ c - target) ||_2 where target is derived from errors.
    # Actually, it's easier to minimize the error vector directly with weights.
    # Error vector e(c) = [rel_err_f; rel_err_D]
    # We want to minimize max(e).
    # Approximation: minimize sum( w_i * e_i^2 ) where w_i = 1/max(e_i, eps)^2 or similar.
    
    # Let's define a function for the weighted LS problem
    # We will fit gamma such that the resulting f and D errors are minimized.
    # This is non-linear in c. So we must use least_squares on the error vector.
    
    def weighted_error_vector(params, weights):
        c0, c1, c2, c3, c4, c5 = params
        gamma = c0 + c1 * q_arr + c2 * q_arr**2 + c3 * u_arr + c4 * wa_arr * a_arr + c5 * u_arr * q_arr
        
        # Soft penalty for out of bounds to keep solver stable
        # If out of bounds, return large values
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            # Return large errors
            return np.full(2 * len(q_arr), 100.0)
            
        f_model = Om_a_arr ** gamma
        
        rel_err_f = np.abs(f_model - f_ref_arr) / np.maximum(f_ref_arr, 1e-10)
        
        x_arr = np.log(a_arr)
        dx = np.diff(x_arr)
        f_avg = 0.5 * (f_model[:-1] + f_model[1:])
        integrands = f_avg * dx
        rev_int = integrands[::-1]
        cumsum_rev = np.cumsum(rev_int)
        idx_map = np.arange(len(a_arr)-2, -1, -1)
        integrals = cumsum_rev[idx_map]
        D_model = np.ones(len(a_arr))
        D_model[:-1] = np.exp(-integrals)
        
        rel_err_D = np.abs(D_model - D_ref_arr) / np.maximum(D_ref_arr, 1e-10)
        
        # Weighted errors
        # weights shape (N, 2) or (2N,)
        # We pass weights as a flat array of length 2N
        w_f, w_D = weights
        err_vec = np.concatenate([rel_err_f * w_f, rel_err_D * w_D])
        return err_vec

    # Iteration loop
    max_iter = 50
    for i_iter in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Compute current errors
        rel_err_f, rel_err_D, curr_score = compute_errors_and_score(best_params)
        if rel_err_f is None:
            break
            
        # Update weights
        # Focus on the largest errors
        # Normalize errors to [0, 1]
        max_f = np.max(rel_err_f)
        max_D = np.max(rel_err_D)
        curr_max = max(max_f, max_D)
        
        if curr_max < 1e-6:
            break
            
        # Soft weighting: w = exp(err / T)
        # Small T focuses on max errors
        T = curr_max * 0.5 
        if T < 1e-4: T = 1e-4
        
        w_f = np.exp(rel_err_f / T)
        w_D = np.exp(rel_err_D / T)
        
        # Normalize weights to avoid overflow/underflow in solver
        w_f = w_f / np.max(w_f)
        w_D = w_D / np.max(w_D)
        
        weights = (w_f, w_D)
        
        # Solve weighted least squares
        try:
            result = least_squares(
                weighted_error_vector, 
                best_params, 
                args=(weights,),
                bounds=([-10]*6, [10]*6),
                max_nfev=100,
                ftol=1e-8,
                xtol=1e-8
            )
            
            new_params = result.x
            _, _, new_score = compute_errors_and_score(new_params)
            
            if new_score < best_score:
                best_score = new_score
                best_params = new_params
            else:
                # If no improvement, try a smaller step or break
                # Sometimes the weighting oscillates.
                pass
                
        except Exception:
            break

    # Phase 3: Local Refinement
    # Try small perturbations around the best found
    if time.time() < deadline - 2:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            # Random small perturbation
            perturb = 0.01 * np.random.randn(6)
            candidate = np.clip(best_params + perturb, -10, 10)
            _, _, cand_score = compute_errors_and_score(candidate)
            if cand_score < best_score:
                best_score = cand_score
                best_params = candidate

    best_params = np.clip(best_params, -10, 10)
    return list(best_params)

def _compute_Om_a(Om0, w0, wa, a):
    """Compute matter density at scale factor a."""
    Om_m = Om0 * a**3
    w_eff = w0 + wa * (1 - a)
    Om_de = (1 - Om0) * a**(-3 * w_eff)
    return Om_m / (Om_m + Om_de)

def _solve_growth_ode_fast(Om0, w0, wa, a_values):
    """Solve the growth ODE efficiently."""
    a_min = min(a_values)
    a_max = max(a_values)
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode_system(x, y):
        D, Dp = y
        a = np.exp(x)
        Om_m = _compute_Om_a(Om0, w0, wa, a)
        coeff = (3/2) * (1 + Om_m / (1 - Om_m))
        return [Dp, -coeff * Dp + (3/2) * Om_m * D]

    x_eval = np.linspace(x_max, x_min, len(a_values))
    sol = solve_ivp(ode_system, [x_max, x_min], [1.0, 0.0], t_eval=x_eval,
                    rtol=1e-6, atol=1e-8, method='RK45')

    if not sol.success:
        return None, None

    D_vals = sol.y[0]
    Dp_vals = sol.y[1]

    # Check for invalid values
    if np.any(D_vals <= 0) or np.any(~np.isfinite(D_vals)):
        return None, None

    f_vals = Dp_vals / D_vals
    D_vals = D_vals / D_vals[0]  # Normalize D(1) = 1

    if np.any(~np.isfinite(f_vals)) or np.any(~np.isfinite(D_vals)):
        return None, None

    return f_vals, D_vals

# EVOLVE-BLOCK-END
