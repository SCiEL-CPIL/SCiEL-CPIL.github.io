"""Growth-index fitting for CPL dark-energy cosmologies."""
import time
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import least_squares

# Deadline guard: 85s hard limit (5s margin before 90s kill)
_T0 = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _T0

def _time_left():
    return _DEADLINE - _elapsed()

def _gamma_of(c, a, Om, w0, wa):
    """Compute gamma from the 6-coefficient formula."""
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    g = (c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q)
    return g

def _f_of(c, a, Om, w0, wa):
    """Growth rate f = Om^gamma."""
    g = _gamma_of(c, a, Om, w0, wa)
    g = np.clip(g, 0.1, 1.2)
    return Om ** g

def _Om_of_x(x, Om0, w0, wa):
    """Compute Om(a) where a = exp(x)."""
    a = np.exp(x)
    integ = (1.0 + w0) * x + wa * (x - a + 1.0)
    Om_m = Om0 * np.exp(-3.0 * x)
    Om_de = (1.0 - Om0) * np.exp(-3.0 * integ)
    return Om_m / (Om_m + Om_de)

def _numerical_f_and_D(a_arr, Om0, w0, wa):
    """Solve the GR growth ODE numerically to get f and D(a)/D(1)."""
    x_arr = np.log(a_arr)
    
    def Om_and_dOm(x):
        a = np.exp(x)
        w = w0 + wa * (1.0 - a)
        integ = (1.0 + w0) * x + wa * (x - a + 1.0)
        Om_m = Om0 * np.exp(-3.0 * x)
        Om_de = (1.0 - Om0) * np.exp(-3.0 * integ)
        total = Om_m + Om_de
        Om = Om_m / total
        Om_m_prime = -3.0 * Om_m
        Om_de_prime = -3.0 * (1.0 + w) * Om_de
        dOm = (Om_m_prime * Om_de - Om_m * Om_de_prime) / (total * total)
        return Om, dOm
    
    x_start = np.log(0.01)  # a=0.01
    x_end = 0.0             # a=1
    
    def ode_f(x, y):
        f = y[0]
        Om, dOm = Om_and_dOm(x)
        return [-1.5*(1.0+Om)*f*f - 0.5*dOm*f + 1.5*Om]
    
    Om_init, _ = Om_and_dOm(x_start)
    f_init = Om_init ** 0.55
    
    sol = solve_ivp(ode_f, [x_start, x_end], [f_init], 
                    t_eval=x_arr, method='RK45', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        f_num = np.array([_Om_of_x(np.log(a), Om0, w0, wa)**0.55 for a in a_arr])
        D_num = np.ones_like(a_arr)
        return f_num, D_num
    
    f_num = sol.y[0]
    
    # Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(x) dx)
    cum_int = cumulative_trapezoid(f_num, x_arr, initial=0.0)
    total_int = cum_int[-1]
    int_to_0 = total_int - cum_int
    D_num = np.exp(-int_to_0)
    
    return f_num, D_num

def construct():
    """Return 6 coefficients for the growth-index formula."""
    # Representative CPL test cosmologies (disjoint from validation)
    test_cosmos = [
        (0.30, -1.0, 0.0),    # LCDM
        (0.30, -0.5, 0.5),    # CPL positive wa
        (0.30, -1.5, 0.5),    # CPL negative w0
        (0.30, -0.8, -0.3),   # CPL negative wa
        (0.25, -1.0, 0.3),    # Low Om, positive wa
        (0.35, -1.2, 0.2),    # High Om
        (0.30, -0.9, 0.4),    # Moderate CPL
        (0.30, -1.1, -0.2),   # Another CPL variant
    ]
    
    # Test a values - use more points for better D integration
    a_test = np.array([0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 
                        0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    # Precompute numerical solutions (expensive, do once)
    print(f"Precomputing numerical solutions... elapsed={_elapsed():.1f}s")
    num_solutions = []
    for (Om0, w0, wa) in test_cosmos:
        f_num, D_num = _numerical_f_and_D(a_test, Om0, w0, wa)
        num_solutions.append((f_num, D_num, w0, wa, Om0))
    print(f"Precomputation done. elapsed={_elapsed():.1f}s")
    
    # Build residual function for least_squares
    # Residuals: weighted f-errors and D-errors combined into one vector
    def residual_func(c):
        residuals = []
        for idx, (f_num, D_num, w0, wa, Om0) in enumerate(num_solutions):
            a_arr = a_test
            Om_arr = np.array([_Om_of_x(np.log(a), Om0, w0, wa) for a in a_arr])
            
            f_model = _f_of(c, a_arr, Om_arr, w0, wa)
            
            # f residuals (relative error)
            valid_f = np.abs(f_num) > 1e-10
            if np.any(valid_f):
                err_f = (f_model[valid_f] - f_num[valid_f]) / np.abs(f_num[valid_f])
                # Weight f residuals moderately
                residuals.extend(err_f * 1.0)
            
            # D residuals (relative error)
            valid_D = np.abs(D_num) > 1e-10
            if np.any(valid_D):
                x_arr = np.log(a_arr)
                cum_int_m = cumulative_trapezoid(f_model, x_arr, initial=0.0)
                total_int_m = cum_int_m[-1]
                int_to_0_m = total_int_m - cum_int_m
                D_model = np.exp(-int_to_0_m)
                
                err_D = (D_model[valid_D] - D_num[valid_D]) / np.abs(D_num[valid_D])
                # Weight D residuals - they capture integrated behavior
                # Use a weight that emphasizes mid-range a values where D changes most
                residuals.extend(err_D * 1.5)
        
        return np.array(residuals)
    
    # Start with a safe constant gamma
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"Starting least_squares optimization. elapsed={_elapsed():.1f}s")
    
    # Run least_squares with bounds
    bounds = ([-10.0]*6, [10.0]*6)
    
    try:
        result = least_squares(
            residual_func,
            c0,
            bounds=bounds,
            method='trf',
            max_nfev=500,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12
        )
        best_c = result.x
        print(f"least_squares converged: {result.success}, cost={result.cost:.6f}")
        print(f"  nfev={result.nfev}, elapsed={_elapsed():.1f}s")
    except Exception as e:
        print(f"least_squares failed: {e}, using initial guess")
        best_c = c0.copy()
    
    # Refine with a second pass if time permits
    if _time_left() > 15.0:
        print(f"Starting refinement pass. elapsed={_elapsed():.1f}s")
        try:
            result2 = least_squares(
                residual_func,
                best_c,
                bounds=bounds,
                method='trf',
                max_nfev=200,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14
            )
            if result2.cost < result.cost:
                best_c = result2.x
                print(f"Refinement improved: cost={result2.cost:.6f}")
            else:
                print(f"Refinement did not improve: cost={result2.cost:.6f} vs {result.cost:.6f}")
        except Exception as e:
            print(f"Refinement failed: {e}")
    
    # Clamp to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Final verification: check gamma bounds on a wider grid
    a_check = np.linspace(0.01, 1.0, 50)
    w0_check = np.linspace(-1.5, 0.5, 15)
    wa_check = np.linspace(-0.5, 0.5, 15)
    
    gamma_ok = True
    for a in a_check:
        for w0 in w0_check:
            for wa in wa_check:
                Om = _Om_of_x(np.log(a), 0.30, w0, wa)
                g = _gamma_of(best_c, a, Om, w0, wa)
                if g < 0.1 - 1e-6 or g > 1.2 + 1e-6:
                    gamma_ok = False
                    break
            if not gamma_ok:
                break
        if not gamma_ok:
            break
    
    if not gamma_ok:
        print("Warning: gamma out of bounds, reverting to safe constant gamma")
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Compute final max relative error for reporting
    max_err = 0.0
    for (f_num, D_num, w0, wa, Om0) in num_solutions:
        a_arr = a_test
        Om_arr = np.array([_Om_of_x(np.log(a), Om0, w0, wa) for a in a_arr])
        f_model = _f_of(best_c, a_arr, Om_arr, w0, wa)
        
        valid_f = np.abs(f_num) > 1e-10
        if np.any(valid_f):
            err_f = np.abs(f_model[valid_f] - f_num[valid_f]) / np.abs(f_num[valid_f])
            max_err = max(max_err, np.max(err_f))
        
        valid_D = np.abs(D_num) > 1e-10
        if np.any(valid_D):
            x_arr = np.log(a_arr)
            cum_int_m = cumulative_trapezoid(f_model, x_arr, initial=0.0)
            total_int_m = cum_int_m[-1]
            int_to_0_m = total_int_m - cum_int_m
            D_model = np.exp(-int_to_0_m)
            err_D = np.abs(D_model[valid_D] - D_num[valid_D]) / np.abs(D_num[valid_D])
            max_err = max(max_err, np.max(err_D))
    
    print(f"Final coefficients: {best_c}")
    print(f"Final max relative error: {max_err:.6f}")
    print(f"Total elapsed: {_elapsed():.1f}s")
    
    return list(best_c)
