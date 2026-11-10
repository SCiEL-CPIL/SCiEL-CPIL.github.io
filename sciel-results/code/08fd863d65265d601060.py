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
    
    x_start = np.log(0.01)
    x_end = 0.0
    
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
    
    cum_int = cumulative_trapezoid(f_num, x_arr, initial=0.0)
    total_int = cum_int[-1]
    int_to_0 = total_int - cum_int
    D_num = np.exp(-int_to_0)
    
    return f_num, D_num

def _check_gamma_bounds(c, a_grid, w0_grid, wa_grid, Om0=0.30):
    """Check if gamma stays in [0.1, 1.2] for all combinations."""
    for a in a_grid:
        for w0 in w0_grid:
            for wa in wa_grid:
                Om = _Om_of_x(np.log(a), Om0, w0, wa)
                g = _gamma_of(c, a, Om, w0, wa)
                if g < 0.1 - 1e-6 or g > 1.2 + 1e-6:
                    return False
    return True

def _objective(c, test_cosmos, a_test, num_solutions):
    """Compute max relative error over all test cosmologies and a values."""
    max_err_f = 0.0
    max_err_D = 0.0
    
    for idx, (Om0, w0, wa) in enumerate(test_cosmos):
        f_num, D_num = num_solutions[idx]
        
        a_arr = a_test
        Om_arr = np.zeros_like(a_arr)
        for i, a in enumerate(a_arr):
            Om_arr[i] = _Om_of_x(np.log(a), Om0, w0, wa)
        
        f_model = _f_of(c, a_arr, Om_arr, w0, wa)
        
        valid = np.abs(f_num) > 1e-10
        if np.any(valid):
            err_f = np.abs(f_model[valid] - f_num[valid]) / np.abs(f_num[valid])
            max_err_f = max(max_err_f, np.max(err_f))
        
        valid_D = np.abs(D_num) > 1e-10
        if np.any(valid_D):
            x_arr = np.log(a_arr)
            cum_int_m = cumulative_trapezoid(f_model, x_arr, initial=0.0)
            total_int_m = cum_int_m[-1]
            int_to_0_m = total_int_m - cum_int_m
            D_model = np.exp(-int_to_0_m)
            
            err_D = np.abs(D_model[valid_D] - D_num[valid_D]) / np.abs(D_num[valid_D])
            max_err_D = max(max_err_D, np.max(err_D))
    
    return max(max_err_f, max_err_D)

def construct():
    """Return 6 coefficients for the growth-index formula."""
    # Dense grid of representative CPL test cosmologies
    Om0_grid = [0.25, 0.28, 0.30, 0.32, 0.35]
    w0_grid = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5]
    wa_grid = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    
    test_cosmos = []
    for Om0 in Om0_grid:
        for w0 in w0_grid:
            for wa in wa_grid:
                test_cosmos.append((Om0, w0, wa))
    
    # Dense a values for fitting
    a_test = np.linspace(0.05, 1.0, 20)
    
    # Precompute numerical solutions
    print(f"Precomputing {len(test_cosmos)} numerical solutions... elapsed={_elapsed():.1f}s")
    num_solutions = []
    for i, (Om0, w0, wa) in enumerate(test_cosmos):
        if _time_left() < 5.0:
            break
        f_num, D_num = _numerical_f_and_D(a_test, Om0, w0, wa)
        num_solutions.append((f_num, D_num))
    print(f"Precomputed {len(num_solutions)} solutions. elapsed={_elapsed():.1f}s")
    
    if len(num_solutions) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Build design matrix for least-squares fit
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # We want to fit gamma such that f = Om^gamma matches f_num
    # Take log: ln(f) = gamma * ln(Om)
    # So gamma_ref = ln(f_num) / ln(Om)
    
    all_Om = []
    all_a = []
    all_w0 = []
    all_wa = []
    all_gamma_ref = []
    
    for idx, (Om0, w0, wa) in enumerate(test_cosmos[:len(num_solutions)]):
        f_num, D_num = num_solutions[idx]
        for i, a in enumerate(a_test):
            Om = _Om_of_x(np.log(a), Om0, w0, wa)
            if Om > 1e-10 and f_num[i] > 1e-10:
                gamma_ref = np.log(f_num[i]) / np.log(Om)
                if 0.0 < gamma_ref < 2.0:  # Filter unreasonable values
                    all_Om.append(Om)
                    all_a.append(a)
                    all_w0.append(w0)
                    all_wa.append(wa)
                    all_gamma_ref.append(gamma_ref)
    
    all_Om = np.array(all_Om)
    all_a = np.array(all_a)
    all_w0 = np.array(all_w0)
    all_wa = np.array(all_wa)
    all_gamma_ref = np.array(all_gamma_ref)
    
    print(f"Built design matrix with {len(all_Om)} points. elapsed={_elapsed():.1f}s")
    
    if len(all_Om) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Build design matrix A where gamma = A @ c
    q = 1.0 - all_Om
    u = 1.0 + all_w0 + all_wa * (1.0 - all_a)
    
    A = np.column_stack([
        np.ones_like(all_Om),  # c0
        q,                      # c1
        q**2,                   # c2
        u,                      # c3
        all_wa * all_a,         # c4
        u * q                   # c5
    ])
    
    # Weighted least squares: weight by 1/Om to emphasize low-z where error matters most
    weights = 1.0 / np.maximum(all_Om, 0.01)
    weights = weights / np.sum(weights)
    
    # Solve weighted least squares
    A_w = A * np.sqrt(weights)[:, np.newaxis]
    b_w = all_gamma_ref * np.sqrt(weights)
    
    try:
        c_ls, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
        print(f"Least-squares solution: {c_ls}")
    except:
        c_ls = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Check gamma bounds
    a_check = np.linspace(0.05, 1.0, 30)
    w0_check = [-1.5, -1.0, -0.5, 0.0, 0.5]
    wa_check = [-0.5, -0.25, 0.0, 0.25, 0.5]
    
    if _check_gamma_bounds(c_ls, a_check, w0_check, wa_check, Om0=0.30):
        best_c = c_ls
    else:
        print("Gamma out of bounds, using safe constant gamma")
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Refine with least_squares on the actual objective
    def obj_func(c):
        if _time_left() < 2.0:
            return np.ones(6) * 100.0
        return _objective(c, test_cosmos[:len(num_solutions)], a_test, num_solutions) * 100
    
    best_obj = obj_func(best_c)
    print(f"Initial objective: {best_obj:.6f}. elapsed={_elapsed():.1f}s")
    
    if _time_left() > 10.0:
        try:
            result = least_squares(obj_func, best_c, bounds=([-10]*6, [10]*6), 
                                   method='trf', max_nfev=200, xtol=1e-10, ftol=1e-10)
            if result.fun < best_obj:
                best_c = result.x
                best_obj = result.fun
                print(f"Refined objective: {best_obj:.6f}")
        except Exception as e:
            print(f"Refinement failed: {e}")
    
    # Final gamma bounds check
    if not _check_gamma_bounds(best_c, a_check, w0_check, wa_check, Om0=0.30):
        print("Final check: gamma out of bounds, reverting to safe constant")
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = np.clip(best_c, -10.0, 10.0)
    print(f"Final coefficients: {best_c}")
    print(f"Total elapsed: {_elapsed():.1f}s")
    
    return list(best_c)
