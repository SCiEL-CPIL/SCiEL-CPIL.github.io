import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_data_cache = None

def _prepare_data():
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    models = [
        (0.27, -1.0, 0.0),
        (0.30, -1.0, 0.0),
        (0.25, -1.2, 0.5),
        (0.30, -0.8, -0.3),
        (0.28, -1.0, 0.3),
        (0.28, -1.0, -0.3),
        (0.35, -1.0, 0.0),
        (0.20, -1.0, 0.0),
        (0.28, -1.5, 0.5),
        (0.28, -0.5, -0.5),
    ]
    a_grid = np.linspace(0.1, 1.0, 50)
    
    f_refs = []
    D_refs = []
    Om_a_refs = []
    u_refs = []
    wa_refs = []
    w0_refs = []
    
    for Om0, w0, wa in models:
        Ode = 1.0 - Om0
        
        def _E2(a):
            term_m = Om0 * a**(-3)
            term_de = Ode * a**(-3*(1+w0+wa)) * np.exp(3*wa*a)
            return term_m + term_de

        def _ode_y(y, y_prime):
            a = np.exp(y)
            E2 = _E2(a)
            E = np.sqrt(E2)
            
            dE2_da = -3*Om0*a**(-4) + Ode * (a**(-3*(1+w0+wa)) * np.exp(3*wa*a)) * (-3*(1+w0+wa)/a + 3*wa)
            dE2_dy = a * dE2_da
            dE_dy = 0.5 * E**(-1) * dE2_dy
            
            coeff = 0.5 + a * (dE_dy / E)
            return -coeff * y_prime + 1.5 * Om0 * a**(-3) * y

        a_min = 0.01
        y_min = np.log(a_min)
        y0 = np.array([a_min, a_min])
        
        sol = solve_ivp(_ode_y, [y_min, 0.0], y0, t_eval=np.log(a_grid), method='RK45', rtol=1e-8, atol=1e-10)
        if not sol.success:
            continue
            
        D = sol.y[0]
        D_y = sol.y[1]
        D_norm = D / D[-1]
        f = D_y / D
        
        E2_vals = _E2(a_grid)
        Om_a = Om0 * a_grid**(-3) / E2_vals
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        f_refs.append(f)
        D_refs.append(D_norm)
        Om_a_refs.append(Om_a)
        u_refs.append(u)
        wa_refs.append(wa)
        w0_refs.append(w0)
        
    _data_cache = {
        'f': np.array(f_refs),
        'D': np.array(D_refs),
        'Om_a': np.array(Om_a_refs),
        'u': np.array(u_refs),
        'a': a_grid,
        'wa': np.array(wa_refs),
        'w0': np.array(w0_refs)
    }
    return _data_cache

def _objective(coeffs, data):
    c0, c1, c2, c3, c4, c5 = coeffs
    a = data['a']
    Om_a = data['Om_a']
    u = data['u']
    wa = data['wa']
    
    q = 1.0 - Om_a
    
    wa_b = wa[:, np.newaxis] * np.ones((1, len(a)))
    a_b = np.ones((len(Om_a), 1)) * a[np.newaxis, :]
    
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa_b*a_b + c5*u*q
    
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        penalty = 1e6 * (np.sum(np.maximum(0, 0.1 - gamma)) + np.sum(np.maximum(0, gamma - 1.2)))
        return penalty + 1e6
    
    f_model = Om_a ** gamma
    
    x = np.log(a)
    f_x = f_model
    dx = np.diff(x)
    f_avg = 0.5 * (f_x[:, :-1] + f_x[:, 1:])
    trapz = f_avg * dx[np.newaxis, :]
    
    trapz_rev = trapz[:, ::-1]
    cumsum_rev = np.cumsum(trapz_rev, axis=1)
    I = cumsum_rev[:, ::-1]
    
    # I has shape (M, N-1). We need D_model for N points.
    # D_model[:, :-1] = exp(-I)
    # D_model[:, -1] = 1.0
    D_model = np.zeros_like(f_model)
    D_model[:, :-1] = np.exp(-I)
    D_model[:, -1] = 1.0
    
    err_f = np.abs(f_model - data['f']) / data['f']
    err_D = np.abs(D_model - data['D']) / data['D']
    
    max_err = np.max(np.concatenate([err_f.flatten(), err_D.flatten()]))
    
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    data = _prepare_data()
    
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_x = x0
    best_err = _objective(x0, data)
    
    try:
        res = minimize(_objective, x0, args=(data,), method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if time.time() < deadline:
            if res.success or res.fun < best_err:
                best_x = res.x
                best_err = res.fun
                
        if time.time() < deadline - 10:
            for _ in range(5):
                x_try = best_x + np.random.normal(0, 0.05, size=6)
                x_try = np.clip(x_try, -5, 5)
                err_try = _objective(x_try, data)
                if err_try < best_err:
                    best_err = err_try
                    best_x = x_try
                    res2 = minimize(_objective, x_try, args=(data,), method='L-BFGS-B', 
                                    options={'maxiter': 200, 'ftol': 1e-10})
                    if res2.fun < best_err:
                        best_err = res2.fun
                        best_x = res2.x
                        
                if time.time() > deadline - 5:
                    break
                    
    except Exception:
        pass

    best_x = np.clip(best_x, -10, 10)
    return list(best_x)
