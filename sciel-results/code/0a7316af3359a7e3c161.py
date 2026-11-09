"""Compact dark-energy growth-index fit (CPL cosmologies)."""
import math
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _Om(a, Om0, w0, wa):
    u = 1.0 + w0 + wa * (1.0 - a)
    return Om0 * a**(-3.0) + (1.0 - Om0) * np.exp(-3.0 * u)

def _Hprime_over_H(a, Om0, w0, wa):
    u = 1.0 + w0 + wa * (1.0 - a)
    Om_a = _Om(a, Om0, w0, wa)
    de_a = 1.0 - Om_a
    w_a = w0 + wa * (1.0 - a)
    return -0.5 * (Om_a + 3.0 * (1.0 + w_a) * de_a)

def _reference_f_D(a_vals, Om0, w0, wa):
    """Compute reference f(a) and D(a)/D(1) via ODE integration."""
    n = len(a_vals)
    f_ref = np.zeros(n)
    D_ref = np.zeros(n)
    
    a_min = a_vals[0]
    x_end = math.log(a_min)
    n_steps = 3000
    dx = x_end / n_steps
    
    F0 = 1.0
    f1 = 0.55 * Om0**0.55
    G0 = f1
    
    F = F0
    G = G0
    x = 0.0
    
    F_fine = np.zeros(n_steps + 1)
    G_fine = np.zeros(n_steps + 1)
    F_fine[0] = F0
    G_fine[0] = G0
    
    for i in range(n_steps):
        x_i = i * dx
        a_i = math.exp(x_i)
        
        def ode_F(x, F, G):
            return G
        def ode_G(x, F, G):
            a = math.exp(x)
            Om_a = _Om(a, Om0, w0, wa)
            Hpp = _Hprime_over_H(a, Om0, w0, wa)
            return 1.5 * (Om_a - 1.0) * F - (3.0 + Hpp) * G
        
        k1F = ode_F(x_i, F, G)
        k1G = ode_G(x_i, F, G)
        k2F = ode_F(x_i + 0.5*dx, F + 0.5*dx*k1F, G + 0.5*dx*k1G)
        k2G = ode_G(x_i + 0.5*dx, F + 0.5*dx*k1F, G + 0.5*dx*k1G)
        k3F = ode_F(x_i + 0.5*dx, F + 0.5*dx*k2F, G + 0.5*dx*k2G)
        k3G = ode_G(x_i + 0.5*dx, F + 0.5*dx*k2F, G + 0.5*dx*k2G)
        k4F = ode_F(x_i + dx, F + dx*k3F, G + dx*k3G)
        k4G = ode_G(x_i + dx, F + dx*k3F, G + dx*k3G)
        
        F = F + (dx/6.0)*(k1F + 2*k2F + 2*k3F + k4F)
        G = G + (dx/6.0)*(k1G + 2*k2G + 2*k3G + k4G)
        
        F_fine[i+1] = F
        G_fine[i+1] = G
    
    x_fine = np.linspace(0, x_end, n_steps + 1)
    x_vals = np.log(a_vals)
    
    for j, x_j in enumerate(x_vals):
        F_j = np.interp(x_j, x_fine, F_fine)
        G_j = np.interp(x_j, x_fine, G_fine)
        D_ref[j] = F_j
        f_ref[j] = G_j / F_j if abs(F_j) > 1e-10 else 0.0
    
    return f_ref, D_ref

def _candidate_f_D_vec(c, a_vals, Om0, w0, wa):
    """Compute candidate f(a) and D(a)/D(1) vectorized."""
    c0, c1, c2, c3, c4, c5 = c
    Om_a = _Om(a_vals, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a_vals)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    f_cand = Om_a ** gamma
    
    # D(a)/D(1) = exp(-int_{ln(a)}^0 f(e^x) dx)
    x_min = math.log(a_vals[0])
    n_fine = 1500
    x_fine = np.linspace(x_min, 0, n_fine)
    a_fine = np.exp(x_fine)
    Om_fine = _Om(a_fine, Om0, w0, wa)
    q_fine = 1.0 - Om_fine
    u_fine = 1.0 + w0 + wa * (1.0 - a_fine)
    gamma_fine = c0 + c1*q_fine + c2*q_fine**2 + c3*u_fine + c4*wa*a_fine + c5*u_fine*q_fine
    f_fine = Om_fine ** gamma_fine
    
    dx_fine = np.diff(x_fine)
    cum_int = np.concatenate([[0], np.cumsum(0.5*(f_fine[1:]+f_fine[:-1])*dx_fine)])
    total_int = cum_int[-1]
    
    x_vals = np.log(a_vals)
    ci_vals = np.interp(x_vals, x_fine, cum_int)
    integral = total_int - ci_vals
    D_cand = np.exp(-integral)
    
    return f_cand, D_cand

def _max_rel_err(f_cand, D_cand, f_ref, D_ref):
    err_f = np.abs(f_cand - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
    err_D = np.abs(D_cand - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
    return np.max(np.concatenate([err_f, err_D]))

def _soft_max_err(f_cand, D_cand, f_ref, D_ref, beta=50.0):
    err_f = np.abs(f_cand - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
    err_D = np.abs(D_cand - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
    errs = np.concatenate([err_f, err_D])
    m = np.max(errs)
    return m + math.log(np.sum(np.exp(beta*(errs - m)))) / beta

def _objective(c, a_vals, models, f_refs, D_refs, beta=50.0):
    total_err = 0.0
    n_models = len(models)
    for i, (Om0, w0, wa) in enumerate(models):
        f_cand, D_cand = _candidate_f_D_vec(c, a_vals, Om0, w0, wa)
        err = _soft_max_err(f_cand, D_cand, f_refs[i], D_refs[i], beta)
        total_err += err
    return total_err / n_models

def _gamma_valid(c, a_vals, Om0, w0, wa):
    c0, c1, c2, c3, c4, c5 = c
    Om_a = _Om(a_vals, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a_vals)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    return np.all(gamma >= 0.1) and np.all(gamma <= 1.2)

def construct():
    start = time.time()
    deadline = start + 85.0
    
    # Reduced model grid for fitting
    Om0_vals = [0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.2, -1.0, -0.8]
    wa_vals = [-0.5, -0.2, 0.0, 0.2, 0.5]
    
    models = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                models.append((Om0, w0, wa))
    
    a_vals = np.linspace(0.3, 1.0, 30)
    
    # Compute references
    f_refs = []
    D_refs = []
    for Om0, w0, wa in models:
        f_ref, D_ref = _reference_f_D(a_vals, Om0, w0, wa)
        f_refs.append(f_ref)
        D_refs.append(D_ref)
        if time.time() - start > deadline - 15:
            break
    
    c0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_c = c0.copy()
    
    # Evaluate initial
    max_err_init = 0.0
    for i, (Om0, w0, wa) in enumerate(models):
        f_cand, D_cand = _candidate_f_D_vec(best_c, a_vals, Om0, w0, wa)
        err = _max_rel_err(f_cand, D_cand, f_refs[i], D_refs[i])
        max_err_init = max(max_err_init, err)
    best_score = 100.0 * max_err_init
    
    # Multi-start Nelder-Mead
    rng = np.random.RandomState(42)
    n_starts = 4
    
    for s in range(n_starts):
        if time.time() - start > deadline - 5:
            break
        
        if s == 0:
            c_init = c0.copy()
        else:
            c_init = c0 + 0.05 * rng.randn(6)
            c_init = np.clip(c_init, -10, 10)
        
        try:
            res = minimize(
                _objective, c_init,
                args=(a_vals, models, f_refs, D_refs, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            c_opt = np.clip(res.x, -10, 10)
            
            valid = True
            for Om0, w0, wa in models[:3]:
                if not _gamma_valid(c_opt, a_vals, Om0, w0, wa):
                    valid = False
                    break
            
            if valid:
                max_err = 0.0
                for i, (Om0, w0, wa) in enumerate(models):
                    f_cand, D_cand = _candidate_f_D_vec(c_opt, a_vals, Om0, w0, wa)
                    err = _max_rel_err(f_cand, D_cand, f_refs[i], D_refs[i])
                    max_err = max(max_err, err)
                
                if max_err < best_score / 100.0:
                    best_score = 100.0 * max_err
                    best_c = c_opt.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() - start < deadline - 10:
        try:
            res = minimize(
                _objective, best_c,
                args=(a_vals, models, f_refs, D_refs, 100.0),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            c_opt = np.clip(res.x, -10, 10)
            
            valid = True
            for Om0, w0, wa in models[:3]:
                if not _gamma_valid(c_opt, a_vals, Om0, w0, wa):
                    valid = False
                    break
            
            if valid:
                max_err = 0.0
                for i, (Om0, w0, wa) in enumerate(models):
                    f_cand, D_cand = _candidate_f_D_vec(c_opt, a_vals, Om0, w0, wa)
                    err = _max_rel_err(f_cand, D_cand, f_refs[i], D_refs[i])
                    max_err = max(max_err, err)
                
                if max_err < best_score / 100.0:
                    best_score = 100.0 * max_err
                    best_c = c_opt.copy()
        except Exception:
            pass
    
    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
