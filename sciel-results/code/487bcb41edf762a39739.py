"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

def _E_lambda_cdm(a, Om=0.3):
    return np.sqrt(Om * a**(-3) + 1.0)

def _E_w_cdm(a, Om=0.3, w=-0.8):
    return np.sqrt(Om * a**(-3*(1+w)) + 1.0)

def _E_cpl(a, Om=0.3, w0=-0.5, wa=-0.5):
    Oml = 1.0 - Om
    term_m = Om * np.exp(-3*(1+w0+wa)*np.log(a))
    term_l = Oml * np.exp(-3*(1+w0+wa)*np.log(a) + 3*wa*(1-a))
    return np.sqrt(term_m + term_l)

def _true_integral(z, E_func):
    def integrand(zp):
        a = 1.0 / (1.0 + zp)
        return 1.0 / E_func(a)
    val, _ = quad(integrand, 0, z, limit=50)
    return val

def construct():
    z_tests = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    models = [_E_lambda_cdm, _E_w_cdm, _E_cpl]
    
    true_vals = []
    for m in models:
        for z in z_tests:
            true_vals.append(_true_integral(z, m))
    true_vals = np.array(true_vals)
    
    x_gl, w_gl = leggauss(8)
    n0 = (x_gl + 1.0) / 2.0
    w0 = w_gl / 2.0
    x_init = np.concatenate([n0, w0])
    
    def cost(params):
        nodes = params[:8]
        weights = params[8:16]
        if np.any(nodes < 0) or np.any(nodes > 1) or np.any(weights < 0):
            return 1e10
        errors = []
        idx = 0
        for m in models:
            for z in z_tests:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = m(a_vals)
                quad_val = z * np.sum(weights / E_vals)
                true_val = true_vals[idx]
                rel_err = (quad_val - true_val) / true_val
                errors.append(rel_err * rel_err)
                idx += 1
        return np.sum(errors)
        
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[8:16]) - 1.0}]
    for i in range(8):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - 0.001})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+8] - 0.001})
    for i in range(7):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - 0.001})
    constraints.append({'type': 'ineq', 'fun': lambda x: 0.999 - x[7]})

    bounds = [(0.001, 0.999)] * 8 + [(0.001, 1.0)] * 8

    try:
        res = minimize(cost, x_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-12})
        if res.success:
            nodes = res.x[:8]
            weights = res.x[8:16]
        else:
            nodes = n0
            weights = w0
    except Exception:
        nodes = n0
        weights = w0
        
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
