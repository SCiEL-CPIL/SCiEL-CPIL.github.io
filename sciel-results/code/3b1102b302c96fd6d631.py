import time
import math
import numpy as np
import random
from scipy.optimize import minimize

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)

    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d == 0:
            continue
        beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]

        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D

        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent with constraint penalties.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6

    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]

    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6

    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _reflectance_and_gradient(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance and its gradient with respect to thicknesses
    using central finite differences.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        return 0.0, []
    
    h = 0.1  # nm step for finite difference
    f0 = _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat)
    grad = np.zeros(n_layers)
    
    for i in range(n_layers):
        th_plus = list(thicknesses)
        th_minus = list(thicknesses)
        th_plus[i] += h
        th_minus[i] -= h
        
        # Ensure bounds
        if th_plus[i] > problem["max_layer_nm"]:
            th_plus[i] = problem["max_layer_nm"]
            h_eff = h - (thicknesses[i] - problem["max_layer_nm"])
        else:
            h_eff = h
            
        if th_minus[i] < 0:
            th_minus[i] = 0.0
            h_eff = h - (0.0 - thicknesses[i])
        else:
            h_eff = h
            
        if h_eff < 1e-10:
            # At boundary, use one-sided difference
            if thicknesses[i] >= problem["max_layer_nm"] - 1e-9:
                f_plus = f0
                f_minus = _mean_reflectance_percent(materials, th_minus, problem, wls, n_mat)
                grad[i] = (f_plus - f_minus) / (thicknesses[i] - th_minus[i])
            else:
                f_plus = _mean_reflectance_percent(materials, th_plus, problem, wls, n_mat)
                f_minus = f0
                grad[i] = (f_plus - f_minus) / (th_plus[i] - thicknesses[i])
        else:
            f_plus = _mean_reflectance_percent(materials, th_plus, problem, wls, n_mat)
            f_minus = _mean_reflectance_percent(materials, th_minus, problem, wls, n_mat)
            grad[i] = (f_plus - f_minus) / (2.0 * h_eff)
            
    return f0, grad

def _optimize_l_bfgs(objective, grad_func, x0, lo, hi, deadline, max_iter=200):
    """
    L-BFGS-B optimization with bound constraints and deadline check.
    """
    n = len(x0)
    
    # Convert to bounds tuple for scipy
    bounds = [(lo[i], hi[i]) for i in range(n)]
    
    # Use L-BFGS-B which handles bounds
    options = {
        'maxiter': max_iter,
        'ftol': 1e-8,
        'gtol': 1e-5,
        'maxls': 50,
        'disp': False
    }
    
    def objective_wrapper(x):
        if time.time() > deadline:
            return 1e10
        return objective(x)
    
    def gradient_wrapper(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        _, g = grad_func(x)
        return g
    
    try:
        result = minimize(objective_wrapper, x0, jac=gradient_wrapper,
                          method='L-BFGS-B', bounds=bounds, options=options)
        return result.x, result.fun
    except Exception as e:
        # Fallback to simple gradient descent if L-BFGS fails
        x = x0.copy()
        lr = 1.0
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            f, g = grad_func(x)
            # Gradient step
            x_new = x - lr * g
            # Clip to bounds
            x_new = np.clip(x_new, lo, hi)
            f_new = objective(x_new)
            if f_new < f:
                x = x_new
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        return x, objective(x)

def _gradient_polish(objective, x0, lo, hi, deadline, max_iter=100):
    """
    Simple gradient descent polish with adaptive learning rate.
    """
    x = x0.copy()
    lr = 5.0  # Initial learning rate
    n = len(x)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        f, g = _reflectance_and_gradient_for_polish(objective, x, lo, hi)
        
        # Check for convergence
        if np.max(np.abs(g)) < 1e-4:
            break
            
        # Gradient step
        x_new = x - lr * g
        x_new = np.clip(x_new, lo, hi)
        
        f_new = objective(x_new)
        if f_new < f:
            x = x_new
            lr = min(lr * 1.1, 50.0)  # Increase LR slightly
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
                
    return x, objective(x)

def _reflectance_and_gradient_for_polish(objective, x, lo, hi):
    """
    Compute objective and gradient for polish step.
    """
    f = objective(x)
    h = 0.5
    n = len(x)
    g = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] = min(x[i] + h, hi[i])
        x_minus[i] = max(x[i] - h, lo[i])
        
        if x_plus[i] == x[i] and x_minus[i] == x[i]:
            g[i] = 0.0
        elif x_plus[i] == x[i]:
            g[i] = (f - objective(x_minus)) / (x[i] - x_minus[i])
        elif x_minus[i] == x[i]:
            g[i] = (objective(x_plus) - f) / (x_plus[i] - x[i])
        else:
            g[i] = (objective(x_plus) - objective(x_minus)) / (x_plus[i] - x_minus[i])
            
    return f, g

def _optimize_thicknesses(materials, x0, problem, wls, n_mat, deadline):
    """
    Optimize thicknesses for a given material sequence using L-BFGS-B.
    """
    n_layers = len(materials)
    lo = np.zeros(n_layers)
    hi = np.full(n_layers, problem["max_layer_nm"])
    
    def objective(th):
        return _mean_reflectance_percent(materials, list(th), problem, wls, n_mat)
    
    def grad_func(th):
        return _reflectance_and_gradient(materials, list(th), problem, wls, n_mat)
    
    # Try L-BFGS-B first
    x_lbgfs, f_lbgfs = _optimize_l_bfgs(objective, grad_func, x0, lo, hi, deadline, max_iter=150)
    
    # Polish with gradient descent
    x_polished, f_polished = _gradient_polish(objective, x_lbgfs, lo, hi, deadline, max_iter=50)
    
    # Use the better result
    if f_polished <= f_lbgfs:
        return x_polished, f_polished
    else:
        return x_lbgfs, f_lbgfs

def design(problem):
    """
    Design a broadband antireflection coating using gradient-based optimization.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]

    best_val = float('inf')
    best_mats = None
    best_ths = None

    def update_best(mats, ths):
        nonlocal best_val, best_mats, best_ths
        ths_clipped = [min(max(t, 0.0), max_layer) for t in ths]
        s = sum(ths_clipped)
        if s > max_total:
            if s > 1e-9:
                f = max_total / s
                ths_clipped = [t * f for t in ths_clipped]
        val = _mean_reflectance_percent(mats, ths_clipped, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(mats)
            best_ths = list(ths_clipped)
        return val

    # Generate initial candidates with good starting points
    # Use quarter-wave designs centered at various wavelengths
    center_wls = [450, 500, 550, 600, 650, 700, 400, 800]
    
    # For each layer count, try multiple material patterns and initial thicknesses
    candidate_configs = []
    
    for n_layers in [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        
        # Different material patterns
        patterns = []
        
        # Alternating starting with L
        patterns.append(['L' if i % 2 == 0 else 'H' for i in range(n_layers)])
        # Alternating starting with H
        patterns.append(['H' if i % 2 == 0 else 'L' for i in range(n_layers)])
        
        # For even layers, try symmetric patterns
        if n_layers % 2 == 0:
            half = n_layers // 2
            # L-H-L-H... symmetric
            sym1 = ['L' if i % 2 == 0 else 'H' for i in range(half)]
            sym1 = sym1 + sym1[::-1]
            patterns.append(sym1)
            
            # H-L-H-L... symmetric
            sym2 = ['H' if i % 2 == 0 else 'L' for i in range(half)]
            sym2 = sym2 + sym2[::-1]
            patterns.append(sym2)
        
        # For each pattern, try multiple initial thickness sets
        for mats in patterns:
            for cw in center_wls[:4]:  # Use first 4 center wavelengths
                ths = []
                for m in mats:
                    n = n_mat[m]
                    q = cw / (4.0 * n)
                    ths.append(q)
                
                # Scale to fit total constraint
                s = sum(ths)
                if s > max_total:
                    f = max_total / s
                    ths = [t * f for t in ths]
                
                candidate_configs.append((mats, ths))
            
            # Add a few random initial points
            for _ in range(2):
                ths = [random.uniform(20, max_layer * 0.5) for _ in range(n_layers)]
                s = sum(ths)
                if s > max_total:
                    f = max_total / s
                    ths = [t * f for t in ths]
                candidate_configs.append((mats, ths))

    # Shuffle for diversity
    random.shuffle(candidate_configs)
    
    # Limit number of candidates based on time
    max_candidates = 60
    candidate_configs = candidate_configs[:max_candidates]
    
    for mats, ths_init in candidate_configs:
        if time.time() > deadline:
            break
        
        n_layers = len(mats)
        x0 = np.clip(np.array(ths_init, dtype=float), 0, max_layer)
        
        # Optimize using L-BFGS-B + gradient polish
        x_opt, f_opt = _optimize_thicknesses(mats, x0, problem, wls, n_mat, deadline)
        
        update_best(mats, x_opt)
        
        # Also try a perturbed version
        if time.time() < deadline - 15:
            x_pert = x0.copy()
            for i in range(n_layers):
                if random.random() < 0.5:
                    x_pert[i] += random.uniform(-0.1, 0.1) * max_layer
                    x_pert[i] = np.clip(x_pert[i], 0, max_layer)
            
            x_opt2, f_opt2 = _optimize_thicknesses(mats, x_pert, problem, wls, n_mat, deadline)
            update_best(mats, x_opt2)

    # Additional refinement: try to improve the best solution found
    if best_mats is not None and time.time() < deadline - 20:
        n_layers = len(best_mats)
        
        # Multiple restarts from perturbed best
        for _ in range(3):
            if time.time() > deadline:
                break
            
            x0 = np.array(best_ths, dtype=float).copy()
            for i in range(n_layers):
                if random.random() < 0.5:
                    x0[i] += random.uniform(-0.05, 0.05) * max_layer
                    x0[i] = np.clip(x0[i], 0, max_layer)
            
            x_opt, f_opt = _optimize_thicknesses(best_mats, x0, problem, wls, n_mat, deadline)
            update_best(best_mats, x_opt)

    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
