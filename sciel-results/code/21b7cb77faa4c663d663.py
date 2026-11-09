"""Broadband antireflection coating design with Nelder-Mead optimization."""
import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

def _mean_reflectance(labels, thicknesses, materials, n0, ns, wl_min, wl_max, n_pts=401):
    """Compute mean reflectance (percent) using characteristic matrix method."""
    nL = materials['L']
    nH = materials['H']
    n_layers = len(thicknesses)
    
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r

    # Wavelengths
    wavelengths = np.linspace(wl_min, wl_max, n_pts)
    k2pi = 2.0 * np.pi
    
    # Pre-compute indices
    n_indices = np.array([nH if l == 'H' else nL for l in labels])
    t = np.array(thicknesses)
    
    total_ref = 0.0
    for wl in wavelengths:
        # Initialize M to identity
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j
        
        for i in range(n_layers):
            ni = n_indices[i]
            ti = t[i]
            phase = k2pi * ni * ti / wl
            c = math.cos(phase)
            s = math.sin(phase)
            
            # Layer matrix
            L11 = c
            L12 = 1j * s / ni
            L21 = 1j * ni * s
            L22 = c
            
            # M_new = L * M
            n11 = L11 * m11 + L12 * m21
            n12 = L11 * m12 + L12 * m22
            n21 = L21 * m11 + L22 * m21
            n22 = L21 * m12 + L22 * m22
            m11, m12, m21, m22 = n11, n12, n21, n22
            
        # Admittance Y
        # Y = (m11 * ns + m12) / (m21 * ns + m22)
        denom = m21 * ns + m22
        if abs(denom) < 1e-12:
            continue
        Y = (m11 * ns + m12) / denom
        r = (n0 - Y) / (n0 + Y)
        total_ref += abs(r) ** 2
        
    return 100.0 * total_ref / n_pts

def _constrain_and_eval(labels, x, materials, n0, ns, wl_min, wl_max, max_layer_nm, max_total_nm):
    """Constrain x to valid thicknesses and evaluate."""
    # x is raw values, we map them to [0, max_layer_nm]
    # Using a simple clamp is sufficient for Nelder-Mead if we start in range
    # But to be safe, let's interpret x as direct thicknesses if we use bounds in optimizer
    # For Nelder-Mead, we must handle bounds manually or transform variables.
    # Here, we assume x is already in [0, max_layer_nm] or we clamp.
    thicknesses = [max(0.0, min(max_layer_nm, xi)) for xi in x]
    
    total = sum(thicknesses)
    if total > max_total_nm:
        # Scale down if necessary? Or return penalty?
        # Returning a high penalty is better for optimizer
        return 100.0 + (total - max_total_nm) # Penalty
    
    return _mean_reflectance(labels, thicknesses, materials, n0, ns, wl_min, wl_max)

def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    nL = materials['L']
    nH = materials['H']
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    deadline = time.time() + 50.0 # Safety margin
    
    # Strategy: Try a few different layer counts and patterns.
    # Optimize thicknesses for each.
    
    best_score = float('inf')
    best_labels = None
    best_thick = None
    
    # Define some candidate structures
    # 1. Alternating L/H starting with L
    # 2. Alternating L/H starting with H
    # 3. Graded index approximations
    
    candidate_structures = []
    # We will try up to 10 layers to keep optimization fast, but allow up to max_layers if time permits
    # Actually, more layers usually better. Let's try 6, 8, 10 layers.
    for n_layers in [4, 6, 8, 10]:
        if n_layers > max_layers:
            break
        # Pattern 1: L H L H ...
        labels1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        candidate_structures.append(labels1)
        # Pattern 2: H L H L ...
        labels2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        candidate_structures.append(labels2)
        
    # Also try a "sandwich" or other patterns? Alternating is usually best for broadband.
    
    for labels in candidate_structures:
        if time.time() > deadline:
            break
            
        n_layers = len(labels)
        
        # Initial guess: Quarter wave at center wavelength
        centre_wl = 0.5 * (wl_min + wl_max)
        init_thick = []
        for l in labels:
            if l == 'L':
                init_thick.append(centre_wl / (4.0 * nL))
            else:
                init_thick.append(centre_wl / (4.0 * nH))
        
        # Randomize slightly
        for i in range(n_layers):
            init_thick[i] *= (0.8 + 0.4 * random.random())
            init_thick[i] = min(init_thick[i], max_layer_nm)
            
        # Optimize
        if _HAVE_SCIPY:
            def objective(x):
                return _constrain_and_eval(labels, x, materials, n0, ns, wl_min, wl_max, max_layer_nm, max_total_nm)
            
            bounds = [(0, max_layer_nm) for _ in range(n_layers)]
            # Nelder-Mead doesn't support bounds directly in all versions, 
            # but L-BFGS-B does. Let's use L-BFGS-B if available and feasible.
            # However, Nelder-Mead is often more robust for non-smooth or complex landscapes if we handle bounds via penalty.
            # Let's try L-BFGS-B first as it's faster for smooth problems.
            try:
                res = _scipy_minimize(objective, init_thick, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'maxfun': 5000})
                if res.success or res.fun < best_score:
                    score = res.fun
                    thick = res.x
                    # Ensure constraints are met strictly for return
                    thick = [max(0.0, min(max_layer_nm, t)) for t in thick]
                    if sum(thick) <= max_total_nm:
                        if score < best_score:
                            best_score = score
                            best_labels = labels[:]
                            best_thick = thick
            except Exception:
                pass
        else:
            # Fallback to simple coordinate descent or Nelder-Mead implementation
            # For brevity and reliability in sandbox, if no SciPy, we do a simple local search
            pass

    # If no result or SciPy failed, do a manual Nelder-Mead or better initial guess check
    if best_score == float('inf') or time.time() < deadline:
        # Let's implement a robust Nelder-Mead in pure Python if needed, 
        # or just do a finer grid search on the best structure found so far.
        
        # If we found a best_labels, do a local refinement
        if best_labels is not None:
            labels = best_labels
            n_layers = len(labels)
            current_thick = best_thick[:]
            current_score = best_score
            
            # Simple local optimization: Nelder-Mead
            # Simple implementation
            simplex = [current_thick[:]]
            for i in range(n_layers):
                pt = current_thick[:]
                pt[i] += 1.0 # Step size
                if pt[i] > max_layer_nm: pt[i] = max_layer_nm
                simplex.append(pt)
            
            values = [_constrain_and_eval(labels, p, materials, n0, ns, wl_min, wl_max, max_layer_nm, max_total_nm) for p in simplex]
            
            alpha, gamma, rho = 1.0, 2.0, 0.5
            max_iter = 200
            it = 0
            while it < max_iter and time.time() < deadline:
                # Sort
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        if values[j] < values[i]:
                            simplex[i], simplex[j] = simplex[j], simplex[i]
                            values[i], values[j] = values[j], values[i]
                            
                # If best improved, update global best
                if values[0] < current_score:
                    current_score = values[0]
                    current_thick = simplex[0][:]
                    best_score = current_score
                    best_labels = labels[:]
                    best_thick = current_thick[:]
                    
                # Check convergence
                if abs(values[0] - values[-1]) < 1e-6:
                    break
                    
                # Reflect
                worst = simplex[-1]
                centroid = [0.0] * n_layers
                for i in range(n_layers):
                    for p in simplex[:-1]:
                        centroid[i] += p[i]
                centroid = [c / n_layers for c in centroid]
                
                x_r = [centroid[i] + alpha * (centroid[i] - worst[i]) for i in range(n_layers)]
                x_r = [max(0.0, min(max_layer_nm, xi)) for xi in x_r]
                f_r = _constrain_and_eval(labels, x_r, materials, n0, ns, wl_min, wl_max, max_layer_nm, max_total_nm)
                
                if f_r < values[0]:
                    # Expand
                    x_e = [centroid[i] + gamma * (x_r[i] - centroid[i]) for i in range(n_layers)]
                    x_e = [max(0.0, min(max_layer_nm, xi)) for xi in x_e]
                    f_e = _constrain_and_eval(labels, x_e, materials, n0, ns, wl_min, wl_max, max_layer_nm, max_total_nm)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        values[-1] = f_r
                elif f_r < values[-2]:
                    simplex[-1] = x_r
                    values[-1] = f_r
                else:
                    # Contract
                    x_c = [centroid[i] + rho * (worst[i] - centroid[i]) for i in range(n_layers)]
                    x_c = [max(0.0, min(max_layer_nm, xi)) for xi in x_c]
                    f_c = _constrain_and_eval(labels, x_c, materials, n0, ns, wl_min, wl_max, max_layer_nm, max_total_nm)
                    if f_c < values[-1]:
                        simplex[-1] = x_c
                        values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n_layers)]
                            values[i] = _constrain_and_eval(labels, simplex[i], materials, n0, ns, wl_min, wl_max, max_layer_nm, max_total_nm)
                
                it += 1
        
        # If still no good result, fallback to single layer
        if best_labels is None:
            best_labels = ['L']
            best_thick = [0.5 * (wl_min + wl_max) / (4.0 * nL)]

    return best_labels, best_thick
