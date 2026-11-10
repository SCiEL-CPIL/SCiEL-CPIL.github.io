"""Broadband antireflection coating design using coordinate descent + Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _reflectance_vectorized(thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """Compute reflectance spectrum using characteristic-matrix method (vectorized over wavelengths)."""
    n_L = materials['L']
    n_H = materials['H']
    n_s = substrate_index
    n_0 = ambient_index
    
    # Build the characteristic matrices for each layer
    # For normal incidence: M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j / lambda
    
    # We'll compute the total matrix product for all wavelengths at once
    # wavelengths shape: (N,)
    # thicknesses shape: (num_layers,)
    
    # Initialize total matrix as identity for each wavelength
    N = len(wavelengths)
    # M_total[0,0], M_total[0,1], M_total[1,0], M_total[1,1] each shape (N,)
    m00 = np.ones(N, dtype=complex)
    m01 = np.zeros(N, dtype=complex)
    m10 = np.zeros(N, dtype=complex)
    m11 = np.ones(N, dtype=complex)
    
    for j, (label, d) in enumerate(zip(thicknesses_labels, thicknesses)):
        n_j = n_L if label == 'L' else n_H
        # delta = 2*pi*n_j*d / lambda
        delta = 2.0 * np.pi * n_j * d / wavelengths  # shape (N,)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix elements
        # M_j = [[cos_d, i*sin_d/n_j], [i*n_j*sin_d, cos_d]]
        m00_j = cos_d
        m01_j = 1j * sin_d / n_j
        m10_j = 1j * n_j * sin_d
        m11_j = cos_d
        
        # Multiply: M_new = M_old @ M_j
        # [[a,b],[c,d]] @ [[e,f],[g,h]] = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        nm00 = m00 * m00_j + m01 * m10_j
        nm01 = m00 * m01_j + m01 * m11_j
        nm10 = m10 * m00_j + m11 * m10_j
        nm11 = m10 * m01_j + m11 * m11_j
        
        m00, m01, m10, m11 = nm00, nm01, nm10, nm11
    
    # Add substrate effect: the admittance at the substrate is n_s (for normal incidence, lossless)
    # Y = (n_0 * m11 + n_s * m10) / (n_0 * m01 + n_s * m00)
    # Actually, the standard formula:
    # Y = (m11 + n_s * m10) / (m01 + n_s * m00)  ... let me be careful.
    
    # Standard characteristic matrix method:
    # The total matrix M = M_1 * M_2 * ... * M_k
    # Y = (M_22 + n_s * M_21) / (M_12 + n_s * M_11)
    # where M_11=m00, M_12=m01, M_21=m10, M_22=m11
    # Then R = |(Y - n_0)/(Y + n_0)|^2
    
    Y = (m11 + n_s * m10) / (m01 + n_s * m00)
    R = np.abs((Y - n_0) / (Y + n_0)) ** 2
    return R * 100.0  # in percent


def _objective(labels, thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """Compute mean reflectance in percent."""
    # Set global for _reflectance_vectorized
    global thicknesses_labels
    thicknesses_labels = labels
    R = _reflectance_vectorized(thicknesses, materials, ambient_index, substrate_index, wavelengths)
    return np.mean(R)


def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-8, timeout_check=None):
    """Nelder-Mead simplex optimization with bounds handling."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        step = 0.05 * (bounds[i][1] - bounds[i][0]) + 0.01
        simplex[i + 1] = x0
        simplex[i + 1][i] += step
    
    # Clip to bounds
    for i in range(n + 1):
        simplex[i] = np.clip(simplex[i], bounds[i][0] if i < n else 0, 
                             [bounds[j][1] for j in range(n)] if i < n else 1)
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        if timeout_check and timeout_check():
            break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i])
    
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


def design(problem):
    """Design broadband AR coating using multi-start optimization."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    materials = problem['materials']
    ambient_index = problem['ambient_index']
    substrate_index = problem['substrate_index']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    # 401 equally spaced wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def check_timeout():
        return time.time() > deadline
    
    # Try different numbers of layers
    best_score = float('inf')
    best_labels = None
    best_thick = None
    
    # Start with quarter-wave stack as initial guess
    centre = 0.5 * (band[0] + band[1])
    dL = centre / (4.0 * materials['L'])
    dH = centre / (4.0 * materials['H'])
    
    # Try various layer counts and patterns
    layer_configs = []
    
    # Quarter-wave stacks with alternating L/H
    for n in range(2, min(max_layers + 1, 17), 2):
        # Start with L, alternate
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        thicks = [dL if l == 'L' else dH for l in labels]
        if sum(thicks) <= max_total_nm:
            layer_configs.append((labels, thicks))
    
    # Also try starting with H
    for n in range(2, min(max_layers + 1, 17), 2):
        labels = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        thicks = [dH if l == 'H' else dL for l in labels]
        if sum(thicks) <= max_total_nm:
            layer_configs.append((labels, thicks))
    
    # Some odd layer counts
    for n in [3, 5, 7, 9, 11]:
        if n <= max_layers:
            labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
            thicks = [dL if l == 'L' else dH for l in labels]
            if sum(thicks) <= max_total_nm:
                layer_configs.append((labels, thicks))
    
    # Random initial configurations
    rng = np.random.RandomState(42)
    for _ in range(min(10, max(0, 20 - len(layer_configs)))):
        n = rng.randint(4, min(max_layers + 1, 13))
        labels = [rng.choice(['L', 'H']) for _ in range(n)]
        thicks = []
        for l in labels:
            t = rng.uniform(50, min(max_layer_nm, 300))
            thicks.append(t)
        if sum(thicks) <= max_total_nm:
            layer_configs.append((labels, thicks))
    
    # Evaluate and optimize each configuration
    for labels, thicks in layer_configs:
        if check_timeout():
            break
        
        n = len(labels)
        bounds = [(0.0, max_layer_nm)] * n
        
        def f(x):
            # Ensure total thickness constraint
            if sum(x) > max_total_nm:
                return 100.0  # penalty
            return _objective(labels, x, materials, ambient_index, substrate_index, wavelengths)
        
        x0 = np.array(thicks, dtype=float)
        x0 = np.clip(x0, 0, max_layer_nm)
        
        try:
            x_opt, f_opt = _nelder_mead(f, x0, bounds, max_iter=150, tol=1e-10, timeout_check=check_timeout)
            if f_opt < best_score:
                best_score = f_opt
                best_labels = labels[:]
                best_thick = [float(t) for t in x_opt]
        except Exception:
            pass
    
    # Coordinate descent refinement on the best solution
    if best_labels is not None and not check_timeout():
        labels = best_labels[:]
        thicks = np.array(best_thick, dtype=float)
        n = len(labels)
        
        def f_cd(x):
            if sum(x) > max_total_nm:
                return 100.0
            return _objective(labels, x, materials, ambient_index, substrate_index, wavelengths)
        
        current_best = f_cd(thicks)
        
        # Multi-scale coordinate descent
        for scale in [50, 20, 10, 5, 2, 1, 0.5, 0.1]:
            if check_timeout():
                break
            improved = True
            while improved and not check_timeout():
                improved = False
                for i in range(n):
                    for direction in [-1, 1]:
                        x_trial = thicks.copy()
                        x_trial[i] += direction * scale
                        x_trial[i] = max(0.0, min(max_layer_nm, x_trial[i]))
                        if sum(x_trial) > max_total_nm:
                            continue
                        f_trial = f_cd(x_trial)
                        if f_trial < current_best:
                            thicks = x_trial
                            current_best = f_trial
                            improved = True
                            if current_best < best_score:
                                best_score = current_best
                                best_thick = [float(t) for t in thicks]
    
    # Final Nelder-Mead polish
    if best_labels is not None and not check_timeout():
        labels = best_labels
        thicks = np.array(best_thick, dtype=float)
        n = len(labels)
        bounds = [(0.0, max_layer_nm)] * n
        
        def f_polish(x):
            if sum(x) > max_total_nm:
                return 100.0
            return _objective(labels, x, materials, ambient_index, substrate_index, wavelengths)
        
        try:
            x_opt, f_opt = _nelder_mead(f_polish, thicks, bounds, max_iter=100, tol=1e-12, timeout_check=check_timeout)
            if f_opt < best_score:
                best_score = f_opt
                best_thick = [float(t) for t in x_opt]
        except Exception:
            pass
    
    # Ensure validity
    if best_labels is None:
        # Fallback: single quarter-wave layer
        return ["L"], [centre / (4.0 * materials['L'])]
    
    # Clamp thicknesses
    best_thick = [max(0.0, min(max_layer_nm, t)) for t in best_thick]
    
    # Ensure total constraint
    total = sum(best_thick)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thick = [t * scale for t in best_thick]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
