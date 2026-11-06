"""Broadband antireflection coating design via differential evolution + local polish."""
import numpy as np
import time

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """
    Objective: mean reflectance in percent. Returns large penalty if constraints violated.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=3000, tol=1e-10, time_deadline=None):
    """
    Nelder-Mead simplex with bounds.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.05 * (hi - lo) if hi > lo else 0.05
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
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
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _differential_evolution(f, bounds, pop_size=30, max_iter=100, F=0.8, CR=0.9, 
                            time_deadline=None, seed=42):
    """
    Differential evolution with bounds.
    """
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    rng = np.random.RandomState(seed)
    
    # Initialize population
    pop = np.zeros((pop_size, n))
    for i in range(pop_size):
        pop[i] = rng.uniform(lo, hi)
    
    fvals = np.array([f(pop[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        for i in range(pop_size):
            if time_deadline is not None and time.time() > time_deadline:
                break
            
            # Select three distinct random indices different from i
            while True:
                a, b, c = rng.randint(0, pop_size, 3)
                if a != b and b != c and a != c and a != i:
                    break
            
            # Create trial vector
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(0, n)
            for j in range(n):
                if rng.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            trial = np.clip(trial, lo, hi)
            
            f_trial = f(trial)
            
            if f_trial <= fvals[i]:
                pop[i] = trial
                fvals[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    best_idx = i
    
    return best_x, best_f

def _polish_with_gradient_descent(f, x0, bounds, max_iter=2000, time_deadline=None):
    """
    Gradient descent with numerical gradient and line search for final polish.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    
    # Numerical gradient
    def numerical_gradient(x):
        eps = 1e-6
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_plus = np.clip(x_plus, lo_bounds, hi_bounds)
            x_minus = x.copy()
            x_minus[i] -= eps
            x_minus = np.clip(x_minus, lo_bounds, hi_bounds)
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    
    # Gradient descent with backtracking line search
    lr = 1.0
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        grad = numerical_gradient(x)
        if np.max(np.abs(grad)) < 1e-8:
            break
        
        # Gradient direction (negative gradient)
        direction = -grad
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-10:
            break
        direction = direction / norm_dir
        
        # Backtracking line search
        step = lr
        f_old = fval
        for ls_iter in range(50):
            x_trial = x + step * direction
            x_trial = np.clip(x_trial, lo_bounds, hi_bounds)
            f_trial = f(x_trial)
            
            if f_trial < f_old - 1e-4 * step * np.dot(grad, direction):
                x = x_trial
                fval = f_trial
                break
            else:
                step *= 0.5
        
        # Adaptive learning rate
        if fval < f_old - 1e-3:
            lr = min(lr * 1.5, 10.0)
        elif fval > f_old + 1e-4:
            lr = max(lr * 0.5, 0.01)
    
    return x, fval

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Best result tracking
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Generate layer patterns
    patterns = []
    
    # Alternating starting with L
    for n in range(3, min(max_layers + 1, 20), 2):
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append((n, labels, "alt_L"))
    
    # Alternating starting with H
    for n in range(3, min(max_layers + 1, 20), 2):
        labels = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        patterns.append((n, labels, "alt_H"))
    
    # L H H L repeating (period 4)
    for n in [8, 12, 16]:
        if n <= max_layers:
            labels = []
            for i in range(n):
                m = i % 4
                if m == 0:
                    labels.append('L')
                elif m in [1, 2]:
                    labels.append('H')
                else:
                    labels.append('L')
            patterns.append((n, labels, "LHHL"))
    
    # H L L H repeating (period 4)
    for n in [8, 12, 16]:
        if n <= max_layers:
            labels = []
            for i in range(n):
                m = i % 4
                if m == 0:
                    labels.append('H')
                elif m in [1, 2]:
                    labels.append('L')
                else:
                    labels.append('H')
            patterns.append((n, labels, "HLLH"))
    
    # Single and two layers
    patterns.append((1, ['L'], "L1"))
    patterns.append((1, ['H'], "H1"))
    patterns.append((2, ['L', 'H'], "LH"))
    patterns.append((2, ['H', 'L'], "HL"))
    
    # Prioritize patterns: 7-13 layers first, then 5-6 and 14-17, then others
    def pattern_priority(p):
        n = p[0]
        if 7 <= n <= 13:
            return 0
        elif 5 <= n < 7 or 13 < n <= 17:
            return 1
        else:
            return 2
    
    patterns.sort(key=pattern_priority)
    
    # For each pattern, do DE + NM polish + gradient polish
    for n_layers, labels, ptype in patterns:
        if time.time() > deadline - 40:
            break
        
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        def make_f(indices):
            def f(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            return f
        
        f = make_f(indices)
        
        # Differential evolution for global search
        remaining = deadline - 30 - time.time()
        if remaining > 5:
            de_deadline = time.time() + min(20, remaining * 0.4)
            try:
                x_de, f_de = _differential_evolution(f, bounds, pop_size=40, max_iter=200, 
                                                     F=0.8, CR=0.9, time_deadline=de_deadline, 
                                                     seed=42)
                if f_de < best_score:
                    best_score = f_de
                    best_labels = labels.copy()
                    best_thicknesses = x_de.copy()
            except Exception:
                pass
        
        # Multiple NM starts from diverse initializations
        n_starts = 8
        for start_i in range(n_starts):
            if time.time() > deadline - 25:
                break
            
            # Generate initial guess
            if start_i == 0:
                # Quarter-wave at center
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
            elif start_i == 1:
                # Quarter-wave at red end
                x0 = np.array([band_nm[1] / (4.0 * n) for n in indices])
            elif start_i == 2:
                # Quarter-wave at blue end
                x0 = np.array([band_nm[0] / (4.0 * n) for n in indices])
            elif start_i == 3:
                # Random but constrained
                x0 = np.random.uniform(0, max_layer_nm, size=n_layers)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
            elif start_i == 4:
                # Small random
                x0 = np.random.uniform(0, 150, size=n_layers)
            elif start_i == 5:
                # All same
                x0 = np.full(n_layers, max_total_nm / n_layers)
            elif start_i == 6:
                # Quarter-wave with noise
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
                x0 = x0 * np.random.uniform(0.7, 1.3, size=n_layers)
            else:
                # Random with total constraint
                x0 = np.random.dirichlet(np.ones(n_layers)) * max_total_nm
                x0 = np.clip(x0, 0, max_layer_nm)
            
            x0 = np.clip(x0, 0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
            
            # Evaluate initial
            f0 = f(x0)
            
            if f0 < best_score:
                best_score = f0
                best_labels = labels.copy()
                best_thicknesses = x0.copy()
            
            if f0 >= 100.0:
                continue  # Constraint violated, skip NM
            
            # NM polish with time-limited budget
            remaining = deadline - 20 - time.time()
            time_per_start = max(0.2, min(3.0, remaining / max(1, n_starts - start_i)))
            nm_deadline = time.time() + time_per_start
            
            try:
                x_opt, f_opt = _nelder_mead(f, x0, bounds, max_iter=3000, tol=1e-10, 
                                             time_deadline=nm_deadline)
            except Exception:
                x_opt, f_opt = x0, f0
            
            if f_opt < best_score:
                best_score = f_opt
                best_labels = labels.copy()
                best_thicknesses = x_opt.copy()
            
            # Gradient polish for final refinement
            if f_opt < best_score + 0.5 and time.time() < deadline - 10:
                gd_deadline = time.time() + 1.0
                try:
                    x_gd, f_gd = _polish_with_gradient_descent(f, x_opt, bounds, 
                                                               max_iter=500, time_deadline=gd_deadline)
                    if f_gd < best_score:
                        best_score = f_gd
                        best_labels = labels.copy()
                        best_thicknesses = x_gd.copy()
                except Exception:
                    pass
            
            # Early exit if very good
            if best_score < 0.2:
                break
    
    # Final fallback
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
