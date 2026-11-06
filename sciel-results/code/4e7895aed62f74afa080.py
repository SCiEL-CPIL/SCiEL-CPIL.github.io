"""
Broadband antireflection coating design via differential evolution + Nelder-Mead polish.

Strategy:
- Encode the full design (layer types + thicknesses) into a single vector
- Use differential evolution for global exploration over the combined space
- Polish the best result with Nelder-Mead
- Multiple layer counts are tried, and the best overall is returned
"""
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

def _objective_from_vector(x, n_layers, materials, wavelengths_nm, ambient_idx, substrate_idx, 
                           max_layer_nm, max_total_nm):
    """
    Decode the vector x into layer labels and thicknesses, then compute mean reflectance.
    x is of length 2*n_layers: first n_layers are continuous values in [0,1] for layer type,
    last n_layers are thicknesses in [0, max_layer_nm].
    """
    # Decode layer types: values in [0, 0.5) -> L, [0.5, 1) -> H
    type_vals = x[:n_layers]
    thicknesses = x[n_layers:]
    
    # Ensure thicknesses are valid
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    
    if np.sum(thicknesses) > max_total_nm:
        # Scale down proportionally
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    # Determine layer types
    labels = ['L' if v < 0.5 else 'H' for v in type_vals]
    indices = [materials[label] for label in labels]
    
    # Remove zero-thickness layers (they don't affect reflectance)
    valid_mask = thicknesses > 1e-6
    if not np.any(valid_mask):
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return r0 * r0 * 100.0
    
    labels_valid = [labels[i] for i in range(n_layers) if valid_mask[i]]
    thicknesses_valid = thicknesses[valid_mask]
    indices_valid = [materials[label] for label in labels_valid]
    
    R = _compute_reflectance(wavelengths_nm, indices_valid, thicknesses_valid, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _differential_evolution(objective, bounds, pop_size=50, max_iter=200, 
                            F=0.8, CR=0.9, time_deadline=None, verbose=False):
    """
    Differential evolution with bounds.
    """
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo
    
    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    fvals = np.array([objective(ind) for ind in pop])
    
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        for i in range(pop_size):
            if time_deadline is not None and time.time() > time_deadline:
                break
            
            # Select three distinct individuals
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)
            
            # Create trial vector
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)
            
            # Crossover
            r = np.random.random(n)
            jrand = np.random.randint(n)
            trial = np.where(r < CR, v, pop[i])
            trial[jrand] = v[jrand]
            trial = np.clip(trial, lo, hi)
            
            # Selection
            f_trial = objective(trial)
            if f_trial <= fvals[i]:
                pop[i] = trial
                fvals[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
        
        if verbose and it % 20 == 0:
            print(f"DE iter {it}: best_f = {best_f:.6f}")
    
    return best_x, best_f

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, time_deadline=None):
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

def _coordinate_descent(f, x0, bounds, max_iter=200, time_deadline=None):
    """
    Cyclic coordinate descent with golden-section line search per coordinate.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    
    # Golden ratio
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        improved = False
        for i in range(n):
            if time_deadline is not None and time.time() > time_deadline:
                break
            
            lo = lo_bounds[i]
            hi = hi_bounds[i]
            x_i = x[i]
            
            # Try to find better value along coordinate i
            # Use a few trial points
            best_x_i = x_i
            best_f = fval
            
            # Try perturbation in both directions
            for direction in [-1, 1]:
                for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
                    trial = x.copy()
                    trial[i] = x_i + direction * scale * (hi - lo) * 0.1
                    trial[i] = np.clip(trial[i], lo, hi)
                    if trial[i] == x_i:
                        continue
                    ft = f(trial)
                    if ft < best_f:
                        best_f = ft
                        best_x_i = trial[i]
            
            if best_x_i != x_i:
                # Do a quick golden-section search around the best found
                a = min(x_i, best_x_i) - 0.1 * (hi - lo)
                b = max(x_i, best_x_i) + 0.1 * (hi - lo)
                a = max(a, lo)
                b = min(b, hi)
                
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                x_c = x.copy()
                x_c[i] = c
                x_d = x.copy()
                x_d[i] = d
                
                fc = f(x_c)
                fd = f(x_d)
                
                for _ in range(30):
                    if time_deadline is not None and time.time() > time_deadline:
                        break
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        x_c = x.copy()
                        x_c[i] = c
                        fc = f(x_c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        x_d = x.copy()
                        x_d[i] = d
                        fd = f(x_d)
                
                best_trial = x.copy()
                best_trial[i] = (a + b) / 2.0
                best_trial[i] = np.clip(best_trial[i], lo, hi)
                ft = f(best_trial)
                if ft < best_f:
                    x = best_trial
                    fval = ft
                    best_f = ft
                    best_x_i = (a + b) / 2.0
                    improved = True
            
            if best_x_i != x_i:
                x[i] = best_x_i
                fval = best_f
                improved = True
        
        if not improved:
            break
    
    return x, fval

def _generate_initial_designs(n_layers, materials, ambient_idx, substrate_idx, center_wl):
    """
    Generate initial designs from quarter-wave stack theory and variations.
    Returns list of (labels, thicknesses) tuples.
    """
    designs = []
    nL = materials["L"]
    nH = materials["H"]
    
    # Design 1: Simple quarter-wave stack (L H L H ...)
    for start in ["L", "H"]:
        labels = []
        thicknesses = []
        for i in range(n_layers):
            if i % 2 == 0:
                lab = start
            else:
                lab = "H" if start == "L" else "L"
            labels.append(lab)
            thicknesses.append(center_wl / (4.0 * materials[lab]))
        # Check total thickness
        if sum(thicknesses) <= 1000:
            designs.append((labels, thicknesses))
    
    # Design 2: Alternating with varying optical thickness (broadband)
    for start in ["L", "H"]:
        labels = []
        thicknesses = []
        for i in range(n_layers):
            if i % 2 == 0:
                lab = start
            else:
                lab = "H" if start == "L" else "L"
            labels.append(lab)
            # Vary thickness to cover bandwidth
            opt_thick = center_wl / (4.0 * materials[lab])
            # Add some variation
            variation = 1.0 + 0.2 * (i / max(n_layers - 1, 1))
            thicknesses.append(opt_thick * variation)
        if sum(thicknesses) <= 1000:
            designs.append((labels, thicknesses))
    
    # Design 3: All-L or all-H with graded thickness
    for lab in ["L", "H"]:
        labels = [lab] * n_layers
        thicknesses = []
        for i in range(n_layers):
            t = center_wl / (4.0 * materials[lab]) * (0.5 + 0.5 * i / max(n_layers - 1, 1))
            thicknesses.append(t)
        if sum(thicknesses) <= 1000:
            designs.append((labels, thicknesses))
    
    # Design 4: Random but constrained
    np.random.seed(42)
    for _ in range(5):
        labels = []
        thicknesses = []
        for i in range(n_layers):
            lab = "L" if np.random.random() < 0.5 else "H"
            labels.append(lab)
            t = np.random.uniform(20, 200)
            thicknesses.append(t)
        if sum(thicknesses) <= 1000:
            designs.append((labels, thicknesses))
    
    return designs

def _gradient_descent_polish(f, x0, bounds, lr=1.0, max_iter=500, time_deadline=None, tol=1e-12):
    """
    Gradient descent with numerical gradients and adaptive step size.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    
    eps = 1e-6
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        # Compute numerical gradient
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] = min(x[i] + eps, hi_bounds[i])
            x_minus = x.copy()
            x_minus[i] = max(x[i] - eps, lo_bounds[i])
            
            if x_plus[i] == x_minus[i]:
                grad[i] = 0.0
            else:
                grad[i] = (f(x_plus) - f(x_minus)) / (x_plus[i] - x_minus[i])
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Adaptive step size
        step = lr / (1.0 + 0.1 * it)
        
        # Try the step
        x_new = x - step * grad
        x_new = np.clip(x_new, lo_bounds, hi_bounds)
        
        f_new = f(x_new)
        if f_new < fval:
            x = x_new
            fval = f_new
            lr = min(lr * 1.1, 100.0)  # Increase step if improving
        else:
            lr = max(lr * 0.5, 0.01)  # Decrease step if not improving
            # Try smaller step
            x_new2 = x - 0.1 * step * grad
            x_new2 = np.clip(x_new2, lo_bounds, hi_bounds)
            f_new2 = f(x_new2)
            if f_new2 < fval:
                x = x_new2
                fval = f_new2
    
    return x, fval

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    def make_f(indices):
        def f(x):
            x = np.clip(x, 0, max_layer_nm)
            if np.sum(x) > max_total_nm:
                x = x * (max_total_nm / np.sum(x))
            R = _compute_reflectance(wavelengths_nm, indices, x, 
                                     ambient_idx, substrate_idx)
            return np.mean(R) * 100.0
        return f
    
    # Try different layer counts
    for n_layers in [4, 6, 8, 10, 12, 14]:
        if n_layers > max_layers:
            continue
        
        if time.time() > deadline - 20:
            break
        
        # Generate initial designs from theory
        initial_designs = _generate_initial_designs(n_layers, materials, ambient_idx, substrate_idx, center_wl)
        
        # Evaluate and polish each initial design
        for labels_init, thicknesses_init in initial_designs:
            if time.time() > deadline - 10:
                break
            
            n_polish = len(labels_init)
            if n_polish == 0:
                continue
            
            indices = [materials[label] for label in labels_init]
            bounds_thick = [(0.0, float(max_layer_nm))] * n_polish
            f = make_f(indices)
            x0 = np.array(thicknesses_init, dtype=float)
            
            # Evaluate initial
            f_init = f(x0)
            if f_init < best_score:
                best_score = f_init
                best_labels = labels_init[:]
                best_thicknesses = thicknesses_init[:]
            
            # Nelder-Mead polish
            if time.time() < deadline - 5:
                nm_deadline = min(time.time() + 10, deadline - 3)
                try:
                    x_opt, f_opt = _nelder_mead(f, x0, bounds_thick, max_iter=500, 
                                                 tol=1e-10, time_deadline=nm_deadline)
                    if f_opt < best_score:
                        best_score = f_opt
                        best_labels = labels_init[:]
                        best_thicknesses = x_opt.tolist()
                except Exception:
                    pass
            
            # Gradient descent polish
            if time.time() < deadline - 3:
                gd_deadline = min(time.time() + 5, deadline - 2)
                try:
                    x_gd, f_gd = _gradient_descent_polish(f, np.array(best_thicknesses), bounds_thick,
                                                          lr=10.0, max_iter=200, time_deadline=gd_deadline)
                    if f_gd < best_score:
                        best_score = f_gd
                        best_labels = labels_init[:]
                        best_thicknesses = x_gd.tolist()
                except Exception:
                    pass
            
            # Coordinate descent polish
            if time.time() < deadline - 2:
                cd_deadline = min(time.time() + 3, deadline - 1)
                try:
                    x_cd, f_cd = _coordinate_descent(f, np.array(best_thicknesses), bounds_thick, 
                                                     max_iter=100, time_deadline=cd_deadline)
                    if f_cd < best_score:
                        best_score = f_cd
                        best_labels = labels_init[:]
                        best_thicknesses = x_cd.tolist()
                except Exception:
                    pass
    
    # Fallback: single quarter-wave layer
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Final constraint check
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
