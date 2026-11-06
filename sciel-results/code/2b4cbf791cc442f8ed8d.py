"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _compute_reflectance_vectorized(wavelengths, n_layers_list, thicknesses, n0, ns):
    """
    Vectorized characteristic-matrix reflectance computation.
    wavelengths: array of shape (N,)
    n_layers_list: list of refractive indices for each layer
    thicknesses: list of physical thicknesses in nm
    Returns reflectance array of shape (N,)
    """
    N = len(wavelengths)
    n_layers = len(n_layers_list)
    
    # Initialize M as identity for all wavelengths: shape (N, 2, 2)
    M = np.zeros((N, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for j in range(n_layers):
        n_j = n_layers_list[j]
        d_j = thicknesses[j]
        # Phase: k_j = 2*pi*n_j/wavelength
        k = 2.0 * np.pi * n_j / wavelengths  # shape (N,)
        cos_k = np.cos(k)
        sin_k = np.sin(k)
        
        # Layer matrix B for each wavelength
        B = np.zeros((N, 2, 2), dtype=complex)
        B[:, 0, 0] = cos_k
        B[:, 0, 1] = 1j * sin_k / n_j
        B[:, 1, 0] = 1j * n_j * sin_k
        B[:, 1, 1] = cos_k
        
        # M = M @ B for each wavelength
        # M_new[i] = M[i] @ B[i]
        M = np.einsum('nij,njk->nik', M, B)
    
    # Admittance Y = (M21 + M22*ns) / (M11 + M12*ns)
    M11 = M[:, 0, 0]
    M12 = M[:, 0, 1]
    M21 = M[:, 1, 0]
    M22 = M[:, 1, 1]
    
    Y = (M21 + M22 * ns) / (M11 + M12 * ns)
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2
    
    return R


def _mean_reflectance(wavelengths, n_layers_list, thicknesses, n0, ns):
    """Compute mean reflectance in percent."""
    R = _compute_reflectance_vectorized(wavelengths, n_layers_list, thicknesses, n0, ns)
    return 100.0 * np.mean(R)


def _nelder_mead(objective, x0, bounds, max_iter=200, tol=1e-6, deadline=None):
    """Custom Nelder-Mead simplex optimizer with deadline guard."""
    n = len(x0)
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        if x[i] != 0:
            x[i] *= 1.05
        else:
            x[i] = 0.05
        simplex.append(x)
    
    # Evaluate initial simplex
    values = [objective(x) for x in simplex]
    
    # Sort simplex by objective value
    order = np.argsort(values)
    simplex = [simplex[i] for i in order]
    values = [values[i] for i in order]
    
    alpha, gamma, rho = 1.0, 2.0, 0.5
    start_time = time.time()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)
        
        if fr < values[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            if fr < values[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = objective(xc)
            
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
        
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if abs(values[-1] - values[0]) < tol:
            break
    
    return simplex[0], values[0]


def _differential_evolution(objective, bounds, pop_size=50, max_iter=100, deadline=None, seed=42):
    """Differential evolution optimizer."""
    rng = np.random.RandomState(seed)
    n = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.zeros((pop_size, n))
    for i in range(pop_size):
        pop[i] = lb + rng.rand(n) * (ub - lb)
    
    values = np.array([objective(x) for x in pop])
    
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    start_time = time.time()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find best
        idx_best = np.argmin(values)
        x_best = pop[idx_best]
        f_best = values[idx_best]
        
        # Update each member
        for i in range(pop_size):
            # Select 3 distinct random indices
            r1, r2 = rng.choice(pop_size, 2, replace=False)
            while r1 == i or r2 == i:
                r1, r2 = rng.choice(pop_size, 2, replace=False)
            
            # Differential mutation
            v = pop[r1] + F * (pop[r2] - pop[i])
            v = np.clip(v, lb, ub)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(n)
            for j in range(n):
                if rng.rand() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]
            
            trial = np.clip(trial, lb, ub)
            f_trial = objective(trial)
            
            if f_trial <= values[i]:
                pop[i] = trial
                values[i] = f_trial
        
        # Check convergence
        if np.std(values) < 1e-8:
            break
    
    idx_best = np.argmin(values)
    return pop[idx_best], values[idx_best]


def _optimize_stack(problem):
    """Optimize the coating stack with focus on higher layer counts."""
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    n_wl = 401
    wavelengths = np.linspace(wl_min, wl_max, n_wl)
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    center_wl = 0.5 * (wl_min + wl_max)
    
    best_result = None
    best_score = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Focus on higher layer counts first (known to achieve lower reflectance)
    # Try layer counts: 12, 14, 16, 18, 20, then 8, 10 if time permits
    layer_counts = [12, 14, 16, 18, 20, 8, 10]
    
    for n_layers in layer_counts:
        if n_layers > max_layers:
            continue
        
        if time.time() > deadline:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate time proportionally: more time for higher layer counts
        # Give at least 10s per configuration, up to remaining time
        time_alloc = min(remaining * 0.4, 30.0)
        config_deadline = time.time() + time_alloc
        
        # Initialize with alternating L/H, starting with L
        mat_labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                mat_labels.append('L')
            else:
                mat_labels.append('H')
        
        # Initial thicknesses: quarter-wave at center wavelength
        init_thicknesses = []
        for m in mat_labels:
            n_mat = nL if m == 'L' else nH
            d = center_wl / (4.0 * n_mat)
            init_thicknesses.append(d)
        
        # Also try starting with H
        mat_labels_h = []
        for i in range(n_layers):
            if i % 2 == 0:
                mat_labels_h.append('H')
            else:
                mat_labels_h.append('L')
        init_thicknesses_h = []
        for m in mat_labels_h:
            n_mat = nL if m == 'L' else nH
            d = center_wl / (4.0 * n_mat)
            init_thicknesses_h.append(d)
        
        # Define objective function
        def make_objective(mat_labels):
            n_layers_list = [nL if m == 'L' else nH for m in mat_labels]
            def objective(x):
                thicknesses = list(x)
                if any(t < 0 or t > max_layer for t in thicknesses):
                    return 100.0
                if sum(thicknesses) > max_total:
                    return 100.0
                return _mean_reflectance(wavelengths, n_layers_list, thicknesses, n0, ns)
            return objective
        
        bounds = [(0.0, max_layer) for _ in range(n_layers)]
        
        # Run differential evolution for global exploration
        try:
            obj_func = make_objective(mat_labels)
            x_de, f_de = _differential_evolution(obj_func, bounds, pop_size=30, 
                                                  max_iter=50, deadline=config_deadline)
            if f_de < best_score:
                best_score = f_de
                best_result = (list(mat_labels), list(x_de))
        except Exception:
            pass
        
        if time.time() > config_deadline:
            continue
        
        # Polish with Nelder-Mead from DE result
        try:
            obj_func = make_objective(mat_labels)
            x_nm, f_nm = _nelder_mead(obj_func, x_de, bounds, max_iter=100, 
                                       tol=1e-6, deadline=config_deadline)
            if f_nm < best_score:
                best_score = f_nm
                best_result = (list(mat_labels), list(x_nm))
        except Exception:
            pass
        
        # Also try from quarter-wave initialization
        if time.time() < config_deadline - 2.0:
            try:
                obj_func = make_objective(mat_labels)
                x_nm2, f_nm2 = _nelder_mead(obj_func, init_thicknesses, bounds, 
                                             max_iter=50, tol=1e-5, deadline=config_deadline)
                if f_nm2 < best_score:
                    best_score = f_nm2
                    best_result = (list(mat_labels), list(x_nm2))
            except Exception:
                pass
        
        # Try H-first pattern
        if time.time() < config_deadline - 2.0:
            try:
                obj_func_h = make_objective(mat_labels_h)
                x_nm3, f_nm3 = _nelder_mead(obj_func_h, init_thicknesses_h, bounds, 
                                             max_iter=50, tol=1e-5, deadline=config_deadline)
                if f_nm3 < best_score:
                    best_score = f_nm3
                    best_result = (list(mat_labels_h), list(x_nm3))
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: single quarter-wave layer
        d = center_wl / (4.0 * nL)
        return ["L"], [d]


def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    
    try:
        result = _optimize_stack(problem)
    except Exception as e:
        center = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        d = center / (4.0 * problem["materials"]["L"])
        result = ["L"], [d]
    
    materials, thicknesses = result
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    if len(materials) > max_layers:
        materials = materials[:max_layers]
        thicknesses = thicknesses[:max_layers]
    
    thicknesses = [max(0.0, min(t, max_layer)) for t in thicknesses]
    
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses
# EVOLVE-BLOCK-END
