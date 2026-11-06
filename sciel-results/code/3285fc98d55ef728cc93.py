"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths using the characteristic matrix method (vectorized).
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    # Initialize total matrix as identity for all wavelengths
    # Shape: (N_wls, 2, 2)
    total = np.eye(2, dtype=complex)
    total = np.broadcast_to(total, (len(wls), 2, 2)).copy()
    
    for mat, thick in zip(materials, thicknesses):
        if thick < 1e-9:
            continue
        n = materials_dict[mat]
        d = thick
        # beta = 2*pi*n*d / wls
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Matrix M for this layer at each wavelength
        # M = [[cos, i*sin/n], [i*n*sin, cos]]
        # We need to compute total @ M for each wavelength
        # total is (N, 2, 2), M is (N, 2, 2)
        # This is a batched matrix multiplication
        
        # M[0,0] = cos, M[0,1] = i*sin/n
        # M[1,0] = i*n*sin, M[1,1] = cos
        
        # new_total[0,0] = total[0,0]*cos + total[0,1]*(i*sin/n)
        # new_total[0,1] = total[0,0]*(i*sin/n) + total[0,1]*cos
        # new_total[1,0] = total[1,0]*cos + total[1,1]*(i*n*sin)
        # new_total[1,1] = total[1,0]*(i*n*sin) + total[1,1]*cos
        
        t00 = total[:, 0, 0]
        t01 = total[:, 0, 1]
        t10 = total[:, 1, 0]
        t11 = total[:, 1, 1]
        
        new_t00 = t00 * cos_b + t01 * (1.0j * sin_b / n)
        new_t01 = t00 * (1.0j * sin_b / n) + t01 * cos_b
        new_t10 = t10 * cos_b + t11 * (1.0j * n * sin_b)
        new_t11 = t10 * (1.0j * n * sin_b) + t11 * cos_b
        
        total[:, 0, 0] = new_t00
        total[:, 0, 1] = new_t01
        total[:, 1, 0] = new_t10
        total[:, 1, 1] = new_t11

    # Calculate admittance Y
    # Y = (T[1,0] + T[1,1]*Y_s) / (T[0,0] + T[0,1]*Y_s)
    Y_s = substrate_index
    Y_0 = ambient_index
    
    B = total[:, 0, 0]
    C = total[:, 0, 1]
    D = total[:, 1, 0]
    E = total[:, 1, 1]
    
    Y = (D + E * Y_s) / (B + C * Y_s)
    
    r = (Y_0 - Y) / (Y_0 + Y)
    return np.abs(r) ** 2

def _mean_reflectance(materials, thicknesses, band_nm=(400, 800), n_points=401,
                       ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute mean reflectance in percent over the band."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _nelder_mead(func, x0, bounds, max_iter=500, tol=1e-8, deadline=None):
    """
    Custom Nelder-Mead implementation with bounds and time check.
    func: objective function
    x0: initial guess
    bounds: list of (min, max) for each dimension
    """
    n = len(x0)
    x = np.array(x0, dtype=float)
    
    # Ensure initial point is within bounds
    for i in range(n):
        x[i] = np.clip(x[i], bounds[i][0], bounds[i][1])
        
    # Initialize simplex
    simplex = [x.copy()]
    for i in range(n):
        new_point = x.copy()
        step = 0.1 if x[i] != 0.0 else 1.0
        new_point[i] += step
        new_point[i] = np.clip(new_point[i], bounds[i][0], bounds[i][1])
        simplex.append(new_point)
        
    values = [func(p) for p in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_val = min(values)
    best_x = simplex[values.index(best_val)].copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        # Clip to bounds
        for i in range(n):
            xr[i] = np.clip(xr[i], bounds[i][0], bounds[i][1])
        fr = func(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            for i in range(n):
                xe[i] = np.clip(xe[i], bounds[i][0], bounds[i][1])
            fe = func(xe)
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
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            for i in range(n):
                xc[i] = np.clip(xc[i], bounds[i][0], bounds[i][1])
            fc = func(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    for j in range(n):
                        simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
                    values[i] = func(simplex[i])
                    
        # Update best
        min_idx = np.argmin(values)
        if values[min_idx] < best_val:
            best_val = values[min_idx]
            best_x = simplex[min_idx].copy()
            
    return best_x, best_val

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    
    best_score = np.inf
    best_materials = ["L"]
    best_thicknesses = [wl_center / (4.0 * nL)]
    
    # Define objective function factory
    def make_obj(mats):
        def obj(thicks):
            # Check bounds quickly (Nelder-Mead handles clipping, but let's be safe)
            if any(t < 0 or t > max_layer_nm for t in thicks):
                return 100.0
            if sum(thicks) > max_total_nm:
                return 100.0
            return _mean_reflectance(mats, thicks, band_nm, 401, n_amb, n_sub, materials_dict)
        return obj
    
    # Try different layer counts and sequences
    # Optimal broadband AR coatings often use alternating L/H layers.
    # Start with 1, 2, 3, 4 layers.
    
    sequences_to_try = []
    
    # 1 layer
    sequences_to_try.append(["L"])
    sequences_to_try.append(["H"])
    
    # 2 layers
    sequences_to_try.append(["L", "H"])
    sequences_to_try.append(["H", "L"])
    
    # 3 layers
    sequences_to_try.append(["L", "H", "L"])
    sequences_to_try.append(["H", "L", "H"])
    sequences_to_try.append(["L", "L", "H"])
    sequences_to_try.append(["H", "H", "L"])
    
    # 4 layers
    sequences_to_try.append(["L", "H", "L", "H"])
    sequences_to_try.append(["H", "L", "H", "L"])
    
    # 5 layers
    sequences_to_try.append(["L", "H", "L", "H", "L"])
    sequences_to_try.append(["H", "L", "H", "L", "H"])
    
    for mats in sequences_to_try:
        if time.time() > deadline:
            break
            
        n_layers = len(mats)
        
        # Initial guess: quarter-wave at center wavelength
        x0 = []
        for m in mats:
            n = nL if m == 'L' else nH
            x0.append(wl_center / (4.0 * n))
            
        # Bounds
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # Check if total thickness is feasible with initial guess
        if sum(x0) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(x0) * 0.9
            x0 = [x * scale for x in x0]
            
        obj = make_obj(mats)
        
        try:
            x_opt, val_opt = _nelder_mead(obj, x0, bounds, max_iter=200, tol=1e-6, deadline=deadline)
            
            # Validate final result
            if _is_valid(mats, x_opt, problem):
                # Re-evaluate to be sure
                val_check = obj(x_opt)
                if val_check < best_score:
                    best_score = val_check
                    best_materials = mats
                    best_thicknesses = x_opt
        except Exception:
            continue

    # If time remains, do a quick local search around the best found
    if time.time() < deadline - 10 and len(best_thicknesses) > 0:
        # Try a few random perturbations
        np.random.seed(42)
        for _ in range(20):
            if time.time() > deadline:
                break
            new_thicks = best_thicknesses[:]
            # Perturb one random thickness
            idx = np.random.randint(0, len(new_thicks))
            new_thicks[idx] = new_thicks[idx] * np.random.uniform(0.8, 1.2)
            
            if _is_valid(best_materials, new_thicks, problem):
                score = _mean_reflectance(best_materials, new_thicks, band_nm, 401, n_amb, n_sub, materials_dict)
                if score < best_score:
                    best_score = score
                    best_thicknesses = new_thicks

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
