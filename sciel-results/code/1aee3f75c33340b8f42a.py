"""Broadband antireflection coating design via multi-start local optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _char_matrix(n, d, wl):
    """Compute the characteristic matrix for a single layer at wavelength wl."""
    beta = 2.0 * math.pi * n * d / wl
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    return np.array([[cos_b, 1j * sin_b / n],
                     [1j * n * sin_b, cos_b]], dtype=complex)

def _compute_reflectance(materials, thicknesses, wl, n_ambient, n_substrate):
    """Compute reflectance at a single wavelength using characteristic matrix method."""
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return abs(r) ** 2
    
    # Build total characteristic matrix
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for i in range(n_layers):
        n_mat = n_ambient if False else None  # placeholder
        # We need the actual refractive index
        pass
    
    # Actually, we need to pass the refractive indices. Let me restructure.
    return 0.0

def _compute_reflectance_full(materials, thicknesses, wl, n_ambient, n_substrate, n_dict):
    """Compute reflectance at a single wavelength."""
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return abs(r) ** 2
    
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for i in range(n_layers):
        n_mat = n_dict[materials[i]]
        d_mat = thicknesses[i]
        Mi = _char_matrix(n_mat, d_mat, wl)
        M = M @ Mi
    
    # Boundary conditions
    # Y = (M[0,0] + M[0,1]*n_substrate) / (M[1,0] + M[1,1]*n_substrate)
    # Actually, the admittance Y = (M11 + M12 * n_s) / (M21 + M22 * n_s)
    # Then r = (n_0 - Y) / (n_0 + Y) where n_0 is ambient
    
    Y = (M[0, 0] + M[0, 1] * n_substrate) / (M[1, 0] + M[1, 1] * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    return abs(r) ** 2

def _mean_reflectance_percent(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wl_start, wl_end = band[0], band[1]
    n_wl = 401
    wls = np.linspace(wl_start, wl_end, n_wl)
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_dict = problem["materials"]
    
    total_R = 0.0
    for wl in wls:
        R = _compute_reflectance_full(materials, thicknesses, float(wl), n_ambient, n_substrate, n_dict)
        total_R += R
    
    mean_R = total_R / n_wl
    return mean_R * 100.0

def _nelder_mead(objective, x0, max_iter=200, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tol=1e-6):
    """Nelder-Mead simplex optimization."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        xi = list(x0)
        xi[i] = xi[i] + 0.1 if xi[i] == 0 else xi[i] * 1.1
        simplex.append(xi)
    
    fvals = [objective(np.array(s)) for s in simplex]
    
    for _ in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean([np.array(simplex[i]) for i in range(n)], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - np.array(simplex[n]))
        fr = objective(xr)
        
        if fvals[0] < fr < fvals[n-1]:
            simplex[n] = list(xr)
            fvals[n] = fr
            continue
        
        # Expansion
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[n] = list(xe)
                fvals[n] = fe
            else:
                simplex[n] = list(xr)
                fvals[n] = fr
            continue
        
        # Contraction
        if fr < fvals[n-1]:
            # Outside contraction
            xc = centroid + sigma * (xr - centroid)
            fc = objective(xc)
            if fc <= fr:
                simplex[n] = list(xc)
                fvals[n] = fc
            else:
                # Inside contraction
                xi = centroid + sigma * (np.array(simplex[n]) - centroid)
                fi = objective(xi)
                if fi < fvals[n-1]:
                    simplex[n] = list(xi)
                    fvals[n] = fi
                else:
                    break
        else:
            # Inside contraction
            xi = centroid + sigma * (np.array(simplex[n]) - centroid)
            fi = objective(xi)
            if fi < fvals[n-1]:
                simplex[n] = list(xi)
                fvals[n] = fi
            else:
                break
        
        # Shrink
        for i in range(1, n+1):
            simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
            fvals[i] = objective(np.array(simplex[i]))
    
    order = np.argsort(fvals)
    return np.array(simplex[order[0]]), fvals[order[0]]

def _validate_and_clamp(materials, thicknesses, problem):
    """Validate and clamp thicknesses to satisfy constraints."""
    max_layer = problem.get("max_layer_nm", 400)
    max_total = problem.get("max_total_nm", 1000)
    
    thicknesses = list(thicknesses)
    # Clamp individual layers
    for i in range(len(thicknesses)):
        thicknesses[i] = max(0.0, min(max_layer, thicknesses[i]))
    
    # Check total
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses

def design(problem):
    """Design a broadband antireflection coating."""
    t_start = time.time()
    deadline = t_start + 120.0  # Safety margin before 150s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_dict = problem["materials"]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    def objective(x):
        """Objective function for optimization. x is a flat array of thicknesses."""
        n_layers = len(x)
        if n_layers == 0:
            return _mean_reflectance_percent([], [], problem)
        
        # Determine materials pattern
        # We'll use a fixed pattern for each starting point
        # This is handled by the closure over materials_pattern
        materials = list(materials_pattern)
        thicknesses = [max(0.0, min(max_layer, float(xi))) for xi in x]
        
        # Check total constraint
        total = sum(thicknesses)
        if total > max_total:
            # Penalize or scale
            scale = max_total / total
            thicknesses = [t * scale for t in thicknesses]
        
        return _mean_reflectance_percent(materials, thicknesses, problem)
    
    # Try several different layer patterns
    patterns = []
    
    # Pattern 1: Alternating L,H starting with L
    for n_layers in range(1, min(max_layers, 8) + 1):
        pattern = []
        for i in range(n_layers):
            pattern.append('L' if i % 2 == 0 else 'H')
        patterns.append(pattern)
    
    # Pattern 2: Alternating H,L starting with H
    for n_layers in range(1, min(max_layers, 8) + 1):
        pattern = []
        for i in range(n_layers):
            pattern.append('H' if i % 2 == 0 else 'L')
        patterns.append(pattern)
    
    # Pattern 3: All L
    for n_layers in range(1, min(max_layers, 5) + 1):
        patterns.append(['L'] * n_layers)
    
    # Pattern 4: All H
    for n_layers in range(1, min(max_layers, 5) + 1):
        patterns.append(['H'] * n_layers)
    
    # Pattern 5: Quarter-wave stack
    centre_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    for n_layers in range(1, min(max_layers, 6) + 1):
        pattern = []
        for i in range(n_layers):
            pattern.append('L' if i % 2 == 0 else 'H')
        patterns.append(pattern)
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    # For each pattern, try to optimize
    for materials_pattern in unique_patterns:
        if time.time() > deadline - 5.0:
            break
        
        n_layers = len(materials_pattern)
        if n_layers == 0:
            continue
        
        # Initial guess: quarter-wave thicknesses
        centre_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        x0 = []
        for i in range(n_layers):
            n_mat = n_dict[materials_pattern[i]]
            d_qw = centre_wl / (4.0 * n_mat)
            x0.append(d_qw)
        
        x0 = np.array(x0)
        
        # Run Nelder-Mead
        try:
            x_opt, f_opt = _nelder_mead(objective, x0, max_iter=100)
            
            if f_opt < best_score:
                best_score = f_opt
                best_materials = list(materials_pattern)
                best_thicknesses = [max(0.0, min(max_layer, float(xi))) for xi in x_opt]
                
                # Check total constraint
                total = sum(best_thicknesses)
                if total > max_total:
                    scale = max_total / total
                    best_thicknesses = [t * scale for t in best_thicknesses]
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to a simple design
    if best_materials is None:
        centre_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        best_materials = ['L']
        best_thicknesses = [centre_wl / (4.0 * n_dict['L'])]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
