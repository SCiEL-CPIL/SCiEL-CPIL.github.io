"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _compute_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using vectorized characteristic matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    
    # 401 equally spaced wavelengths from 400 to 800 nm
    wavelengths = np.linspace(band[0], band[1], 401)
    
    n_dict = {'L': nL, 'H': nH}
    n_layers = len(materials)
    
    if n_layers == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * np.abs(r) ** 2
    
    k = 2 * np.pi / wavelengths  # shape (401,)
    
    # Initialize M as identity for all wavelengths: shape (401, 2, 2)
    M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for mat, d in zip(materials, thicknesses):
        n = n_dict[mat]
        delta = k * n * d  # shape (401,)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # layer_M for each wavelength: shape (401, 2, 2)
        layer_M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        layer_M[:, 0, 0] = cos_d
        layer_M[:, 0, 1] = 1j * sin_d / n
        layer_M[:, 1, 0] = 1j * n * sin_d
        layer_M[:, 1, 1] = cos_d
        
        # M = M @ layer_M for each wavelength (batched matmul)
        M = np.einsum('wij,wjk->wik', M, layer_M)
    
    # Reflectance at normal incidence
    numerator = n0 * M[:, 0, 0] + n0 * M[:, 0, 1] - ns * M[:, 1, 0] - ns * M[:, 1, 1]
    denominator = n0 * M[:, 0, 0] + n0 * M[:, 0, 1] + ns * M[:, 1, 0] + ns * M[:, 1, 1]
    r = numerator / denominator
    R = np.abs(r) ** 2
    
    return 100.0 * np.mean(R)

def _quarter_wave_stack(n_layers, problem):
    """Generate a quarter-wave stack design."""
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # For a quarter-wave stack, we alternate L and H starting with L
    # Thickness of each quarter-wave layer at center wavelength
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)
    
    materials = []
    thicknesses = []
    
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append('L')
            thicknesses.append(dL)
        else:
            materials.append('H')
            thicknesses.append(dH)
    
    return materials, thicknesses

def _nelder_mead(objective, x0, max_iter=200, tol=1e-4):
    """Simple Nelder-Mead implementation using only NumPy."""
    n = len(x0)
    # Initialize simplex
    simplex = [x0]
    for i in range(n):
        new_point = x0.copy()
        new_point[i] += 0.1 if x0[i] != 0 else 1.0
        simplex.append(new_point)
    
    f_vals = [objective(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort simplex
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            gamma = 2.0
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        # Contraction
        rho = 0.5
        x_c = centroid + rho * (simplex[-1] - centroid)
        f_c = objective(x_c)
        
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
        else:
            # Shrink
            for i in range(1, n + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                f_vals[i] = objective(simplex[i])
                
    return simplex[0]

def _optimize_stack(materials, problem, max_iter=500):
    """Optimize thicknesses for a given material sequence using custom Nelder-Mead."""
    n_layers = len(materials)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Initial guess: quarter-wave thicknesses
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    x0 = []
    for mat in materials:
        n = nL if mat == 'L' else nH
        x0.append(centre / (4.0 * n))
    
    def objective(x):
        # Check constraints
        if any(d < 0 or d > max_layer_nm for d in x):
            return 100.0
        if sum(x) > max_total_nm:
            return 100.0
        return _compute_reflectance(materials, x, problem)
    
    # Use custom Nelder-Mead
    x_opt = _nelder_mead(objective, x0, max_iter=max_iter)
    
    # Project back to feasible region
    x_opt = np.clip(x_opt, 0, max_layer_nm)
    total = np.sum(x_opt)
    if total > max_total_nm:
        x_opt = x_opt * (max_total_nm / total)
    
    return list(x_opt)

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    max_layers = problem["max_layers"]
    best_score = float('inf')
    best_design = None
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Try different numbers of layers
    # Focus on 3-12 layers which give good broadband AR performance
    for n_layers in range(1, min(max_layers + 1, 13)):
        if time.time() > deadline:
            break
        
        # Try alternating L-H stack starting with L
        materials, thicknesses = _quarter_wave_stack(n_layers, problem)
        
        # Check if it fits within constraints
        if all(0 <= d <= max_layer_nm for d in thicknesses) and sum(thicknesses) <= max_total_nm:
            score = _compute_reflectance(materials, thicknesses, problem)
            if score < best_score:
                best_score = score
                best_design = (list(materials), list(thicknesses))
        
        # Optimize the thicknesses with more iterations now that eval is faster
        if time.time() > deadline:
            break
        
        # Scale max_iter based on remaining time
        remaining = deadline - time.time()
        if remaining > 30:
            max_iter = 500
        elif remaining > 15:
            max_iter = 300
        else:
            max_iter = 100
        
        opt_thicknesses = _optimize_stack(materials, problem, max_iter=max_iter)
        
        # Check constraints
        if all(0 <= d <= max_layer_nm for d in opt_thicknesses) and sum(opt_thicknesses) <= max_total_nm:
            score = _compute_reflectance(materials, opt_thicknesses, problem)
            if score < best_score:
                best_score = score
                best_design = (list(materials), list(opt_thicknesses))
        
        # Also try starting with H
        if n_layers >= 2 and time.time() < deadline - 5:
            materials_h = []
            thicknesses_h = []
            for i in range(n_layers):
                if i % 2 == 0:
                    materials_h.append('H')
                    thicknesses_h.append(thicknesses[1] if n_layers > 1 else thicknesses[0])
                else:
                    materials_h.append('L')
                    thicknesses_h.append(thicknesses[0] if n_layers > 1 else thicknesses[1])
            
            # Rebuild properly
            materials_h = []
            thicknesses_h = []
            centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
            for i in range(n_layers):
                if i % 2 == 0:
                    materials_h.append('H')
                    thicknesses_h.append(centre / (4.0 * problem["materials"]["H"]))
                else:
                    materials_h.append('L')
                    thicknesses_h.append(centre / (4.0 * problem["materials"]["L"]))
            
            if all(0 <= d <= max_layer_nm for d in thicknesses_h) and sum(thicknesses_h) <= max_total_nm:
                score = _compute_reflectance(materials_h, thicknesses_h, problem)
                if score < best_score:
                    best_score = score
                    best_design = (list(materials_h), list(thicknesses_h))
                
                if time.time() < deadline - 5:
                    opt_thicknesses_h = _optimize_stack(materials_h, problem, max_iter=200)
                    if all(0 <= d <= max_layer_nm for d in opt_thicknesses_h) and sum(opt_thicknesses_h) <= max_total_nm:
                        score = _compute_reflectance(materials_h, opt_thicknesses_h, problem)
                        if score < best_score:
                            best_score = score
                            best_design = (list(materials_h), list(opt_thicknesses_h))
    
    # If no valid design found, return a simple quarter-wave layer
    if best_design is None:
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]
    
    return best_design
# EVOLVE-BLOCK-END
