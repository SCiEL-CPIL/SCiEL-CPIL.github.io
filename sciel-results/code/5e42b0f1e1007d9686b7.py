"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import random

def _reflectance_spectrum(materials, thicknesses, band_nm, mat_dict, n_ambient=1.0, n_substrate=1.52, n_wavelengths=401):
    """Compute mean reflectance (%) over equally spaced wavelengths using characteristic matrix method."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, n_wavelengths)
    n_layers = len(materials)
    
    # Pre-compute indices using mat_dict
    n_list = np.array([n_ambient] + [mat_dict[m] for m in materials] + [n_substrate])
    d_list = np.array([0.0] + thicknesses + [0.0])
    
    total_R = 0.0
    for lam in wavelengths:
        # Phase thickness: delta = 2*pi*n*d/lam
        # Build characteristic matrix product
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        
        for i in range(n_layers):
            n_i = n_list[i+1]
            d_i = d_list[i+1]
            delta = 2.0 * np.pi * n_i * d_i / lam
            c = np.cos(delta)
            s = np.sin(delta)
            # Layer matrix
            M_i = np.array([[c, 1j*s/n_i], [1j*n_i*s, c]])
            M = M @ M_i
        
        M11, M12 = M[0,0], M[0,1]
        M21, M22 = M[1,0], M[1,1]
        
        denom = M11 + M12 / n_substrate
        if abs(denom) < 1e-30:
            total_R += 1.0
            continue
        Y = (M21 + M22 / n_substrate) / denom
        R = abs((n_ambient - Y) / (n_ambient + Y))**2
        total_R += R * 100.0  # Convert to percent
    
    return total_R / len(wavelengths)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) == 0:
        return False
    total = sum(thicknesses)
    if total > problem["max_total_nm"] + 1e-6:
        return False
    for t in thicknesses:
        if t < -1e-6 or t > problem["max_layer_nm"] + 1e-6:
            return False
    for m in materials:
        if m not in problem["materials"]:
            return False
    return True

def _optimize_thicknesses(materials, problem, max_iter=200, tol=1e-8, n_wavelengths=401):
    """Optimize thicknesses for a fixed set of materials using L-BFGS-B."""
    n_layers = len(materials)
    if n_layers == 0:
        return [], 100.0
    
    best_thick = list(thicknesses_init(materials, problem))
    mat_dict = problem["materials"]
    best_score = _reflectance_spectrum(materials, best_thick, problem["band_nm"], mat_dict, n_wavelengths=n_wavelengths)
    
    try:
        from scipy.optimize import minimize
        
        def objective(thick):
            # Use a coarser grid for optimization speed if requested, but here we assume n_wavelengths is passed
            # We use the same n_wavelengths as the caller for consistency in this function's context
            # However, to be safe and fast, we might use a lower res during optimization.
            # Let's stick to the provided n_wavelengths parameter for the objective.
            if any(t < -1e-6 for t in thick):
                return 100.0
            return _reflectance_spectrum(materials, list(thick), problem["band_nm"], mat_dict, n_wavelengths=n_wavelengths)
        
        x0 = np.array(best_thick)
        bounds = [(0, problem["max_layer_nm"])] * n_layers
        
        # L-BFGS-B is generally better for bounded smooth problems
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        
        if result.success or result.fun < best_score:
            new_thick = [max(0, min(problem["max_layer_nm"], t)) for t in result.x]
            # Check total thickness constraint manually if L-BFGS-B didn't strictly enforce it (it should)
            if sum(new_thick) <= problem["max_total_nm"] + 1e-6:
                if _is_valid(materials, new_thick, problem):
                    best_thick = new_thick
                    best_score = _reflectance_spectrum(materials, best_thick, problem["band_nm"], mat_dict, n_wavelengths=n_wavelengths)
    except Exception:
        pass
    
    return best_thick, best_score

def thicknesses_init(materials, problem):
    """Initialize thicknesses as quarter-wave at center wavelength."""
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    return [centre / (4.0 * problem["materials"][m]) for m in materials]

def design(problem):
    """Design a broadband antireflection coating using Simulated Annealing on layer sequence."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n_ambient = problem.get("ambient_index", 1.0)
    n_substrate = problem.get("substrate_index", 1.52)
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    best_materials = None
    best_thicknesses = None
    best_score = 100.0
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    def get_initial_thicknesses(mats):
        qw_thick = [centre / (4.0 * materials_dict[m]) for m in mats]
        if sum(qw_thick) > max_total_nm:
            scale = max_total_nm / sum(qw_thick)
            qw_thick = [t * scale for t in qw_thick]
        return qw_thick

    # 1. Evaluate some good initial candidates
    initial_candidates = [
        ["L", "H", "L"],
        ["H", "L", "H"],
        ["L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L", "H", "L"],
    ]
    
    for mats in initial_candidates:
        if time.time() > deadline - 30.0:
            break
        if len(mats) > max_layers:
            continue
        qw_thick = get_initial_thicknesses(mats)
        if not _is_valid(mats, qw_thick, problem):
            continue
        
        # Use a coarser grid for initial evaluation to save time
        score = _reflectance_spectrum(mats, qw_thick, band_nm, materials_dict, n_wavelengths=100)
        if score < best_score:
            best_materials = mats
            best_thicknesses = qw_thick
            best_score = score
        
        # Optimize thicknesses with coarse grid first, then fine
        opt_thick, opt_score = _optimize_thicknesses(mats, problem, max_iter=100, n_wavelengths=100)
        if opt_score < best_score and _is_valid(mats, opt_thick, problem):
            # Refine with fine grid
            opt_thick_fine, opt_score_fine = _optimize_thicknesses(mats, problem, max_iter=200, n_wavelengths=401)
            if opt_score_fine < best_score and _is_valid(mats, opt_thick_fine, problem):
                best_materials = mats
                best_thicknesses = opt_thick_fine
                best_score = opt_score_fine

    # 2. Simulated Annealing on the layer sequence
    if best_materials is None:
        best_materials = ["L", "H", "L"]
        best_thicknesses = get_initial_thicknesses(best_materials)
        best_score = _reflectance_spectrum(best_materials, best_thicknesses, band_nm, materials_dict, n_wavelengths=401)
    
    current_materials = list(best_materials)
    current_score = best_score
    
    T = 1.0  # Temperature
    T_min = 0.01
    cooling_rate = 0.95
    
    # Limit the number of SA steps based on time
    while time.time() < deadline - 10.0 and T > T_min:
        # Generate a neighbor
        new_materials = list(current_materials)
        op = random.choice(['swap', 'insert', 'delete', 'flip'])
        
        if op == 'swap' and len(new_materials) >= 2:
            i, j = random.sample(range(len(new_materials)), 2)
            new_materials[i], new_materials[j] = new_materials[j], new_materials[i]
        elif op == 'insert' and len(new_materials) < max_layers:
            i = random.randint(0, len(new_materials))
            new_materials.insert(i, random.choice(['L', 'H']))
        elif op == 'delete' and len(new_materials) > 1:
            i = random.randint(0, len(new_materials) - 1)
            new_materials.pop(i)
        elif op == 'flip' and len(new_materials) > 0:
            i = random.randint(0, len(new_materials) - 1)
            new_materials[i] = 'H' if new_materials[i] == 'L' else 'L'
        else:
            T *= cooling_rate
            continue
            
        if len(new_materials) == 0:
            continue
            
        # Check validity of total thickness with QWOT scaling
        qw_thick = get_initial_thicknesses(new_materials)
        if not _is_valid(new_materials, qw_thick, problem):
            continue
            
        # Evaluate with coarse grid
        new_score = _reflectance_spectrum(new_materials, qw_thick, band_nm, materials_dict, n_wavelengths=50)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_materials = new_materials
            current_score = new_score
            
            # If improved, do a quick local optimization on thicknesses
            if delta < 0:
                opt_thick, opt_score = _optimize_thicknesses(current_materials, problem, max_iter=50, n_wavelengths=100)
                if _is_valid(current_materials, opt_thick, problem):
                    # Check if this is a new global best with fine grid
                    fine_score = _reflectance_spectrum(current_materials, opt_thick, band_nm, materials_dict, n_wavelengths=401)
                    if fine_score < best_score:
                        best_score = fine_score
                        best_materials = list(current_materials)
                        best_thicknesses = list(opt_thick)
                        # Final polish on the best found so far
                        if time.time() < deadline - 5.0:
                            final_thick, final_score = _optimize_thicknesses(best_materials, problem, max_iter=300, n_wavelengths=401)
                            if _is_valid(best_materials, final_thick, problem) and final_score < best_score:
                                best_thicknesses = final_thick
                                best_score = final_score
        else:
            # Reject, but still cool down
            pass
            
        T *= cooling_rate
        
    # Final polish on the absolute best
    if time.time() < deadline - 2.0 and best_materials is not None:
        final_thick, final_score = _optimize_thicknesses(best_materials, problem, max_iter=500, n_wavelengths=401)
        if _is_valid(best_materials, final_thick, problem) and final_score < best_score:
            best_thicknesses = final_thick
            best_score = final_score
            
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * materials_dict["L"])]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
