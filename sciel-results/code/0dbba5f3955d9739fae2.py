"""Broadband antireflection coating design via differential evolution + Nelder-Mead polish."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _reflectance_spectrum(materials, thicknesses, band_nm, ambient_index, substrate_index):
    """Compute reflectance (fraction) at each wavelength using characteristic matrix method."""
    wl = np.linspace(band_nm[0], band_nm[1], 401)
    n_mats = [ambient_index] + materials + [substrate_index]
    # materials list corresponds to thicknesses list
    n_layers = len(thicknesses)
    n_vals = [ambient_index] + materials + [substrate_index]
    
    # Build the matrix M for each wavelength
    # For normal incidence, the characteristic matrix for a layer is:
    # [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d/lambda
    
    refl = np.zeros(len(wl))
    for i, lam in enumerate(wl):
        M = np.array([[1.0, 0.0]], dtype=complex)
        for j in range(n_layers):
            n_j = materials[j]
            d_j = thicknesses[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            c = np.cos(delta)
            s = np.sin(delta)
            M_j = np.array([[c, 1j * s / n_j],
                            [1j * n_j * s, c]], dtype=complex)
            M = M @ M_j
        
        # The total field matrix
        # M_total = M_stack @ [[1, 0], [0, 1/n_substrate]]
        # Actually, the standard formulation:
        # Let M = M_1 M_2 ... M_N
        # Then [E, H]_air = M @ [E, H]_substrate
        # At the substrate, H/E = n_substrate (for forward wave)
        # So [E, H]_air = M @ [1, n_substrate] (up to a constant)
        
        M_stack = M
        # The admittance at the air side
        Y = M_stack[0, 1] / M_stack[0, 0]  # This gives H/E in normalized units? Let me be careful.
        
        # Standard characteristic matrix method:
        # M = [[m11, m12], [m21, m22]]
        # The reflectance is | (m11 + m12 * n_s - n_0 * (m21 + m22 * n_s)) / (m11 + m12 * n_s + n_0 * (m21 + m22 * n_s)) |^2
        # Wait, let me use the standard formula:
        # r = (m11 + m12*n_s - n_0*(m21 + m22*n_s)) / (m11 + m12*n_s + n_0*(m21 + m22*n_s))
        
        m11 = M_stack[0, 0]
        m12 = M_stack[0, 1]
        m21 = M_stack[1, 0]
        m22 = M_stack[1, 1]
        
        n_0 = ambient_index
        n_s = substrate_index
        
        num = m11 + m12 * n_s - n_0 * (m21 + m22 * n_s)
        den = m11 + m12 * n_s + n_0 * (m21 + m22 * n_s)
        
        r = num / den
        refl[i] = np.abs(r) ** 2
    
    return refl

def _objective(params, problem):
    """Objective function: mean reflectance in percent, with penalty for constraint violations."""
    n_layers = problem["n_layers"]
    thicknesses = params[:n_layers]
    
    # Constraint: total thickness
    total = np.sum(thicknesses)
    if total > problem["max_total_nm"]:
        return 100.0 + (total - problem["max_total_nm"]) * 0.1
    
    # Constraint: individual layer thickness
    if np.any(thicknesses < 0) or np.any(thicknesses > problem["max_layer_nm"]):
        return 100.0 + 10.0
    
    materials = problem["materials_list"]
    band_nm = problem["band_nm"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    
    refl = _reflectance_spectrum(materials, thicknesses, band_nm, ambient_index, substrate_index)
    return 100.0 * np.mean(refl)

def design(problem):
    """Design an antireflection coating using differential evolution + Nelder-Mead polish."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Try different numbers of layers with alternating L/H pattern
    # Start with a reasonable number of layers
    best_score = float('inf')
    best_materials = ["L"]
    best_thicknesses = [0.5 * (band_nm[0] + band_nm[1]) / (4.0 * materials["L"])]
    
    # Candidate layer counts
    layer_counts = [4, 6, 8, 10, 12]
    
    for n_layers in layer_counts:
        if time.time() > deadline - 20.0:
            break
        
        # Create alternating L/H pattern starting with L
        mat_list = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        n_vals = [materials[m] for m in mat_list]
        
        # Set up bounds for differential evolution
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        # Add a constraint penalty for total thickness
        # We'll handle this in the objective function
        
        p = {
            "n_layers": n_layers,
            "materials_list": n_vals,
            "max_total_nm": max_total_nm,
            "max_layer_nm": max_layer_nm,
            "band_nm": band_nm,
            "ambient_index": ambient_index,
            "substrate_index": substrate_index
        }
        
        # Differential evolution
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(p,),
                maxiter=50,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            
            if result.success or result.fun < best_score:
                best_score = result.fun
                best_params = result.x
                
                # Nelder-Mead polish
                try:
                    nm_result = minimize(
                        _objective,
                        best_params,
                        args=(p,),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    if nm_result.fun < best_score:
                        best_score = nm_result.fun
                        best_params = nm_result.x
                except:
                    pass
                
                # Check constraints on final result
                if np.sum(best_params) <= max_total_nm and np.all(best_params >= 0) and np.all(best_params <= max_layer_nm):
                    best_materials = mat_list
                    best_thicknesses = list(best_params)
        except:
            continue
    
    # Return the best result found
    # Round to avoid floating point issues
    best_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    
    return best_materials, best_thicknesses
