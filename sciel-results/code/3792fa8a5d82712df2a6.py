"""Design broadband antireflection coating using numerical optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance_spectrum(thicknesses, indices, n_layers, wavelengths, ambient, substrate):
    """
    Compute normal-incidence reflectance using characteristic matrix method.
    Returns reflectance as a fraction (0 to 1).
    """
    n = n_layers
    if n == 0:
        # Bare substrate
        r = (ambient - substrate) / (ambient + substrate)
        return np.abs(r)**2 * np.ones_like(wavelengths)

    # Initialize M as identity
    m11 = 1.0 + 0.0j
    m12 = 0.0 + 0.0j
    m21 = 0.0 + 0.0j
    m22 = 1.0 + 0.0j
    
    for i in range(n):
        ni = indices[i]
        di = thicknesses[i]
        delta = 2.0 * np.pi * ni * di / wavelengths
        cd = np.cos(delta)
        sd = np.sin(delta)
        
        # Layer matrix
        li11 = cd
        li12 = 1j * sd / ni
        li21 = 1j * ni * sd
        li22 = cd
        
        # M_new = M_old * M_layer
        nm11 = m11 * li11 + m12 * li21
        nm12 = m11 * li12 + m12 * li22
        nm21 = m21 * li11 + m22 * li21
        nm22 = m21 * li12 + m22 * li22
        
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22
    
    # Effective admittance
    y0 = (m21 + m22 * substrate) / (m11 + m12 * substrate)
    r = (y0 - ambient) / (y0 + ambient)
    R = np.abs(r) ** 2
    return R

def design(problem):
    """
    Design a broadband antireflection coating using differential evolution
    to optimize layer thicknesses for various candidate material sequences.
    """
    ambient = problem["ambient_index"]
    substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points as per spec)
    wavelengths = np.linspace(wl_min, wl_max, 401)
    
    # Objective function: mean reflectance
    def objective(thicknesses_flat):
        # Reshape thicknesses to match the sequence length
        # We pass the sequence and indices via closure or global state, 
        # but differential_evolution requires a single array.
        # We will define specific objective functions for each sequence.
        pass # Placeholder

    best_R = np.inf
    best_labels = ["L"]
    best_thicknesses = [wl_min / (4.0 * n_L)] # Fallback
    
    # Generate candidate sequences
    # We want sequences that allow for a graded index profile.
    # Common good structures:
    # 1. Single layer L
    # 2. L H
    # 3. L H L
    # 4. L H L H L
    # 5. L H L H L H L H L
    # 6. L H L H L H L H L H L H L H L
    # Also try starting with H:
    # 7. H L
    # 8. H L H
    # etc.
    
    # To save time, we will test a few representative sequences.
    # We limit the number of layers to keep the optimization fast.
    # Differential evolution is stochastic and can be slow for high dimensions.
    # Let's try up to 9 layers.
    
    candidates = []
    # Sequence 1: L
    candidates.append(["L"])
    # Sequence 2: L H
    candidates.append(["L", "H"])
    # Sequence 3: L H L
    candidates.append(["L", "H", "L"])
    # Sequence 4: L H L H L
    candidates.append(["L", "H", "L", "H", "L"])
    # Sequence 5: L H L H L H L H L
    candidates.append(["L", "H", "L", "H", "L", "H", "L", "H", "L"])
    # Sequence 6: H L H L H L H L H
    candidates.append(["H", "L", "H", "L", "H", "L", "H", "L", "H"])
    # Sequence 7: L H L H L H L (7 layers)
    candidates.append(["L", "H", "L", "H", "L", "H", "L"])
    # Sequence 8: H L H L H L H
    candidates.append(["H", "L", "H", "L", "H", "L", "H"])
    
    for seq in candidates:
        n_layers = len(seq)
        indices = [n_H if l == 'H' else n_L for l in seq]
        
        # Define bounds for thicknesses
        # Each layer thickness is in [0, max_layer_nm]
        # Additionally, sum of thicknesses <= max_total_nm
        # Differential evolution doesn't handle linear inequality constraints directly easily.
        # We can penalize violations in the objective function.
        
        bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
        
        def obj(thicknesses):
            # Check total thickness constraint
            if np.sum(thicknesses) > max_total_nm:
                return 1.0 # Penalize heavily
            
            R = _reflectance_spectrum(thicknesses, indices, n_layers, wavelengths, ambient, substrate)
            return np.mean(R)
        
        # Run differential evolution
        # Max iterations and popsize tuned for speed (60s budget)
        # For n_layers up to 9, this should be fast enough.
        try:
            result = differential_evolution(
                obj, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-4, 
                seed=42,
                polish=False # Disable Nelder-Mead polish to save time, DE is usually good enough
            )
            
            if result.success or result.fun < best_R:
                # Check if the result is valid (it might have converged to a penalty value)
                if result.fun < 1.0: # If not penalized
                    R_val = result.fun
                    # Verify constraints strictly
                    if np.sum(result.x) <= max_total_nm:
                        if R_val < best_R:
                            best_R = R_val
                            best_labels = seq
                            best_thicknesses = list(result.x)
        except Exception:
            continue

    # If optimization failed or didn't improve, try a simple heuristic fallback
    # which is better than the seed's fixed quarter-wave if the seed's was bad,
    # but here we assume DE found something.
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
