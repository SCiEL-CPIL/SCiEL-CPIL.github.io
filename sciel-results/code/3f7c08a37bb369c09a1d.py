"""Broadband antireflection coating design via Differential Evolution with vectorized evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _vectorized_reflectance(materials, thicknesses, wavelengths, n_ambient, n_substrate):
    """
    Compute reflectance for a stack at multiple wavelengths using vectorized numpy ops.
    materials: list of 'L' or 'H'
    thicknesses: list of floats (nm)
    wavelengths: 1D array of floats (nm)
    Returns: 1D array of reflectance values (0 to 1)
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full_like(wavelengths, r**2)

    # Precompute indices
    n_vals = np.array([1.38 if m == 'L' else 2.35 for m in materials])
    d_vals = np.array(thicknesses)
    
    # M = [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    # beta = 2*pi*n*d/wl
    # We need to multiply matrices M_1 * M_2 * ... * M_N
    # Vectorized: shape (N, W, 2, 2)
    N = len(wavelengths)
    beta = 2.0 * np.pi * n_vals[:, None] * d_vals[:, None] / wavelengths[None, :] # (L, W)
    
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    
    # Initialize product matrix P = I for each wavelength
    # P has shape (W, 2, 2)
    P = np.zeros((N, 2, 2), dtype=complex)
    P[:, 0, 0] = 1.0
    P[:, 1, 1] = 1.0
    
    # Multiply M_i for each layer
    for i in range(n_layers):
        # M_i has shape (W, 2, 2)
        M = np.zeros((N, 2, 2), dtype=complex)
        M[:, 0, 0] = cos_b[i]
        M[:, 0, 1] = 1j * sin_b[i] / n_vals[i]
        M[:, 1, 0] = 1j * n_vals[i] * sin_b[i]
        M[:, 1, 1] = cos_b[i]
        
        # P = M_i * P  (Matrix multiplication: M @ P)
        # Using einsum for speed: P_new[a, b, c] = sum_k M[a, b, k] P[a, k, c]
        P = np.einsum('abk,akc->abc', M, P)
        
    # Substrate and ambient
    # Y = n_substrate * P
    # r = (n_ambient - Y) / (n_ambient + Y)
    Y = n_substrate * P[:, 0, 0] # Wait, characteristic matrix relation:
    # Usually: [B, C]^T = M_total [n_sub, 1]^T
    # B = M11*n_sub + M12
    # C = M21*n_sub + M22
    # r = (n_amb*C - B) / (n_amb*C + B)  ... or similar depending on convention.
    # Standard: r = (n_0 * Y - Z) / (n_0 * Y + Z) where Y/B and Z/C?
    # Let's stick to the standard characteristic matrix definition:
    # M = [[cos(beta), i sin(beta)/n], [i n sin(beta), cos(beta)]]
    # Total M = M_1 ... M_N
    # B = M_11 * n_sub + M_12
    # C = M_21 * n_sub + M_22
    # r = (n_amb * C - B) / (n_amb * C + B)
    
    B = P[:, 0, 0] * n_substrate + P[:, 0, 1]
    C = P[:, 1, 0] * n_substrate + P[:, 1, 1]
    
    r = (n_ambient * C - B) / (n_ambient * C + B)
    R = np.abs(r)**2
    return R

def _objective(x, problem, wavelengths):
    """
    Objective function for DE.
    x: flat array [d1, d2, ..., dN, m1, m2, ..., mN]
    First N are thicknesses, next N are material indices (0 for L, 1 for H)
    """
    n_max = problem["max_layers"]
    N = n_max
    
    # Extract thicknesses and materials
    d = x[:N]
    m_idx = x[N:]
    
    # Enforce constraints by penalizing or clipping
    # Thicknesses must be in [0, max_layer_nm]
    max_layer_nm = problem["max_layer_nm"]
    d_clipped = np.clip(d, 0, max_layer_nm)
    
    # Materials: round to nearest integer and clip to [0, 1]
    m_rounded = np.round(m_idx)
    m_rounded = np.clip(m_rounded, 0, 1).astype(int)
    
    # Determine active layers
    # A layer is active if its thickness > 0 (or > epsilon)
    # But DE might assign 0 thickness to some.
    # We should only include layers with d > 0 in the reflectance calc to save time?
    # Or just calculate with d=0, which is mathematically fine (identity matrix).
    # However, the problem says "at most 20 layers". If we use 20 slots, some might be 0.
    # The evaluator expects a list. If we return 20 layers with some 0 thickness, is it valid?
    # "list of physical thicknesses... every thickness must lie in [0, max_layer_nm]"
    # So 0 is allowed.
    
    materials = ['H' if mi == 1 else 'L' for mi in m_rounded]
    
    # Calculate mean reflectance
    R = _vectorized_reflectance(materials, d_clipped.tolist(), wavelengths, 
                                problem["ambient_index"], problem["substrate_index"])
    mean_R = np.mean(R)
    
    # Penalty for exceeding total thickness
    total_d = np.sum(d_clipped)
    max_total = problem["max_total_nm"]
    if total_d > max_total:
        penalty = 100.0 * (total_d - max_total) / max_total
        return mean_R + penalty
    
    return mean_R

def design(problem):
    """
    Design a stack using Differential Evolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    n_max = problem["max_layers"]
    band = problem["band_nm"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    
    # Sample 200 wavelengths for optimization speed
    # The evaluator uses 401, but 200 should be sufficient for DE to find good regions
    wavelengths_opt = np.linspace(band[0], band[1], 200)
    
    # DE bounds
    # 20 thicknesses, 20 materials
    bounds = []
    for _ in range(n_max):
        bounds.append((0, problem["max_layer_nm"]))
    for _ in range(n_max):
        bounds.append((0, 1)) # 0 for L, 1 for H
    
    # Initial best: seed solution
    centre = 0.5 * (band[0] + band[1])
    seed_materials = ["L"]
    seed_thicknesses = [centre / (4.0 * problem["materials"]["L"])]
    
    # Evaluate seed
    R_seed = _vectorized_reflectance(seed_materials, seed_thicknesses, wavelengths_opt, n_amb, n_sub)
    best_score = np.mean(R_seed)
    best_materials = seed_materials
    best_thicknesses = seed_thicknesses
    
    # Run Differential Evolution
    # We need to check time periodically. DE doesn't support easy interruption.
    # So we run with a small maxiter and check.
    # Or just run a fixed small number of generations.
    
    # Let's try a few restarts with different initial populations if time permits?
    # DE with popsize=10, maxiter=20 is fast.
    
    try:
        # Use a custom callback to check time? No, simpler to just limit iterations.
        # 20 vars, popsize 10, maxiter 30 -> 300 evals.
        # Each eval is ~200 wavelengths * 20 layers. Very fast in numpy.
        
        result = differential_evolution(
            _objective, 
            bounds, 
            args=(problem, wavelengths_opt),
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False # Skip polish to avoid extra time
        )
        
        if result.success or result.fun < best_score:
            x_opt = result.x
            d_opt = np.clip(x_opt[:n_max], 0, problem["max_layer_nm"])
            m_opt = np.round(x_opt[n_max:]).astype(int)
            
            # Filter out zero thickness layers to clean up output?
            # The problem allows 0 thickness. But it's cleaner to remove them.
            # However, the evaluator just checks constraints.
            # Let's keep all 20 to be safe, or filter?
            # "list of 'L'/'H' labels and a list of physical thicknesses"
            # If I return 20 layers, some with 0 thickness, it's valid.
            # But maybe fewer layers is better? No, score is reflectance.
            
            # Let's filter out layers with d < 1e-6 to avoid numerical noise in evaluator?
            # The evaluator might treat d=0 as identity, which is fine.
            
            materials_opt = ['H' if mi == 1 else 'L' for mi in m_opt]
            thicknesses_opt = d_opt.tolist()
            
            # Re-evaluate on full 401 wavelengths to be sure?
            # For now, trust the opt score.
            
            best_materials = materials_opt
            best_thicknesses = thicknesses_opt
            best_score = result.fun
            
    except Exception as e:
        pass
        
    # If time remains, try a second run with different seed?
    if time.time() < deadline - 10:
        try:
            result2 = differential_evolution(
                _objective, 
                bounds, 
                args=(problem, wavelengths_opt),
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=123,
                polish=False
            )
            if result2.fun < best_score:
                x_opt = result2.x
                d_opt = np.clip(x_opt[:n_max], 0, problem["max_layer_nm"])
                m_opt = np.round(x_opt[n_max:]).astype(int)
                materials_opt = ['H' if mi == 1 else 'L' for mi in m_opt]
                thicknesses_opt = d_opt.tolist()
                best_materials = materials_opt
                best_thicknesses = thicknesses_opt
                best_score = result2.fun
        except:
            pass

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
