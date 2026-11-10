"""Broadband AR coating design using continuous admittance profile optimization."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

def _nelder_mead(objective, x0, bounds=None, max_iter=100, tol=1e-5, time_limit=None, start_time=None):
    """
    Simple Nelder-Mead implementation using only NumPy.
    objective: function to minimize
    x0: initial guess (1D array)
    bounds: list of (low, high) tuples or None
    max_iter: maximum iterations
    tol: convergence tolerance
    time_limit: absolute time limit (seconds) or None
    start_time: start time for time_limit check
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * xi[i] if abs(xi[i]) > 1e-6 else 0.1
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate initial points
    values = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
        
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            # Accept reflection
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            if fr < values[-1]:
                # Outside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = objective(simplex[i])
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
                fc = objective(xc)
                if fc <= values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = objective(simplex[i])
        
        # Apply bounds if provided
        if bounds is not None:
            for i in range(n + 1):
                for j in range(n):
                    if bounds[j] is not None:
                        low, high = bounds[j]
                        if simplex[i][j] < low:
                            simplex[i][j] = low
                        elif simplex[i][j] > high:
                            simplex[i][j] = high
    
    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]

def _reflectance_vectorized(materials, thicknesses, lambdas, n0, ns):
    """
    Compute reflectance spectrum using characteristic matrix method, vectorized over wavelengths.
    materials: list of refractive indices (one per layer)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    n0: ambient index, ns: substrate index
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    M = len(lambdas)
    
    # Initialize total matrix as identity
    M11 = np.ones((M,), dtype=complex)
    M12 = np.zeros((M,), dtype=complex)
    M21 = np.zeros((M,), dtype=complex)
    M22 = np.ones((M,), dtype=complex)
    
    # Multiply by each layer matrix from left (air side) to right (substrate side)
    for j in range(N):
        nj = materials[j]
        tj = thicknesses[j]
        d = 2.0 * np.pi * nj * tj / lambdas  # shape (M,)
        
        cd = np.cos(d)
        sd = np.sin(d)
        
        i_sd_nj = 1j * sd / nj
        i_nj_sd = 1j * nj * sd
        
        new_M11 = M11 * cd + M12 * i_sd_nj
        new_M12 = M11 * i_sd_nj + M12 * cd
        new_M21 = M21 * cd + M22 * i_nj_sd
        new_M22 = M21 * i_nj_sd + M22 * cd
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Compute input admittance
    B = M11 + M12 * ns
    C = M21 + M22 * ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2
    return R

def _quantize_profile(profile, n0, ns, nL, nH, max_layers, max_total_nm, max_layer_nm):
    """
    Quantize a continuous admittance profile into discrete L/H layers.
    profile: 1D array of admittance values (length = number of segments)
    Returns: (materials, thicknesses)
    """
    # The profile represents the desired input admittance at each depth
    # We map each segment to the nearest material (L or H) based on its index
    # and determine thicknesses based on the number of segments assigned to each material
    
    # Simple approach: use the profile to determine the sequence of L/H layers
    # and distribute the total thickness budget
    
    # First, determine the sequence of materials
    # Map admittance values to nearest material index
    materials = []
    for y in profile:
        # Choose L if closer to nL, H if closer to nH
        if abs(y - nL) <= abs(y - nH):
            materials.append('L')
        else:
            materials.append('H')
    
    # Merge consecutive same-material layers
    merged = []
    for m in materials:
        if not merged or merged[-1] != m:
            merged.append(m)
    
    # Limit to max_layers
    if len(merged) > max_layers:
        # Keep only the first max_layers layers
        merged = merged[:max_layers]
    
    # Determine thicknesses: distribute max_total_nm evenly among layers
    # But weight by the "importance" of each layer based on the profile
    n_layers = len(merged)
    if n_layers == 0:
        return ['L'], [0.0]
    
    # Equal distribution as a starting point
    t_per_layer = min(max_total_nm / n_layers, max_layer_nm)
    thicknesses = [t_per_layer] * n_layers
    
    # Adjust to respect max_total_nm
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return merged, thicknesses

def _objective(params, problem, lambdas, n0, ns, max_layer_nm, max_total_nm):
    """
    Objective function for optimization.
    params: array of [n_layers, t_1, t_2, ..., t_n_layers]
    First element is the number of layers (will be rounded), rest are thicknesses.
    """
    n_layers = int(round(params[0]))
    n_layers = max(0, min(n_layers, 20))
    if n_layers == 0:
        return 10.0  # penalize empty stack
    
    thicknesses = params[1:1+n_layers]
    # Clip thicknesses to valid range
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    total_thick = np.sum(thicknesses)
    if total_thick > max_total_nm:
        return 10.0 + (total_thick - max_total_nm) / max_total_nm * 10.0
    
    # Assign materials: alternate L, H, L, H, ... starting with L
    materials = []
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append(problem["materials"]["L"])
        else:
            materials.append(problem["materials"]["H"])
    
    R = _reflectance_vectorized(materials, thicknesses, lambdas, n0, ns)
    return np.mean(R) * 100.0  # in percent

def _objective_continuous(profile_params, problem, lambdas, n0, ns, max_layer_nm, max_total_nm):
    """
    Objective function for continuous profile optimization.
    profile_params: array of admittance values defining the continuous profile
    We optimize the shape of the admittance profile, then quantize it.
    """
    n0_val = problem["ambient_index"]
    ns_val = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    
    # profile_params represents the target admittance at each depth segment
    # We'll use a simple quantization: map each segment to L or H
    # and use equal thicknesses for now
    
    # Quantize the profile
    materials, thicknesses = _quantize_profile(profile_params, n0_val, ns_val, nL, nH, 
                                                 max_layers, max_total_nm, max_layer_nm)
    
    if not materials or len(thicknesses) == 0:
        return 10.0
    
    # Convert to numeric indices
    mat_indices = [nL if m == 'L' else nH for m in materials]
    
    R = _reflectance_vectorized(mat_indices, thicknesses, lambdas, n0_val, ns_val)
    return np.mean(R) * 100.0

def design(problem):
    """Design broadband AR coating using continuous profile optimization."""
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat_L = problem["materials"]["L"]
    mat_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths: 401 equally spaced points in [400, 800]
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Strategy: Optimize a continuous admittance profile, then quantize to L/H layers
    # The profile is defined by a set of admittance values at different depths
    # We'll use 20 segments (corresponding to max_layers)
    
    n_segments = max_layers  # 20 segments
    
    # Initial guess: linear taper from n0 to ns
    profile_init = np.linspace(n0, ns, n_segments)
    
    # Bounds for admittance: between 1.0 and 2.5 (reasonable range)
    lower_bound = np.ones(n_segments) * 1.0
    upper_bound = np.ones(n_segments) * 2.5
    
    # Optimize the continuous profile using Nelder-Mead
    try:
        def _obj_wrapper(profile_params):
            return _objective_continuous(profile_params, problem, lambdas, n0, ns, max_layer_nm, max_total_nm)
        
        bounds_list = list(zip(lower_bound, upper_bound))
        profile_opt, fun_opt = _nelder_mead(
            _obj_wrapper,
            profile_init,
            bounds=bounds_list,
            max_iter=100,
            tol=1e-4,
            time_limit=deadline - time.time() - 5.0,
            start_time=t_start
        )
        
        if fun_opt < 10.0:
            pass # profile_opt is already set
        else:
            profile_opt = profile_init
            
    except Exception as e:
        profile_opt = profile_init
    
    # Check time
    if time.time() > deadline - 10.0:
        # Not enough time, use fallback
        centre = 0.5 * (band[0] + band[1])
        t_L = centre / (4.0 * mat_L)
        t_H = centre / (4.0 * mat_H)
        if t_L + t_H <= max_total_nm and t_L <= max_layer_nm and t_H <= max_layer_nm:
            return ["L", "H"], [t_L, t_H]
        else:
            return ["L"], [min(t_L, max_layer_nm)]
    
    # Quantize the optimized profile
    materials, thicknesses = _quantize_profile(profile_opt, n0, ns, mat_L, mat_H,
                                                 max_layers, max_total_nm, max_layer_nm)
    
    # If quantization didn't work well, try a local refinement using DE on the quantized result
    if materials and len(thicknesses) > 0:
        # Compute current score
        mat_indices = [mat_L if m == 'L' else mat_H for m in materials]
        R = _reflectance_vectorized(mat_indices, thicknesses, lambdas, n0, ns)
        current_score = np.mean(R) * 100.0
        
        # Try to improve with a short DE run on the thicknesses
        if time.time() < deadline - 20.0:
            n_layers = len(materials)
            if n_layers > 0 and n_layers <= max_layers:
                # Optimize thicknesses with fixed material sequence
                def _obj_thicknesses(t_params):
                    t_params = np.clip(t_params, 0.0, max_layer_nm)
                    if np.sum(t_params) > max_total_nm:
                        return 10.0 + (np.sum(t_params) - max_total_nm) / max_total_nm
                    mat_idx = [mat_L if m == 'L' else mat_H for m in materials]
                    R = _reflectance_vectorized(mat_idx, t_params, lambdas, n0, ns)
                    return np.mean(R) * 100.0
                
                t0 = np.array(thicknesses)
                bounds_t = [(0.0, max_layer_nm)] * n_layers
                
                try:
                    t_opt, fun_t = _nelder_mead(
                        _obj_thicknesses,
                        t0,
                        bounds=bounds_t,
                        max_iter=50,
                        tol=1e-4,
                        time_limit=deadline - time.time() - 5.0,
                        start_time=t_start
                    )
                    if fun_t < current_score:
                        thicknesses = list(np.clip(t_opt, 0.0, max_layer_nm))
                        # Check total
                        if sum(thicknesses) > max_total_nm:
                            scale = max_total_nm / sum(thicknesses)
                            thicknesses = [t * scale for t in thicknesses]
                except Exception:
                    pass
    
    # Final verification and cleanup
    if not materials or len(thicknesses) == 0:
        # Fallback: simple 2-layer quarter-wave stack
        centre = 0.5 * (band[0] + band[1])
        t_L = centre / (4.0 * mat_L)
        t_H = centre / (4.0 * mat_H)
        if t_L + t_H <= max_total_nm and t_L <= max_layer_nm and t_H <= max_layer_nm:
            return ["L", "H"], [t_L, t_H]
        else:
            return ["L"], [min(t_L, max_layer_nm)]
    
    # Remove zero-thickness layers
    valid_indices = [i for i, t in enumerate(thicknesses) if t > 1.0]
    if not valid_indices:
        return ["L"], [0.0]
    
    materials = [materials[i] for i in valid_indices]
    thicknesses = [thicknesses[i] for i in valid_indices]
    
    # Verify constraints
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses

# EVOLVE-BLOCK-END
