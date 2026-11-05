"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _is_valid(materials, thicknesses, problem):
    """
    Check if the design is valid.
    """
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True


def design(problem):
    """
    Design a broadband antireflection coating.
    Uses a combination of analytical initialization and local optimization.
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    band_nm = problem["band_nm"]
    wl_min, wl_max = band_nm[0], band_nm[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Wavelengths for evaluation (401 equally spaced)
    wls = np.linspace(wl_min, wl_max, 401)
    
    def _fast_reflectance(materials, thicknesses, wl_array, n_ambient, n_substrate):
        """
        Compute reflectance at all wavelengths in wl_array for given materials/thicknesses.
        Returns array of reflectance values (fraction, 0-1).
        """
        n_layers = len(materials)
        if n_layers == 0:
            r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
            return np.full(len(wl_array), abs(r)**2)
        
        # Precompute layer indices
        n_vals = [problem["materials"][m] for m in materials]
        
        # For each wavelength, compute the total characteristic matrix
        R = np.zeros(len(wl_array))
        
        for i, wl in enumerate(wl_array):
            # Start with identity
            M11, M12, M21, M22 = 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j
            
            for j in range(n_layers):
                n = n_vals[j]
                d = thicknesses[j]
                beta = 2.0 * np.pi * n * d / wl
                cos_b = np.cos(beta)
                sin_b = np.sin(beta)
                
                # Layer matrix elements
                m11 = cos_b
                m12 = -1j * sin_b / n
                m21 = 1j * n * sin_b
                m22 = cos_b
                
                # Multiply: new_M = old_M @ layer_M
                nm11 = M11 * m11 + M12 * m21
                nm12 = M11 * m12 + M12 * m22
                nm21 = M21 * m11 + M22 * m21
                nm22 = M21 * m12 + M22 * m22
                
                M11, M12, M21, M22 = nm11, nm12, nm21, nm22
            
            # Compute Y and reflectance
            Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
            r = (n_ambient - Y) / (n_ambient + Y)
            R[i] = abs(r)**2
        
        return R
    
    def score(materials, thicknesses):
        """Compute mean reflectance in percent."""
        if not _is_valid(materials, thicknesses, problem):
            return float('inf')
        R_array = _fast_reflectance(materials, thicknesses, wls, n_ambient, n_substrate)
        return float(np.mean(R_array) * 100.0)
    
    # Initialize with multi-layer quarter-wave stacks
    wl_center = 0.5 * (wl_min + wl_max)
    d_L = wl_center / (4.0 * n_L)
    d_H = wl_center / (4.0 * n_H)
    
    # Generate initial designs
    initial_designs = []
    
    # 1-layer
    initial_designs.append((["L"], [d_L]))
    initial_designs.append((["H"], [d_H]))
    
    # 3-layer
    initial_designs.append((["H", "L", "H"], [d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L"], [d_L, d_H, d_L]))
    
    # 5-layer
    initial_designs.append((["H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L]))
    
    # 7-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    
    # 9-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    
    # 11-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    
    # 13-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    
    # 15-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    
    # 17-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    
    # 19-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H]))
    initial_designs.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L]))
    
    # Evaluate all initial designs
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, ths in initial_designs:
        if not _is_valid(mats, ths, problem):
            continue
        s = score(mats, ths)
        if s < best_score:
            best_score = s
            best_materials = mats[:]
            best_thicknesses = ths[:]
    
    if best_materials is None:
        # Fallback: single L layer
        return ["L"], [wl_center / (4.0 * n_L)]
    
    # Local optimization using scipy
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    
    if minimize is not None:
        def objective(thicknesses):
            """Objective function for optimization."""
            if not _is_valid(best_materials, list(thicknesses), problem):
                return 1e10
            R_array = _fast_reflectance(best_materials, list(thicknesses), wls, n_ambient, n_substrate)
            return float(np.mean(R_array) * 100.0)
        
        x0 = np.array(best_thicknesses, dtype=float)
        bounds = [(0.0, max_layer_nm)] * len(x0)
        
        # Try L-BFGS-B
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.fun < best_score:
            new_ths = [max(0.0, min(max_layer_nm, t)) for t in result.x]
            if _is_valid(best_materials, new_ths, problem):
                best_thicknesses = new_ths
                best_score = result.fun
        
        # Try Nelder-Mead for robustness
        elapsed = time.time() - start_time
        if elapsed < deadline - 15.0:
            result_nm = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result_nm.fun < best_score:
                new_ths = [max(0.0, min(max_layer_nm, t)) for t in result_nm.x]
                if _is_valid(best_materials, new_ths, problem):
                    best_thicknesses = new_ths
                    best_score = result_nm.fun
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed >= deadline - 10.0:
        if not _is_valid(best_materials, best_thicknesses, problem):
            return ["L"], [wl_center / (4.0 * n_L)]
        return best_materials, best_thicknesses
    
    # Try additional patterns if time permits
    if elapsed < deadline - 30.0:
        # Try some non-alternating patterns
        additional_patterns = []
        
        # Pattern: H-L-H-L-H with varying thickness ratios
        for ratio in [0.5, 0.7, 1.0, 1.3, 1.5]:
            ths = [d_H, d_L * ratio, d_H, d_L * ratio, d_H]
            if sum(ths) <= max_total_nm and all(0 <= t <= max_layer_nm for t in ths):
                additional_patterns.append((["H", "L", "H", "L", "H"], ths))
        
        # Pattern: L-H-L-H-L with varying thickness ratios
        for ratio in [0.5, 0.7, 1.0, 1.3, 1.5]:
            ths = [d_L, d_H * ratio, d_L, d_H * ratio, d_L]
            if sum(ths) <= max_total_nm and all(0 <= t <= max_layer_nm for t in ths):
                additional_patterns.append((["L", "H", "L", "H", "L"], ths))
        
        for mats, ths in additional_patterns:
            if not _is_valid(mats, ths, problem):
                continue
            s = score(mats, ths)
            if s < best_score:
                best_score = s
                best_materials = mats[:]
                best_thicknesses = ths[:]
                
                # Optimize this new best
                elapsed = time.time() - start_time
                if elapsed < deadline - 15.0 and minimize is not None:
                    x0 = np.array(best_thicknesses, dtype=float)
                    bounds = [(0.0, max_layer_nm)] * len(x0)
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if result.fun < best_score:
                        new_ths = [max(0.0, min(max_layer_nm, t)) for t in result.x]
                        if _is_valid(best_materials, new_ths, problem):
                            best_thicknesses = new_ths
                            best_score = result.fun
    
    # Final check
    if not _is_valid(best_materials, best_thicknesses, problem):
        return ["L"], [wl_center / (4.0 * n_L)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
