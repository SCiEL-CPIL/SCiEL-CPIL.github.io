"""Antireflection coating design using a discretized index-gradient model."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_spectrum(materials, thicknesses, problem, wavelengths):
    """Compute reflectance spectrum using the characteristic matrix method."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_layers = len(materials)
    
    if n_layers == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        R = np.full(len(wavelengths), abs(r) ** 2 * 100.0)
        return R
    
    n_vals = np.array([problem["materials"][m] for m in materials])
    d_vals = np.array(thicknesses)
    n_waves = len(wavelengths)
    R = np.zeros(n_waves)
    
    for i, lam in enumerate(wavelengths):
        # Build the characteristic matrix product
        M = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], dtype=complex)
        
        for j in range(n_layers):
            delta = 2.0 * np.pi * n_vals[j] * d_vals[j] / lam
            cd = np.cos(delta)
            sd = np.sin(delta)
            
            if abs(sd) < 1e-14:
                inv_n_sd = 0.0 + 0j
            else:
                inv_n_sd = 1.0 / (n_vals[j] * sd)
            
            B = n_vals[j] * sd
            
            layer_M = np.array([[cd, 1j * inv_n_sd], [1j * B, cd]], dtype=complex)
            M = M @ layer_M
        
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]
        
        Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
        r = (n_amb - Y_in) / (n_amb + Y_in)
        R[i] = np.abs(r) ** 2 * 100.0
    
    return R

def _nelder_mead(objective, x0, bounds, maxiter=200, tol=1e-10):
    """Nelder-Mead optimization with bounds."""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    # Clamp initial point
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    
    # Simplex initialization
    simplex = [x0.copy()]
    for i in range(n):
        point = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) + 1e-6
        point[i] += step
        point[i] = np.clip(point[i], bounds[i][0], bounds[i][1])
        simplex.append(point)
    
    values = [objective(p) for p in simplex]
    
    for it in range(maxiter):
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i].copy() for i in order]
        values = [values[i] for i in order]
        
        if abs(values[0] - values[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
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
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    values[i] = objective(simplex[i])
    
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]

def design(problem):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    centre = 0.5 * (band[0] + band[1])
    
    best_score = 1e6
    best_materials = ["L"]
    best_thicknesses = [centre / (4.0 * n_L)]
    
    # Strategy: Use a fixed 20-layer structure that approximates an index gradient
    # from air (1.0) to glass (1.52), using only L and H materials.
    # The ideal index profile for broadband AR is roughly exponential or power-law.
    # We'll try several candidate patterns and optimize their thicknesses.
    
    # Pattern 1: Alternating L/H starting with L (low index near air)
    # This is a standard Bragg stack but we'll optimize thicknesses.
    # For a gradient, we want the effective index to transition smoothly.
    
    # Let's try a few different structural patterns:
    candidate_patterns = []
    
    # Pattern A: L H L H ... (20 layers, starting with L)
    for i in range(20):
        candidate_patterns.append("L" if i % 2 == 0 else "H")
    
    # Pattern B: H L H L ... (20 layers, starting with H)
    for i in range(20):
        candidate_patterns.append("H" if i % 2 == 0 else "L")
    
    # Pattern C: Symmetric L H H L H H L ... 
    # Actually, let's think about what gives a good index gradient.
    # Near air (n=1.0), we want low index (L=1.38).
    # Near glass (n=1.52), we want... well, the substrate is 1.52, 
    # so the last layer should ideally be close to 1.52. 
    # H=2.35 is too high, L=1.38 is too low. 
    # A quarter-wave stack with optimized thicknesses can do better.
    
    # Let's try a pattern that has more L's near the air side and more H's near the glass side,
    # but since we can only use L and H, we'll use the alternating pattern and rely on 
    # thickness optimization to create an effective gradient.
    
    # Actually, the key insight for broadband AR with two materials is to use 
    # a quarter-wave stack at the center wavelength, but with optimized thicknesses.
    # The classic result is that a 4-layer or 6-layer stack can give good broadband performance.
    
    # Let's try multiple layer counts and patterns:
    patterns_to_try = []
    
    # For each layer count from 2 to 20, try alternating patterns
    for n_layers in range(2, min(max_layers + 1, 21)):
        # Pattern 1: Start with L
        pattern1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        patterns_to_try.append(pattern1)
        
        # Pattern 2: Start with H
        pattern2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        patterns_to_try.append(pattern2)
    
    # Also try some non-alternating patterns for small layer counts
    if max_layers >= 4:
        patterns_to_try.append(["L", "H", "H", "L"])
    if max_layers >= 6:
        patterns_to_try.append(["L", "H", "H", "L", "L", "H"])
        patterns_to_try.append(["L", "L", "H", "H", "L", "L"])
    if max_layers >= 8:
        patterns_to_try.append(["L", "H", "H", "L", "L", "H", "H", "L"])
    
    # Deduplicate
    seen = set()
    unique_patterns = []
    for p in patterns_to_try:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    # Sort patterns by layer count (try smaller first to get quick wins)
    unique_patterns.sort(key=lambda p: len(p))
    
    def _objective(params, materials):
        thicknesses = np.array(params)
        
        # Check constraints
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 1e6
        if np.sum(thicknesses) > max_total_nm:
            return 1e6
        
        R = _reflectance_spectrum(materials, thicknesses, problem, wavelengths)
        return np.mean(R)
    
    # Try each pattern with multi-start Nelder-Mead
    for materials in unique_patterns:
        if time.time() > deadline - 5:
            break
        
        n_layers = len(materials)
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # Multiple starting points
        starts = []
        
        # Start 1: Quarter-wave at center wavelength
        x0 = []
        for m in materials:
            n_m = problem["materials"][m]
            d0 = centre / (4.0 * n_m)
            x0.append(d0)
        if sum(x0) <= max_total_nm:
            starts.append(x0)
        else:
            scale = max_total_nm / sum(x0)
            starts.append([d * scale for d in x0])
        
        # Start 2: Quarter-wave at band center, scaled
        x0 = []
        for m in materials:
            n_m = problem["materials"][m]
            d0 = centre / (4.0 * n_m)
            x0.append(d0)
        if sum(x0) > max_total_nm:
            scale = max_total_nm / sum(x0)
            x0 = [d * scale for d in x0]
        starts.append(x0)
        
        # Start 3: Uniform thickness
        uniform = max_total_nm / n_layers
        if uniform <= max_layer_nm:
            starts.append([uniform] * n_layers)
        
        # Start 4: Gradient-inspired (thinner near air, thicker near substrate)
        x0 = []
        total = 0
        for i, m in enumerate(materials):
            # Linear gradient from 0.5 to 1.5 times average
            frac = 0.5 + (i / max(1, n_layers - 1))
            d0 = (centre / (4.0 * problem["materials"][m])) * frac
            x0.append(d0)
        if sum(x0) > max_total_nm:
            scale = max_total_nm / sum(x0)
            x0 = [d * scale for d in x0]
        starts.append(x0)
        
        best_local = 1e6
        best_local_x = None
        
        for x0 in starts:
            if time.time() > deadline - 2:
                break
            
            try:
                result_x, result_f = _nelder_mead(
                    lambda p: _objective(p, materials),
                    x0,
                    bounds,
                    maxiter=150
                )
                
                # Verify constraints
                if all(0 <= t <= max_layer_nm for t in result_x) and sum(result_x) <= max_total_nm:
                    if result_f < best_local:
                        best_local = result_f
                        best_local_x = result_x
            except Exception:
                pass
        
        if best_local_x is not None and best_local < best_score:
            best_score = best_local
            best_materials = materials
            best_thicknesses = list(best_local_x)
    
    # Final validation
    if len(best_materials) != len(best_thicknesses):
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    if not all(0 <= t <= max_layer_nm for t in best_thicknesses):
        best_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
