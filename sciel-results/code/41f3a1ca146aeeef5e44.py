"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_dict = problem["materials"]
    lam0, lam1 = problem["band_nm"]
    n_waves = 401
    lambdas = np.linspace(lam0, lam1, n_waves)
    
    total_r = 0.0
    for lam in lambdas:
        # Build total characteristic matrix
        A = 1.0 + 0j
        B = 0.0 + 0j
        C = 0.0 + 0j
        D = 1.0 + 0j
        
        for i, mat in enumerate(materials):
            n_i = n_dict[mat]
            d_i = thicknesses[i]
            # Phase thickness delta = 2*pi*n*d/lambda
            delta = 2.0 * np.pi * n_i * d_i / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer matrix:
            # [cos_d, i*sin_d/n_i]
            # [i*n_i*sin_d, cos_d]
            M11 = cos_d
            M12 = 1j * sin_d / n_i
            M21 = 1j * n_i * sin_d
            M22 = cos_d
            
            # Multiply current total matrix M by layer matrix M_i
            # New M = M * M_i
            nA = A * M11 + B * M21
            nB = A * M12 + B * M22
            nC = C * M11 + D * M21
            nD = C * M12 + D * M22
            
            A, B, C, D = nA, nB, nC, nD
        
        # Input admittance Y_in = 1 / (C + D/n0) ... wait, standard formula:
        # Y_in = (C + D/n0) / (A + B/n0) is wrong.
        # The characteristic matrix M relates:
        # [E0, H0] = M [Es, Hs]
        # So E0 = A*Es + B*Hs, H0 = C*Es + D*Hs
        # Admittance Y = H/E.
        # At substrate, Ys = ns (assuming no reflection from infinity, so only forward wave)
        # Es = 1, Hs = ns (normalized)
        # E0 = A*1 + B*ns
        # H0 = C*1 + D*ns
        # Y_in = H0/E0 = (C + D*ns) / (A + B*ns)
        
        Y_in = (C + D * ns) / (A + B * ns)
        
        # Reflectance R = | (Y0 - Y_in) / (Y0 + Y_in) |^2
        # Y0 = n0 = 1.0
        r = (n0 - Y_in) / (n0 + Y_in)
        R = np.abs(r) ** 2
        total_r += R
        
    return 100.0 * total_r / n_waves

def _evaluate_candidate(materials, thicknesses, problem):
    """Check constraints and return reflectance or infinity if invalid."""
    if not materials:
        # No layers, just bare glass
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
        
    if len(materials) > problem["max_layers"]:
        return float('inf')
    if len(materials) != len(thicknesses):
        return float('inf')
    
    total_thick = 0.0
    for i, mat in enumerate(materials):
        if mat not in problem["materials"]:
            return float('inf')
        d = thicknesses[i]
        if d < 0 or d > problem["max_layer_nm"]:
            return float('inf')
        total_thick += d
        
    if total_thick > problem["max_total_nm"]:
        return float('inf')
        
    return _mean_reflectance(materials, thicknesses, problem)

def design(problem):
    """Design a stack by evaluating several candidate stacks and returning the best."""
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    lam_center = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Quarter wave thicknesses at center wavelength
    tqL = lam_center / (4.0 * nL)
    tqH = lam_center / (4.0 * nH)
    
    candidates = []
    
    # 1. Single layer L
    candidates.append((["L"], [tqL]))
    
    # 2. Doublet H L (starting with H often better for matching air to glass)
    candidates.append((["H", "L"], [tqH, tqL]))
    candidates.append((["L", "H"], [tqL, tqH]))
    
    # 3. Triplet L H L or H L H
    candidates.append((["L", "H", "L"], [tqL, tqH, tqL]))
    candidates.append((["H", "L", "H"], [tqH, tqL, tqH]))
    
    # 4. Quadruplet
    candidates.append((["L", "H", "L", "H"], [tqL, tqH, tqL, tqH]))
    candidates.append((["H", "L", "H", "L"], [tqH, tqL, tqH, tqL]))
    
    # 5. Quintuplet
    candidates.append((["L", "H", "L", "H", "L"], [tqL, tqH, tqL, tqH, tqL]))
    candidates.append((["H", "L", "H", "L", "H"], [tqH, tqL, tqH, tqL, tqH]))
    
    # 6. Sextuplet
    candidates.append((["L", "H", "L", "H", "L", "H"], [tqL, tqH, tqL, tqH, tqL, tqH]))
    candidates.append((["H", "L", "H", "L", "H", "L"], [tqH, tqL, tqH, tqL, tqH, tqL]))
    
    # 7. Septuplet
    candidates.append((["L", "H", "L", "H", "L", "H", "L"], [tqL, tqH, tqL, tqH, tqL, tqH, tqL]))
    candidates.append((["H", "L", "H", "L", "H", "L", "H"], [tqH, tqL, tqH, tqL, tqH, tqL, tqH]))
    
    # 8. Octuplet
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H"], [tqL, tqH, tqL, tqH, tqL, tqH, tqL, tqH]))
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L"], [tqH, tqL, tqH, tqL, tqH, tqL, tqH, tqL]))
    
    # 9. Nonuplet
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], [tqL, tqH, tqL, tqH, tqL, tqH, tqL, tqH, tqL]))
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"], [tqH, tqL, tqH, tqL, tqH, tqL, tqH, tqL, tqH]))
    
    # 10. Decuplet
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], [tqL, tqH, tqL, tqH, tqL, tqH, tqL, tqH, tqL, tqH]))
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], [tqH, tqL, tqH, tqL, tqH, tqL, tqH, tqL, tqH, tqL]))
    
    # Also try some half-wave or other variations?
    # Maybe a 2-layer doublet with different thicknesses?
    # Let's stick to quarter-wave stacks first as they are robust.
    # We can add a few "optimized" ones if time permits, but quarter-wave is a strong baseline.
    
    best_materials = None
    best_thicknesses = None
    best_r = float('inf')
    
    for mats, thicks in candidates:
        r = _evaluate_candidate(mats, thicks, problem)
        if r < best_r:
            best_r = r
            best_materials = mats
            best_thicknesses = thicks
            
    if best_materials is None:
        # Fallback to single L layer
        return ["L"], [tqL]
        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
