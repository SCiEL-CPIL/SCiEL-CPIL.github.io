"""Broadband antireflection coating design via numerical optimization."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _char_matrix_stack(thetas, indices, thicknesses, n0, ns):
    """Compute reflectance using characteristic matrix method.
    thetas: array of phase thicknesses (2*pi*n*d/lambda)
    indices: array of layer indices
    thicknesses: not directly needed if thetas given
    Returns reflectance (scalar) for the stack.
    """
    N = len(thetas)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return abs(r) ** 2
    
    # Build characteristic matrix
    # Each layer has matrix: [[cos(theta), i*sin(theta)/n], [i*n*sin(theta), cos(theta)]]
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    for i in range(N):
        n_i = indices[i]
        theta_i = thetas[i]
        c = np.cos(theta_i)
        s = np.sin(theta_i)
        M_i = np.array([[c, 1j * s / n_i],
                        [1j * n_i * s, c]], dtype=complex)
        M = M @ M_i
    
    # Admittance at the front surface
    # Y = M[1,1]/M[0,1] if using standard convention
    # Actually: Y = (M[1,0]*ns + M[1,1]) / (M[0,0]*ns + M[0,1]) ... 
    # Let me use the standard formula:
    # Y = (M11*ns + M12) / (M21*ns + M22)
    # where M is the total matrix from layer 1 to N
    
    # Wait, let me be more careful. The characteristic matrix for a stack is:
    # [B, C] = M1 @ M2 @ ... @ MN
    # Y = (B*ns + C) / (D*ns + E) ... no
    # Standard: Y = (M[0,0]*ns + M[0,1]) / (M[1,0]*ns + M[1,1])
    # Actually the standard formula is:
    # Y = (B*ns + C) / (D*ns + E) where [B C; D E] = total matrix
    # Let me just use: Y = (M[0,0]*ns + M[0,1]) / (M[1,0]*ns + M[1,1])
    
    B = M[0, 0]
    C = M[0, 1]
    D = M[1, 0]
    E = M[1, 1]
    
    Y = (B * ns + C) / (D * ns + E)
    
    r = (n0 - Y) / (n0 + Y)
    return float(np.real(r * np.conj(r)))


def _reflectance_from_thick(materials, thicknesses, n0, ns, wavelengths):
    """Compute mean reflectance over wavelengths for given stack."""
    N = len(thicknesses)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return abs(r) ** 2 * 100.0
    
    total_ref = 0.0
    for lam in wavelengths:
        thetas = 2.0 * np.pi * np.array(materials) * np.array(thicknesses) / lam
        total_ref += _char_matrix_stack(thetas, materials, thicknesses, n0, ns)
    
    return total_ref / len(wavelengths) * 100.0


def _objective(x, materials, n0, ns, wavelengths, max_layer_nm, max_total_nm):
    """Objective function for optimization. Returns mean reflectance in percent."""
    # x contains thicknesses for all layers
    thicknesses = np.abs(x)  # Use absolute value to handle negative values
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    
    total = np.sum(thicknesses)
    if total > max_total_nm:
        return 100.0 + (total - max_total_nm) * 0.1  # Penalty
    
    return _reflectance_from_thick(materials, thicknesses, n0, ns, wavelengths)


def design(problem):
    """Design broadband antireflection coating using numerical optimization."""
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin before 150s limit
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat_L = problem["materials"]["L"]
    mat_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Use 401 equally spaced wavelengths as the evaluator does
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Try different layer counts
    # Start with common AR coating designs
    layer_counts = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    layer_counts = [lc for lc in layer_counts if lc <= max_layers]
    
    # Sort by complexity - try simpler first to get a good initial guess
    for n_layers in layer_counts:
        if time.time() > deadline - 5.0:
            break
        
        # Try different material sequences
        # For AR coatings, common patterns start with L (low index) near air
        # and can alternate or use specific patterns
        
        if n_layers == 1:
            sequences = [["L"], ["H"]]
        elif n_layers == 2:
            sequences = [["L", "L"], ["L", "H"], ["H", "L"], ["H", "H"]]
        elif n_layers == 3:
            sequences = [["L", "L", "L"], ["L", "L", "H"], ["L", "H", "L"],
                        ["L", "H", "H"], ["H", "L", "L"], ["H", "L", "H"],
                        ["H", "H", "L"], ["H", "H", "H"]]
        else:
            # For larger stacks, try alternating and some patterns
            sequences = []
            # Alternating starting with L
            seq_alt_L = [("L" if i % 2 == 0 else "H") for i in range(n_layers)]
            sequences.append(seq_alt_L)
            # Alternating starting with H
            seq_alt_H = [("H" if i % 2 == 0 else "L") for i in range(n_layers)]
            sequences.append(seq_alt_H)
            # All L
            sequences.append(["L"] * n_layers)
            # L...LH pattern (most L, one H at substrate)
            if n_layers >= 2:
                sequences.append(["L"] * (n_layers - 1) + ["H"])
            # LH...L pattern
            if n_layers >= 2:
                sequences.append(["L", "H"] + ["L"] * (n_layers - 2))
        
        for seq in sequences:
            if time.time() > deadline - 5.0:
                break
            
            materials = [problem["materials"][s] for s in seq]
            
            # Initial guess: quarter-wave at center wavelength
            center_wl = 0.5 * (band[0] + band[1])
            x0 = np.array([center_wl / (4.0 * n) for n in materials])
            
            # Also try half-wave and other initializations
            initializations = [
                x0,
                np.array([center_wl / (2.0 * n) for n in materials]),
                np.array([center_wl / (4.0 * n) * 0.5 for n in materials]),
                np.array([center_wl / (4.0 * n) * 1.5 for n in materials]),
            ]
            
            for x_init in initializations:
                if time.time() > deadline - 10.0:
                    break
                
                # Clip initial values
                x_init = np.clip(x_init, 0, max_layer_nm)
                
                # Use Nelder-Mead for robustness (no gradient needed)
                try:
                    result = minimize(
                        _objective, x_init,
                        args=(materials, n0, ns, wavelengths, max_layer_nm, max_total_nm),
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 0.1, 'fatol': 0.001,
                                 'adaptive': True}
                    )
                    
                    if result.fun < best_score:
                        # Verify constraints
                        thicknesses = np.clip(np.abs(result.x), 0, max_layer_nm)
                        if np.sum(thicknesses) <= max_total_nm + 1e-6:
                            best_score = result.fun
                            best_materials = list(seq)
                            best_thicknesses = list(thicknesses)
                except Exception:
                    pass
        
        # After finding a good solution for this layer count, do a local polish
        if best_materials is not None and best_thicknesses is not None:
            if time.time() > deadline - 15.0:
                break
            
            # Polish with tighter tolerance
            materials = [problem["materials"][s] for s in best_materials]
            x0 = np.array(best_thicknesses)
            try:
                result = minimize(
                    _objective, x0,
                    args=(materials, n0, ns, wavelengths, max_layer_nm, max_total_nm),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 0.01, 'fatol': 0.0001,
                             'adaptive': True}
                )
                if result.fun < best_score:
                    thicknesses = np.clip(np.abs(result.x), 0, max_layer_nm)
                    if np.sum(thicknesses) <= max_total_nm + 1e-6:
                        best_score = result.fun
                        best_thicknesses = list(thicknesses)
            except Exception:
                pass
    
    # If no valid solution found, return a simple quarter-wave
    if best_materials is None or best_thicknesses is None:
        center = 0.5 * (band[0] + band[1])
        return ["L"], [center / (4.0 * mat_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
