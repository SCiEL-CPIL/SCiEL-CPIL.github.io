"""Broadband antireflection coating design via characteristic-matrix optimization."""
import numpy as np
import time

def _reflectance(materials, thicknesses, wavelengths, n_air, n_sub):
    """
    Compute reflectance spectrum using characteristic matrix method.
    
    Parameters:
    - materials: list of 'L' or 'H' labels
    - thicknesses: list of thicknesses in nm
    - wavelengths: array of wavelengths in nm
    - n_air: ambient index
    - n_sub: substrate index
    
    Returns:
    - reflectance array (fraction, 0 to 1)
    """
    n_mats = {'L': 1.38, 'H': 2.35}
    num_layers = len(materials)
    
    # Initialize total matrix as identity
    M = np.ones((len(wavelengths), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for i in range(num_layers):
        n = n_mats[materials[i]]
        d = thicknesses[i]
        beta = 2 * np.pi * n * d / wavelengths
        # Characteristic matrix for each layer
        # B = [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        B = np.ones((len(wavelengths), 2, 2), dtype=complex)
        B[:, 0, 0] = cos_b
        B[:, 0, 1] = 1j * sin_b / n
        B[:, 1, 0] = 1j * n * sin_b
        B[:, 1, 1] = cos_b
        M = M @ B
    
    # Total matrix M = [[m11, m12], [m21, m22]]
    m11 = M[:, 0, 0]
    m12 = M[:, 0, 1]
    m21 = M[:, 1, 0]
    m22 = M[:, 1, 1]
    
    # Admittance at air interface: Y = (m11 + m21*n_air) / (m12*n_air + m22)
    # Actually, for normal incidence, the characteristic matrix relates fields
    # Let me use the standard formula:
    # Y = (m11 + m21 * n0) / (m12 * n0 + m22) where n0 is substrate? No...
    
    # Standard approach:
    # The characteristic matrix M gives:
    # [E0]   = M [En]
    # [H0]       [Hn]
    # where n is the substrate side
    # H = Y * E, so [E0] = M [En] [1; Yn]
    # [H0]         [En]
    # H0 = M21*En + M22*Yn*En = (M21 + M22*Yn)*En
    # E0 = M11*En + M12*Yn*En = (M11 + M12*Yn)*En
    # Y0 = H0/E0 = (M21 + M22*Yn)/(M11 + M12*Yn)
    
    Yn = n_sub  # admittance of substrate (for E field, Y = n for TE/TM at normal incidence)
    Y0 = (m21 + m22 * Yn) / (m11 + m12 * Yn)
    
    # Reflectance: R = |(Y0 - n_air)/(Y0 + n_air)|^2
    r = (Y0 - n_air) / (Y0 + n_air)
    R = np.abs(r)**2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n_points = 401
    wavelengths = np.linspace(band[0], band[1], n_points)
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    
    R = _reflectance(materials, thicknesses, wavelengths, n_air, n_sub)
    return 100.0 * np.mean(R)

def _gradient_step(materials, thicknesses, problem, lr=5.0, n_eval=201):
    """
    Compute gradient of mean reflectance with respect to thicknesses using finite differences.
    """
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], n_eval)
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_mats = {'L': 1.38, 'H': 2.35}
    
    f0 = _mean_reflectance_percent(materials, thicknesses, problem)
    grad = np.zeros(len(thicknesses))
    eps = 0.5  # nm
    
    for i in range(len(thicknesses)):
        t_plus = list(thicknesses)
        t_minus = list(thicknesses)
        t_plus[i] = min(400.0, t_plus[i] + eps)
        t_minus[i] = max(0.0, t_minus[i] - eps)
        
        R_plus = _reflectance(materials, t_plus, wavelengths, n_air, n_sub)
        R_minus = _reflectance(materials, t_minus, wavelengths, n_air, n_sub)
        
        f_plus = 100.0 * np.mean(R_plus)
        f_minus = 100.0 * np.mean(R_minus)
        
        # Central difference approximation
        if t_plus[i] > t_minus[i]:
            grad[i] = (f_plus - f_minus) / (t_plus[i] - t_minus[i])
        else:
            grad[i] = 0.0
    
    return f0, grad

def design(problem):
    """
    Design a broadband antireflection coating using gradient-based optimization.
    """
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Start with a reasonable initial guess: quarter-wave stack
    # Try a few different starting points
    best_materials = ["L"]
    best_thicknesses = [centre / (4.0 * 1.38)]
    best_score = _mean_reflectance_percent(best_materials, best_thicknesses, problem)
    
    # Try several initial configurations
    init_configs = [
        (["L"], [centre / (4.0 * 1.38)]),
        (["H"], [centre / (4.0 * 2.35)]),
        (["L", "H"], [centre / (4.0 * 1.38), centre / (4.0 * 2.35)]),
        (["H", "L"], [centre / (4.0 * 2.35), centre / (4.0 * 1.38)]),
        (["L", "H", "L"], [centre / (4.0 * 1.38), centre / (4.0 * 2.35), centre / (4.0 * 1.38)]),
        (["H", "L", "H"], [centre / (4.0 * 2.35), centre / (4.0 * 1.38), centre / (4.0 * 2.35)]),
        (["L", "H", "L", "H"], [centre / (4.0 * 1.38), centre / (4.0 * 2.35), centre / (4.0 * 1.38), centre / (4.0 * 2.35)]),
        (["H", "L", "H", "L"], [centre / (4.0 * 2.35), centre / (4.0 * 1.38), centre / (4.0 * 2.35), centre / (4.0 * 1.38)]),
    ]
    
    for mats, thicks in init_configs:
        if len(mats) > max_layers:
            continue
        if sum(thicks) > max_total_nm:
            continue
        score = _mean_reflectance_percent(mats, thicks, problem)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thicknesses = thicks[:]
    
    # Gradient-based optimization
    materials = best_materials[:]
    thicknesses = np.array(best_thicknesses, dtype=float)
    current_score = best_score
    
    # Learning rate schedule
    lr = 3.0
    
    while time.time() < deadline - 10.0:
        # Check constraints
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            # Project back
            thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        
        if np.sum(thicknesses) > max_total_nm:
            # Scale down
            thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
        
        # Compute gradient
        f0, grad = _gradient_step(materials, list(thicknesses), problem)
        
        if f0 < current_score:
            current_score = f0
            best_materials = materials[:]
            best_thicknesses = list(thicknesses)
        
        # Update
        thicknesses = thicknesses - lr * grad
        
        # Project to feasible region
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        total = np.sum(thicknesses)
        if total > max_total_nm:
            thicknesses = thicknesses * (max_total_nm / total)
        
        # Decay learning rate
        lr *= 0.95
        
        # If learning rate is too small, break
        if lr < 0.1:
            break
    
    # Final check and return
    final_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in thicknesses]
    total = sum(final_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        final_thicknesses = [t * scale for t in final_thicknesses]
    
    # Use the best found
    if current_score <= best_score:
        return materials, final_thicknesses
    else:
        return best_materials, best_thicknesses
