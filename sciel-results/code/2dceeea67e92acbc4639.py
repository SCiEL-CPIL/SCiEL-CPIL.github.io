"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _char_matrix_method(wavelengths, indices, optical_thicknesses, n_ambient, n_substrate):
    """
    Compute reflectance using the characteristic matrix method.
    
    Parameters:
    - wavelengths: array of wavelengths in nm
    - indices: array of refractive indices for each layer
    - optical_thicknesses: array of optical thicknesses (n*d) in nm
    - n_ambient: refractive index of ambient medium
    - n_substrate: refractive index of substrate
    
    Returns:
    - reflectance: array of reflectance values (0 to 1)
    """
    n_layers = len(indices)
    # Convert wavelengths to wavenumbers (2*pi/lambda)
    k = 2 * np.pi / wavelengths
    
    # Initialize the characteristic matrix for the stack
    # Start with the substrate
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    # Build the characteristic matrix for each layer
    for i in range(n_layers):
        n_i = indices[i]
        d_i = optical_thicknesses[i] / n_i  # physical thickness
        # Phase thickness
        delta = k * n_i * d_i
        # Characteristic matrix for a homogeneous layer
        M_layer = np.array([
            [np.cos(delta), 1j / n_i * np.sin(delta)],
            [1j * n_i * np.sin(delta), np.cos(delta)]
        ], dtype=complex)
        M = M @ M_layer
    
    # Compute the effective admittance
    # The total field is related to the characteristic matrix
    # For normal incidence, we use the ratio of fields
    # The reflectance is computed from the characteristic matrix
    # Y = (M[0,0] * n_substrate + M[0,1]) / (M[1,0] * n_substrate + M[1,1])
    Y = (M[0, 0] * n_substrate + M[0, 1]) / (M[1, 0] * n_substrate + M[1, 1])
    
    # Reflectance
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return np.real(R)

def _mean_reflectance(params, labels, problem):
    """
    Compute the mean reflectance for a given set of parameters.
    
    Parameters:
    - params: array of physical thicknesses
    - labels: list of 'L'/'H' labels
    - problem: problem dictionary
    
    Returns:
    - mean reflectance in percent
    """
    if len(params) != len(labels):
        return 100.0  # Invalid
    
    # Check constraints
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if np.any(params < -1e-9) or np.any(params > max_layer_nm + 1e-9):
        return 100.0
    if np.sum(params) > max_total_nm + 1e-9:
        return 100.0
    
    # Clip to valid range
    params = np.clip(params, 0.0, max_layer_nm)
    
    # Build indices and optical thicknesses
    materials = problem["materials"]
    indices = []
    for label in labels:
        indices.append(materials[label])
    indices = np.array(indices)
    
    optical_thicknesses = indices * params
    
    # Compute reflectance over the band
    band_nm = problem["band_nm"]
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    R = _char_matrix_method(wavelengths, indices, optical_thicknesses,
                            problem["ambient_index"], problem["substrate_index"])
    
    return np.mean(R) * 100.0

def _is_valid(materials_list, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials_list) != len(thicknesses):
        return False
    if len(materials_list) > max_layers:
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    if any(m not in ('L', 'H') for m in materials_list):
        return False
    
    return True

def _quarter_wave_stack(labels, problem):
    """Generate a quarter-wave stack with given labels."""
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    materials = problem["materials"]
    thicknesses = []
    for label in labels:
        n = materials[label]
        d = centre / (4.0 * n)
        thicknesses.append(d)
    return labels, thicknesses

def design(problem):
    """
    Design an antireflection coating using continuous optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Try several candidate layer counts and patterns
    best_score = 100.0
    best_materials = ["L"]
    best_thicknesses = [0.5 * (problem["band_nm"][0] + problem["band_nm"][1]) / (4.0 * problem["materials"]["L"])]
    
    # Candidate patterns: number of layers and alternating pattern
    # Start with symmetric patterns
    patterns = []
    
    # Quarter-wave designs with different layer counts
    for n_layers in range(1, min(problem["max_layers"] + 1, 21)):
        # Alternating L, H, L, H, ... starting with L
        labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                labels.append('L')
            else:
                labels.append('H')
        patterns.append(labels)
        
        # Also try starting with H
        if n_layers > 1:
            labels_h = []
            for i in range(n_layers):
                if i % 2 == 0:
                    labels_h.append('H')
                else:
                    labels_h.append('L')
            patterns.append(labels_h)
    
    # Add the classic 10-layer design
    classic = ['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    if classic not in patterns:
        patterns.append(classic)
    
    # Also try some non-alternating patterns
    # e.g., LHLH, HLHL, etc. are already covered by alternating
    # Try symmetric patterns like HLLH, HLHLH, etc.
    extra_patterns = [
        ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'],  # 11 layers
        ['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],  # 11 layers
        ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'],  # 13 layers
        ['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'],  # 13 layers
    ]
    for p in extra_patterns:
        if len(p) <= problem["max_layers"] and p not in patterns:
            patterns.append(p)
    
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Sort patterns by length (shorter first for quick wins)
    patterns.sort(key=lambda x: len(x))
    
    for labels in patterns:
        if time.time() > deadline:
            break
        
        n_layers = len(labels)
        
        # Initial guess: quarter-wave thicknesses
        x0 = np.array([centre / (4.0 * materials[label]) for label in labels])
        
        # Check if initial guess is valid
        if not _is_valid(labels, list(x0), problem):
            # Scale down if total thickness exceeds limit
            total = np.sum(x0)
            if total > problem["max_total_nm"]:
                x0 = x0 * (problem["max_total_nm"] / total)
        
        # Define bounds
        max_layer_nm = problem["max_layer_nm"]
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # Objective function
        def objective(x):
            return _mean_reflectance(x, labels, problem)
        
        # Try L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_materials = labels
                best_thicknesses = list(result.x)
        except Exception:
            pass
        
        # Also try a few random restarts if time permits
        if time.time() < deadline - 5.0:
            n_restarts = 3
            for _ in range(n_restarts):
                if time.time() > deadline - 2.0:
                    break
                # Random initial guess within bounds
                x_rand = np.random.uniform(0.0, max_layer_nm, size=n_layers)
                # Ensure total thickness constraint
                total = np.sum(x_rand)
                if total > problem["max_total_nm"]:
                    x_rand = x_rand * (problem["max_total_nm"] / total)
                
                try:
                    result = minimize(objective, x_rand, method='L-BFGS-B', bounds=bounds,
                                     options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.success and result.fun < best_score:
                        best_score = result.fun
                        best_materials = labels
                        best_thicknesses = list(result.x)
                except Exception:
                    pass
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to a simple valid design
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * problem["materials"]["L"])]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
