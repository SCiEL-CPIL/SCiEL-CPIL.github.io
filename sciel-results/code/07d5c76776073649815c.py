"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating stack."""
    deadline = time.time() + 140.0  # 20s safety margin before 160s kill
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    n_L = materials["L"]
    n_H = materials["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points from 400 to 800 nm)
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def compute_reflectance(labels, thicknesses):
        """Compute mean reflectance in percent using characteristic matrix method."""
        n_layers = len(labels)
        if n_layers == 0:
            # Bare substrate
            r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
            return 100.0 * r * r
        
        reflectances = np.zeros(len(wavelengths))
        
        for i, wl in enumerate(wavelengths):
            # Build characteristic matrix
            # M = product of all layer matrices * substrate matrix
            # For normal incidence:
            # Layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
            # where delta = 2*pi*n*d/wl
            # Substrate matrix: [[1, i/n_sub], [0, 1]] (for transmission into substrate)
            
            M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
            
            for j in range(n_layers):
                n_j = n_L if labels[j] == 'L' else n_H
                d_j = thicknesses[j]
                delta = 2.0 * np.pi * n_j * d_j / wl
                cos_d = np.cos(delta)
                sin_d = np.sin(delta)
                
                # Layer matrix for normal incidence
                L = np.array([
                    [cos_d, 1j * sin_d / n_j],
                    [1j * n_j * sin_d, cos_d]
                ])
                M = M @ L
            
            # Substrate matrix (going into substrate)
            S = np.array([
                [1.0+0j, 1j / substrate_index],
                [0.0+0j, 1.0+0j]
            ])
            M = M @ S
            
            # Reflectance: R = |M21/M11|^2 * (n0/n_sub) ... wait, let me be careful
            # Actually for the characteristic matrix method:
            # Y = M21 / M11 (admittance at the front surface)
            # R = |(n0 - Y)/(n0 + Y)|^2
            Y = M[1, 0] / M[0, 0]
            r = (ambient_index - Y) / (ambient_index + Y)
            reflectances[i] = np.abs(r)**2
        
        return np.mean(reflectances) * 100.0
    
    def objective(params):
        """Objective function for optimization."""
        if time.time() > deadline:
            return 1e10
        n_layers = len(params)
        labels = []
        thicknesses = []
        for p in params:
            if p < 0:
                labels.append('L')
                thicknesses.append(-p)
            else:
                labels.append('H')
                thicknesses.append(p)
        
        # Check constraints
        if sum(thicknesses) > max_total_nm:
            return 1e10 + sum(thicknesses) - max_total_nm
        for t in thicknesses:
            if t > max_layer_nm:
                return 1e10 + t - max_layer_nm
        
        return compute_reflectance(labels, thicknesses)
    
    def make_params(labels, thicknesses):
        """Convert labels and thicknesses to parameter vector."""
        params = []
        for l, t in zip(labels, thicknesses):
            if l == 'L':
                params.append(-t)
            else:
                params.append(t)
        return np.array(params)
    
    # Generate initial guesses
    initial_guesses = []
    
    # 1. Quarter-wave stack at center wavelength
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    qw_L = center_wl / (4.0 * n_L)
    qw_H = center_wl / (4.0 * n_H)
    
    # Single layer
    initial_guesses.append(make_params(['L'], [qw_L]))
    initial_guesses.append(make_params(['H'], [qw_H]))
    
    # Double layer (L-H)
    initial_guesses.append(make_params(['L', 'H'], [qw_L, qw_H]))
    initial_guesses.append(make_params(['H', 'L'], [qw_H, qw_L]))
    
    # Triple layer (L-H-L)
    initial_guesses.append(make_params(['L', 'H', 'L'], [qw_L, qw_H, qw_L]))
    initial_guesses.append(make_params(['H', 'L', 'H'], [qw_H, qw_L, qw_H]))
    
    # 4-layer stacks
    for pattern in [['L','H','L','H'], ['H','L','H','L'], ['L','L','H','H'], ['H','H','L','L']]:
        th = [qw_L if p=='L' else qw_H for p in pattern]
        initial_guesses.append(make_params(pattern, th))
    
    # 5-layer stacks
    for pattern in [['L','H','L','H','L'], ['H','L','H','L','H']]:
        th = [qw_L if p=='L' else qw_H for p in pattern]
        initial_guesses.append(make_params(pattern, th))
    
    # 6-layer stacks
    for pattern in [['L','H','L','H','L','H'], ['H','L','H','L','H','L']]:
        th = [qw_L if p=='L' else qw_H for p in pattern]
        initial_guesses.append(make_params(pattern, th))
    
    # Add some random initial guesses
    np.random.seed(42)
    for _ in range(10):
        n_l = np.random.randint(2, 8)
        labels = [np.random.choice(['L', 'H']) for _ in range(n_l)]
        thicknesses = np.random.uniform(10, 200, size=n_l)
        # Scale to fit within constraints
        total = sum(thicknesses)
        if total > max_total_nm:
            thicknesses = thicknesses * (max_total_nm / total)
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
        initial_guesses.append(make_params(labels, thicknesses))
    
    # Optimize each initial guess
    best_score = 1e10
    best_params = None
    best_labels = None
    best_thicknesses = None
    
    for x0 in initial_guesses:
        if time.time() > deadline - 5.0:
            break
        
        n_params = len(x0)
        bounds = [(-max_layer_nm, max_layer_nm)] * n_params
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'maxfev': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to single quarter-wave layer
        return ["L"], [center_wl / (4.0 * n_L)]
    
    # Convert back to labels and thicknesses
    labels = []
    thicknesses = []
    for p in best_params:
        if p < 0:
            labels.append('L')
            thicknesses.append(-p)
        else:
            labels.append('H')
            thicknesses.append(p)
    
    # Clean up very thin layers
    final_labels = []
    final_thicknesses = []
    for l, t in zip(labels, thicknesses):
        if t > 1.0:  # Keep only layers thicker than 1 nm
            final_labels.append(l)
            final_thicknesses.append(min(t, max_layer_nm))
    
    # Ensure total thickness constraint
    total = sum(final_thicknesses)
    if total > max_total_nm and len(final_thicknesses) > 0:
        scale = max_total_nm / total
        final_thicknesses = [t * scale for t in final_thicknesses]
    
    # Ensure at least one layer
    if not final_labels:
        final_labels = ["L"]
        final_thicknesses = [center_wl / (4.0 * n_L)]
    
    return final_labels, final_thicknesses
# EVOLVE-BLOCK-END
