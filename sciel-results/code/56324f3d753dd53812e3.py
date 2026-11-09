"""
Broadband antireflection coating design using differential evolution.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution

def _characteristic_matrix(wavelengths, layers, thicknesses, n_ambient, n_substrate):
    """
    Compute the reflectance spectrum using the characteristic matrix method.
    
    Parameters:
    -----------
    wavelengths : array_like
        Wavelengths in nm
    layers : list of str
        Layer labels ('L' or 'H'), from air to substrate
    thicknesses : list of float
        Physical thicknesses in nm
    n_ambient : float
        Refractive index of ambient medium
    n_substrate : float
        Refractive index of substrate
    
    Returns:
    --------
    reflectance : ndarray
        Reflectance (fraction, not percent) at each wavelength
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    n_layers = len(layers)
    
    # Get refractive indices for each layer
    n_layers_arr = np.zeros(n_layers)
    for i, label in enumerate(layers):
        n_layers_arr[i] = n_map[label]
    
    # Compute optical thicknesses (n*d) and phase deltas
    # delta_i = 2*pi*n_i*d_i / lambda
    # We'll compute for each wavelength
    
    reflectance = np.zeros(len(wavelengths))
    
    for idx, lam in enumerate(wavelengths):
        # Phase thickness for each layer: delta = 2*pi*n*d/lambda
        deltas = 2.0 * np.pi * n_layers_arr * thicknesses / lam
        
        # Characteristic matrix for each layer
        # M_i = [[cos(delta_i), (i/sin(delta_i)) / n_i],
        #        [i*n_i*sin(delta_i), cos(delta_i)]]
        
        # Start with identity matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for i in range(n_layers):
            c = np.cos(deltas[i])
            s = np.sin(deltas[i])
            n_i = n_layers_arr[i]
            
            # Avoid division by zero when sin(delta) is very small
            if abs(s) < 1e-12:
                s_inv = 0.0
            else:
                s_inv = 1.0 / s
            
            # Layer matrix
            M_i = np.array([
                [c, 1j * s_inv / n_i],
                [1j * n_i * s, c]
            ], dtype=complex)
            
            M = M @ M_i
        
        # Compute admittance
        # Y = M[1,0] / M[0,0]
        # R = |(Y_s - Y) / (Y_s + Y)|^2 where Y = n_ambient * (M[0,0] + M[0,1]) / (M[1,0] + M[1,1])
        # Actually, for normal incidence:
        # B = M[0,0] + M[0,1]
        # C = M[1,0] + M[1,1]
        # Y = C / B
        # R = |(n_ambient * Y - n_substrate) / (n_ambient * Y + n_substrate)|^2
        
        B = M[0, 0] + M[0, 1]
        C = M[1, 0] + M[1, 1]
        
        if abs(B) < 1e-15:
            reflectance[idx] = 1.0
            continue
        
        Y = C / B
        R = np.abs((n_ambient * Y - n_substrate) / (n_ambient * Y + n_substrate)) ** 2
        reflectance[idx] = float(np.real(R))
    
    return reflectance

def _evaluate_design(layers, thicknesses, problem):
    """Evaluate a design, returning mean reflectance in percent."""
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    
    refl = _characteristic_matrix(wavelengths, layers, thicknesses, n_ambient, n_substrate)
    return float(np.mean(refl) * 100.0)

def design(problem):
    """
    Design an antireflection coating using differential evolution.
    """
    global n_map
    n_map = problem["materials"]
    
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # We'll optimize over a fixed number of layers (up to max_layers)
    # Each layer has a type (0=L, 1=H) and a thickness
    # We'll use differential evolution on a continuous parameter vector
    
    # Strategy: fix number of layers, optimize thicknesses and types
    # For types, we can encode as continuous variables and round
    
    # Let's try a few different layer counts and pick the best
    best_score = float('inf')
    best_result = None
    
    # Try different numbers of layers
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if time.time() > deadline:
            break
        
        # Parameters: for each layer, one continuous variable for thickness
        # We'll handle layer types separately with a simpler approach
        # Actually, let's encode both type and thickness
        # For each layer: variable in [0, 1] where <0.5 means L, >=0.5 means H
        # And a separate variable for thickness in [0, max_layer_nm]
        
        # Total parameters: 2 * n_layers
        # But we need to handle the constraint that total thickness <= max_total_nm
        
        def objective(params):
            if time.time() > deadline:
                return float('inf')
            
            types = []
            thicknesses = []
            total_thick = 0.0
            
            for i in range(n_layers):
                type_var = params[2*i]
                thick_var = params[2*i + 1]
                
                label = 'L' if type_var < 0.5 else 'H'
                types.append(label)
                thicknesses.append(thick_var)
                total_thick += thick_var
            
            if total_thick > max_total_nm:
                return float('inf')
            
            score = _evaluate_design(types, thicknesses, problem)
            return score
        
        # Bounds
        bounds = []
        for i in range(n_layers):
            bounds.append((0.0, 1.0))  # type
            bounds.append((0.0, max_layer_nm))  # thickness
        
        # Run differential evolution
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42
            )
            
            if result.success or result.fun < best_score:
                if result.fun < best_score and np.isfinite(result.fun):
                    best_score = result.fun
                    types = []
                    thicknesses = []
                    for i in range(n_layers):
                        type_var = result.x[2*i]
                        thick_var = result.x[2*i + 1]
                        label = 'L' if type_var < 0.5 else 'H'
                        types.append(label)
                        thicknesses.append(float(thick_var))
                    best_result = (types, thicknesses)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: single quarter-wave layer
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]
    
    return best_result
