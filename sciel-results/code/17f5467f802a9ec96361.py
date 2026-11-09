"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _compute_reflectance(wavelengths, n_layers, t_layers, n_ambient, n_substrate, n_materials):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    
    Parameters:
    - wavelengths: array of wavelengths in nm
    - n_layers: list of layer indices (L or H)
    - t_layers: list of thicknesses in nm
    - n_ambient: refractive index of ambient
    - n_substrate: refractive index of substrate
    - n_materials: dict mapping 'L' -> n_L, 'H' -> n_H
    
    Returns:
    - reflectance: array of reflectance values (0 to 1)
    """
    # Convert wavelengths to radians? No, we work in physical units.
    # The characteristic matrix method uses phase delta = 2*pi*n*t/lambda.
    
    # Build the total characteristic matrix M = M1 * M2 * ... * M_N
    # For each layer j: M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    
    N = len(n_layers)
    if N == 0:
        # Bare substrate
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r)**2
    
    # Initialize total matrix as identity
    M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
    
    for i in range(N):
        n_j = n_materials[n_layers[i]]
        t_j = t_layers[i]
        delta_j = 2.0 * np.pi * n_j * t_j / wavelengths  # phase thickness
        
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        # Layer matrix
        M_j = np.array([
            [cos_d, 1j * sin_d / n_j],
            [1j * n_j * sin_d, cos_d]
        ])
        
        # Accumulate: M = M_j * M (since light goes from air to substrate, first layer is first)
        # Actually, the convention is M_total = M_1 * M_2 * ... * M_N where M_1 is the first layer from incident side.
        # The characteristic matrix is defined such that [E, H]^T at interface 0 = M_total * [E, H]^T at interface N.
        # So we multiply on the left: M = M_j @ M
        M = M_j @ M
    
    # The admittance Y = n_ambient * (M[1,0] + M[1,1]) / (M[0,0] + M[0,1])
    # Wait, the standard formula:
    # Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate) * n_ambient? 
    # Let me recall the exact formula.
    
    # The characteristic matrix M relates the field at the input plane (air side) to the field at the output plane (substrate side).
    # [E_in]   = [M11 M12] [E_out]
    # [H_in]     [M21 M22] [H_out]
    # 
    # At the substrate side, the wave is only forward propagating, so:
    # E_out = A, H_out = A / n_substrate (in normalized units where H = E/n for forward wave)
    # Actually, the standard relation is:
    # Y = (M11 + M12 * n_s) / (M21 + M22 * n_s)
    # where Y is the input admittance. Then reflectance R = |(Y - n_0)/(Y + n_0)|^2
    
    n_s = n_substrate
    Y = (M[0,0] + M[0,1] * n_s) / (M[1,0] + M[1,1] * n_s)
    
    r = (Y - n_ambient) / (Y + n_ambient)
    R = np.abs(r)**2
    return R

def _objective(params, problem):
    """
    Objective function for differential evolution.
    params: [base_thickness, gradient_factor]
    """
    base_t, grad = params
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    max_layers = problem["max_layers"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    band = problem["band_nm"]
    
    # We use a periodic sequence of L and H layers.
    # Let's use a fixed number of layers, say 10 (5 pairs), to keep it simple but effective.
    # Actually, let's allow up to 10 layers (5 L-H pairs) for a good balance.
    n_pairs = 5
    n_layers_total = 2 * n_pairs
    
    # Generate thicknesses with a linear gradient
    # The gradient factor controls how much the thickness changes from one layer to the next.
    # t_i = base_t * (1 + grad * i) for i = 0, 1, ..., n_layers_total - 1
    # We need to ensure all thicknesses are in [0, max_layer_nm] and sum <= max_total_nm.
    
    thicknesses = []
    for i in range(n_layers_total):
        t_i = base_t * (1.0 + grad * i)
        if t_i < 0:
            t_i = 0.0
        if t_i > max_layer_nm:
            t_i = max_layer_nm
        thicknesses.append(t_i)
    
    total_t = sum(thicknesses)
    if total_t > max_total_nm:
        # Scale down
        scale = max_total_nm / total_t
        thicknesses = [t * scale for t in thicknesses]
        total_t = sum(thicknesses)
    
    # Create the layer sequence: L, H, L, H, ...
    labels = []
    for i in range(n_layers_total):
        if i % 2 == 0:
            labels.append("L")
        else:
            labels.append("H")
    
    # Compute reflectance
    wavelengths = np.linspace(band[0], band[1], 401)
    R = _compute_reflectance(wavelengths, labels, thicknesses, n_ambient, n_substrate, problem["materials"])
    mean_R = np.mean(R) * 100.0  # in percent
    
    return mean_R

def design(problem):
    """
    Design a broadband antireflection coating using differential evolution
    to optimize a periodic L-H stack with linear gradient thicknesses.
    """
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Define bounds for the optimization
    # base_t: [0, max_layer_nm]
    # grad: [-0.5, 0.5] (reasonable range for linear gradient)
    bounds = [(0.0, max_layer_nm), (-0.5, 0.5)]
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds,
        args=(problem,),
        maxiter=50,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=False
    )
    
    base_t, grad = result.x
    
    # Reconstruct the optimal design
    n_pairs = 5
    n_layers_total = 2 * n_pairs
    
    thicknesses = []
    for i in range(n_layers_total):
        t_i = base_t * (1.0 + grad * i)
        if t_i < 0:
            t_i = 0.0
        if t_i > max_layer_nm:
            t_i = max_layer_nm
        thicknesses.append(t_i)
    
    total_t = sum(thicknesses)
    if total_t > max_total_nm:
        scale = max_total_nm / total_t
        thicknesses = [t * scale for t in thicknesses]
    
    labels = []
    for i in range(n_layers_total):
        if i % 2 == 0:
            labels.append("L")
        else:
            labels.append("H")
    
    return labels, thicknesses
# EVOLVE-BLOCK-END
