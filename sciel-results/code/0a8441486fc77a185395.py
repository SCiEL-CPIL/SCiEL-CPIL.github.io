"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance_stack(labels, thicknesses, n_ambient, n_substrate, n_materials, wavelengths):
    """
    Compute reflectance spectrum using the characteristic matrix method.
    wavelengths: array of wavelengths in nm
    Returns: array of reflectance in percent (0-100)
    """
    # Convert wavelengths to angular frequencies or use direct phase calculation
    # Phase thickness delta_j = 2*pi * n_j * d_j / lambda
    # We work in the complex plane. Since materials are lossless, n is real.
    
    # Number of wavelengths
    N = len(wavelengths)
    
    # Initialize the characteristic matrix M as identity for each wavelength
    # M = [[M11, M12], [M21, M22]]
    # We'll compute M for each wavelength
    
    # Pre-compute phase factors for each layer at each wavelength
    # delta_j = 2*pi*n_j*d_j/lambda
    # cos(delta) and sin(delta)
    
    # For efficiency, we can vectorize over wavelengths
    # Let's build the total matrix M = product of layer matrices
    
    # Layer matrices:
    # For a layer with index n_j and thickness d_j:
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j/lambda
    
    # The total matrix is M_total = M_1 * M_2 * ... * M_k
    # Then Y = (m11*n_sub + m12*n_sub) / (m21*n_sub + m22) ... wait
    # Actually, the admittance Y = (m11 + m12*n_sub) / (m21 + m22*n_sub) * n_sub? No.
    
    # Standard formula:
    # Let M = [[M11, M12], [M21, M22]]
    # The characteristic admittance is Y = (M11 + M12 * n_s) / (M21 + M22 * n_s)
    # where n_s is the substrate index? No, that's not right either.
    
    # Let me recall the standard thin-film characteristic matrix method.
    # For a stack of k layers on a substrate with index n_s, in a medium with index n_0:
    # The total characteristic matrix M = M_1 M_2 ... M_k
    # where each layer matrix M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # Then the input admittance Y = (M11 + M12 * n_s) / (M21 + M22 * n_s)
    # Wait, I think the formula is:
    # Y = (M11 * n_s + M12) / (M21 * n_s + M22)
    # Let me verify with a single quarter-wave layer.
    # If d = lambda/(4*n), delta = pi/2, cos(delta)=0, sin(delta)=1
    # M = [[0, i/n], [i*n, 0]]
    # Y = (0*n_s + i/n) / (i*n*n_s + 0) = (i/n)/(i*n*n_s) = 1/(n^2*n_s)
    # Reflectance R = |(Y - n_0)/(Y + n_0)|^2
    # For antireflection, we want Y = n_0, so 1/(n^2*n_s) = n_0 => n = sqrt(1/(n_0*n_s))
    # For n_0=1, n_s=1.52, n = sqrt(1/1.52) ≈ 0.81, which is not right.
    
    # Let me reconsider. The standard formula from Hecht or Macleod:
    # The characteristic matrix of the stack is M = M_1 M_2 ... M_k
    # The input admittance Y is given by:
    # Y = (M11 + M12 * n_s) / (M21 + M22 * n_s)
    # where n_s is the substrate refractive index.
    # Let's re-check with the single quarter-wave layer.
    # M = [[0, i/n], [i*n, 0]]
    # Y = (0 + i/n * n_s) / (i*n + 0 * n_s) = (i*n_s/n) / (i*n) = n_s/n^2
    # For AR, Y = n_0, so n_s/n^2 = n_0 => n = sqrt(n_s/n_0)
    # For n_0=1, n_s=1.52, n = sqrt(1.52) ≈ 1.23, which is closer but still not the standard result.
    
    # Actually, the standard AR condition for a single layer is n = sqrt(n_0 * n_s).
    # Let me look up the correct formula.
    
    # From Macleod's "Thin-Film Optical Filters":
    # The characteristic matrix of a layer is:
    # M_j = [[cos(delta_j), i*sin(delta_j)/eta_j], [i*eta_j*sin(delta_j), cos(delta_j)]]
    # where eta_j is the optical admittance (for non-magnetic materials, eta_j = n_j).
    # The total matrix is M = M_1 M_2 ... M_k
    # The input admittance Y is:
    # Y = (M11 + M12 * eta_s) / (M21 + M22 * eta_s)
    # where eta_s is the substrate admittance.
    
    # Let's re-check with the single quarter-wave layer.
    # M = [[0, i/eta], [i*eta, 0]]
    # Y = (0 + i/eta * eta_s) / (i*eta + 0) = (i*eta_s/eta) / (i*eta) = eta_s/eta^2
    # For AR, Y = eta_0, so eta_s/eta^2 = eta_0 => eta = sqrt(eta_s/eta_0)
    # For non-magnetic materials, eta = n, so n = sqrt(n_s/n_0)
    # This gives n = sqrt(1.52/1) ≈ 1.23 for our case.
    
    # But the standard AR condition is n = sqrt(n_0 * n_s) ≈ sqrt(1*1.52) ≈ 1.23.
    # Wait, sqrt(1.52) ≈ 1.233 and sqrt(1*1.52) ≈ 1.233. They are the same!
    # So the formula is correct.
    
    # Now, the reflectance is:
    # R = |(Y - eta_0)/(Y + eta_0)|^2
    
    # Let's implement this.
    
    # For each wavelength, compute the total matrix M.
    
    # To vectorize, we can compute the matrix for each wavelength separately,
    # but that would be slow. Instead, let's use a loop over wavelengths but vectorize over layers.
    
    # Actually, since we have at most 20 layers and 401 wavelengths, a simple loop should be fine.
    
    reflectances = np.zeros(N)
    
    for i, lam in enumerate(wavelengths):
        # Initialize M as identity matrix
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        
        for j, (label, d) in enumerate(zip(labels, thicknesses)):
            n_j = n_materials[label]
            delta_j = 2.0 * np.pi * n_j * d / lam
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Layer matrix
            M_j = np.array([[cos_d, 1j*sin_d/n_j], [1j*n_j*sin_d, cos_d]])
            
            M = M_j @ M
        
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]
        
        eta_s = n_substrate
        eta_0 = n_ambient
        
        Y = (M11 + M12 * eta_s) / (M21 + M22 * eta_s)
        
        R = np.abs((Y - eta_0) / (Y + eta_0)) ** 2
        reflectances[i] = R * 100.0  # Convert to percent
    
    return reflectances

def _mean_reflectance(labels, thicknesses, problem):
    """Compute the mean reflectance in percent over the band."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_materials = problem["materials"]
    band_nm = problem["band_nm"]
    
    # 401 equally spaced wavelengths
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    reflectances = _reflectance_stack(labels, thicknesses, n_ambient, n_substrate, n_materials, wavelengths)
    return np.mean(reflectances)

def _build_stack(base_thickness, gradient, n_periods, problem):
    """
    Build a periodic stack of alternating L and H layers.
    The thickness of each layer is modulated by a linear gradient factor.
    
    Parameters:
    - base_thickness: base thickness in nm
    - gradient: linear gradient factor (applied to thickness)
    - n_periods: number of LH pairs (so total layers = 2*n_periods)
    - problem: problem dictionary
    
    Returns:
    - labels: list of 'L'/'H' labels
    - thicknesses: list of physical thicknesses in nm
    """
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Ensure we don't exceed max_layers
    n_pairs = min(n_periods, max_layers // 2)
    
    if n_pairs <= 0:
        return [], []
    
    labels = []
    thicknesses = []
    
    for i in range(n_pairs):
        # Position in the stack (0 to 2*n_pairs-1)
        pos = 2 * i
        # Linear gradient factor: from 1.0 at the air side to (1 + gradient) at the substrate side
        # Or we can use a symmetric gradient. Let's use a linear ramp.
        # t_i = base_thickness * (1 + gradient * (i / (n_pairs - 1))) if n_pairs > 1 else base_thickness
        
        if n_pairs > 1:
            frac = i / (n_pairs - 1)
        else:
            frac = 0.0
        
        scale = 1.0 + gradient * frac
        
        # L layer
        d_L = base_thickness * scale
        # H layer
        d_H = base_thickness * scale
        
        # Clamp to [0, max_layer_nm]
        d_L = max(0.0, min(d_L, max_layer_nm))
        d_H = max(0.0, min(d_H, max_layer_nm))
        
        labels.append('L')
        thicknesses.append(d_L)
        labels.append('H')
        thicknesses.append(d_H)
    
    # Check total thickness
    total = sum(thicknesses)
    if total > max_total_nm:
        # Scale down all thicknesses
        scale_factor = max_total_nm / total
        thicknesses = [d * scale_factor for d in thicknesses]
    
    return labels, thicknesses

def _objective(params, problem):
    """
    Objective function for differential evolution.
    params: [base_thickness, gradient, n_periods]
    """
    base_thickness, gradient, n_periods = params
    n_periods = int(round(n_periods))
    
    # Ensure n_periods is at least 1
    n_periods = max(1, n_periods)
    
    labels, thicknesses = _build_stack(base_thickness, gradient, n_periods, problem)
    
    if not labels:
        return 100.0  # High penalty
    
    return _mean_reflectance(labels, thicknesses, problem)

def design(problem):
    """
    Design a stack of at most twenty low- and high-index layers (1.38 and 2.35) on glass
    that minimizes the mean reflectance between 400 and 800 nm at normal incidence.
    """
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Bounds for differential evolution
    # base_thickness: [10, max_layer_nm]
    # gradient: [-1.0, 1.0]
    # n_periods: [1, max_layers//2]
    
    bounds = [
        (10.0, max_layer_nm),
        (-1.0, 1.0),
        (1, max_layers // 2)
    ]
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds,
        args=(problem,),
        seed=42,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        polish=False
    )
    
    base_thickness, gradient, n_periods = result.x
    n_periods = int(round(n_periods))
    n_periods = max(1, n_periods)
    
    labels, thicknesses = _build_stack(base_thickness, gradient, n_periods, problem)
    
    # Ensure constraints are met
    if not labels:
        # Fallback to a simple single layer
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]
    
    # Final check on constraints
    total = sum(thicknesses)
    if total > max_total_nm:
        scale_factor = max_total_nm / total
        thicknesses = [d * scale_factor for d in thicknesses]
    
    for d in thicknesses:
        if d > max_layer_nm:
            thicknesses[thicknesses.index(d)] = max_layer_nm
    
    return labels, thicknesses
# EVOLVE-BLOCK-END
