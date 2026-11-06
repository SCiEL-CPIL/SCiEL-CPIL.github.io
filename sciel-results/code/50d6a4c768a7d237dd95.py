"""Rewritten program: QWOT-based combinatorial design with simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np

def _reflectance_spectrum(materials, thicknesses, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance (fraction) for a stack of layers.
    materials: list of refractive indices (real, lossless)
    thicknesses: list of physical thicknesses in nm
    wavelengths: array of wavelengths in nm
    Returns: array of reflectance fractions.
    """
    n0 = ambient_index
    ns = substrate_index
    n_layers = len(materials)
    n_total = n_layers + 1  # include substrate as last medium
    
    # Build array of refractive indices: ambient, layers..., substrate
    n_all = [n0] + list(materials) + [ns]
    # Optical thicknesses in radians: phi_j = 2*pi*n_j*d_j / lambda
    # We'll compute the characteristic matrix product for each wavelength
    
    num_waves = len(wavelengths)
    R = np.zeros(num_waves)
    
    for iw, lam in enumerate(wavelengths):
        # Compute phase factors for each layer
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # where delta_j = 2*pi*n_j*d_j/lambda
        
        # Start with identity matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = materials[j]
            d_j = thicknesses[j]
            delta_j = 2.0 * np.pi * n_j * d_j / lam
            c = np.cos(delta_j)
            s = np.sin(delta_j)
            
            # Characteristic matrix for layer j
            Mj = np.array([[c, 1j * s / n_j],
                           [1j * n_j * s, c]], dtype=complex)
            M = M @ Mj
        
        # Boundary conditions:
        # [B, C] = M @ [1, 1/n_s]  (at substrate side, E and H normalized)
        # Actually, the standard formulation:
        # Let M_total = M_1 * M_2 * ... * M_N
        # Then [E_0, H_0] = M_total * [E_s, H_s]
        # At substrate: E_s = 1, H_s = 1/n_s (for unit incident wave from left)
        # Wait, let me use the standard transfer matrix approach.
        
        # The characteristic matrix for the stack is M = M_1 M_2 ... M_N
        # [B, C]^T = M [1, 1/n_s]^T  where B and C are at the front surface
        # Then the admittance at the front surface is Y = C/B (normalized by n0)
        # Reflectance amplitude r = (n0 - Y) / (n0 + Y) where Y = C/B * n0? 
        
        # Let me be more careful. The standard thin-film characteristic matrix:
        # For a stack, the relation between fields at the entrance and exit is:
        # [E_0, H_0]^T = M_total [E_N, H_N]^T
        # where E_N, H_N are at the substrate interface.
        # For a wave incident from the left, at the substrate we have:
        # E_N = 1 + r_N, H_N = (1/n_s)(1 - r_N)
        # But it's easier to use the admittance method.
        
        # Alternative: Use the formula directly.
        # B = M[0,0] + M[0,1]/n_s
        # C = M[1,0] + M[1,1]/n_s
        # Y = C/B  (this is the normalized admittance at the entrance)
        # r = (n0 - Y) / (n0 + Y)
        # R = |r|^2
        
        B = M[0, 0] + M[0, 1] / substrate_index
        C = M[1, 0] + M[1, 1] / substrate_index
        
        Y = C / B
        r = (n0 - Y) / (n0 + Y)
        R[iw] = np.abs(r) ** 2
    
    return R

def _evaluate(materials_seq, scale, problem):
    """
    Evaluate a design given a sequence of 'L'/'H' and a QWOT scale factor.
    Each layer has optical thickness = scale * (pi/2) at the center wavelength.
    Wait, QWOT means optical thickness = lambda_0/4, so physical thickness = lambda_0/(4*n).
    If we scale by a factor s, then optical thickness = s * lambda_0/4.
    So physical thickness = s * lambda_0 / (4 * n).
    """
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    
    materials = []
    thicknesses = []
    for label in materials_seq:
        n = mat_dict[label]
        # Optical thickness = scale * lambda_0 / 4
        # Physical thickness = optical_thickness / n = scale * lambda_0 / (4 * n)
        d = scale * lambda_0 / (4.0 * n)
        materials.append(n)
        thicknesses.append(d)
    
    # Check constraints
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    
    if len(materials) > max_layers:
        return 1.0  # invalid, return worst case
    if any(d < 0 or d > max_layer_nm for d in thicknesses):
        return 1.0
    if sum(thicknesses) > max_total_nm:
        return 1.0
    
    # Compute reflectance
    wavelengths = np.linspace(band[0], band[1], 401)
    R = _reflectance_spectrum(materials, thicknesses, problem["ambient_index"], 
                               problem["substrate_index"], wavelengths)
    mean_R = np.mean(R) * 100.0  # in percent
    return mean_R

def design(problem):
    """
    Design an antireflection coating using QWOT-based simulated annealing.
    """
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    n_L = mat_dict["L"]
    n_H = mat_dict["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Precompute wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # QWOT scale: each layer has optical thickness = scale * lambda_0 / 4
    # Physical thickness for L: scale * lambda_0 / (4 * n_L)
    # Physical thickness for H: scale * lambda_0 / (4 * n_H)
    
    # The scale factor s determines the actual thicknesses.
    # For a given sequence, we can find the optimal scale by minimizing mean reflectance.
    # But to keep it simple, we'll do SA over sequences and scales.
    
    # Start with a good initial design: alternating L and H, starting with L
    # Let's try a few initial sequences
    
    best_seq = ['L']
    best_scale = lambda_0 / (4.0 * n_L) / (lambda_0 / 4.0)  # This is 1/n_L? No.
    # Let me re-define: scale is the factor such that optical_thickness = scale * lambda_0/4
    # So for the seed, optical thickness = lambda_0/4, so scale = 1.0
    best_scale = 1.0
    best_score = _evaluate(best_seq, best_scale, problem)
    
    # Simulated annealing
    num_iter = 5000
    T_init = 0.5
    T_final = 0.001
    cooling = (T_final / T_init) ** (1.0 / num_iter)
    
    current_seq = list(best_seq)
    current_scale = best_scale
    current_score = best_score
    
    for i in range(num_iter):
        T = T_init * (cooling ** i)
        
        # Generate a neighbor
        r = np.random.rand()
        if r < 0.4:
            # Change a random layer
            new_seq = list(current_seq)
            if len(new_seq) > 0:
                idx = np.random.randint(len(new_seq))
                new_seq[idx] = 'H' if new_seq[idx] == 'L' else 'L'
            else:
                new_seq = ['L']
            new_scale = current_scale
        elif r < 0.6:
            # Add a layer
            if len(current_seq) < max_layers:
                new_seq = list(current_seq)
                pos = np.random.randint(len(new_seq) + 1)
                new_layer = 'L' if np.random.rand() < 0.5 else 'H'
                new_seq.insert(pos, new_layer)
            else:
                new_seq = list(current_seq)
            new_scale = current_scale
        elif r < 0.8:
            # Remove a layer
            if len(current_seq) > 1:
                new_seq = list(current_seq)
                idx = np.random.randint(len(new_seq))
                new_seq.pop(idx)
            else:
                new_seq = list(current_seq)
            new_scale = current_scale
        else:
            # Perturb the scale
            new_seq = list(current_seq)
            new_scale = current_scale * (1.0 + 0.1 * np.random.randn())
            new_scale = max(0.5, min(2.0, new_scale))  # Keep in reasonable range
        
        # Evaluate
        new_score = _evaluate(new_seq, new_scale, problem)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score < current_score:
            current_seq = new_seq
            current_scale = new_scale
            current_score = new_score
            if current_score < best_score:
                best_seq = list(current_seq)
                best_scale = current_scale
                best_score = current_score
        else:
            delta = new_score - current_score
            if delta < 0:
                # This shouldn't happen if new_score >= current_score
                pass
            accept_prob = np.exp(-delta / T) if T > 0 else 0.0
            if np.random.rand() < accept_prob:
                current_seq = new_seq
                current_scale = new_scale
                current_score = new_score
    
    # Refine: for the best sequence, do a finer search over scale
    # Try multiple scale values around the best
    for s in np.linspace(0.7, 1.3, 25):
        score = _evaluate(best_seq, s, problem)
        if score < best_score:
            best_scale = s
            best_score = score
    
    # Also try adding/removing layers around the best
    for trial in range(100):
        trial_seq = list(best_seq)
        r = np.random.rand()
        if r < 0.5 and len(trial_seq) < max_layers:
            pos = np.random.randint(len(trial_seq) + 1)
            trial_seq.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
        elif r < 0.8 and len(trial_seq) > 1:
            idx = np.random.randint(len(trial_seq))
            trial_seq.pop(idx)
        else:
            if len(trial_seq) > 0:
                idx = np.random.randint(len(trial_seq))
                trial_seq[idx] = 'H' if trial_seq[idx] == 'L' else 'L'
        
        for s in np.linspace(0.7, 1.3, 5):
            score = _evaluate(trial_seq, s, problem)
            if score < best_score:
                best_seq = list(trial_seq)
                best_scale = s
                best_score = score
    
    # Convert back to labels and thicknesses
    materials_labels = best_seq
    thicknesses = []
    for label in best_seq:
        n = mat_dict[label]
        d = best_scale * lambda_0 / (4.0 * n)
        thicknesses.append(d)
    
    return materials_labels, thicknesses
# EVOLVE-BLOCK-END
