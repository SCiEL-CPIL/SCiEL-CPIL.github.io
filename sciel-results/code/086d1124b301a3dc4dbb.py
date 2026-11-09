"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _characteristic_matrix(n, d, k, lam):
    """
    Compute the 2x2 characteristic matrix for a single layer.
    n: complex refractive index
    d: thickness in nm
    k: wavenumber factor = 2*pi*n/lam (but we use beta = 2*pi*n*d/lam)
    Actually, the characteristic matrix for a layer with index n, thickness d at wavelength lam:
    M = [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    where beta = 2*pi*n*d/lam
    """
    beta = 2.0 * np.pi * n * d / lam
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    i = 1j
    M = np.array([
        [cos_b, i * sin_b / n],
        [i * n * sin_b, cos_b]
    ], dtype=complex)
    return M

def _reflectance(mats, thicknesses, lam, n_s, n_0):
    """
    Compute normal-incidence reflectance for a given wavelength.
    mats: list of complex or real indices for each layer
    thicknesses: list of thicknesses in nm
    lam: wavelength in nm
    n_s: substrate index
    n_0: ambient index
    """
    # Start from the substrate
    # The total matrix is product of layer matrices times the substrate matrix
    # For the substrate (semi-infinite), the matrix is [[1, 0], [1/n_s, 0]]... 
    # Actually, standard method:
    # M_total = M_1 * M_2 * ... * M_N
    # Then [B, C]^T = M_total * [1, 1/n_s]^T
    # Then Y = C/B (admittance at front surface)
    # R = |(n_0 - Y)/(n_0 + Y)|^2  ... wait, let me be careful.
    
    # Standard thin-film characteristic matrix method:
    # For a stack of N layers on a substrate with index n_s, in ambient n_0:
    # The characteristic matrix for each layer j is:
    # M_j = [[cos(delta_j), i*sin(delta_j)/eta_j], [i*eta_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j/(lam) (for normal incidence, n_j is the phase index)
    # eta_j = n_j (for normal incidence, s- and p-polarization have same eta = n_j)
    
    # The total matrix M = M_1 * M_2 * ... * M_N
    # Then [B, C]^T = M * [1, 1/n_s]^T
    # The input admittance Y = C/B
    # Reflectance R = |(Y - n_0)/(Y + n_0)|^2
    
    if len(mats) == 0:
        # Bare substrate
        Y = 1.0 / n_s
        r = (Y - n_0) / (Y + n_0)
        return abs(r) ** 2
    
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for i in range(len(mats)):
        n_j = mats[i]
        d_j = thicknesses[i]
        M_j = _characteristic_matrix(n_j, d_j, 0, lam)
        M = M @ M_j
    
    # Multiply by the substrate vector [1, 1/n_s]
    vec = np.array([1.0, 1.0 / n_s], dtype=complex)
    BC = M @ vec
    B = BC[0]
    C = BC[1]
    
    if abs(B) < 1e-12:
        return 1.0
    
    Y = C / B
    r = (Y - n_0) / (Y + n_0)
    R = abs(r) ** 2
    return float(np.real(R))

def _mean_reflectance(mats, thicknesses, band_nm, n_s, n_0):
    """Compute mean reflectance in percent over the band."""
    lam0, lam1 = band_nm
    lams = np.linspace(lam0, lam1, 401)
    Rs = []
    for lam in lams:
        R = _reflectance(mats, thicknesses, lam, n_s, n_0)
        Rs.append(R)
    return 100.0 * np.mean(Rs)

def _greedy_add_layer(problem, current_labels, current_thicknesses):
    """
    Try adding one more layer of each type (L or H) with optimal thickness.
    Return the best (label, thickness) or None if no improvement.
    """
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    current_total = sum(current_thicknesses)
    current_mats = [n_L if l == 'L' else n_H for l in current_labels]
    current_R = _mean_reflectance(current_mats, current_thicknesses, band_nm, n_s, n_0)
    
    best_R = current_R
    best_label = None
    best_thickness = None
    
    for label, n_layer in [('L', n_L), ('H', n_H)]:
        if current_total + max_layer_nm > max_total_nm:
            continue
        
        # Search for optimal thickness
        # Use a coarse-to-fine search
        best_local_R = current_R
        best_local_t = 0.0
        
        # Coarse search
        for t in np.linspace(0, max_layer_nm, 50):
            if current_total + t > max_total_nm:
                continue
            new_mats = current_mats + [n_layer]
            new_ths = current_thicknesses + [t]
            R = _mean_reflectance(new_mats, new_ths, band_nm, n_s, n_0)
            if R < best_local_R:
                best_local_R = R
                best_local_t = t
        
        if best_local_t > 0:
            # Fine search around best_local_t
            t_min = max(0, best_local_t - max_layer_nm / 50)
            t_max = min(max_layer_nm, best_local_t + max_layer_nm / 50)
            for t in np.linspace(t_min, t_max, 20):
                if current_total + t > max_total_nm:
                    continue
                new_mats = current_mats + [n_layer]
                new_ths = current_thicknesses + [t]
                R = _mean_reflectance(new_mats, new_ths, band_nm, n_s, n_0)
                if R < best_local_R:
                    best_local_R = R
                    best_local_t = t
        
        if best_local_R < best_R:
            best_R = best_local_R
            best_label = label
            best_thickness = best_local_t
    
    if best_label is not None:
        return best_label, best_thickness, best_R
    return None

def _optimize_existing_thicknesses(problem, labels, thicknesses, iterations=30):
    """
    Optimize the thicknesses of existing layers using a simple coordinate descent.
    """
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    mats = [n_L if l == 'L' else n_H for l in labels]
    ths = list(thicknesses)
    current_R = _mean_reflectance(mats, ths, band_nm, n_s, n_0)
    
    for _ in range(iterations):
        improved = False
        for i in range(len(ths)):
            # Try to optimize thickness i
            best_R = current_R
            best_t = ths[i]
            
            # Coarse search
            for t in np.linspace(0, max_layer_nm, 30):
                # Check total thickness constraint
                new_total = sum(ths) - ths[i] + t
                if new_total > max_total_nm:
                    continue
                new_ths = ths[:i] + [t] + ths[i+1:]
                R = _mean_reflectance(mats, new_ths, band_nm, n_s, n_0)
                if R < best_R:
                    best_R = R
                    best_t = t
            
            if best_t != ths[i]:
                ths[i] = best_t
                current_R = best_R
                improved = True
        
        if not improved:
            break
    
    return labels, ths, current_R

def design(problem):
    """
    Design a broadband antireflection coating using a greedy layer addition approach
    followed by thickness optimization.
    """
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Start with no layers
    labels = []
    thicknesses = []
    
    # Greedily add layers
    for _ in range(max_layers):
        result = _greedy_add_layer(problem, labels, thicknesses)
        if result is None:
            break
        label, thickness, new_R = result
        labels.append(label)
        thicknesses.append(thickness)
    
    # Optimize thicknesses
    if len(labels) > 0:
        labels, thicknesses, final_R = _optimize_existing_thicknesses(problem, labels, thicknesses, iterations=20)
    
    # Ensure constraints
    total = sum(thicknesses)
    if total > max_total_nm:
        # Scale down
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    # Clip thicknesses
    thicknesses = [max(0, min(max_layer_nm, t)) for t in thicknesses]
    
    return labels, thicknesses
# EVOLVE-BLOCK-END
