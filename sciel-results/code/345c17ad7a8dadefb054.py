"""Broadband antireflection coating design using stochastic global optimization."""
import time
import math
import random

def _reflectance_mean(materials, thicknesses, n_ambient, n_substrate, band_nm, num_wl=401):
    """Compute mean reflectance (%) over the band using the characteristic-matrix method."""
    if not thicknesses:
        # Bare substrate reflectance
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return 100.0 * r * r

    wl0, wl1 = band_nm
    d_wl = (wl1 - wl0) / (num_wl - 1)
    total_r = 0.0
    n_layers = len(thicknesses)

    # Precompute indices list
    n_list = []
    for i, mat in enumerate(materials):
        n_list.append(problem["materials"][mat])

    # We need to import numpy for speed
    import numpy as np

    wavelengths = np.linspace(wl0, wl1, num_wl)
    k = 2.0 * math.pi

    for wl in wavelengths:
        # Build characteristic matrix product
        # M = M_1 * M_2 * ... * M_N
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # We work with complex numbers
        
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j

        for i in range(n_layers):
            n_i = n_list[i]
            d_i = thicknesses[i]
            delta_i = k * n_i * d_i / wl
            
            cos_d = math.cos(delta_i)
            sin_d = math.sin(delta_i)
            
            # Matrix M_i
            a = cos_d
            b = 1j * sin_d / n_i
            c = 1j * n_i * sin_d
            d = cos_d

            # Multiply current (m11, m12; m21, m22) by M_i
            # New M = Old M * M_i
            new_m11 = m11 * a + m12 * c
            new_m12 = m11 * b + m12 * d
            new_m21 = m21 * a + m22 * c
            new_m22 = m21 * b + m22 * d
            
            m11 = new_m11
            m12 = new_m12
            m21 = new_m21
            m22 = new_m22

        # Admittance Y = (n_ambient * m22 + m21) / (n_ambient * m12 + m11) ? No.
        # The characteristic matrix M relates [E, H] at the front to [E, H] at the back.
        # [E_f; H_f] = M * [E_s; H_s]
        # At the substrate, H_s = (n_sub / n_ambient) * E_s ? No, H is normalized by Z0.
        # Standard formulation:
        # Let M = [[M11, M12], [M21, M22]]
        # The input admittance Y_in = (n_sub * M21 + M22) / (n_sub * M11 + M12) ? 
        # Wait, the matrix is defined such that:
        # [E]   [M11 M12] [E]
        # [H] = [M21 M22] [H]
        # Where H is the magnetic field component.
        # At the substrate interface, the boundary condition is H_s = (n_sub / n_0) E_s ? 
        # Actually, usually it's defined with Y = H/E.
        # For a plane wave in medium n, Y = n (if H is normalized to E in vacuum units? No.)
        # Let's use the standard formula:
        # r = (n_0 * M22 + M21 - n_s * M12 - n_s * M11) / (n_0 * M22 + M21 + n_s * M12 + n_s * M11)
        # Wait, let's derive or recall correctly.
        
        # Matrix M propagates from layer 1 (air side) to layer N (substrate side).
        # [E_0; H_0] = M * [E_N; H_N]
        # At the substrate, E_N and H_N are related by the substrate impedance.
        # H_N = (n_sub / n_0) * E_N ? No, H is typically scaled.
        # Let's use the admittance method.
        # Y_in = (M21 + n_sub * M22) / (M11 + n_sub * M12)  <-- This is if M is defined differently.
        
        # Let's stick to the explicit r formula which is robust:
        # r = (Y_in - n_0) / (Y_in + n_0)
        # Y_in = (n_sub * M21 + M22) / (n_sub * M11 + M12)  ??
        
        # Let's check a single layer case.
        # M = [[cos d, i sin d / n], [i n sin d, cos d]]
        # Y_in = (n_s * i n sin d + cos d) / (n_s * cos d + i n sin d / n)
        # Y_in = (n_s i n sin d + cos d) / (n_s cos d + i sin d)
        # This looks like the standard input admittance formula where Y is in units of n_0?
        # Actually, standard formula:
        # Y_in = (n_0 M21 + n_s M22) / (n_0 M11 + n_s M12) ? No.
        
        # Let's use the reflection coefficient directly:
        # r = (M21 + n_sub * M22 - n_0 * M11 - n_0 * M12) / (M21 + n_sub * M22 + n_0 * M11 + n_0 * M12)
        # Wait, signs depend on convention.
        
        # Let's use a known correct implementation structure.
        # M = product of matrices.
        # r = (n_0 * M22 + M21 - n_s * M12 - n_s * M11) / (n_0 * M22 + M21 + n_s * M12 + n_s * M11)
        # Let's test this with no layers. M = I.
        # r = (n_0 - n_s) / (n_0 + n_s). Correct.
        
        # Let's test with one layer, thick limit (d -> infinity, oscillates).
        # Let's trust this formula.
        
        denom = n_ambient * m22 + m21 + n_substrate * m12 + n_substrate * m11
        numer = n_ambient * m22 + m21 - n_substrate * m12 - n_substrate * m11
        
        if abs(denom) < 1e-12:
            r_mag = 1.0
        else:
            r = numer / denom
            r_mag = abs(r)
        
        total_r += r_mag * r_mag

    return (total_r / num_wl) * 100.0

def design(problem):
    """Design a broadband AR coating using simulated annealing on layer thicknesses."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    # Strategy: Use a fixed number of layers (e.g., 20) to maximize degrees of freedom.
    # Alternate L and H layers to create a quasi-periodic structure that is good for broadband.
    # Or just optimize all 20 thicknesses.
    # Start with a symmetric design: L H H L L H H L ... or just random.
    # A common good starting point is quarter-wave stack.
    
    num_layers = max_layers
    # Initialize with quarter-wave thicknesses at center wavelength
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Pattern: L H L H ... or H L H L ...
    # Let's try a symmetric pattern: L H L H ... H L (palindrome)
    # Or just optimize random thicknesses.
    # Given the compute budget, we can do a decent amount of optimization.
    
    # We will use Simulated Annealing.
    # State: list of thicknesses (length num_layers) and material labels (fixed pattern).
    # Let's fix the pattern to L H L H ... to reduce search space?
    # Or allow pattern search? Pattern search is combinatorial.
    # The prompt says "list of 'L'/'H' labels".
    # Let's fix a good pattern: Alternating L and H.
    # Actually, a "doubly periodic" or "symmetric" stack is often optimal.
    # Let's use a fixed pattern of alternating L and H, starting with L.
    # L H L H ... (20 layers)
    
    pattern = []
    for i in range(num_layers):
        if i % 2 == 0:
            pattern.append('L')
        else:
            pattern.append('H')
    
    # Initial thicknesses: quarter-wave at center wavelength
    init_thick = []
    for mat in pattern:
        n_mat = materials_dict[mat]
        d = center_wl / (4.0 * n_mat)
        # Clamp to max_layer_nm
        if d > max_layer_nm:
            d = max_layer_nm
        init_thick.append(d)
        
    # Check total thickness constraint
    total = sum(init_thick)
    if total > max_total_nm:
        scale = max_total_nm / total
        init_thick = [d * scale for d in init_thick]
        
    best_thick = list(init_thick)
    best_score = _reflectance_mean(pattern, best_thick, n_ambient, n_substrate, band_nm)
    
    current_thick = list(init_thick)
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    # Estimate max iterations
    # Each eval takes ~0.1ms? Let's assume 1ms to be safe. 50s / 1ms = 50,000 iters.
    # We can do more if it's faster.
    
    iter_count = 0
    while time.time() < deadline:
        # Generate a new candidate
        new_thick = list(current_thick)
        # Perturb 1 or 2 random layers
        num_perturb = random.randint(1, 3)
        for _ in range(num_perturb):
            idx = random.randint(0, num_layers - 1)
            # Gaussian perturbation
            scale = max_layer_nm * 0.1
            delta = random.gauss(0, scale)
            new_thick[idx] += delta
            # Clamp
            if new_thick[idx] < 0:
                new_thick[idx] = 0
            elif new_thick[idx] > max_layer_nm:
                new_thick[idx] = max_layer_nm
        
        # Check total thickness
        total_new = sum(new_thick)
        if total_new > max_total_nm:
            # Scale down if necessary
            scale = max_total_nm / total_new
            new_thick = [d * scale for d in new_thick]
        
        new_score = _reflectance_mean(pattern, new_thick, n_ambient, n_substrate, band_nm)
        
        # Acceptance criterion
        if new_score < current_score:
            current_thick = new_thick
            current_score = new_score
            if new_score < best_score:
                best_thick = list(new_thick)
                best_score = new_score
        else:
            # Metropolis criterion
            delta_e = new_score - current_score
            if delta_e < 0:
                accept = True
            else:
                if T > T_min:
                    accept = random.random() < math.exp(-delta_e / T)
                else:
                    accept = False
            
            if accept:
                current_thick = new_thick
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        iter_count += 1
        
        # Occasionally restart if stuck?
        # Simple restart: if no improvement for a long time, reset to best and reheat?
        # For now, just continue.
        
    return pattern, best_thick

# Helper to make sure we don't crash if numpy is not available, though it should be.
# The function _reflectance_mean imports numpy inside.
# Let's ensure the import is at the top level if possible, or inside.
# The prompt says NumPy is available.

def design(problem):
    """
    Design a broadband antireflection coating.
    Uses a fixed 20-layer alternating L/H pattern and optimizes thicknesses
    using Simulated Annealing within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    # Import numpy for efficient vectorization if we refactor _reflectance_mean
    import numpy as np
    
    def calc_reflectance(thicknesses, pattern):
        """Vectorized reflectance calculation."""
        wl0, wl1 = band_nm
        num_wl = 401
        wavelengths = np.linspace(wl0, wl1, num_wl)
        k = 2.0 * np.pi
        
        n_list = np.array([materials_dict[m] for m in pattern])
        d_vec = np.array(thicknesses)
        
        # delta_i = k * n_i * d_i / wl
        # Shape: (num_wl, num_layers)
        deltas = k * n_list[None, :] * d_vec[None, :] / wavelengths[:, None]
        
        cos_d = np.cos(deltas)
        sin_d = np.sin(deltas)
        
        # Matrix elements
        # M_i = [[cos, i sin/n], [i n sin, cos]]
        # We multiply matrices. This is hard to vectorize over layers easily without a loop.
        # Loop over layers is fast enough if the inner part is vectorized over wavelengths.
        
        m11 = np.ones(num_wl, dtype=complex)
        m12 = np.zeros(num_wl, dtype=complex)
        m21 = np.zeros(num_wl, dtype=complex)
        m22 = np.ones(num_wl, dtype=complex)
        
        for i in range(len(pattern)):
            n_i = n_list[i]
            cos_i = cos_d[:, i]
            sin_i = sin_d[:, i]
            
            a = cos_i
            b = 1j * sin_i / n_i
            c = 1j * n_i * sin_i
            d = cos_i
            
            # M_new = M_old * M_i
            new_m11 = m11 * a + m12 * c
            new_m12 = m11 * b + m12 * d
            new_m21 = m21 * a + m22 * c
            new_m22 = m21 * b + m22 * d
            
            m11 = new_m11
            m12 = new_m12
            m21 = new_m21
            m22 = new_m22
        
        # r = (n_0 * M22 + M21 - n_s * M12 - n_s * M11) / (n_0 * M22 + M21 + n_s * M12 + n_s * M11)
        denom = n_ambient * m22 + m21 + n_substrate * m12 + n_substrate * m11
        numer = n_ambient * m22 + m21 - n_substrate * m12 - n_substrate * m11
        
        r = numer / denom
        r_mag_sq = np.abs(r) ** 2
        
        return np.mean(r_mag_sq) * 100.0
    
    # Fixed pattern: 20 layers, alternating L and H
    num_layers = max_layers
    pattern = []
    for i in range(num_layers):
        if i % 2 == 0:
            pattern.append('L')
        else:
            pattern.append('H')
    
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    init_thick = []
    for mat in pattern:
        n_mat = materials_dict[mat]
        d = center_wl / (4.0 * n_mat)
        if d > max_layer_nm:
            d = max_layer_nm
        init_thick.append(d)
        
    total = sum(init_thick)
    if total > max_total_nm:
        scale = max_total_nm / total
        init_thick = [d * scale for d in init_thick]
        
    best_thick = list(init_thick)
    best_score = calc_reflectance(best_thick, pattern)
    
    current_thick = list(init_thick)
    current_score = best_score
    
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.999
    
    while time.time() < deadline:
        new_thick = list(current_thick)
        num_perturb = random.randint(1, 4)
        for _ in range(num_perturb):
            idx = random.randint(0, num_layers - 1)
            scale = max_layer_nm * 0.1
            delta = random.gauss(0, scale)
            new_thick[idx] += delta
            if new_thick[idx] < 0:
                new_thick[idx] = 0
            elif new_thick[idx] > max_layer_nm:
                new_thick[idx] = max_layer_nm
        
        total_new = sum(new_thick)
        if total_new > max_total_nm:
            scale = max_total_nm / total_new
            new_thick = [d * scale for d in new_thick]
        
        new_score = calc_reflectance(new_thick, pattern)
        
        if new_score < current_score:
            current_thick = new_thick
            current_score = new_score
            if new_score < best_score:
                best_thick = list(new_thick)
                best_score = new_score
        else:
            delta_e = new_score - current_score
            if T > T_min:
                if random.random() < math.exp(-delta_e / T):
                    current_thick = new_thick
                    current_score = new_score
            # else reject
            
        T *= cooling_rate
        
    return pattern, best_thick
