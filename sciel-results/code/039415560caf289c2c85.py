"""Broadband antireflection coating design via Simulated Annealing with variable structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    Uses vectorized operations over the wavelength axis.
    
    thicknesses: array of physical thicknesses in nm
    materials: array of 'L' or 'H' strings
    wls: array of wavelengths in nm
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r)
    
    # Pre-compute refractive indices for each layer
    n_layers_arr = np.array([n_H if m == 'H' else n_L for m in materials])
    
    # M = Identity
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        n = n_layers_arr[i]
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance: Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.real(r * r.conjugate())
    
    return float(np.mean(R))

def _evaluate_design(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm):
    """
    Evaluate a design. Returns reflectance percentage if valid, inf otherwise.
    """
    if len(thicknesses) == 0:
        return float('inf')
    
    # Check constraints
    if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in thicknesses):
        return float('inf')
    if sum(thicknesses) > max_total_nm + 1e-6:
        return float('inf')
        
    # Clip tiny negative values to 0 for calculation stability
    t_calc = np.clip(thicknesses, 0, max_layer_nm)
    
    R_frac = _vectorized_reflectance(t_calc, materials, n_ambient, n_substrate, n_L, n_H, wls)
    return R_frac * 100.0

def design(problem):
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min, wl_max)[0] + 0.5 * (wl_min, wl_max)[1]
    
    # Initial best: Single quarter-wave L layer
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    # Fallback solution
    best_score = float('inf')
    best_thicknesses = [qw_L]
    best_materials = ['L']
    
    # Evaluate fallback
    score = _evaluate_design(best_thicknesses, best_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if score < best_score:
        best_score = score

    # Simulated Annealing Parameters
    # We represent the state as (materials_list, thicknesses_array)
    # Start with a reasonable initial guess: 5 alternating layers
    init_n_layers = 5
    current_materials = []
    current_thicknesses = []
    for i in range(init_n_layers):
        if i % 2 == 0:
            current_materials.append('H')
            current_thicknesses.append(qw_H)
        else:
            current_materials.append('L')
            current_thicknesses.append(qw_L)
    current_thicknesses = np.array(current_thicknesses)
    
    current_score = _evaluate_design(current_thicknesses, current_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if current_score < best_score:
        best_score = current_score
        best_thicknesses = current_thicknesses.tolist()
        best_materials = current_materials[:]

    # SA Loop
    T = 100.0  # Initial temperature (in percent reflectance units, scaled)
    T_min = 0.01
    alpha = 0.999  # Cooling rate
    max_iterations = 50000  # Max steps
    
    for step in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Generate a neighbor
        # 1. Perturb thicknesses
        # 2. Maybe change material of a layer
        # 3. Maybe add/remove a layer
        
        new_materials = current_materials[:]
        new_thicknesses = current_thicknesses.copy()
        
        move_type = np.random.rand()
        
        if move_type < 0.7:
            # Perturb 1-3 random thicknesses
            n_perturb = np.random.randint(1, 4)
            if len(new_thicknesses) > 0:
                indices = np.random.choice(len(new_thicknesses), n_perturb, replace=False)
                for idx in indices:
                    # Gaussian perturbation
                    delta = np.random.normal(0, 20)
                    new_thicknesses[idx] += delta
                    # Clamp
                    new_thicknesses[idx] = np.clip(new_thicknesses[idx], 0, max_layer_nm)
                    
        elif move_type < 0.9:
            # Change material of a random layer
            if len(new_materials) > 0:
                idx = np.random.randint(len(new_materials))
                if new_materials[idx] == 'H':
                    new_materials[idx] = 'L'
                else:
                    new_materials[idx] = 'H'
        else:
            # Add or remove a layer
            if len(new_materials) < max_layers and np.random.rand() < 0.5:
                # Add layer at random position
                pos = np.random.randint(0, len(new_materials) + 1)
                mat = 'H' if np.random.rand() < 0.5 else 'L'
                thick = np.random.uniform(50, 200)
                new_materials.insert(pos, mat)
                new_thicknesses = np.insert(new_thicknesses, pos, thick)
            elif len(new_materials) > 1 and np.random.rand() < 0.5:
                # Remove random layer
                pos = np.random.randint(0, len(new_materials))
                new_materials.pop(pos)
                new_thicknesses = np.delete(new_thicknesses, pos)
            
        # Check validity and evaluate
        if len(new_thicknesses) == 0:
            continue
            
        # Quick constraint check before expensive eval
        if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in new_thicknesses):
            continue
        if sum(new_thicknesses) > max_total_nm + 1e-6:
            continue
            
        new_score = _evaluate_design(new_thicknesses, new_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
        
        if np.isnan(new_score) or np.isinf(new_score):
            continue
            
        # Acceptance criterion
        if new_score < current_score:
            # Accept better
            current_score = new_score
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            
            if current_score < best_score:
                best_score = current_score
                best_thicknesses = current_thicknesses.tolist()
                best_materials = current_materials[:]
        else:
            # Accept worse with probability exp(-delta/T)
            delta = new_score - current_score
            # Prevent overflow
            if delta > 50 * T:
                prob = 0
            else:
                prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                current_score = new_score
                current_materials = new_materials
                current_thicknesses = new_thicknesses
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min

    # Final validity check and cleanup
    # Ensure we return a valid design
    if not best_thicknesses or any(t < 0 or t > max_layer_nm for t in best_thicknesses) or sum(best_thicknesses) > max_total_nm:
        # Fallback to single layer
        return ["L"], [qw_L]
        
    # Clean up tiny negative values if any slipped through
    cleaned_thick = [max(0.0, t) for t in best_thicknesses]
    # Re-check sum after cleaning
    if sum(cleaned_thick) > max_total_nm:
        # Scale down if necessary (rare)
        scale = max_total_nm / sum(cleaned_thick)
        cleaned_thick = [t * scale for t in cleaned_thick]
        
    return best_materials, cleaned_thick

# EVOLVE-BLOCK-END
