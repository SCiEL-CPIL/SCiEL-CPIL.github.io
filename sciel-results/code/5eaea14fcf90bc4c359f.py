"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_reflectance(materials, thicknesses, n_air, n_sub, wavelengths):
    """
    Compute reflectance for a stack across all wavelengths.
    materials: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    n_air: ambient index
    n_sub: substrate index
    wavelengths: array of wavelengths in nm
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_air - n_sub) / (n_air + n_sub)
        return np.full_like(wavelengths, abs(r)**2)
    
    M_total = np.ones((len(wavelengths), 2, 2), dtype=complex)
    # Map materials to indices
    n_vals = []
    for m in materials:
        if m == 'L':
            n_vals.append(problem_global["materials"]["L"])
        else:
            n_vals.append(problem_global["materials"]["H"])
            
    for i in range(n_layers):
        n = n_vals[i]
        d = thicknesses[i]
        delta = 2.0 * np.pi * n * d / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        M_i = np.ones((len(wavelengths), 2, 2), dtype=complex)
        M_i[:, 0, 1] = 1j * sin_d / n
        M_i[:, 1, 0] = 1j * n * sin_d
        M_i[:, 0, 0] = cos_d
        M_i[:, 1, 1] = cos_d
        
        M_total = np.einsum('nij,njk->nik', M_total, M_i)
        
    m11 = M_total[:, 0, 0]
    m12 = M_total[:, 0, 1]
    m21 = M_total[:, 1, 0]
    m22 = M_total[:, 1, 1]
    
    Y_in = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
    r = (n_air - Y_in) / (n_air + Y_in)
    R = np.abs(r)**2
    return R

problem_global = None

def design(problem):
    global problem_global
    problem_global = problem
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def eval_stack(mats, thicks):
        if not mats:
             return 0.0 # Should not happen if we enforce min 1 layer, but safe
        # Check constraints
        if len(mats) > max_layers:
            return float('inf')
        if np.sum(thicks) > max_total_nm:
            return float('inf')
        if np.any(np.array(thicks) < -1e-9) or np.any(np.array(thicks) > max_layer_nm + 1e-9):
            return float('inf')
            
        # Clip small negatives to 0 for physical validity during eval if needed, 
        # but SA moves should prevent this.
        thicks_clipped = np.clip(np.array(thicks), 0, max_layer_nm)
        R = _compute_reflectance(mats, thicks_clipped, n_air, n_sub, wavelengths)
        return np.mean(R) * 100.0

    # Simulated Annealing
    start_time = time.time()
    time_budget = 55.0 # Leave 5s for overhead/return
    
    # Initial state: Random stack or a good heuristic start
    # Let's start with a reasonable heuristic: 5-layer quarter wave stack at center wavelength
    lam_c = 0.5 * (band_nm[0] + band_nm[1])
    init_mats = ['L', 'H', 'L', 'H', 'L']
    init_thicks = [lam_c / (4*n_L), lam_c / (4*n_H), lam_c / (4*n_L), lam_c / (4*n_H), lam_c / (4*n_L)]
    
    current_mats = init_mats[:]
    current_thicks = np.array(init_thicks, dtype=float)
    current_score = eval_stack(current_mats, current_thicks)
    
    best_mats = current_mats[:]
    best_thicks = current_thicks.copy()
    best_score = current_score
    
    T = 10.0 # Initial temperature (arbitrary scale, score is % reflectance)
    T_min = 1e-4
    alpha = 0.999 # Cooling factor
    
    # Pre-allocate for speed? No, list ops are fine for small N.
    
    while time.time() - start_time < time_budget:
        if T < T_min:
            break
            
        # Generate a neighbor
        move_type = random.random()
        new_mats = current_mats[:]
        new_thicks = current_thicks.copy()
        
        if move_type < 0.6: # Change thickness
            if len(current_mats) > 0:
                idx = random.randint(0, len(current_mats) - 1)
                # Gaussian perturbation scaled by current thickness or max
                scale = max(10.0, current_thicks[idx] * 0.5)
                delta_t = random.gauss(0, scale)
                new_thicks[idx] += delta_t
                # Ensure bounds
                new_thicks[idx] = np.clip(new_thicks[idx], 0, max_layer_nm)
                
        elif move_type < 0.8: # Change material
            if len(current_mats) > 0:
                idx = random.randint(0, len(current_mats) - 1)
                new_mats[idx] = 'H' if current_mats[idx] == 'L' else 'L'
                
        elif move_type < 0.9: # Add layer
            if len(current_mats) < max_layers:
                pos = random.randint(0, len(current_mats))
                new_mat = random.choice(['L', 'H'])
                # Insert at pos
                new_mats.insert(pos, new_mat)
                # Initial thickness: small random or quarter wave
                init_t = random.uniform(20, 100)
                new_thicks = np.insert(new_thicks, pos, init_t)
                
        else: # Remove layer
            if len(current_mats) > 1:
                idx = random.randint(0, len(current_mats) - 1)
                new_mats.pop(idx)
                new_thicks = np.delete(new_thicks, idx)
                
        # Evaluate
        new_score = eval_stack(new_mats, new_thicks)
        
        # Acceptance criterion
        if new_score < current_score:
            # Always accept better
            current_mats = new_mats
            current_thicks = new_thicks
            current_score = new_score
            
            if current_score < best_score:
                best_mats = current_mats[:]
                best_thicks = current_thicks.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            delta_E = new_score - current_score
            if delta_E > 0:
                prob = np.exp(-delta_E / T)
                if random.random() < prob:
                    current_mats = new_mats
                    current_thicks = new_thicks
                    current_score = new_score
            # else reject
            
        # Cool down
        T *= alpha
        
    # Final check: ensure best is valid
    if best_score >= float('inf'):
         # Fallback to simple quarter wave
         best_mats = ['L']
         best_thicks = [lam_c / (4*n_L)]
         
    # Clean up: remove zero thickness layers
    final_mats = []
    final_thicks = []
    for m, t in zip(best_mats, best_thicks):
        if t > 1e-6:
            final_mats.append(m)
            final_thicks.append(float(t))
            
    if len(final_mats) == 0:
        final_mats = ["L"]
        final_thicks = [lam_c / (4.0 * n_L)]
        
    return final_mats, final_thicks
# EVOLVE-BLOCK-END
