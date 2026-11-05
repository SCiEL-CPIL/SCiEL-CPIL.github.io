"""Broadband antireflection coating design via Multi-Start Basin Hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _vectorized_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance fraction for a single design across all wavelengths.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return float(r * r)
    
    n_layers_arr = np.array([n_H if m == 'H' else n_L for m in materials])
    
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
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
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
    
    if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in thicknesses):
        return float('inf')
    if sum(thicknesses) > max_total_nm + 1e-6:
        return float('inf')
        
    t_calc = np.clip(thicknesses, 0, max_layer_nm)
    R_frac = _vectorized_reflectance(t_calc, materials, n_ambient, n_substrate, n_L, n_H, wls)
    return R_frac * 100.0

def _local_optimize(current_thicknesses, current_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm, time_limit=1.0):
    """
    Perform a short Simulated Annealing run to refine the current state.
    """
    start_time = time.time()
    current_score = _evaluate_design(current_thicknesses, current_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    
    if np.isinf(current_score):
        return current_thicknesses, current_materials, current_score

    best_score = current_score
    best_thicknesses = current_thicknesses.copy()
    best_materials = current_materials[:]
    
    T = 20.0
    T_min = 0.01
    alpha = 0.99
    
    while time.time() - start_time < time_limit:
        new_materials = current_materials[:]
        new_thicknesses = current_thicknesses.copy()
        
        move_type = np.random.rand()
        
        if move_type < 0.8:
            n_perturb = np.random.randint(1, 4)
            if len(new_thicknesses) > 0:
                indices = np.random.choice(len(new_thicknesses), n_perturb, replace=False)
                for idx in indices:
                    delta = np.random.normal(0, 15)
                    new_thicknesses[idx] += delta
                    new_thicknesses[idx] = np.clip(new_thicknesses[idx], 0, max_layer_nm)
        else:
            if len(new_materials) > 0:
                idx = np.random.randint(len(new_materials))
                new_materials[idx] = 'L' if new_materials[idx] == 'H' else 'H'

        if len(new_thicknesses) == 0:
            continue
        
        if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in new_thicknesses):
            continue
        if sum(new_thicknesses) > max_total_nm + 1e-6:
            continue
            
        new_score = _evaluate_design(new_thicknesses, new_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
        
        if np.isnan(new_score) or np.isinf(new_score):
            continue
            
        if new_score < current_score:
            current_score = new_score
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            
            if current_score < best_score:
                best_score = current_score
                best_thicknesses = current_thicknesses.copy()
                best_materials = current_materials[:]
        else:
            delta = new_score - current_score
            if delta < 50 * T:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current_score = new_score
                    current_materials = new_materials
                    current_thicknesses = new_thicknesses
                    
        T *= alpha
        if T < T_min:
            T = T_min

    return best_thicknesses, best_materials, best_score

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
    deadline = start_time + 145.0

    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    best_score = float('inf')
    best_thicknesses = [qw_L]
    best_materials = ['L']
    
    score = _evaluate_design(best_thicknesses, best_materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm)
    if score < best_score:
        best_score = score

    # Generate diverse starting points
    starts = []
    
    # 1. Standard quarter-wave stack (H-L-H-L-H)
    mat1 = ['H', 'L', 'H', 'L', 'H']
    th1 = [qw_H, qw_L, qw_H, qw_L, qw_H]
    starts.append((mat1, th1))
    
    # 2. Alternating starting with L (L-H-L-H-L)
    mat2 = ['L', 'H', 'L', 'H', 'L']
    th2 = [qw_L, qw_H, qw_L, qw_H, qw_L]
    starts.append((mat2, th2))
    
    # 3. Random 7-layer stack
    mat3 = ['H' if np.random.rand() < 0.5 else 'L' for _ in range(7)]
    th3 = [np.random.uniform(50, 150) for _ in range(7)]
    # Ensure sum constraint roughly
    if sum(th3) > max_total_nm:
        th3 = [t * (max_total_nm / sum(th3)) for t in th3]
    starts.append((mat3, th3))
    
    # 4. Random 10-layer stack
    mat4 = ['H' if np.random.rand() < 0.5 else 'L' for _ in range(10)]
    th4 = [np.random.uniform(30, 100) for _ in range(10)]
    if sum(th4) > max_total_nm:
        th4 = [t * (max_total_nm / sum(th4)) for t in th4]
    starts.append((mat4, th4))
    
    # 5. Thick single layer (near optimal for single layer)
    # Optimal single layer thickness is around lambda/4n, but for broadband, sometimes slightly different helps?
    # Actually, for a single layer, the optimal is exactly lambda/4n for the center wavelength.
    # Let's try a random thick layer.
    mat5 = ['H']
    th5 = [np.random.uniform(100, 300)]
    starts.append((mat5, th5))
    
    # Basin Hopping Loop
    for mat, th in starts:
        if time.time() > deadline:
            break
            
        # Ensure validity of start
        th_arr = np.array(th, dtype=float)
        if len(th_arr) == 0:
            continue
        if any(t < 0 or t > max_layer_nm for t in th_arr):
            continue
        if sum(th_arr) > max_total_nm:
            continue
            
        # Run local optimization
        opt_th, opt_mat, opt_score = _local_optimize(th_arr, mat, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm, time_limit=20.0)
        
        if opt_score < best_score:
            best_score = opt_score
            best_thicknesses = opt_th.tolist()
            best_materials = opt_mat[:]
            
    # If time remains, do a final polish on the best solution found
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 1:
            best_thicknesses, best_materials, best_score = _local_optimize(
                np.array(best_thicknesses), 
                best_materials, 
                n_ambient, n_substrate, n_L, n_H, wls, 
                max_total_nm, max_layer_nm, 
                time_limit=remaining_time
            )
            
    # Final validity check and cleanup
    if not best_thicknesses or any(t < 0 or t > max_layer_nm for t in best_thicknesses) or sum(best_thicknesses) > max_total_nm:
        return ["L"], [qw_L]
        
    cleaned_thick = [max(0.0, t) for t in best_thicknesses]
    if sum(cleaned_thick) > max_total_nm:
        scale = max_total_nm / sum(cleaned_thick)
        cleaned_thick = [t * scale for t in cleaned_thick]
        
    return best_materials, cleaned_thick

# EVOLVE-BLOCK-END
