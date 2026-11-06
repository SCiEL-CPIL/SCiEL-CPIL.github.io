"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _reflectance_vectorized(materials, thicknesses, wls, n_amb, n_sub, mat_dict):
    """
    Vectorized reflectance calculation for a stack of layers.
    Returns an array of reflectance values (fraction, not percent) for the given wavelengths.
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full_like(wls, r * r)
    
    # Precompute n and d for each layer
    ns = np.array([mat_dict[m] for m in materials])
    ds = np.array(thicknesses)
    
    # Initialize total matrix as identity for all wavelengths
    # Shape: (n_wls, 2, 2)
    M_total = np.zeros((len(wls), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n = ns[i]
        d = ds[i]
        if d < 1e-9:
            continue
        beta = 2.0 * np.pi * n * d / wls
        c = np.cos(beta)
        s = np.sin(beta)
        
        # Characteristic matrix for layer i:
        # [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
        # M_layer shape: (n_wls, 2, 2)
        M_layer = np.zeros((len(wls), 2, 2), dtype=complex)
        M_layer[:, 0, 0] = c
        M_layer[:, 0, 1] = 1j * s / n
        M_layer[:, 1, 0] = 1j * n * s
        M_layer[:, 1, 1] = c
        
        # M_total = M_total @ M_layer
        # Using einsum for batched matrix multiplication
        # (n_wls, 2, 2) @ (n_wls, 2, 2) -> (n_wls, 2, 2)
        M_total = np.einsum('wij,wjk->wik', M_total, M_layer)
        
    # Calculate input admittance Y
    # Y = (M[1,0] + M[1,1]*n_sub) / (M[0,0] + M[0,1]*n_sub)
    Y = (M_total[:, 1, 0] + M_total[:, 1, 1] * n_sub) / (M_total[:, 0, 0] + M_total[:, 0, 1] * n_sub)
    
    # Reflectance
    r = (n_amb - Y) / (n_amb + Y)
    return np.abs(r) ** 2

def _mean_reflectance(materials, thicknesses, wls, n_amb, n_sub, mat_dict):
    """Calculate mean reflectance in percent."""
    refls = _reflectance_vectorized(materials, thicknesses, wls, n_amb, n_sub, mat_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def design(problem):
    """
    Design a broadband antireflection coating.
    Strategy:
    1. Initialize with a good seed (quarter-wave stack).
    2. Use a local search (Nelder-Mead style or simple gradient-free) to optimize thicknesses.
    3. Try different stack patterns (number of layers, sequence).
    4. Use vectorized reflectance calculation for speed.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_materials = ["L"]
    best_thicknesses = [600.0 / (4.0 * mat_dict["L"])]
    best_score = _mean_reflectance(best_materials, best_thicknesses, wls, n_amb, n_sub, mat_dict)
    
    def try_candidate(materials, thicknesses):
        nonlocal best_materials, best_thicknesses, best_score
        if not _is_valid(materials, thicknesses, problem):
            return
        score = _mean_reflectance(materials, thicknesses, wls, n_amb, n_sub, mat_dict)
        if score < best_score:
            best_score = score
            best_materials = materials[:]
            best_thicknesses = thicknesses[:]
    
    # Generate initial candidates based on common AR coating structures
    # 1. Single layer
    for mat in ['L', 'H']:
        d = 600.0 / (4.0 * mat_dict[mat])
        try_candidate([mat], [d])
        
    # 2. Two-layer quarter-wave stacks
    for wl_ref in [500, 550, 600]:
        for pattern in [('L', 'H'), ('H', 'L')]:
            mats = list(pattern)
            thicks = [wl_ref / (4.0 * mat_dict[m]) for m in mats]
            try_candidate(mats, thicks)
            
    # 3. Multi-layer quarter-wave stacks (alternating)
    for n_layers in range(3, 11):
        for wl_ref in [500, 550, 600]:
            mats = []
            thicks = []
            for i in range(n_layers):
                mat = 'L' if i % 2 == 0 else 'H'
                mats.append(mat)
                thicks.append(wl_ref / (4.0 * mat_dict[mat]))
            try_candidate(mats, thicks)
            
    # 4. Local optimization using Nelder-Mead or similar
    # We will refine the best candidate found so far and some other promising ones.
    
    def optimize_thicknesses(materials, thicknesses, max_iters=2000):
        """
        Simple local search optimization of thicknesses.
        Uses a coordinate descent or small random perturbations.
        """
        current_thicks = np.array(thicknesses, dtype=float)
        current_score = _mean_reflectance(materials, current_thicks, wls, n_amb, n_sub, mat_dict)
        
        if not _is_valid(materials, current_thicks, problem):
            return None, current_score
            
        step_size = 10.0
        best_local_score = current_score
        best_local_thicks = current_thicks.copy()
        
        for i in range(max_iters):
            if time.time() - start_time > deadline:
                break
                
            improved = False
            for j in range(len(current_thicks)):
                for delta in [step_size, -step_size, step_size/2, -step_size/2]:
                    new_thicks = current_thicks.copy()
                    new_thicks[j] += delta
                    if new_thicks[j] < 0 or new_thicks[j] > max_layer_nm:
                        continue
                    if sum(new_thicks) > max_total_nm:
                        continue
                    new_score = _mean_reflectance(materials, new_thicks, wls, n_amb, n_sub, mat_dict)
                    if new_score < current_score - 1e-5:
                        current_thicks = new_thicks
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Reduce step size if no improvement
                step_size *= 0.8
                if step_size < 0.1:
                    break
            
            # Update global best if local best is better
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_thicks = current_thicks.copy()
                
        return best_local_thicks, best_local_score

    # Optimize the best candidate found in the initial generation
    if best_materials:
        opt_thicks, opt_score = optimize_thicknesses(best_materials, best_thicknesses)
        if opt_thicks is not None and opt_score < best_score:
            best_score = opt_score
            best_thicknesses = opt_thicks.tolist()
            
    # Try optimizing other common patterns
    for n_layers in range(2, 11):
        if time.time() - start_time > deadline:
            break
            
        # Generate a few random-ish starting points for this number of layers
        for _ in range(3):
            mats = []
            thicks = []
            for i in range(n_layers):
                # Alternate L and H, but start randomly
                start_mat = 'L' if (i + _ * 2) % 2 == 0 else 'H'
                mats.append(start_mat)
                # Random thickness in a reasonable range
                thicks.append(np.random.uniform(50, 200))
            
            if _is_valid(mats, thicks, problem):
                opt_thicks, opt_score = optimize_thicknesses(mats, thicks, max_iters=1000)
                if opt_thicks is not None:
                    try_candidate(mats, opt_thicks.tolist())
                    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
