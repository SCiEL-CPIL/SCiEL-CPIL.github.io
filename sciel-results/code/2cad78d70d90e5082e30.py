"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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
    1. Generate diverse initial candidates (quarter-wave stacks of varying lengths).
    2. Use Nelder-Mead simplex optimization to refine thicknesses for each candidate.
    3. Return the best valid solution found.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_materials = ["L"]
    # Initial best: simple quarter wave L
    best_thicknesses = [600.0 / (4.0 * mat_dict["L"])]
    best_score = _mean_reflectance(best_materials, best_thicknesses, wls, n_amb, n_sub, mat_dict)

    def objective(thicknesses):
        """
        Objective function for scipy.optimize.minimize.
        Returns mean reflectance plus a penalty if constraints are violated.
        """
        # Check constraints
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return 100.0 + 1e6  # High penalty
        if sum(thicknesses) > max_total_nm:
            return 100.0 + 1e6  # High penalty
            
        score = _mean_reflectance(best_materials, thicknesses, wls, n_amb, n_sub, mat_dict)
        return score

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
    # We will optimize these using Nelder-Mead

    candidates = []
    
    # 1. Single layer
    for mat in ['L', 'H']:
        d = 600.0 / (4.0 * mat_dict[mat])
        candidates.append(([mat], [d]))
        
    # 2. Two-layer quarter-wave stacks
    for wl_ref in [500, 550, 600]:
        for pattern in [('L', 'H'), ('H', 'L')]:
            mats = list(pattern)
            thicks = [wl_ref / (4.0 * mat_dict[m]) for m in mats]
            candidates.append((mats, thicks))
            
    # 3. Multi-layer quarter-wave stacks (alternating)
    for n_layers in range(3, 11):
        for wl_ref in [500, 550, 600]:
            mats = []
            thicks = []
            for i in range(n_layers):
                mat = 'L' if i % 2 == 0 else 'H'
                mats.append(mat)
                thicks.append(wl_ref / (4.0 * mat_dict[mat]))
            candidates.append((mats, thicks))
            
    # 4. Add some random patterns to diversify
    np.random.seed(42)
    for n_layers in range(2, 8):
        for _ in range(2):
            mats = []
            thicks = []
            for i in range(n_layers):
                mat = np.random.choice(['L', 'H'])
                mats.append(mat)
                thicks.append(np.random.uniform(50, 150))
            candidates.append((mats, thicks))

    # Optimize each candidate
    for mats, thicks in candidates:
        if time.time() - start_time > deadline:
            break
            
        # Check if we have enough time for an optimization run
        # Nelder-Mead can be expensive, so we limit maxiter
        # We set a time limit per optimization by checking time in a wrapper? 
        # SciPy minimize doesn't support time limits directly, but we can check after.
        
        # Update nonlocal for objective function
        # Note: objective uses best_materials, so we must set it temporarily
        # Or better, pass materials to objective via closure or partial
        # Let's create a specific objective for this candidate
        
        def obj_wrapper(thicknesses, m=mats):
             if any(t < 0 or t > max_layer_nm for t in thicknesses):
                 return 100.0 + 1e6
             if sum(thicknesses) > max_total_nm:
                 return 100.0 + 1e6
             return _mean_reflectance(m, thicknesses, wls, n_amb, n_sub, mat_dict)

        x0 = np.array(thicks, dtype=float)
        
        # Constraints for Nelder-Mead are not directly supported in the same way as SLSQP
        # We rely on the penalty method in the objective function.
        # However, Nelder-Mead can wander far. 
        # We can use bounds? No, Nelder-Mead doesn't support bounds in scipy.optimize.minimize directly.
        # We must handle it via penalty or by projecting.
        # Penalty is simpler but might converge to boundary.
        
        try:
            result = minimize(
                obj_wrapper, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-5, 'adaptive': True}
            )
        except Exception:
            continue
            
        if time.time() - start_time > deadline:
            break
            
        if result.success or result.fun < 100.0: # If it didn't fail catastrophically
            new_thicks = result.x
            # Clip to valid range just in case
            new_thicks = np.clip(new_thicks, 0, max_layer_nm)
            # Check total thickness constraint
            if sum(new_thicks) > max_total_nm:
                # Scale down if necessary? Or just reject?
                # Scaling down might break the optimal ratio. Rejecting is safer for now.
                # Or we can try to scale to fit.
                scale = max_total_nm / sum(new_thicks)
                new_thicks = new_thicks * scale
                if any(t > max_layer_nm for t in new_thicks):
                    continue
                    
            try_candidate(mats, new_thicks.tolist())

    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
