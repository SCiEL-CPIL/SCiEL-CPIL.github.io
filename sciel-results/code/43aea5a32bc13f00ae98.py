"""Broadband antireflection coating design via multi-start Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _fast_reflectance(thicknesses, n_ambient, n_substrate, n_L, n_H, wl):
    """
    Compute reflectance fraction at a single wavelength using characteristic matrix method.
    thicknesses: list of physical thicknesses in nm
    Materials alternate starting with H: H, L, H, L, ...
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return r * r
    
    # Total characteristic matrix M = M1 * M2 * ... * MN
    # Start with identity
    M11, M12 = 1.0 + 0.0j, 0.0 + 0.0j
    M21, M22 = 0.0 + 0.0j, 1.0 + 0.0j
    
    for i in range(n_layers):
        if i % 2 == 0:
            n = n_H
        else:
            n = n_L
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wl
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix: [[cos_b, -j*sin_b/n], [j*n*sin_b, cos_b]]
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
    R = (r * r.conjugate()).real
    return float(R)


def _mean_reflectance_pct(thicknesses, n_ambient, n_substrate, n_L, n_H, wls):
    """Compute mean reflectance in percent over all wavelengths."""
    total_R = 0.0
    for wl in wls:
        total_R += _fast_reflectance(thicknesses, n_ambient, n_substrate, n_L, n_H, wl)
    return (total_R / len(wls)) * 100.0


def design(problem):
    """Design a broadband antireflection coating."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    # Quarter-wave thicknesses at center wavelength
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    def is_valid(thicknesses):
        if len(thicknesses) > max_layers:
            return False
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm:
            return False
        return True
    
    def evaluate(thicknesses):
        if not is_valid(thicknesses):
            return float('inf')
        return _mean_reflectance_pct(thicknesses, n_ambient, n_substrate, n_L, n_H, wls)
    
    # Generate initial designs: alternating H-L-H-L-... stacks
    initial_designs = []
    for n_layers in range(1, min(max_layers, 15) + 1, 2):
        # Alternating starting with H
        th = []
        for i in range(n_layers):
            if i % 2 == 0:
                th.append(qw_H)
            else:
                th.append(qw_L)
        if is_valid(th):
            initial_designs.append(th[:])
    
    # Also try some non-quarter-wave initializations
    initial_designs.append([qw_L])
    initial_designs.append([qw_H])
    
    # Evaluate all initial designs
    best_score = float('inf')
    best_thicknesses = None
    for th in initial_designs:
        s = evaluate(th)
        if s < best_score:
            best_score = s
            best_thicknesses = th[:]
    
    if best_thicknesses is None:
        # Fallback
        return ["H"], [qw_H]
    
    # Nelder-Mead optimization on the best initial design
    def objective(x):
        if not is_valid(x):
            return float('inf')
        return _mean_reflectance_pct(x, n_ambient, n_substrate, n_L, n_H, wls)
    
    # Try optimizing the best design found
    x0 = np.array(best_thicknesses, dtype=float)
    n_opt = len(x0)
    
    # Nelder-Mead with adaptive step size
    step = 50.0  # nm initial step
    simplex = [x0.copy()]
    for i in range(n_opt):
        perturbed = x0.copy()
        perturbed[i] += step
        simplex.append(perturbed)
    
    # Sort simplex by objective
    simplex_scores = [objective(v) for v in simplex]
    order = np.argsort(simplex_scores)
    simplex = [simplex[i] for i in order]
    simplex_scores = [simplex_scores[i] for i in order]
    
    best_trial_score = simplex_scores[0]
    best_trial_x = simplex[0].copy()
    
    max_iter = 300
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Reflection
        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + (centroid - simplex[-1])
        
        if objective(x_r) < simplex_scores[0]:
            # Expansion
            x_e = centroid + 2.0 * (centroid - simplex[-1])
            if objective(x_e) < objective(x_r):
                simplex[-1] = x_e
            else:
                simplex[-1] = x_r
        elif objective(x_r) < simplex_scores[-2]:
            simplex[-1] = x_r
        else:
            # Contraction
            x_c = centroid + 0.5 * (simplex[-1] - centroid)
            if objective(x_c) < simplex_scores[-1]:
                simplex[-1] = x_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
        
        # Re-sort
        simplex_scores = [objective(v) for v in simplex]
        order = np.argsort(simplex_scores)
        simplex = [simplex[i] for i in order]
        simplex_scores = [simplex_scores[i] for i in order]
        
        if simplex_scores[0] < best_trial_score:
            best_trial_score = simplex_scores[0]
            best_trial_x = simplex[0].copy()
        
        # Check convergence
        if len(simplex) > 1 and abs(simplex_scores[0] - simplex_scores[-1]) < 1e-10:
            break
    
    # Compare with best initial
    if best_trial_score < best_score:
        best_score = best_trial_score
        best_thicknesses = best_trial_x.tolist()
    
    # Try a few more initial designs if time permits
    if time.time() < deadline - 20.0:
        for th in initial_designs:
            if time.time() > deadline - 10.0:
                break
            x0 = np.array(th, dtype=float)
            # Quick local optimization
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                # Simple coordinate descent
                improved = False
                for i in range(len(x0)):
                    for delta in [step, -step, step/2, -step/2, step/4, -step/4]:
                        trial = x0.copy()
                        trial[i] += delta
                        if is_valid(trial):
                            s = _mean_reflectance_pct(trial, n_ambient, n_substrate, n_L, n_H, wls)
                            if s < best_score - 1e-8:
                                x0 = trial
                                best_score = s
                                best_thicknesses = x0.tolist()
                                improved = True
                                break
                    step *= 0.8
                if not improved:
                    break
    
    # Build final result
    n_final = len(best_thicknesses)
    materials = []
    for i in range(n_final):
        if i % 2 == 0:
            materials.append("H")
        else:
            materials.append("L")
    
    # Ensure all thicknesses are within bounds
    for i in range(n_final):
        best_thicknesses[i] = max(0.0, min(max_layer_nm, best_thicknesses[i]))
    
    # Final validity check
    if not is_valid(best_thicknesses):
        # Fallback to single quarter-wave L layer
        return ["L"], [qw_L]
    
    return materials, best_thicknesses
# EVOLVE-BLOCK-END
