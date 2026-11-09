"""Broadband antireflection coating design: Global Simulated Annealing on sequence and thicknesses."""
import time
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[800 - 400 + 1 if False else 401], 401) # Just use linspace(400, 800, 401)
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)
    
    # delta_j = 2*pi*n_j*d_j/lambda  -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)
        
    # Initialize with a random valid stack
    # Start with a moderate number of layers, e.g., 10
    current_n_layers = 10
    current_materials = []
    current_ths = []
    for i in range(current_n_layers):
        # Alternate L/H to start, but randomize a bit
        if i % 2 == 0:
            current_materials.append('L')
        else:
            current_materials.append('H')
        current_ths.append(qw[current_materials[-1]])
        
    current_ths = np.array(current_ths)
    
    def get_score(mats, ths):
        if len(mats) == 0:
            return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                            (problem["substrate_index"] + problem["ambient_index"])) ** 2
        # Check constraints
        if sum(ths) > max_total + 1e-6:
            return 1e6
        if any(t > max_layer + 1e-6 or t < -1e-6 for t in ths):
            return 1e6
        return _mean_reflectance(mats, ths, problem)

    current_score = get_score(current_materials, current_ths)
    best_materials = list(current_materials)
    best_ths = list(current_ths)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.5  # Initial temperature (in percent reflectance units)
    T_min = 0.01
    cooling_rate = 0.995
    
    rng = np.random.default_rng(42)
    
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline and T > T_min and steps < max_steps:
        steps += 1
        
        # Generate a neighbor
        r = rng.random()
        new_materials = list(current_materials)
        new_ths = list(current_ths)
        
        if r < 0.4:
            # Change thickness of a random layer
            idx = rng.integers(0, len(new_ths))
            # Gaussian perturbation
            step_size = 10.0 * T / 0.1 # Scale step with temperature? Or fixed?
            step_size = 5.0 
            new_ths[idx] += rng.normal(0, step_size)
            new_ths[idx] = max(0.0, min(max_layer, new_ths[idx]))
            
        elif r < 0.7:
            # Swap two layers
            if len(new_materials) > 1:
                i, j = rng.choice(len(new_materials), size=2, replace=False)
                new_materials[i], new_materials[j] = new_materials[j], new_materials[i]
                new_ths[i], new_ths[j] = new_ths[j], new_ths[i]
                
        elif r < 0.85:
            # Change material type of a random layer
            idx = rng.integers(0, len(new_materials))
            old_mat = new_materials[idx]
            new_mat = 'H' if old_mat == 'L' else 'L'
            new_materials[idx] = new_mat
            # Rescale thickness to maintain optical thickness roughly? Or just clamp.
            # Let's keep thickness same, it might be out of range but constraint check handles it.
            # Actually, better to scale thickness to quarter wave of new material to stay in reasonable bounds?
            # No, let the optimizer find it. But if we just flip, thickness might be bad.
            # Let's keep thickness.
            
        else:
            # Add or remove a layer
            if len(new_materials) < max_layers and rng.random() < 0.5:
                # Add layer
                pos = rng.integers(0, len(new_materials) + 1)
                mat = 'L' if rng.random() < 0.5 else 'H'
                new_materials.insert(pos, mat)
                new_ths.insert(pos, qw[mat])
            elif len(new_materials) > 2:
                # Remove layer
                pos = rng.integers(0, len(new_materials))
                new_materials.pop(pos)
                new_ths.pop(pos)
        
        new_score = get_score(new_materials, new_ths)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_materials = new_materials
            current_ths = new_ths
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_materials = list(current_materials)
                best_ths = list(current_ths)
                
        T *= cooling_rate
        
        # If we are stuck for a long time, cool faster
        # Simple check: if no improvement in last 500 steps, cool harder
        # (Omitted for simplicity, standard cooling usually works)
        
    # Final polish with Nelder-Mead on the best found sequence
    # This helps fix local minima in thickness space for the fixed sequence
    
    def nelder_mead(mats, init_ths, max_iters=200):
        ths = list(init_ths)
        n = len(ths)
        if n == 0:
            return ths, get_score(mats, ths)
            
        def constrain(ths_in):
            ths_out = list(ths_in)
            total = sum(ths_out)
            if total > max_total:
                scale = max_total / total
                ths_out = [t * scale for t in ths_out]
            for i in range(len(ths_out)):
                ths_out[i] = max(0.0, min(max_layer, ths_out[i]))
            total2 = sum(ths_out)
            if total2 > max_total:
                scale = max_total / total2
                ths_out = [t * scale for t in ths_out]
            return ths_out

        simplex = [constrain(ths)]
        for i in range(n):
            pt = constrain(ths)
            pt[i] += 5.0
            simplex.append(pt)
            
        scores = [get_score(mats, simplex[i]) for i in range(n + 1)]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        it = 0
        while it < max_iters and time.time() < deadline - 5:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = constrain(list(xr))
            fr = get_score(mats, xr)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = constrain(list(xe))
                fe = get_score(mats, xe)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                simplex[-1] = xr
                scores[-1] = fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = constrain(list(xc))
                fc = get_score(mats, xc)
                if fc <= fr:
                    simplex[-1] = xc
                    scores[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = constrain(simplex[i])
                        scores[i] = get_score(mats, simplex[i])
                    if np.max(scores) - np.min(scores) < 1e-7:
                        break
                        
        best_idx = np.argmin(scores)
        return simplex[best_idx], scores[best_idx]

    # Run Nelder-Mead on the best sequence found by SA
    if best_ths:
        polished_ths, polished_score = nelder_mead(best_materials, best_ths)
        if polished_score < best_score:
            best_ths = polished_ths
            best_score = polished_score
            best_materials = list(best_materials) # Sequence didn't change in NM

    # Ensure return types are lists of primitives
    if best_ths is None:
        return ["L"], [qw["L"]]
        
    return list(best_materials), list(best_ths)
