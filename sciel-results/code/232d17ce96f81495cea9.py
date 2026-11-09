"""Broadband antireflection coating design: Continuous refractive index profile optimization."""
import time
import random
import numpy as np

def _mean_reflectance(indices, thicknesses, problem):
    """
    Compute mean reflectance (percent) over the band using characteristic matrices.
    indices: list of float refractive indices (continuous)
    thicknesses: list of float thicknesses in nm
    """
    if not indices:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    
    ns = np.array(indices, dtype=float)
    ds = np.asarray(thicknesses, dtype=float)
    
    # delta_j = 2*pi*n_j*d_j/lambda
    # Shape: (N_layers, K_wavelengths)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
    # Initialize characteristic matrix product as identity for all wavelengths
    # M = [[A, B], [C, D]]
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    for j in range(len(indices)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        
        # Matrix for layer j:
        # [[cos(delta), i*sin(delta)/n_j], [i*n_j*sin(delta), cos(delta)]]
        # M_total = M_1 * M_2 * ... * M_N
        # New A = A*cj + B*(i*sj/n_j)
        # New B = A*(i*n_j*sj) + B*cj
        # New C = C*cj + D*(i*sj/n_j)
        # New D = C*(i*n_j*sj) + D*cj
        
        A_new = A * cj + 1j * B * sj / n_j
        B_new = 1j * A * n_j * sj + B * cj
        C_new = C * cj + 1j * D * sj / n_j
        D_new = 1j * C * n_j * sj + D * cj
        
        A, B, C, D = A_new, B_new, C_new, D_new
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    
    # Reflection coefficient r = (n_0*(A+B*n_s) - (C+D*n_s)) / (n_0*(A+B*n_s) + (C+D*n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_min = materials_dict['L'] # 1.38
    n_max = materials_dict['H'] # 2.35
    
    # We fix the number of layers to max_layers (20) to have a high-dimensional search space
    # but allow thicknesses to be 0. This effectively allows fewer layers if the optimizer
    # sets thicknesses to 0.
    N = max_layers
    
    def clamp_indices(indices):
        return [max(n_min, min(n_max, idx)) for idx in indices]
    
    def clamp_thicknesses(ths):
        # Clip to [0, max_layer]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        # Scale down if sum > max_total
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        return ths
    
    def evaluate(params):
        # params: [n_0, ..., n_{N-1}, d_0, ..., d_{N-1}]
        indices = clamp_indices(params[:N])
        thicknesses = clamp_thicknesses(params[N:])
        return _mean_reflectance(indices, thicknesses, problem)
    
    def nelder_mead(params, max_iters=200):
        """
        Standard Nelder-Mead implementation.
        """
        n = len(params)
        # Initialize simplex
        simplex = [list(params)]
        # Initial step size: 10% of range for indices, 10nm for thicknesses
        for i in range(n):
            p = list(params)
            if i < N:
                step = 0.1 * (n_max - n_min)
            else:
                step = 10.0
            p[i] += step
            simplex.append(p)
        
        scores = [evaluate(simplex[i]) for i in range(n + 1)]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            # Sort simplex
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            # Centroid of best n points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = evaluate(xr)
            
            if fr < scores[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                # Accept reflection
                simplex[-1] = xr
                scores[-1] = fr
            else:
                # Contraction
                if fr < scores[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = evaluate(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    # Inside contraction
                    xc = centroid + rho * (centroid - simplex[-1])
                    fc = evaluate(xc)
                    if fc <= scores[-1]:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                            scores[i] = evaluate(simplex[i])
                        if np.max(scores) - np.min(scores) < 1e-7:
                            break
        
        # Return best
        best_idx = np.argmin(scores)
        return simplex[best_idx], scores[best_idx]

    # Multi-start optimization
    best_params = None
    best_score = float('inf')
    
    random.seed(42)
    
    # Generate diverse initial guesses
    # 1. Geometric progression from n_min to n_max (graded index)
    # 2. Arithmetic progression
    # 3. Random
    # 4. All L, All H, Alternating
    
    inits = []
    
    # Geometric
    geo_indices = np.geomspace(n_min, n_max, N)
    geo_ths = np.full(N, 20.0) # Small initial thicknesses
    inits.append(list(geo_indices) + list(geo_ths))
    
    # Arithmetic
    arith_indices = np.linspace(n_min, n_max, N)
    arith_ths = np.full(N, 20.0)
    inits.append(list(arith_indices) + list(arith_ths))
    
    # Alternating discrete
    alt_indices = [n_min if i % 2 == 0 else n_max for i in range(N)]
    alt_ths = [20.0] * N
    inits.append(alt_indices + alt_ths)
    
    # All L
    all_l_indices = [n_min] * N
    all_l_ths = [20.0] * N
    inits.append(all_l_indices + all_l_ths)
    
    # All H
    all_h_indices = [n_max] * N
    all_h_ths = [20.0] * N
    inits.append(all_h_indices + all_h_ths)
    
    # Random
    for _ in range(5):
        rand_indices = np.random.uniform(n_min, n_max, N)
        rand_ths = np.random.uniform(10, 50, N)
        inits.append(list(rand_indices) + list(rand_ths))
    
    # Optimize each
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # Add some noise to start
        noisy_init = [p + random.uniform(-0.5, 0.5) for p in init]
        
        res_params, res_score = nelder_mead(noisy_init, max_iters=150)
        
        if res_score < best_score:
            best_score = res_score
            best_params = res_params

    # Final polish on the best found
    if best_params is not None:
        final_params, final_score = nelder_mead(best_params, max_iters=200)
        if final_score < best_score:
            best_score = final_score
            best_params = final_params

    # Extract best result
    if best_params is None:
        # Fallback
        return ["L"], [100.0]
        
    best_indices = best_params[:N]
    best_thicknesses = best_params[N:]
    
    # Post-processing: Snap indices to nearest material and remove zero-thickness layers
    # This is important because the evaluator expects 'L' or 'H' labels.
    # If an index is close to 1.38, it's L. If close to 2.35, it's H.
    # If it's in between, we must choose the nearest one.
    # However, the continuous optimization might find a "better" index that isn't available.
    # Snapping might degrade the score.
    # But the problem statement says: "Design a stack of ... low- and high-index layers (1.38 and 2.35)".
    # So we MUST return only L and H.
    
    # Strategy for snapping:
    # For each layer, if thickness is very small (< 1nm), remove it.
    # Otherwise, snap index to nearest of 1.38 or 2.35.
    
    final_materials = []
    final_ths = []
    
    for i in range(N):
        n_i = best_indices[i]
        d_i = best_thicknesses[i]
        
        if d_i < 1.0:
            continue
            
        # Determine label
        if abs(n_i - n_min) < abs(n_i - n_max):
            final_materials.append('L')
        else:
            final_materials.append('H')
            
        final_ths.append(d_i)
        
    # If all layers were removed, fallback
    if not final_materials:
        return ["L"], [100.0]
        
    # Verify constraints
    total_th = sum(final_ths)
    if total_th > max_total:
        scale = max_total / total_th
        final_ths = [t * scale for t in final_ths]
        
    return final_materials, final_ths
