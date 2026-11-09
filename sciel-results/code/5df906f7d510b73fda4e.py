"""Broadband antireflection coating design: Fixed graded-index profile with Nelder-Mead optimization."""
import time
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """
    Compute mean reflectance (percent) over the band using characteristic matrices.
    Vectorized for speed.
    """
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)
    
    # delta_j = 2*pi*n_j*d_j/lambda
    # Shape: (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    
    c = np.cos(delta)
    s = np.sin(delta)
    
    # Initialize characteristic matrix product
    # M = [[A, B], [C, D]]
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    
    # Multiply matrices for each layer
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        
        A_new = A * cj + 1j * B * sj / n_j
        B_new = 1j * A * n_j * sj + B * cj
        C_new = C * cj + 1j * D * sj / n_j
        D_new = 1j * C * n_j * sj + D * cj
        
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
    deadline = start_time + 145  # Safety margin for 150s kill
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # 1. Fix the layer sequence to a graded index profile.
    # We want to go from Air (1.0) to Substrate (1.52).
    # Available indices: L=1.38, H=2.35.
    # A monotonic profile starting with H and ending with L provides a "graded" transition
    # that is known to be effective for broadband AR.
    # Pattern: H H H H H H H H H H L L L L L L L L L L
    # This creates a high-index entry layer (reducing initial mismatch via interference)
    # and a low-index exit layer (closer to substrate? No, 1.38 < 1.52, so it's a step down).
    # Actually, standard broadband AR often uses H/L pairs.
    # Let's try a specific graded sequence:
    # Start with H (2.35) to match the high index contrast, then gradually introduce L.
    # Or start with L? 
    # Inspiration 1 (score 3.04) used "alternating H L H L..." or similar graded.
    # Let's use a sequence that minimizes the number of index jumps.
    # Sequence: H, H, H, H, H, H, H, H, H, H, L, L, L, L, L, L, L, L, L, L
    # This is a two-layer effective stack with 10 sub-layers each.
    # This allows Nelder-Mead to optimize the "effective" thickness of each block 
    # by varying the individual layers, but the constraint is that they are all same material.
    # Wait, if they are all same material, we can just optimize 2 variables? 
    # No, the evaluator requires a list of thicknesses. 
    # If we fix materials to [H]*10 + [L]*10, we have 20 variables.
    # But physically, adjacent layers of same index can be merged.
    # However, the problem allows up to 20 layers. Merging them into 2 layers is allowed 
    # if we return fewer layers? 
    # "list of 'L'/'H' labels and a list of physical thicknesses"
    # If I return 2 layers, it's valid. 
    # BUT, the inspiration says "fixed 20-layer grid". 
    # Let's stick to 20 layers to allow for more complex interference structures 
    # if the optimizer wants to break the symmetry, but initialize with the graded pattern.
    # Actually, if I fix the materials, I can't change them. 
    # Let's try: Materials = ['H']*10 + ['L']*10.
    # This effectively creates a 2-layer stack where each "layer" is a 10-layer stack.
    # This reduces the effective dimensionality of the problem significantly compared to random 20 layers.
    
    # Alternative: Alternating H L H L ... is also a graded profile in terms of effective index.
    # Let's try the block approach first as it's a very strong prior for broadband AR.
    
    materials = ['H'] * 10 + ['L'] * 10
    
    # Initial thicknesses: Quarter wave at center wavelength for each layer
    # Or, since they are blocks, maybe total thickness for H block is ~400nm and L block ~400nm?
    # Let's start with uniform thicknesses such that total is ~800nm.
    # 40nm per layer.
    init_ths = [40.0] * 20
    
    # 2. Nelder-Mead Optimization
    # We need to handle constraints:
    # 1. 0 <= t_i <= max_layer (400)
    # 2. sum(t_i) <= max_total (1000)
    
    def objective(ths):
        # Check time
        if time.time() > deadline:
            return 1e10 # Penalize heavily to stop
        
        # Enforce bounds
        ths = np.clip(ths, 0.0, max_layer)
        
        # Enforce total sum
        total = np.sum(ths)
        if total > max_total:
            # Scale down
            ths = ths * (max_total / total)
            # Re-clip in case scaling pushed something weird (unlikely to go negative or > max if scaled down)
            ths = np.clip(ths, 0.0, max_layer)
            
        return _mean_reflectance(materials, ths.tolist(), problem)
    
    # Nelder-Mead implementation
    def nelder_mead(f, x0, max_iter=5000, tol=1e-8, deadline_check=None):
        n = len(x0)
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            point = x0.copy()
            point[i] = point[i] + 0.05 if point[i] == 0 else point[i] * 1.05
            simplex.append(point)
        
        # Evaluate initial simplex
        fvals = [f(p) for p in simplex]
        
        for it in range(max_iter):
            if deadline_check and deadline_check():
                break
                
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
                
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Reflection is good
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                beta = 0.5
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + beta * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - beta * (centroid - simplex[-1])
                    
                fc = f(xc)
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
                        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
    
    # Run Nelder-Mead
    # Start from initial guess
    # To ensure robustness, we might run multiple restarts if time permits, 
    # but NM is generally good with a good start.
    # Let's run one long NM.
    
    best_ths, best_score = nelder_mead(
        objective, 
        np.array(init_ths, dtype=float),
        max_iter=20000,
        deadline_check=lambda: time.time() > deadline
    )
    
    # Ensure final constraints are strictly met for the return value
    best_ths = np.clip(best_ths, 0.0, max_layer)
    total = np.sum(best_ths)
    if total > max_total:
        best_ths = best_ths * (max_total / total)
        best_ths = np.clip(best_ths, 0.0, max_layer)
        
    return materials, best_ths.tolist()
