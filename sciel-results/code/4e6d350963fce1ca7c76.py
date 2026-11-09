"""
Broadband antireflection coating design:
Global search over layer materials and thicknesses.
Uses Differential Evolution for thickness optimization and 
Randomized Hill Climbing for material sequence optimization.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
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
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Precompute quarter-wave thicknesses at center wavelength for initialization
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)
    
    # Bounds for Differential Evolution
    # We optimize N thicknesses.
    # We will handle the total thickness constraint by scaling or by checking in the objective.
    # However, DE requires box constraints.
    # Strategy: Optimize thicknesses in [0, max_layer].
    # If sum > max_total, we penalize heavily or scale.
    # Scaling inside the objective is safer for DE to find the feasible region.
    
    def objective(ths, materials):
        # ths is a numpy array
        # Check total thickness
        total = np.sum(ths)
        if total > max_total:
            # Penalize: return a high value
            # Scale down to feasible region for evaluation? 
            # No, better to just return a high penalty to guide DE away.
            # Or scale:
            if total > 1e-6:
                scale = max_total / total
                ths = ths * scale
            else:
                return 100.0
        # Evaluate
        return _mean_reflectance(materials, list(ths), problem)
    
    def optimize_thicknesses(materials):
        """Run DE to find optimal thicknesses for a given material sequence."""
        n = len(materials)
        if n == 0:
            return [], 100.0
            
        bounds = [(0.0, max_layer) for _ in range(n)]
        
        # Initial guess: quarter-wave
        x0 = [qw[m] for m in materials]
        
        # DE settings
        # Maxiter: limit to keep within time budget
        # Popsize: keep small to allow more iterations or restarts
        # We will run DE with a limited number of iterations and rely on multiple restarts/sequences
        
        try:
            res = differential_evolution(
                lambda ths: objective(ths, materials),
                bounds,
                maxiter=50,  # Keep low to allow many sequences
                popsize=5,
                tol=1e-4,
                seed=42,
                polish=True
            )
            if res.success:
                return list(res.x), res.fun
            else:
                return list(res.x), res.fun
        except Exception:
            return x0, objective(x0, materials)

    # Global Search Strategy:
    # 1. Generate a set of candidate material sequences.
    # 2. For each, optimize thicknesses.
    # 3. Keep the best.
    # 4. Additionally, perform local search on the best sequence (swap/insert/delete layers)
    
    best_score = float('inf')
    best_materials = None
    best_ths = None
    
    # Candidate 1: Quarter wave stack (LH... or HL...)
    # Try starting with L (closer to air) and H (closer to sub)
    # Standard AR: L H L H ... or H L H L ...
    # Let's try a few fixed patterns first as seeds.
    
    candidates = []
    
    # Pattern 1: L H L H ...
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        mat = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        candidates.append(mat)
        mat = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        candidates.append(mat)
        
    # Pattern 2: Graded index approximation
    # More L at the front, more H at the back?
    # e.g. L L H H L L H H ...
    for n_layers in [8, 12, 16, 20]:
        half = n_layers // 2
        mat = ['L'] * half + ['H'] * (n_layers - half)
        candidates.append(mat)
        mat = ['H'] * half + ['L'] * (n_layers - half)
        candidates.append(mat)
        
    # Pattern 3: Random sequences
    import random
    random.seed(123)
    for _ in range(20):
        n_layers = random.randint(4, 20)
        mat = [random.choice(['L', 'H']) for _ in range(n_layers)]
        candidates.append(mat)
        
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique_candidates.append(c)
            
    # Evaluate candidates
    for mat in unique_candidates:
        if time.time() > deadline - 20:
            break
            
        ths, score = optimize_thicknesses(mat)
        if score < best_score:
            best_score = score
            best_materials = list(mat)
            best_ths = list(ths)
            
    # Local Search on the best material sequence
    # Try swapping adjacent layers, adding/removing layers
    # This allows finding better material sequences than random init
    
    if best_materials is not None:
        current_mat = best_materials
        current_ths = best_ths
        current_score = best_score
        
        # Local search loop
        # We will try to improve the material sequence by small changes
        # and re-optimize thicknesses (with a smaller DE budget for speed)
        
        max_local_iters = 50
        local_iter = 0
        while local_iter < max_local_iters and time.time() < deadline - 10:
            local_iter += 1
            
            improved = False
            n_layers = len(current_mat)
            
            # 1. Try swapping adjacent layers
            for i in range(n_layers - 1):
                new_mat = list(current_mat)
                new_mat[i], new_mat[i+1] = new_mat[i+1], new_mat[i]
                # Skip if same
                if new_mat == current_mat:
                    continue
                
                ths, score = optimize_thicknesses(new_mat)
                if score < current_score - 1e-6:
                    current_mat = new_mat
                    current_ths = ths
                    current_score = score
                    best_score = score
                    best_materials = list(current_mat)
                    best_ths = list(current_ths)
                    improved = True
                    break # Restart search from new state
                    
            if improved:
                continue
                
            # 2. Try inserting a layer
            if len(current_mat) < max_layers:
                for pos in range(len(current_mat) + 1):
                    for mat_type in ['L', 'H']:
                        new_mat = list(current_mat)
                        new_mat.insert(pos, mat_type)
                        ths, score = optimize_thicknesses(new_mat)
                        if score < current_score - 1e-6:
                            current_mat = new_mat
                            current_ths = ths
                            current_score = score
                            best_score = score
                            best_materials = list(current_mat)
                            best_ths = list(current_ths)
                            improved = True
                            break
                    if improved:
                        break
                        
            if improved:
                continue
                
            # 3. Try removing a layer
            if len(current_mat) > 2:
                for pos in range(len(current_mat)):
                    new_mat = list(current_mat)
                    new_mat.pop(pos)
                    ths, score = optimize_thicknesses(new_mat)
                    if score < current_score - 1e-6:
                        current_mat = new_mat
                        current_ths = ths
                        current_score = score
                        best_score = score
                        best_materials = list(current_mat)
                        best_ths = list(current_ths)
                        improved = True
                        break
                        
            if not improved:
                break # Local minimum reached for this sequence type

    # Final Polish: Run DE with higher maxiter on the best sequence found
    if best_materials is not None and time.time() < deadline - 5:
        # Increase precision
        def final_objective(ths):
            total = np.sum(ths)
            if total > max_total:
                if total > 1e-6:
                    scale = max_total / total
                    ths = ths * scale
                else:
                    return 100.0
            return _mean_reflectance(best_materials, list(ths), problem)
            
        bounds = [(0.0, max_layer) for _ in range(len(best_materials))]
        try:
            res = differential_evolution(
                final_objective,
                bounds,
                maxiter=100,
                popsize=10,
                tol=1e-6,
                seed=999,
                polish=True
            )
            if res.fun < best_score:
                best_score = res.fun
                best_ths = list(res.x)
        except Exception:
            pass

    # Fallback
    if best_materials is None:
        return ["L"], [qw["L"]]
        
    # Ensure constraints are strictly met in the return value
    # DE might return slightly out of bounds due to floating point
    final_ths = []
    total = 0
    for t in best_ths:
        t = max(0.0, min(max_layer, t))
        final_ths.append(t)
        total += t
        
    if total > max_total:
        scale = max_total / total
        final_ths = [t * scale for t in final_ths]
        
    return list(best_materials), final_ths
