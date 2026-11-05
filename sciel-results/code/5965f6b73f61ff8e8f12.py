"""Design broadband antireflection coating using characteristic-matrix method."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _compute_reflectance_all(materials, thicknesses, wavelengths, n0, ns, nL, nH):
    """Vectorized reflectance computation for all wavelengths at once.
    
    materials: list of 'L'/'H'
    thicknesses: list of physical thicknesses in nm
    wavelengths: numpy array of wavelengths in nm
    Returns: numpy array of reflectance fractions.
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full(len(wavelengths), abs(r) ** 2)
    
    # Precompute layer indices
    layer_indices = np.array([nL if m == 'L' else nH for m in materials])
    
    # Build matrices for each wavelength
    # beta[i, j] = 2*pi * n_i * d_i / wl_j
    # Shape: (n_layers, n_wavelengths)
    beta = 2.0 * np.pi * np.outer(layer_indices, thicknesses) / wavelengths[np.newaxis, :]
    
    cos_b = np.cos(beta)  # (n_layers, n_wl)
    sin_b = np.sin(beta)  # (n_layers, n_wl)
    
    # For each wavelength, multiply all layer matrices
    # M_total = M_1 @ M_2 @ ... @ M_N
    # M_i = [[cos_i, sin_i/n_i], [n_i*sin_i, cos_i]]
    
    # Start with identity
    M11 = np.ones(len(wavelengths))
    M12 = np.zeros(len(wavelengths))
    M21 = np.zeros(len(wavelengths))
    M22 = np.ones(len(wavelengths))
    
    for i in range(n_layers):
        n_i = layer_indices[i]
        c = cos_b[i]
        s = sin_b[i]
        
        # M_new = M_old @ M_i
        # M_old = [[M11, M12], [M21, M22]]
        # M_i = [[c, s/n_i], [n_i*s, c]]
        
        new_M11 = M11 * c + M12 * n_i * s
        new_M12 = M11 * (s / n_i) + M12 * c
        new_M21 = M21 * c + M22 * n_i * s
        new_M22 = M21 * (s / n_i) + M22 * c
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Input admittance
    Y_in = (M21 + M22 * ns) / (M11 + M12 * ns)
    
    # Reflectance
    r = (n0 - Y_in) / (n0 + Y_in)
    R = np.abs(r) ** 2
    
    return np.real(R)


def _mean_reflectance_percent(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band0, band1 = problem["band_nm"]
    
    wavelengths = np.linspace(band0, band1, 401)
    R = _compute_reflectance_all(materials, thicknesses, wavelengths, n0, ns, nL, nH)
    return float(np.mean(R)) * 100.0


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    
    if sum(thicknesses) > max_total_nm:
        return False
    
    for m in materials:
        if m not in ('L', 'H'):
            return False
    
    return True


def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n0 = problem["ambient_index"]  # 1.0
    ns = problem["substrate_index"]  # 1.52
    nL = problem["materials"]["L"]   # 1.38
    nH = problem["materials"]["H"]   # 2.35
    band0, band1 = problem["band_nm"]  # [400, 800]
    max_layers = problem["max_layers"]  # 20
    max_total_nm = problem["max_total_nm"]  # 1000
    max_layer_nm = problem["max_layer_nm"]  # 400
    
    centre_wl = 0.5 * (band0 + band1)  # 600 nm
    wavelengths = np.linspace(band0, band1, 401)
    
    best_score = float('inf')
    best_materials = []
    best_thick = []
    
    # Strategy: Try many candidate structures and optimize the best ones.
    # Key insight: broadband AR needs a graded-index profile.
    # A symmetric stack with many layers approximating smooth index transition works well.
    
    def try_optimize(materials, thicknesses, time_limit):
        """Local optimization of thicknesses using coordinate descent."""
        nonlocal best_score, best_materials, best_thick
        
        if not _is_valid(materials, thicknesses, problem):
            return
        
        current_thick = np.array(thicknesses, dtype=float)
        current_score = _mean_reflectance_percent(materials, list(current_thick), problem)
        
        if current_score < best_score:
            best_score = current_score
            best_materials = materials[:]
            best_thick = list(current_thick)
        
        n_layers = len(current_thick)
        if n_layers == 0:
            return
        
        # Coordinate descent with adaptive step sizes
        step_sizes = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]
        
        for step in step_sizes:
            if time.time() > time_limit:
                break
            
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for i in range(n_layers):
                    if time.time() > time_limit:
                        break
                    
                    orig = current_thick[i]
                    # Try perturbations in both directions
                    for delta in [step, -step]:
                        new_val = orig + delta
                        if new_val < 0 or new_val > max_layer_nm:
                            continue
                        new_thick = current_thick.copy()
                        new_thick[i] = new_val
                        if np.sum(new_thick) > max_total_nm:
                            continue
                        
                        trial_score = _mean_reflectance_percent(materials, list(new_thick), problem)
                        if trial_score < current_score - 1e-12:
                            current_thick = new_thick
                            current_score = trial_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_materials = materials[:]
                                best_thick = list(current_thick)
                            break  # Move to next layer
    
    # Generate candidate structures
    # For broadband AR, we want effective index to go from ~1.0 to ~1.52
    # Symmetric stacks with odd number of layers, highest index in middle
    
    candidates = []
    
    # Single layer (baseline)
    d1 = centre_wl / (4.0 * nL)
    candidates.append((["L"], [d1]))
    
    # Two layers
    dL = centre_wl / (4.0 * nL)
    dH = centre_wl / (4.0 * nH)
    candidates.append((["L", "H"], [dL, dH]))
    candidates.append((["H", "L"], [dH, dL]))
    
    # For N layers, create symmetric patterns
    # Pattern: start with L (close to air), end with L or H (close to glass)
    # Middle layers alternate to create graded effect
    
    for n_layers in range(3, min(max_layers + 1, 21)):
        # Symmetric pattern: mirror around center
        # For odd n: L H L H ... H L (center is H if even count of pairs)
        # Let's create patterns that approximate graded index
        
        # Pattern 1: L...H...L (symmetric, L on both ends)
        mats1 = []
        for i in range(n_layers):
            if i < n_layers // 2:
                mats1.append('L' if i % 2 == 0 else 'H')
            else:
                mirror_i = n_layers - 1 - i
                mats1.append(mats1[mirror_i] if mirror_i < len(mats1) else 'L')
        
        # Simpler: generate all reasonable symmetric patterns
        # For n layers, the pattern is determined by the first ceil(n/2) layers
        # and mirrored
        
        half = (n_layers + 1) // 2
        
        # Try a few patterns for the first half
        # Pattern A: L H L H ... 
        # Pattern B: L L H H ... (graded)
        # Pattern C: H L H L ...
        
        # For graded index approximation:
        # We want effective index to increase from air side to glass side
        # Then decrease back (symmetric)
        
        # Pattern: alternate starting with L
        for start_char in ['L', 'H']:
            mats = []
            for i in range(half):
                mats.append(start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L'))
            # Mirror
            if n_layers % 2 == 0:
                full_mats = mats + mats[::-1]
            else:
                full_mats = mats + mats[-2::-1]
            
            # Check validity of pattern (no consecutive same material at boundaries)
            # Actually, consecutive same materials are fine, they just combine
            
            # Assign quarter-wave thicknesses
            thks = []
            for m in full_mats:
                if m == 'L':
                    thks.append(dL)
                else:
                    thks.append(dH)
            
            # Check total thickness constraint
            if sum(thks) <= max_total_nm:
                candidates.append((full_mats, thks))
        
        # Also try graded thickness patterns
        # For the first half, use varying optical thicknesses
        # to create a smoother index transition
        
        # Pattern with graded optical thicknesses
        # First half: L with increasing optical thickness, then H
        if n_layers >= 4:
            # Try: L L H H ... H H L L (with graded thicknesses)
            mats_graded = []
            thks_graded = []
            
            for i in range(half):
                if i < half // 2:
                    mats_graded.append('L')
                    # Optical thickness increases from 0.2 to 0.3 * lambda
                    frac = 0.2 + 0.1 * (2 * i / max(1, half - 1))
                    thks_graded.append(frac * centre_wl / nL)
                else:
                    mats_graded.append('H')
                    frac = 0.3 - 0.1 * (2 * (half - 1 - i) / max(1, half - 1))
                    thks_graded.append(frac * centre_wl / nH)
            
            if n_layers % 2 == 0:
                full_mats_g = mats_graded + mats_graded[::-1]
                full_thks_g = thks_graded + thks_graded[::-1]
            else:
                full_mats_g = mats_graded + mats_graded[-2::-1]
                full_thks_g = thks_graded + thks_graded[-2::-1]
            
            if sum(full_thks_g) <= max_total_nm and all(0 <= t <= max_layer_nm for t in full_thks_g):
                candidates.append((full_mats_g, full_thks_g))
    
    # Also add some well-known AR coating structures
    # 3-layer: H L H or L H L
    candidates.append((["H", "L", "H"], [dH, dL, dH]))
    candidates.append((["L", "H", "L"], [dL, dH, dL]))
    
    # Evaluate all candidates quickly
    for mats, thks in candidates:
        if not _is_valid(mats, thks, problem):
            continue
        score = _mean_reflectance_percent(mats, thks, problem)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thick = thks[:]
    
    # Now optimize the best candidates
    # Sort candidates by initial score and optimize the top few
    candidate_scores = []
    for mats, thks in candidates:
        if not _is_valid(mats, thks, problem):
            continue
        score = _mean_reflectance_percent(mats, thks, problem)
        candidate_scores.append((score, mats, thks))
    
    candidate_scores.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    n_to_optimize = min(10, len(candidate_scores))
    time_per_candidate = (deadline - time.time()) / max(1, n_to_optimize)
    
    for idx in range(n_to_optimize):
        if time.time() > deadline - 5:
            break
        
        _, mats, thks = candidate_scores[idx]
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 1:
            break
        
        # Allocate time based on rank
        if idx == 0:
            t_alloc = remaining_time * 0.4
        elif idx < 3:
            t_alloc = remaining_time * 0.15
        else:
            t_alloc = remaining_time * 0.05
        
        try_optimize(mats, thks, time.time() + t_alloc)
    
    # Try random restarts if time remains
    if time.time() < deadline - 10:
        import random
        rng = random.Random(42)
        
        n_restarts = 5
        time_per_restart = (deadline - 10 - time.time()) / n_restarts
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random number of layers
            n_layers = rng.randint(3, min(max_layers, 15))
            
            # Random material pattern
            mats = []
            for i in range(n_layers):
                # Bias toward alternating pattern
                if i == 0:
                    mats.append(rng.choice(['L', 'H']))
                else:
                    if rng.random() < 0.7:
                        mats.append('H' if mats[-1] == 'L' else 'L')
                    else:
                        mats.append(rng.choice(['L', 'H']))
            
            # Random thicknesses around quarter-wave
            thks = []
            for m in mats:
                base = centre_wl / (4.0 * (nL if m == 'L' else nH))
                t = base * rng.uniform(0.5, 1.5)
                t = min(max(t, 0.0), max_layer_nm)
                thks.append(t)
            
            # Check total thickness
            if sum(thks) > max_total_nm:
                # Scale down
                scale = max_total_nm / sum(thks)
                thks = [t * scale for t in thks]
            
            if _is_valid(mats, thks, problem):
                score = _mean_reflectance_percent(mats, thks, problem)
                if score < best_score:
                    best_score = score
                    best_materials = mats[:]
                    best_thick = thks[:]
                
                # Quick optimization
                remaining = deadline - 5 - time.time()
                if remaining > 2:
                    try_optimize(mats, thks, time.time() + min(time_per_restart, remaining * 0.5))
    
    # Final validation
    if not _is_valid(best_materials, best_thick, problem):
        # Fallback
        d1 = centre_wl / (4.0 * nL)
        best_materials = ["L"]
        best_thick = [d1]
    
    return best_materials, best_thick

# EVOLVE-BLOCK-END
