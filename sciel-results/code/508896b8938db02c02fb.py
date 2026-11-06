"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrix method."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    total_R = 0.0
    for wl in wls:
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for i in range(len(materials)):
            n_i = n_mat[i]
            phase = 2.0 * np.pi * n_i * d[i] / wl
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            B = np.array([[cos_p, 1j * sin_p / n_i],
                          [1j * n_i * sin_p, cos_p]], dtype=complex)
            M = M @ B
        Y = M[0, 1] / M[1, 1]
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        total_R += R * 100.0
    return total_R / len(wls)

def _evaluate(mats, thick, problem):
    """Evaluate a design, returning mean reflectance or inf if invalid."""
    if not mats or not thick or len(mats) != len(thick):
        return float('inf')
    if len(mats) > problem["max_layers"]:
        return float('inf')
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-9 for t in thick):
        return float('inf')
    if sum(thick) > problem["max_total_nm"] + 1e-9:
        return float('inf')
    return _mean_reflectance(mats, [max(0, t) for t in thick], problem)

def _optimize_sequence(mats, problem, t0, deadline):
    """Optimize thicknesses for a fixed material sequence using L-BFGS-B."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    n_layers = len(mats)
    if n_layers == 0:
        return None, float('inf')
    
    # Initial guess: quarter-wave at center wavelength
    x0 = np.array([centre / (4.0 * (n_L if m == 'L' else n_H)) for m in mats])
    
    # Bounds: each layer in [0, max_layer_nm], sum <= max_total_nm
    # We'll handle the sum constraint via penalty or by scaling
    
    def obj(x):
        if time.time() > deadline:
            return 1000.0
        if np.any(x < 0) or np.any(x > max_layer_nm):
            return 1000.0
        if np.sum(x) > max_total_nm:
            # Penalize sum violation
            penalty = (np.sum(x) - max_total_nm) * 10
            return 1000.0 + penalty
        return _mean_reflectance(mats, list(x), problem)
    
    try:
        from scipy.optimize import minimize
        res = minimize(obj, x0, method='L-BFGS-B',
                      bounds=[(0, max_layer_nm)] * n_layers,
                      options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.fun < 100.0:  # valid solution
            x_opt = np.clip(res.x, 0, max_layer_nm)
            if np.sum(x_opt) > max_total_nm:
                # Scale down to fit
                scale = max_total_nm / np.sum(x_opt)
                x_opt = x_opt * scale
            score = _mean_reflectance(mats, list(x_opt), problem)
            return list(x_opt), score
    except Exception:
        pass
    
    # Fallback: try Nelder-Mead
    try:
        from scipy.optimize import minimize
        def obj2(x):
            if time.time() > deadline:
                return 1000.0
            if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                return 1000.0
            return _mean_reflectance(mats, list(x), problem)
        
        res = minimize(obj2, x0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-7})
        
        if res.fun < 100.0:
            x_opt = np.clip(res.x, 0, max_layer_nm)
            if np.sum(x_opt) > max_total_nm:
                scale = max_total_nm / np.sum(x_opt)
                x_opt = x_opt * scale
            score = _mean_reflectance(mats, list(x_opt), problem)
            return list(x_opt), score
    except Exception:
        pass
    
    return None, float('inf')

def _generate_candidate_sequences(problem, max_layers):
    """Generate a diverse set of candidate material sequences to optimize."""
    sequences = []
    
    # Quarter-wave alternating stacks starting with L (air side)
    for n in range(1, max_layers + 1):
        mats = []
        for i in range(n):
            if i % 2 == 0:
                mats.append('L')
            else:
                mats.append('H')
        sequences.append(mats)
    
    # Quarter-wave alternating stacks starting with H (air side)
    for n in range(1, max_layers + 1):
        mats = []
        for i in range(n):
            if i % 2 == 0:
                mats.append('H')
            else:
                mats.append('L')
        sequences.append(mats)
    
    # Some structured patterns: pairs of same material
    for n_pairs in range(1, max_layers // 2 + 1):
        # LH LH ... pattern
        mats = []
        for _ in range(n_pairs):
            mats.extend(['L', 'H'])
        sequences.append(mats)
        
        # HL HL ... pattern
        mats = []
        for _ in range(n_pairs):
            mats.extend(['H', 'L'])
        sequences.append(mats)
    
    # Triplet patterns
    for n_trip in range(1, max_layers // 3 + 1):
        # LHL LHL ...
        mats = []
        for _ in range(n_trip):
            mats.extend(['L', 'H', 'L'])
        sequences.append(mats)
        
        # H L H H L H ...
        mats = []
        for _ in range(n_trip):
            mats.extend(['H', 'L', 'H'])
        sequences.append(mats)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sequences = []
    for seq in sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_sequences.append(seq)
    
    return unique_sequences

def design(problem):
    """Multi-layer AR coating design with L-BFGS-B optimization over candidate sequences."""
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Fallback: single quarter-wave layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance(best_mats, best_thick, problem)
    
    # Also try single H layer
    thick_h = [centre / (4.0 * n_H)]
    score_h = _mean_reflectance(["H"], thick_h, problem)
    if score_h < best_score:
        best_score = score_h
        best_mats = ["H"]
        best_thick = thick_h
    
    # Generate candidate sequences
    candidates = _generate_candidate_sequences(problem, max_layers)
    
    # Sort candidates: prefer shorter sequences first (faster to evaluate), 
    # then by a heuristic score
    def seq_heuristic(mats):
        # Prefer sequences that start with L (lower index on air side for AR)
        score = 0.0
        if mats[0] == 'H':
            score += 1.0
        # Prefer even number of layers for symmetric designs
        if len(mats) % 2 != 0:
            score += 0.1
        return (len(mats), score)
    
    candidates.sort(key=seq_heuristic)
    
    # Optimize each candidate sequence
    for mats in candidates:
        if time.time() > deadline - 15:
            break
        
        thick_opt, score = _optimize_sequence(mats, problem, t0, deadline)
        
        if thick_opt is not None and score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thick_opt[:]
    
    # Final refinement on the best design with tighter tolerances
    if time.time() < deadline - 5 and len(best_mats) >= 2:
        try:
            from scipy.optimize import minimize
            n_layers = len(best_mats)
            x0 = np.array(best_thick, dtype=float)
            
            def obj(x):
                if time.time() > deadline:
                    return 1000.0
                if np.any(x < 0) or np.any(x > max_layer_nm):
                    return 1000.0
                if np.sum(x) > max_total_nm:
                    return 1000.0 + (np.sum(x) - max_total_nm) * 10
                return _mean_reflectance(best_mats, list(x), problem)
            
            res = minimize(obj, x0, method='L-BFGS-B',
                          bounds=[(0, max_layer_nm)] * n_layers,
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_score and res.fun < 100.0:
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm + 1e-9:
                    best_score = float(res.fun)
                    best_thick = list(x_opt)
        except Exception:
            pass
    
    return best_mats, best_thick
# EVOLVE-BLOCK-END
