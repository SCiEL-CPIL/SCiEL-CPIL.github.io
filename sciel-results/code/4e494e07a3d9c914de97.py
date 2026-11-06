"""Rewritten program: QWOT-based combinatorial design with simulated annealing and adaptive thicknesses."""
# EVOLVE-BLOCK-START
import numpy as np

def _reflectance_spectrum_fast(n_layers_arr, d_layers_arr, n0, ns, wavelengths):
    """
    Vectorized normal-incidence reflectance computation using characteristic matrices.
    n_layers_arr: array of refractive indices for each layer
    d_layers_arr: array of physical thicknesses in nm
    n0: ambient index
    ns: substrate index
    wavelengths: array of wavelengths in nm
    Returns: array of reflectance fractions.
    """
    n_layers = len(n_layers_arr)
    if n_layers == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return np.full(len(wavelengths), np.abs(r)**2)
    
    num_waves = len(wavelengths)
    R = np.zeros(num_waves)
    
    for iw in range(num_waves):
        lam = wavelengths[iw]
        # Accumulate 2x2 complex matrix
        m00 = 1.0 + 0j
        m01 = 0.0 + 0j
        m10 = 0.0 + 0j
        m11 = 1.0 + 0j
        
        for j in range(n_layers):
            nj = n_layers_arr[j]
            dj = d_layers_arr[j]
            delta = 2.0 * np.pi * nj * dj / lam
            c = np.cos(delta)
            s = np.sin(delta)
            # Mj = [[c, i*s/nj], [i*nj*s, c]]
            nm00 = m00 * c + m01 * 1j * s / nj
            nm01 = m00 * 1j * s / nj + m01 * c
            nm10 = m10 * c + m11 * 1j * nj * s
            nm11 = m10 * 1j * nj * s + m11 * c
            m00, m01, m10, m11 = nm00, nm01, nm10, nm11
        
        B = m00 + m01 / ns
        C = m10 + m11 / ns
        Y = C / B
        r = (n0 - Y) / (n0 + Y)
        R[iw] = np.abs(r)**2
    
    return R

def _evaluate_thick(materials_seq, thicknesses, problem):
    """
    Evaluate a design given a sequence of 'L'/'H' labels and a list of physical thicknesses.
    Returns mean reflectance in percent, or 1.0 if constraints violated.
    """
    band = problem["band_nm"]
    mat_dict = problem["materials"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    
    if len(materials_seq) > max_layers:
        return 1.0
    if len(materials_seq) == 0:
        return 1.0
    if any(d < 0 or d > max_layer_nm for d in thicknesses):
        return 1.0
    if sum(thicknesses) > max_total_nm:
        return 1.0
    
    n_arr = np.array([mat_dict[lbl] for lbl in materials_seq])
    d_arr = np.array(thicknesses, dtype=float)
    
    wavelengths = np.linspace(band[0], band[1], 401)
    R = _reflectance_spectrum_fast(n_arr, d_arr, problem["ambient_index"], 
                                    problem["substrate_index"], wavelengths)
    return np.mean(R) * 100.0

def _qpot_thicknesses(materials_seq, scale, problem):
    """Compute physical thicknesses from QWOT scale."""
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    thicknesses = []
    for label in materials_seq:
        n = mat_dict[label]
        d = scale * lambda_0 / (4.0 * n)
        thicknesses.append(d)
    return thicknesses

def _optimize_scale_for_seq(materials_seq, problem, n_samples=50):
    """
    For a given layer sequence, find the optimal global QWOT scale by minimizing
    mean reflectance over a grid of scale values.
    """
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    n_L = mat_dict["L"]
    n_H = mat_dict["H"]
    
    # Determine valid scale range based on constraints
    # For each layer: d = scale * lambda_0 / (4*n) <= max_layer_nm
    # => scale <= 4 * n * max_layer_nm / lambda_0
    # And sum(d) = scale * lambda_0 / 4 * sum(1/n_i) <= max_total_nm
    # => scale <= 4 * max_total_nm / (lambda_0 * sum(1/n_i))
    
    inv_n_sum = sum(1.0 / mat_dict[lbl] for lbl in materials_seq)
    scale_max_from_layer = min(4.0 * mat_dict[lbl] * max_layer_nm / lambda_0 for lbl in materials_seq)
    scale_max_from_total = 4.0 * max_total_nm / (lambda_0 * inv_n_sum)
    scale_max = min(scale_max_from_layer, scale_max_from_total)
    scale_max = max(scale_max, 0.01)
    
    # Search scales from 0.3 to min(scale_max, 2.5)
    scale_upper = min(scale_max, 2.5)
    scales = np.linspace(0.3, scale_upper, n_samples)
    
    best_scale = scales[0]
    best_score = 1.0
    for s in scales:
        thicknesses = _qpot_thicknesses(materials_seq, s, problem)
        score = _evaluate_thick(materials_seq, thicknesses, problem)
        if score < best_score:
            best_score = score
            best_scale = s
    
    # Local refinement around best
    if best_score < 1.0:
        for s in np.linspace(best_scale * 0.85, best_scale * 1.15, 20):
            if s < 0.1 or s > scale_max:
                continue
            thicknesses = _qpot_thicknesses(materials_seq, s, problem)
            score = _evaluate_thick(materials_seq, thicknesses, problem)
            if score < best_score:
                best_score = score
                best_scale = s
    
    return best_scale, best_score

def _optimize_individual_thicknesses(materials_seq, initial_thicknesses, problem, n_iter=300):
    """
    Optimize individual layer thicknesses for a fixed sequence using gradient-free
    coordinate descent with random restarts.
    """
    band = problem["band_nm"]
    mat_dict = problem["materials"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    n_layers = len(materials_seq)
    
    best_thick = list(initial_thicknesses)
    best_score = _evaluate_thick(materials_seq, best_thick, problem)
    
    # Coordinate descent
    for it in range(n_iter):
        # Pick a random layer to perturb
        idx = np.random.randint(n_layers)
        d_old = best_thick[idx]
        # Gaussian perturbation scaled by current thickness
        scale = max(5.0, 0.15 * d_old)
        d_new = d_old + np.random.randn() * scale
        d_new = max(0.0, min(max_layer_nm, d_new))
        
        new_thick = list(best_thick)
        new_thick[idx] = d_new
        
        # Check total constraint
        if sum(new_thick) > max_total_nm:
            # Scale down proportionally
            total = sum(new_thick)
            factor = max_total_nm / total
            new_thick = [d * factor for d in new_thick]
            new_thick[idx] = min(d_new * factor, max_layer_nm)
        
        score = _evaluate_thick(materials_seq, new_thick, problem)
        
        # Accept if better or with some probability
        if score < best_score:
            best_thick = new_thick
            best_score = score
        else:
            # Temperature-based acceptance
            T = 0.1 * (1.0 - it / n_iter)
            delta = score - best_score
            if delta < 0:
                pass
            elif T > 0.001:
                if np.random.rand() < np.exp(-delta / T):
                    best_thick = new_thick
                    best_score = score
    
    return best_thick, best_score

def design(problem):
    """
    Design an antireflection coating using QWOT-based simulated annealing
    with adaptive per-layer thickness optimization.
    """
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    n_L = mat_dict["L"]
    n_H = mat_dict["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Precompute wavelengths for quick evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Try several initial sequences to find a good starting point
    initial_seqs = [
        ['L'],
        ['L', 'H'],
        ['H', 'L'],
        ['L', 'H', 'L'],
        ['L', 'H', 'H', 'L'],
        ['H', 'L', 'H'],
        ['L', 'H', 'L', 'H'],
        ['L', 'H', 'H', 'L', 'H', 'L'],
    ]
    
    best_seq = None
    best_thick = None
    best_score = 1.0
    
    for seq in initial_seqs:
        if len(seq) > max_layers:
            continue
        scale, score = _optimize_scale_for_seq(seq, problem, n_samples=30)
        if score < best_score:
            best_seq = list(seq)
            best_score = score
            best_thick = _qpot_thicknesses(seq, scale, problem)
    
    if best_seq is None:
        best_seq = ['L']
        best_thick = [lambda_0 / (4.0 * n_L)]
        best_score = _evaluate_thick(best_seq, best_thick, problem)
    
    # Simulated annealing over sequences
    num_iter = 3000
    T_init = 0.3
    T_final = 0.001
    cooling = (T_final / T_init) ** (1.0 / num_iter)
    
    current_seq = list(best_seq)
    current_thick = list(best_thick)
    current_score = best_score
    
    for i in range(num_iter):
        T = T_init * (cooling ** i)
        
        # Generate a neighbor
        r = np.random.rand()
        if r < 0.35:
            # Change a random layer type
            new_seq = list(current_seq)
            idx = np.random.randint(len(new_seq))
            new_seq[idx] = 'H' if new_seq[idx] == 'L' else 'L'
            new_thick = list(current_thick)
        elif r < 0.55:
            # Add a layer
            if len(current_seq) < max_layers:
                new_seq = list(current_seq)
                pos = np.random.randint(len(new_seq) + 1)
                new_layer = 'L' if np.random.rand() < 0.5 else 'H'
                new_seq.insert(pos, new_layer)
                # Initialize new thickness to QWOT at center wavelength
                n_new = mat_dict[new_layer]
                d_new = lambda_0 / (4.0 * n_new)
                new_thick = list(current_thick)
                new_thick.insert(pos, d_new)
            else:
                new_seq = list(current_seq)
                new_thick = list(current_thick)
        elif r < 0.75:
            # Remove a layer
            if len(current_seq) > 1:
                new_seq = list(current_seq)
                idx = np.random.randint(len(new_seq))
                new_seq.pop(idx)
                new_thick = list(current_thick)
                new_thick.pop(idx)
            else:
                new_seq = list(current_seq)
                new_thick = list(current_thick)
        else:
            # Perturb one or more thicknesses
            new_seq = list(current_seq)
            new_thick = list(current_thick)
            n_perturb = np.random.randint(1, min(3, len(new_thick) + 1))
            for _ in range(n_perturb):
                idx = np.random.randint(len(new_thick))
                d_old = new_thick[idx]
                scale = max(5.0, 0.2 * d_old)
                d_new = d_old + np.random.randn() * scale
                d_new = max(0.0, min(max_layer_nm, d_new))
                new_thick[idx] = d_new
            # Check total constraint
            if sum(new_thick) > max_total_nm:
                total = sum(new_thick)
                factor = max_total_nm / total
                new_thick = [d * factor for d in new_thick]
        
        # Evaluate
        new_score = _evaluate_thick(new_seq, new_thick, problem)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score < current_score:
            current_seq = new_seq
            current_thick = new_thick
            current_score = new_score
            if current_score < best_score:
                best_seq = list(current_seq)
                best_thick = list(current_thick)
                best_score = current_score
        else:
            delta = new_score - current_score
            if T > 0.001:
                accept_prob = np.exp(-delta / T)
                if np.random.rand() < accept_prob:
                    current_seq = new_seq
                    current_thick = new_thick
                    current_score = new_score
    
    # Final refinement: optimize individual thicknesses for the best sequence
    for trial in range(5):
        # Start from current best or slightly perturbed
        if trial == 0:
            start_thick = list(best_thick)
        else:
            start_thick = list(best_thick)
            for idx in range(len(start_thick)):
                start_thick[idx] = max(0, min(max_layer_nm, 
                    start_thick[idx] * (1.0 + 0.1 * np.random.randn())))
            if sum(start_thick) > max_total_nm:
                factor = max_total_nm / sum(start_thick)
                start_thick = [d * factor for d in start_thick]
        
        refined_thick, refined_score = _optimize_individual_thicknesses(
            best_seq, start_thick, problem, n_iter=200)
        
        if refined_score < best_score:
            best_thick = refined_thick
            best_score = refined_score
    
    # Also try a few alternative sequences near the best
    for trial in range(20):
        trial_seq = list(best_seq)
        r = np.random.rand()
        if r < 0.4 and len(trial_seq) < max_layers:
            pos = np.random.randint(len(trial_seq) + 1)
            trial_seq.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
        elif r < 0.7 and len(trial_seq) > 1:
            idx = np.random.randint(len(trial_seq))
            trial_seq.pop(idx)
        else:
            if len(trial_seq) > 0:
                idx = np.random.randint(len(trial_seq))
                trial_seq[idx] = 'H' if trial_seq[idx] == 'L' else 'L'
        
        # Optimize scale for this sequence
        scale, score = _optimize_scale_for_seq(trial_seq, problem, n_samples=20)
        if score < best_score:
            best_seq = list(trial_seq)
            best_thick = _qpot_thicknesses(trial_seq, scale, problem)
            best_score = score
    
    # Final individual thickness optimization
    if len(best_seq) > 0:
        refined_thick, refined_score = _optimize_individual_thicknesses(
            best_seq, best_thick, problem, n_iter=300)
        if refined_score < best_score:
            best_thick = refined_thick
            best_score = refined_score
    
    # Ensure all constraints are met
    final_thick = []
    for d in best_thick:
        final_thick.append(max(0.0, min(max_layer_nm, d)))
    
    # Check total
    if sum(final_thick) > max_total_nm:
        factor = max_total_nm / sum(final_thick)
        final_thick = [d * factor for d in final_thick]
    
    # Verify final score
    final_score = _evaluate_thick(best_seq, final_thick, problem)
    if final_score > best_score:
        # Fallback to QWOT-based thicknesses
        scale, _ = _optimize_scale_for_seq(best_seq, problem, n_samples=30)
        final_thick = _qpot_thicknesses(best_seq, scale, problem)
    
    return best_seq, final_thick
# EVOLVE-BLOCK-END
