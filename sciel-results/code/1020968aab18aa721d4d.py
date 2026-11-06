"""Improved QWOT-based antireflection coating design with vectorized evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _reflectance_spectrum_vec(n_indices, thicknesses, n0, ns, wavelengths):
    """
    Vectorized normal-incidence reflectance using characteristic matrix method.
    n_indices: array of refractive indices (length N)
    thicknesses: array of physical thicknesses in nm (length N)
    n0, ns: ambient and substrate indices
    wavelengths: array of wavelengths in nm (length W)
    Returns: array of reflectance fractions (length W)
    """
    N = len(n_indices)
    W = len(wavelengths)
    
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full(W, r**2)
    
    # Precompute 2*pi/lambda for all wavelengths
    inv_lambda = 2.0 * np.pi / wavelengths  # shape (W,)
    
    # For each layer, precompute cos and sin of delta for all wavelengths
    # delta_j = n_j * d_j * 2*pi/lambda
    n_arr = np.asarray(n_indices, dtype=np.float64)  # (N,)
    d_arr = np.asarray(thicknesses, dtype=np.float64)  # (N,)
    
    # delta shape: (N, W)
    delta = np.outer(n_arr * d_arr, inv_lambda)  # (N, W)
    cos_d = np.cos(delta)  # (N, W)
    sin_d = np.sin(delta)  # (N, W)
    
    # Initialize M matrices: M00, M01, M10, M11 each shape (W,)
    M00 = np.ones(W, dtype=np.complex128)
    M01 = np.zeros(W, dtype=np.complex128)
    M10 = np.zeros(W, dtype=np.complex128)
    M11 = np.ones(W, dtype=np.complex128)
    
    for j in range(N):
        n_j = n_arr[j]
        c = cos_d[j]  # (W,)
        s = sin_d[j]  # (W,)
        
        # Layer matrix: [[c, i*s/n_j], [i*n_j*s, c]]
        # M_new = M @ M_j
        i_s_over_n = 1j * s / n_j  # (W,)
        i_n_s = 1j * n_j * s  # (W,)
        
        new_M00 = M00 * c + M01 * i_s_over_n
        new_M01 = M00 * i_s_over_n + M01 * c
        new_M10 = M10 * c + M11 * i_n_s
        new_M11 = M10 * i_n_s + M11 * c
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    B = M00 + M01 / ns
    C = M10 + M11 / ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = r.real**2 + r.imag**2
    return R

def _eval_score(seq, scale, chirp, problem):
    """
    Fast evaluation: given sequence of 'L'/'H' labels, scale, and chirp factor,
    compute mean reflectance in percent.
    """
    if len(seq) == 0:
        # Bare substrate
        r = (problem["ambient_index"] - problem["substrate_index"]) / (problem["ambient_index"] + problem["substrate_index"])
        return r**2 * 100.0
    
    mat_dict = problem["materials"]
    lambda_0 = problem["_lambda0"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    wavelengths = problem["_wavelengths"]
    N = len(seq)
    
    n_indices = np.array([mat_dict[label] for label in seq], dtype=np.float64)
    
    # Calculate thicknesses with chirp
    # d_j = (lambda_0 / 4 n_j) * scale * (1 + chirp * pos_factor)
    # pos_factor ranges from -1 to 1
    if N > 1:
        pos_factors = 2.0 * np.arange(N) / (N - 1) - 1.0
    else:
        pos_factors = np.zeros(1)
        
    factors = 1.0 + chirp * pos_factors
    # Ensure factors are positive to avoid negative thicknesses
    if np.any(factors <= 0):
        return 100.0
        
    thicknesses = scale * lambda_0 / (4.0 * n_indices) * factors
    
    # Constraint checks
    if np.any(thicknesses < 0) or np.any(thicknesses > problem["max_layer_nm"]):
        return 100.0
    if np.sum(thicknesses) > problem["max_total_nm"]:
        return 100.0
    
    R = _reflectance_spectrum_vec(n_indices, thicknesses, n0, ns, wavelengths)
    return float(np.mean(R) * 100.0)

def _optimize_scale_chirp(seq, problem):
    """
    Optimization of scale and chirp for a fixed sequence.
    Returns (best_scale, best_chirp, best_score).
    """
    if len(seq) == 0:
        return 1.0, 0.0, 100.0
    
    mat_dict = problem["materials"]
    lambda_0 = problem["_lambda0"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    N = len(seq)
    
    # Determine valid scale range based on max possible thickness
    # Max thickness occurs when chirp factor is max (1+|chirp|)
    # Let's assume chirp in [-1, 1]. Max factor is 2.
    # d_j <= max_layer_nm => scale * lambda_0 / (4 n_j) * 2 <= max_layer_nm
    s_max = 10.0
    for label in seq:
        n = mat_dict[label]
        s_max = min(s_max, max_layer_nm * 4.0 * n / (2.0 * lambda_0))
        
    # Total thickness constraint
    # sum(d_j) = scale * lambda_0 / 4 * sum(1/n_j * factors_j) <= max_total_nm
    # Worst case for sum is when factors are all 2 (if chirp allows)
    # Actually factors depend on position.
    # Let's just use a conservative bound.
    sum_inv_n = sum(1.0 / mat_dict[label] for label in seq)
    # Max sum factor is roughly 2 * N if all positive? No, factors are 1 + chirp*pos.
    # If chirp=1, factors go 1->2. Sum factors = N * 1.5.
    # If chirp=-1, factors go 1->0. Sum factors = N * 0.5.
    # So max sum factor is 1.5 * N.
    max_sum_factor = 1.5 * N
    if sum_inv_n > 0:
        s_max = min(s_max, max_total_nm * 4.0 / (lambda_0 * sum_inv_n * max_sum_factor))
        
    s_min = 0.05
    s_max = max(s_max, s_min)
    
    best_s = 1.0
    best_c = 0.0
    best_score = 100.0
    
    # Grid search over chirp
    chirp_values = np.linspace(-1.0, 1.0, 21)
    
    for c_val in chirp_values:
        # Golden section search for scale
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = s_min, s_max
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = _eval_score(seq, c, c_val, problem)
        fd = _eval_score(seq, d, c_val, problem)
        
        for _ in range(30):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - phi * (b - a)
                fc = _eval_score(seq, c, c_val, problem)
            else:
                a = c
                c = d
                fc = fd
                d = a + phi * (b - a)
                fd = _eval_score(seq, d, c_val, problem)
        
        best_s_local = (a + b) / 2.0
        best_score_local = _eval_score(seq, best_s_local, c_val, problem)
        
        if best_score_local < best_score:
            best_score = best_score_local
            best_s = best_s_local
            best_c = c_val
            
    # Refine around best chirp
    if best_score < 100.0:
        c_refine_grid = np.linspace(best_c - 0.2, best_c + 0.2, 11)
        for c_val in c_refine_grid:
            # Quick golden section
            phi = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = s_min, s_max
            c = b - phi * (b - a)
            d = a + phi * (b - a)
            fc = _eval_score(seq, c, c_val, problem)
            fd = _eval_score(seq, d, c_val, problem)
            
            for _ in range(20):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - phi * (b - a)
                    fc = _eval_score(seq, c, c_val, problem)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + phi * (b - a)
                    fd = _eval_score(seq, d, c_val, problem)
                    
            best_s_local = (a + b) / 2.0
            best_score_local = _eval_score(seq, best_s_local, c_val, problem)
            
            if best_score_local < best_score:
                best_score = best_score_local
                best_s = best_s_local
                best_c = c_val
                
    return best_s, best_c, best_score

def design(problem):
    """
    Design an antireflection coating using QWOT-based combinatorial search.
    """
    start_time = time.time()
    time_budget = 55.0
    
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    max_layers = problem["max_layers"]
    
    # Precompute
    problem["_lambda0"] = lambda_0
    problem["_wavelengths"] = np.linspace(band[0], band[1], 401)
    
    np.random.seed(12345)
    
    # Build initial candidate sequences: alternating patterns with various lengths
    # plus some structured patterns known to be effective
    initial_seqs = []
    
    # Alternating patterns starting with L or H, lengths 1-20
    for length in range(1, max_layers + 1):
        initial_seqs.append(['L' if i % 2 == 0 else 'H' for i in range(length)])
        initial_seqs.append(['H' if i % 2 == 0 else 'L' for i in range(length)])
    
    # Some structured patterns: blocks of same type
    for length in range(3, max_layers + 1, 2):
        # LHHLH pattern
        pat = []
        for i in range(length):
            if i % 4 in [0, 1]:
                pat.append('L')
            else:
                pat.append('H')
        initial_seqs.append(pat)
        
        # HLLHH pattern
        pat = []
        for i in range(length):
            if i % 4 in [0, 1]:
                pat.append('H')
            else:
                pat.append('L')
        initial_seqs.append(pat)
    
    # Evaluate all initial sequences
    best_seq = None
    best_scale = 1.0
    best_chirp = 0.0
    best_score = 100.0
    
    for seq in initial_seqs:
        if len(seq) > max_layers:
            continue
        s, c, score = _optimize_scale_chirp(seq, problem)
        if score < best_score:
            best_score = score
            best_seq = list(seq)
            best_scale = s
            best_chirp = c
    
    if best_seq is None:
        best_seq = ['L']
        best_scale, best_chirp, best_score = _optimize_scale_chirp(best_seq, problem)
    
    # Local search / simulated annealing on the discrete sequence
    # with periodic scale/chirp re-optimization
    T_init = 0.5
    T_final = 0.001
    total_iters = 30000
    cooling = (T_final / T_init) ** (1.0 / total_iters)
    
    current_seq = list(best_seq)
    current_scale = best_scale
    current_chirp = best_chirp
    current_score = best_score
    
    reopt_interval = 50  # re-optimize scale/chirp every N iterations
    
    for it in range(total_iters):
        if time.time() - start_time > time_budget - 5:
            break
        
        T = T_init * (cooling ** it)
        
        # Generate neighbor
        r = np.random.rand()
        new_seq = list(current_seq)
        
        if r < 0.30:
            # Flip a random layer
            if len(new_seq) > 0:
                idx = np.random.randint(len(new_seq))
                new_seq[idx] = 'H' if new_seq[idx] == 'L' else 'L'
        elif r < 0.45:
            # Insert a layer
            if len(new_seq) < max_layers:
                pos = np.random.randint(len(new_seq) + 1)
                new_seq.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
        elif r < 0.60:
            # Remove a layer
            if len(new_seq) > 1:
                idx = np.random.randint(len(new_seq))
                new_seq.pop(idx)
        elif r < 0.75:
            # Swap two adjacent layers
            if len(new_seq) >= 2:
                idx = np.random.randint(len(new_seq) - 1)
                new_seq[idx], new_seq[idx+1] = new_seq[idx+1], new_seq[idx]
        elif r < 0.85:
            # Replace a random contiguous block with random layers
            if len(new_seq) >= 2:
                block_len = np.random.randint(2, min(4, len(new_seq) + 1))
                start = np.random.randint(0, len(new_seq) - block_len + 1)
                for k in range(block_len):
                    new_seq[start + k] = 'L' if np.random.rand() < 0.5 else 'H'
        else:
            # Move a layer to a different position
            if len(new_seq) >= 3:
                idx = np.random.randint(len(new_seq))
                layer = new_seq.pop(idx)
                new_idx = np.random.randint(len(new_seq) + 1)
                new_seq.insert(new_idx, layer)
        
        # Evaluate with current scale/chirp first (fast check)
        new_score_fast = _eval_score(new_seq, current_scale, current_chirp, problem)
        
        # If significantly worse, skip detailed evaluation
        if new_score_fast > current_score + 5.0:
            continue
        
        # Re-optimize scale/chirp periodically or when sequence changes
        if it % reopt_interval == 0 or (new_score_fast < current_score):
            new_scale, new_chirp, new_score = _optimize_scale_chirp(new_seq, problem)
        else:
            new_scale = current_scale
            new_chirp = current_chirp
            new_score = new_score_fast
        
        # Accept/reject
        if new_score < current_score:
            current_seq = new_seq
            current_scale = new_scale
            current_chirp = new_chirp
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_seq = list(current_seq)
                best_scale = current_scale
                best_chirp = current_chirp
        else:
            delta = new_score - current_score
            if delta > 0 and T > 0.001:
                accept_prob = np.exp(-delta / T)
                if np.random.rand() < accept_prob:
                    current_seq = new_seq
                    current_scale = new_scale
                    current_chirp = new_chirp
                    current_score = new_score
    
    # Final refinement
    best_scale, best_chirp, best_score = _optimize_scale_chirp(best_seq, problem)
    
    # Local polish: try all single-layer modifications
    if time.time() - start_time < time_budget - 3:
        for idx in range(len(best_seq)):
            trial = list(best_seq)
            trial[idx] = 'H' if trial[idx] == 'L' else 'L'
            s, c, sc = _optimize_scale_chirp(trial, problem)
            if sc < best_score:
                best_score = sc
                best_seq = trial
                best_scale = s
                best_chirp = c
        
        # Try insertions
        for pos in range(len(best_seq) + 1):
            for label in ['L', 'H']:
                trial = list(best_seq)
                trial.insert(pos, label)
                if len(trial) <= max_layers:
                    s, c, sc = _optimize_scale_chirp(trial, problem)
                    if sc < best_score:
                        best_score = sc
                        best_seq = trial
                        best_scale = s
                        best_chirp = c
        
        # Try deletions
        for idx in range(len(best_seq)):
            if len(best_seq) > 1:
                trial = list(best_seq)
                trial.pop(idx)
                s, c, sc = _optimize_scale_chirp(trial, problem)
                if sc < best_score:
                    best_score = sc
                    best_seq = trial
                    best_scale = s
                    best_chirp = c
    
    # Convert to output
    materials_labels = list(best_seq)
    thicknesses = []
    N = len(best_seq)
    if N > 1:
        pos_factors = 2.0 * np.arange(N) / (N - 1) - 1.0
    else:
        pos_factors = np.zeros(1)
        
    factors = 1.0 + best_chirp * pos_factors
    
    for i, label in enumerate(best_seq):
        n = mat_dict[label]
        d = best_scale * lambda_0 / (4.0 * n) * factors[i]
        thicknesses.append(float(d))
    
    return materials_labels, thicknesses
# EVOLVE-BLOCK-END
