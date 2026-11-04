import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps_fft(h):
    """Compute linear overlap of h with shifted (1-h) for all shifts using FFT."""
    g = 1.0 - h
    # Zero-pad to avoid circular convolution effects
    size = 2 * N
    h_pad = np.zeros(size)
    g_pad = np.zeros(size)
    h_pad[:N] = h
    g_pad[:N] = g
    
    # Linear correlation via FFT
    H = np.fft.rfft(h_pad, n=size)
    G = np.fft.rfft(g_pad, n=size)
    c = np.fft.irfft(H * np.conj(G), n=size)
    
    # Extract the linear correlation part
    # For two sequences of length N, the full correlation has 2N-1 elements
    # The FFT gives us size elements, but we need the right portion
    # c[0:N] corresponds to shifts -(N-1) to 0
    # c[N:2N-1] corresponds to shifts 1 to N-1
    c1 = c[:N]      # negative shifts (shift -1 to -(N-1))
    c2 = c[N+1:]    # positive shifts (shift 1 to N-1), skip index N which is zero shift
    return np.concatenate([c1, c2])

def _compute_overlaps(h):
    """Compute linear overlap of h with shifted (1-h) for all shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # Exclude the zero shift (center)
    c1 = c[:N]      # negative shifts
    c2 = c[N+1:]    # positive shifts
    return np.concatenate([c1, c2])

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) > 1e-10:
        for _ in range(50):
            diff = target - np.sum(h)
            if abs(diff) < 1e-13:
                break
            if diff > 0:
                room = 1.0 - h
                room[h >= 1.0] = 0
                total_room = room.sum()
                if total_room < 1e-15:
                    break
                h += room * (diff / total_room)
                h = np.clip(h, 0, 1)
            else:
                avail = h
                avail[h <= 0.0] = 0
                total_avail = avail.sum()
                if total_avail < 1e-15:
                    break
                h -= avail * (-diff / total_avail)
                h = np.clip(h, 0, 1)
    # Final exact projection using bisection on a scaling parameter
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) > 1e-12:
        # Use a simple iterative approach with tighter tolerance
        for _ in range(100):
            diff = target - np.sum(h)
            if abs(diff) < 1e-14:
                break
            if diff > 0:
                # Add to elements that have room, proportional to room
                room = 1.0 - h
                room[h >= 1.0 - 1e-15] = 0
                total_room = room.sum()
                if total_room < 1e-15:
                    break
                h += room * (diff / total_room)
                h = np.clip(h, 0, 1)
            else:
                avail = h
                avail[h <= 1e-15] = 0
                total_avail = avail.sum()
                if total_avail < 1e-15:
                    break
                h -= avail * (-diff / total_avail)
                h = np.clip(h, 0, 1)
    return h

def _objective_lse(x, beta, lam_sum):
    """Objective: minimize max overlap using LogSumExp approximation."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_overlaps(h)
    
    max_ov = np.max(overlaps)
    if max_ov > 500:
        max_ov = 500
    # Numerically stable LSE
    shifted = beta * (overlaps - max_ov)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    lse = max_ov + np.log(np.sum(np.exp(shifted))) / beta
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _objective_epigraph(x, beta, lam_sum):
    """Epigraph formulation: minimize t subject to overlaps <= t, sum constraint.
    x = [h_0, ..., h_{N-1}, t] where t is the auxiliary variable."""
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    overlaps = _compute_overlaps(h)
    
    # Smooth max via LSE for the constraint violation
    max_ov = np.max(overlaps)
    if max_ov > 500:
        max_ov = 500
    shifted = beta * (overlaps - max_ov)
    shifted = np.clip(shifted, -500, 500)
    lse_overlaps = max_ov + np.log(np.sum(np.exp(shifted))) / beta
    
    # Penalty for exceeding t
    excess = lse_overlaps - t
    if excess > 0:
        constraint_penalty = excess ** 2 + 0.5 * excess
    else:
        constraint_penalty = 0.0
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    
    # Objective: minimize t + penalties
    return t + constraint_penalty + sum_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds = bounds_h + [(-1.0, 2.0)]  # Add bound for t
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # Stronger alternating
    h_alt2 = np.zeros(N)
    h_alt2[::2] = 0.7
    h_alt2[1::2] = 0.3
    candidates.append(h_alt2)
    
    # Sinusoidal modulations
    t_idx = np.arange(N)
    for freq in range(1, 25):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
            for phase in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t_idx/N + phase)
                candidates.append(trial.copy())
    
    # Cosine variants
    for freq in range(1, 25):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
            trial = 0.5 + amp * np.cos(2*np.pi*freq*t_idx/N)
            candidates.append(trial.copy())
    
    # Sawtooth
    h_saw = np.linspace(0.1, 0.9, N)
    candidates.append(h_saw.copy())
    
    # Triangular wave
    h_tri = 0.5 + 0.4 * np.abs(2 * (t_idx / N - 0.5) - 1)
    candidates.append(h_tri.copy())
    
    # Step functions with various splits and values
    for split in range(10, 90, 5):
        for v1 in [0.55, 0.6, 0.65, 0.7]:
            v2 = 1.0 - v1
            h_step = np.zeros(N)
            h_step[:split] = v1
            h_step[split:] = v2
            candidates.append(h_step.copy())
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 10]:
        for v1 in [0.55, 0.6, 0.65]:
            v2 = 1.0 - v1
            h_block = np.zeros(N)
            for i in range(0, N, block_size * 2):
                h_block[i:i+block_size] = v1
                h_block[i+block_size:i+2*block_size] = v2
            candidates.append(h_block.copy())
    
    # Random restarts
    for i in range(80):
        h_rand = rng.uniform(0.05, 0.95, size=N)
        candidates.append(h_rand.copy())
    
    # Evaluate all candidates first
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # LSE optimization with multiple betas - start with lower betas for smoothness
    beta_schedule = [3, 5, 8, 12, 18, 25, 40, 60, 90, 130, 180]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        # Start from best so far
        x0 = best_h.copy()
        
        def obj_func(x, beta=beta):
            return _objective_lse(x, beta, 500.0)
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Epigraph optimization for better max-overlap handling
    if time.time() < deadline - 30.0:
        # Initialize t to current max overlap
        t_init = best_score
        x0_epigraph = np.concatenate([best_h, [t_init]])
        
        def obj_epigraph(x):
            return _objective_epigraph(x, 100.0, 1000.0)
        
        try:
            result = minimize(
                obj_epigraph,
                x0_epigraph,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Random restarts with LSE optimization
    restart_count = 0
    for i in range(30):
        if time.time() > deadline - 35.0:
            break
        restart_count += 1
        
        x0 = rng.uniform(0.1, 0.9, size=N)
        x0 = _project_to_feasible(x0)
        
        beta = 10.0 + i * 4.0
        
        def obj_rand(x, beta=beta):
            return _objective_lse(x, beta, 500.0)
        
        try:
            result = minimize(
                obj_rand,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Final high-precision polish with very high beta
    if time.time() < deadline - 15.0:
        def obj_polish(x):
            return _objective_lse(x, 300.0, 5000.0)
        
        try:
            result = minimize(
                obj_polish,
                best_h,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Another epigraph polish
    if time.time() < deadline - 10.0:
        t_init = best_score
        x0_epigraph = np.concatenate([best_h, [t_init]])
        
        def obj_epigraph_polish(x):
            return _objective_epigraph(x, 200.0, 5000.0)
        
        try:
            result = minimize(
                obj_epigraph_polish,
                x0_epigraph,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Final local search: try small perturbations
    if time.time() < deadline - 3.0:
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            # Random small perturbation
            perturbation = rng.normal(0, 0.01, size=N)
            h_perturbed = best_h + perturbation
            h_perturbed = _project_to_feasible(h_perturbed)
            score_perturbed = _max_overlap(h_perturbed)
            if score_perturbed < best_score:
                best_score = score_perturbed
                best_h = h_perturbed
    
    return best_h
