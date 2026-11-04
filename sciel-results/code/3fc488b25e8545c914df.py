"""Combined approach: rational generating function parameterization with efficient correlate-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute linear overlap of h with shifted (1-h) for all shifts using correlate."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # For circular-like shifts, we take the relevant portion
    # c has length 2N-1; index N-1 is zero shift
    # We want shifts k=1..N-1 and k=-1..-(N-1)
    c1 = c[:N-1]      # negative shifts (k = -(N-1) to -1)
    c2 = c[N:]        # positive shifts (k = 0 to N-1), but we exclude k=0
    c2 = c2[1:]       # exclude zero shift
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
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
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

def _rational_to_h(coeffs, order=4):
    """
    Generate h from rational generating function coefficients.
    h[n] = sum_{j=0}^{order-1} coeffs[j] * cos(2*pi*freq_j*n/N + phase_j) + 0.5
    
    We use a parameterization where coeffs encode frequency, amplitude, and phase
    for each harmonic component. This ensures smooth structure.
    """
    n_harmonics = len(coeffs) // 3
    if n_harmonics == 0:
        return np.full(N, 0.5)
    
    t = np.arange(N)
    h = np.full(N, 0.5)
    
    for i in range(n_harmonics):
        freq = coeffs[3*i]
        amp = coeffs[3*i + 1]
        phase = coeffs[3*i + 2]
        h += amp * np.sin(2*np.pi*freq*t/N + phase)
    
    return h

def _h_to_rational_coeffs(h, n_harmonics=5):
    """Extract dominant harmonic components from h."""
    h_centered = h - np.mean(h)
    fft_h = np.fft.rfft(h_centered)
    n_freqs = len(fft_h)
    
    coeffs = []
    for i in range(1, n_freqs):
        if i > n_harmonics:
            break
        freq = i
        amp = 2.0 * np.abs(fft_h[i]) / N
        phase = np.angle(fft_h[i])
        coeffs.extend([freq, amp, phase])
    
    # Pad with zeros if needed
    while len(coeffs) < 3 * n_harmonics:
        coeffs.extend([0, 0, 0])
    
    return np.array(coeffs)

def _objective(x, beta, lam_sum):
    """Objective: minimize max overlap using LogSumExp approximation."""
    coeffs = x
    h = _rational_to_h(coeffs)
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps(h)
    
    max_ov = np.max(overlaps)
    if max_ov > 500:
        max_ov = 500
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Initialize with diverse candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # Sinusoidal modulations
    t_idx = np.arange(N)
    for freq in range(1, 15):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t_idx/N + phase)
                candidates.append(trial.copy())
    
    # Multi-harmonic
    for n_harm in [2, 3, 4]:
        for freqs in [(1, 3), (2, 5), (1, 4, 7), (3, 5, 8)]:
            for amps in [(0.2, 0.1), (0.15, 0.1), (0.2, 0.1, 0.05), (0.1, 0.1, 0.05)]:
                trial = np.full(N, 0.5)
                for f, a in zip(freqs, amps):
                    trial += a * np.sin(2*np.pi*f*t_idx/N)
                candidates.append(trial.copy())
    
    # Evaluate all candidates
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Optimize using rational parameterization
    n_harmonics = 5
    n_params = 3 * n_harmonics
    
    # Initialize from best candidate
    coeffs0 = _h_to_rational_coeffs(best_h, n_harmonics)
    
    # LSE optimization with multiple betas
    beta_schedule = [5, 10, 20, 40, 80, 150]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        x0 = coeffs0.copy()
        
        def obj_func(x):
            return _objective(x, beta, 500.0)
        
        # Bounds: freq in [0.5, 10], amp in [0, 0.45], phase in [0, 2*pi]
        bounds = []
        for i in range(n_harmonics):
            bounds.append((0.5, 10.0))   # freq
            bounds.append((0.0, 0.45))   # amp
            bounds.append((0.0, 2*np.pi)) # phase
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            h_new = _rational_to_h(result.x)
            h_new = _project_to_feasible(h_new)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                coeffs0 = _h_to_rational_coeffs(best_h, n_harmonics)
        except Exception:
            pass
    
    # Random restarts with rational optimization
    for i in range(15):
        if time.time() > deadline - 10.0:
            break
        
        # Random initial coefficients
        coeffs_rand = []
        for j in range(n_harmonics):
            freq = rng.uniform(0.5, 8.0)
            amp = rng.uniform(0.0, 0.4)
            phase = rng.uniform(0, 2*np.pi)
            coeffs_rand.extend([freq, amp, phase])
        coeffs_rand = np.array(coeffs_rand)
        
        beta = 20.0 + i * 10.0
        
        def obj_rand(x):
            return _objective(x, beta, 500.0)
        
        bounds = []
        for i in range(n_harmonics):
            bounds.append((0.5, 10.0))
            bounds.append((0.0, 0.45))
            bounds.append((0.0, 2*np.pi))
        
        try:
            result = minimize(
                obj_rand,
                coeffs_rand,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            h_new = _rational_to_h(result.x)
            h_new = _project_to_feasible(h_new)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Final verification and return
    h_final = _project_to_feasible(best_h)
    
    return h_final
