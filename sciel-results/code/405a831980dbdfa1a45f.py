"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time

N = 100

# EVOLVE-BLOCK-START
def _max_overlap_fast(h):
    """
    Compute maximum overlap of h with shifted (1-h) using FFT for speed.
    overlap_k = sum_i h[i] * (1 - h[(i-k) % N]) = N/2 - autocorr[k]
    where autocorr[k] = sum_i h[i] * h[(i-k) % N]
    """
    N_local = len(h)
    # Compute circular autocorrelation via FFT
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.real(np.fft.ifft(power))
    # autocorr[k] = sum_i h[i] * h[(i-k) % N]
    # overlap at shift k = N/2 - autocorr[k]
    overlaps = N_local / 2.0 - autocorr
    return np.max(overlaps)


def _autocorr_min(h):
    """Return the minimum autocorrelation value (we want to maximize this)."""
    N_local = len(h)
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.real(np.fft.ifft(power))
    return np.min(autocorr)


def _objective(h):
    """Objective to minimize: max overlap = N/2 - min_autocorr."""
    return N / 2.0 - _autocorr_min(h)


def _project_to_feasible(h):
    """Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2."""
    h = h.copy()
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    # Adjust sum to exactly N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    if abs(diff) < 1e-12:
        return h
    # Distribute the difference
    if diff > 0:
        # Need to add; distribute among elements with room
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room < diff:
            # Can't reach target, set as many to 1 as possible
            h = np.ones(N)
            h[N//2:] = 0.0
            return h
        h += diff * room / total_room
    else:
        # Need to subtract; distribute among elements with value > 0
        total_val = np.sum(h)
        if total_val < -diff:
            h = np.zeros(N)
            h[:N//2] = 1.0
            return h
        h += diff * h / total_val
    h = np.clip(h, 0.0, 1.0)
    # Final correction
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        # Add/subtract evenly
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h


def _sine_candidate(freq=1, phase=0.0, amp=0.5):
    """Generate a sinusoidal candidate."""
    x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
    h = 0.5 + amp * np.sin(x + phase)
    return _project_to_feasible(h)


def _two_sine_candidate(freq1=1, freq2=3, phase1=0.0, phase2=0.0, amp=0.3):
    """Generate a two-sine candidate."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = 0.5 + amp * np.sin(freq1 * x + phase1) + amp * np.sin(freq2 * x + phase2)
    return _project_to_feasible(h)


def _modular_candidate(mod, offset=0):
    """
    Generate a candidate based on modular arithmetic.
    h[i] = 1 if i mod mod < mod//2 else 0, with offset.
    """
    h = np.zeros(N)
    half = mod // 2
    for i in range(N):
        if ((i + offset) % mod) < half:
            h[i] = 1.0
    return _project_to_feasible(h)


def _random_candidate():
    """Generate a random candidate with sum exactly N/2."""
    h = np.random.uniform(0, 1, N)
    return _project_to_feasible(h)


def _binary_candidate():
    """Binary candidate: half 0s and half 1s, randomly arranged."""
    h = np.zeros(N)
    h[:N // 2] = 1.0
    idx = np.random.permutation(N)
    h = h[idx]
    return h


def _coordinate_descent(h, max_iters=5000, step=0.1):
    """Coordinate descent: adjust one element at a time."""
    best = h.copy()
    best_obj = _objective(best)
    N_local = len(best)
    
    for it in range(max_iters):
        if time.time() > 145.0:
            break
        i = np.random.randint(N_local)
        # Try increasing and decreasing h[i]
        for delta in [step, -step, step/2, -step/2]:
            new_h = best.copy()
            new_h[i] = np.clip(new_h[i] + delta, 0, 1)
            # Adjust another element to maintain sum
            diff = np.sum(new_h) - N / 2.0
            if abs(diff) < 1e-10:
                pass
            elif diff > 0:
                # Need to decrease some other element
                j = np.random.randint(N_local)
                if j != i:
                    new_h[j] = np.clip(new_h[j] - diff, 0, 1)
                    # Final check
                    if abs(np.sum(new_h) - N / 2.0) > 1e-8:
                        continue
            else:
                # Need to increase some other element
                j = np.random.randint(N_local)
                if j != i:
                    new_h[j] = np.clip(new_h[j] - diff, 0, 1)
                    if abs(np.sum(new_h) - N / 2.0) > 1e-8:
                        continue
            
            new_obj = _objective(new_h)
            if new_obj < best_obj:
                best = new_h
                best_obj = new_obj
    
    return best, best_obj


def _nelder_mead_search(h, max_iters=2000, alpha=0.5, gamma=2.0, rho=0.5, sigma=2.0):
    """Simplified Nelder-Mead for the projected feasible set."""
    from scipy.optimize import minimize
    
    def obj_func(x):
        h_proj = _project_to_feasible(x)
        return _objective(h_proj)
    
    try:
        result = minimize(obj_func, h, method='Nelder-Mead', 
                         options={'maxiter': max_iters, 'xatol': 1e-6, 'fatol': 1e-6})
        return _project_to_feasible(result.x), _objective(_project_to_feasible(result.x))
    except:
        return h, _objective(h)


def construct():
    start_time = time.time()
    deadline = 145.0  # 5 second safety margin
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fast(best_h)
    
    # Generate diverse candidates
    candidates = []
    
    # Sine candidates
    for freq in range(1, 15):
        for phase in [0.0, np.pi/2, np.pi, np.pi/4, 3*np.pi/4]:
            for amp in [0.3, 0.4, 0.5, 0.6]:
                h = _sine_candidate(freq, phase, amp)
                candidates.append(h)
    
    # Two-sine candidates
    for f1 in [1, 2, 3, 5]:
        for f2 in [2, 3, 5, 7]:
            if f1 != f2:
                for p1 in [0.0, np.pi/2]:
                    for p2 in [0.0, np.pi/2]:
                        h = _two_sine_candidate(f1, f2, p1, p2, 0.25)
                        candidates.append(h)
    
    # Modular candidates
    for mod in range(2, 20):
        for offset in range(mod):
            h = _modular_candidate(mod, offset)
            candidates.append(h)
    
    # Binary candidates
    for _ in range(30):
        candidates.append(_binary_candidate())
    
    # Random candidates
    for _ in range(100):
        candidates.append(_random_candidate())
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        score = _max_overlap_fast(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Coordinate descent from best candidate
    if time.time() < deadline - 10:
        best_h, best_score = _coordinate_descent(best_h, max_iters=3000, step=0.05)
    
    # Try Nelder-Mead from best candidate
    if time.time() < deadline - 5:
        h_nm, score_nm = _nelder_mead_search(best_h, max_iters=500)
        if score_nm < best_score:
            best_score = score_nm
            best_h = h_nm
    
    # Coordinate descent from other good candidates
    good_candidates = []
    for h in candidates[:100]:
        score = _max_overlap_fast(h)
        if score < best_score + 0.5:
            good_candidates.append((h, score))
    
    good_candidates.sort(key=lambda x: x[1])
    
    for h, score in good_candidates[:10]:
        if time.time() > deadline - 5:
            break
        h2, score2 = _coordinate_descent(h, max_iters=2000, step=0.05)
        if score2 < best_score:
            best_score = score2
            best_h = h2
    
    # Final coordinate descent
    if time.time() < deadline:
        best_h, best_score = _coordinate_descent(best_h, max_iters=3000, step=0.03)
    
    return best_h
# EVOLVE-BLOCK-END
