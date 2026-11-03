"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _max_overlap_fast(h):
    """
    Compute max overlap of h with shifted (1-h) using FFT-based circular autocorrelation.
    overlap_k = N/2 - autocorr_k, where autocorr_k = sum_i h[i]*h[(i-k)%N]
    We want to minimize max_k overlap_k = N/2 - min_k autocorr_k
    So we want to maximize min_k autocorr_k.
    """
    # Circular autocorrelation via FFT
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i-k) % N]
    min_autocorr = np.min(autocorr)
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap


def _autocorr_min(h):
    """Return min circular autocorrelation (we want to maximize this)."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    return np.min(autocorr)


def _objective(h):
    """Objective to minimize: max overlap."""
    return _max_overlap_fast(h)


def _project_to_feasible(h):
    """Project h to satisfy: sum = N/2, 0 <= h[i] <= 1."""
    h = h.copy()
    # First, clip to [0, 1]
    h = np.clip(h, 0, 1)
    # Adjust sum to N/2
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    if abs(diff) > 1e-12:
        if diff > 0:
            # Need to increase; distribute to elements below 1
            room = 1.0 - h
            if np.sum(room) > 1e-12:
                h += diff * room / np.sum(room)
            else:
                # All at 1, can't increase further (shouldn't happen if sum < target)
                pass
        else:
            # Need to decrease; distribute from elements above 0
            if np.sum(h) > 1e-12:
                h += diff * h / np.sum(h)
            else:
                pass
        h = np.clip(h, 0, 1)
    # Final adjustment for numerical precision
    current_sum = np.sum(h)
    if abs(current_sum - target) > 1e-10:
        # Distribute remaining difference uniformly among elements with room
        residual = target - current_sum
        if residual > 0:
            room = 1.0 - h
            mask = room > 1e-10
            if np.sum(room[mask]) > 0:
                h[mask] += residual / np.sum(mask)
        else:
            mask = h > 1e-10
            if np.sum(h[mask]) > 0:
                h[mask] += residual / np.sum(mask)
        h = np.clip(h, 0, 1)
    return h


def _smooth_candidate(freq, phase=0.0):
    """Generate a smooth sinusoidal candidate with exact sum."""
    x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
    h = 0.5 + 0.4 * np.sin(x + phase)
    h = _project_to_feasible(h)
    return h


def _random_candidate():
    """Generate a random feasible candidate."""
    h = np.random.uniform(0, 1, N)
    h = _project_to_feasible(h)
    return h


def _nelder_mead_search(h0, max_iters=500, tol=1e-8, time_limit=10.0):
    """Run Nelder-Mead optimization with time limit."""
    start_time = time.time()
    
    def obj(x):
        h = _project_to_feasible(x)
        return _max_overlap_fast(h)
    
    try:
        result = minimize(
            obj, h0, method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol}
        )
        best_h = _project_to_feasible(result.x)
        best_score = _max_overlap_fast(best_h)
    except Exception:
        best_h = h0.copy()
        best_score = _max_overlap_fast(h0)
    
    elapsed = time.time() - start_time
    return best_h, best_score, elapsed


def _coordinate_descent(h0, max_iters=2000, time_limit=10.0):
    """Coordinate descent: perturb one coordinate at a time."""
    h = h0.copy()
    best_score = _max_overlap_fast(h)
    start_time = time.time()
    step_size = 0.1
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        i = np.random.randint(N)
        for delta in [step_size, -step_size, step_size/2, -step_size/2]:
            new_h = h.copy()
            new_h[i] = np.clip(new_h[i] + delta, 0, 1)
            new_h = _project_to_feasible(new_h)
            score = _max_overlap_fast(new_h)
            if score < best_score:
                h = new_h
                best_score = score
                break
    return h, best_score


def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fast(best_h)  # Should be 25.0
    
    # Generate diverse starting candidates
    candidates = []
    
    # Sinusoidal candidates with various frequencies and phases
    for freq in range(1, 15):
        for phase in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 7 * np.pi / 4]:
            h = _smooth_candidate(freq, phase)
            candidates.append(h)
    
    # Random candidates
    for _ in range(100):
        candidates.append(_random_candidate())
    
    # Binary-like candidates (alternating patterns)
    for pattern in ['alternating', 'blocks']:
        for _ in range(20):
            h = np.random.uniform(0, 1, N)
            if pattern == 'alternating':
                # Push toward binary
                h = np.where(h > 0.5, 1.0, 0.0)
                # Ensure sum is N/2
                n_ones = np.sum(h)
                if n_ones > N / 2:
                    idx = np.where(h == 1.0)[0]
                    to_zero = idx[:int(n_ones - N / 2)]
                    h[to_zero] = 0.0
                elif n_ones < N / 2:
                    idx = np.where(h == 0.0)[0]
                    to_one = idx[:int(N / 2 - n_ones)]
                    h[to_one] = 1.0
            else:
                # Block pattern
                block_size = np.random.randint(2, 10)
                h = np.zeros(N)
                for i in range(0, N, block_size * 2):
                    h[i:i + block_size] = 1.0
                h = _project_to_feasible(h)
            candidates.append(h)
    
    # Evaluate all candidates to find good starting points
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 30:
            break
        score = _max_overlap_fast(h)
        scored_candidates.append((score, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top candidates for optimization
    top_candidates = [h for _, h in scored_candidates[:20]]
    
    # Run Nelder-Mead from each top candidate
    for h0 in top_candidates:
        if time.time() > deadline - 10:
            break
        h_opt, score_opt, elapsed = _nelder_mead_search(
            h0, max_iters=200, tol=1e-7, 
            time_limit=min(10.0, max(0, deadline - time.time() - 5))
        )
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Coordinate descent refinement from best
    if time.time() < deadline - 5:
        h_cd, score_cd = _coordinate_descent(
            best_h, max_iters=5000,
            time_limit=min(15.0, max(0, deadline - time.time() - 3))
        )
        if score_cd < best_score:
            best_score = score_cd
            best_h = h_cd
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 3:
        h_final, score_final, _ = _nelder_mead_search(
            best_h, max_iters=300, tol=1e-8,
            time_limit=min(8.0, max(0, deadline - time.time() - 2))
        )
        if score_final < best_score:
            best_score = score_final
            best_h = h_final
    
    # Ensure the result is valid
    best_h = _project_to_feasible(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
