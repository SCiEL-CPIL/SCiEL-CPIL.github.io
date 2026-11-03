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
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    min_autocorr = np.min(autocorr)
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap


def _project_to_feasible(h):
    """Project h to satisfy: sum = N/2, 0 <= h[i] <= 1."""
    h = h.copy()
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    if abs(diff) > 1e-12:
        if diff > 0:
            room = 1.0 - h
            if np.sum(room) > 1e-12:
                h += diff * room / np.sum(room)
            else:
                pass
        else:
            if np.sum(h) > 1e-12:
                h += diff * h / np.sum(h)
            else:
                pass
        h = np.clip(h, 0, 1)
    current_sum = np.sum(h)
    if abs(current_sum - target) > 1e-10:
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


def _sine_candidate(freq, amplitude=0.45, phase=0.0):
    """Generate a sinusoidal candidate with exact sum."""
    x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
    h = 0.5 + amplitude * np.sin(x + phase)
    h = _project_to_feasible(h)
    return h


def _two_sine_candidate(freq1, amp1, freq2, amp2, phase1=0.0, phase2=0.0):
    """Generate a two-sine candidate with exact sum."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = 0.5 + amp1 * np.sin(2 * np.pi * freq1 * x / N + phase1) + amp2 * np.sin(2 * np.pi * freq2 * x / N + phase2)
    h = _project_to_feasible(h)
    return h


def _random_candidate():
    """Generate a random feasible candidate."""
    h = np.random.uniform(0, 1, N)
    h = _project_to_feasible(h)
    return h


def _binary_candidate(seed=None):
    """Generate a binary-like candidate with exactly N/2 ones."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    idx = rng.permutation(N)
    h = np.zeros(N)
    h[idx[:N//2]] = 1.0
    return h


def _local_search(h0, max_iters=2000, time_limit=20.0):
    """Local search with adaptive step sizes."""
    h = h0.copy()
    best_score = _max_overlap_fast(h)
    start_time = time.time()
    step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01]
    current_step_idx = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if time.time() - start_time > time_limit * 0.7 and current_step_idx < len(step_sizes) - 1:
            current_step_idx += 1
        step = step_sizes[current_step_idx]
        
        i = np.random.randint(N)
        improved = False
        for delta in [step, -step]:
            new_h = h.copy()
            new_h[i] = np.clip(new_h[i] + delta, 0, 1)
            new_h = _project_to_feasible(new_h)
            score = _max_overlap_fast(new_h)
            if score < best_score - 1e-10:
                h = new_h
                best_score = score
                improved = True
                break
        
        if not improved and it % 100 == 0:
            # Try larger perturbation occasionally
            for _ in range(5):
                new_h = h + np.random.uniform(-0.3, 0.3, N)
                new_h = _project_to_feasible(new_h)
                score = _max_overlap_fast(new_h)
                if score < best_score - 1e-10:
                    h = new_h
                    best_score = score
                    break
    
    return h, best_score


def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10 second safety margin
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fast(best_h)
    
    # Generate diverse starting candidates
    candidates = []
    
    # Single sine waves - focus on frequencies that might give good overlap properties
    for freq in [1, 2, 3, 4, 5, 7, 10, 13, 17, 20, 25, 30, 35, 40, 50]:
        for amp in [0.3, 0.4, 0.45, 0.48]:
            for phase in [0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
                h = _sine_candidate(freq, amp, phase)
                candidates.append(h)
    
    # Two-sine combinations
    for f1 in [1, 2, 3, 5, 7, 10]:
        for f2 in [4, 6, 8, 12, 15, 20]:
            if f1 == f2:
                continue
            for a1 in [0.25, 0.35]:
                for a2 in [0.15, 0.25]:
                    h = _two_sine_candidate(f1, a1, f2, a2)
                    candidates.append(h)
    
    # Random candidates
    for _ in range(50):
        candidates.append(_random_candidate())
    
    # Binary candidates
    for seed in range(30):
        candidates.append(_binary_candidate(seed))
    
    # Evaluate all candidates to find good starting points
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 40:
            break
        score = _max_overlap_fast(h)
        scored_candidates.append((score, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top candidates for optimization
    top_candidates = [h for _, h in scored_candidates[:15]]
    
    # Run local search from each top candidate
    for h0 in top_candidates:
        if time.time() > deadline - 15:
            break
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        h_opt, score_opt = _local_search(
            h0, max_iters=3000,
            time_limit=min(15.0, remaining)
        )
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final refinement from best
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            h_final, score_final = _local_search(
                best_h, max_iters=5000,
                time_limit=min(20.0, remaining)
            )
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
    
    # Ensure the result is valid
    best_h = _project_to_feasible(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
