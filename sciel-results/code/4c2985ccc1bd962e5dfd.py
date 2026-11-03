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


def _smooth_max_overlap(h, beta=50.0):
    """
    Smooth surrogate for max overlap using soft-max.
    overlap_k = N/2 - autocorr_k
    We use a log-sum-exp style soft-max to approximate max_k overlap_k smoothly.
    """
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    overlaps = N / 2.0 - autocorr
    # Soft max: (1/beta) * log(sum(exp(beta * overlaps)))
    # Use numerically stable version
    max_ov = np.max(overlaps)
    if max_ov > 1e10:
        return max_ov
    exp_shifted = np.exp(beta * (overlaps - max_ov))
    soft_max = max_ov + np.log(np.sum(exp_shifted)) / beta
    return soft_max


def _smooth_max_overlap_grad(h, beta=50.0):
    """
    Gradient of smooth max overlap with respect to h.
    overlap_k = N/2 - sum_i h[i]*h[(i-k)%N]
    d(overlap_k)/d(h[j]) = -sum_i [h[(i-k)%N]*delta_{i,j} + h[i]*delta_{(i-k)%N,j}]
                         = -h[(j+k)%N] - h[(j-k)%N]  ... need to be careful
    Actually: autocorr_k = sum_i h[i]*h[(i-k)%N]
    d(autocorr_k)/d(h[j]) = h[(j+k)%N] + h[(j-k)%N]  (for k != 0 and k != N/2)
    For k=0: d(autocorr_0)/d(h[j]) = 2*h[j]
    For k=N/2 (even N): d(autocorr_{N/2})/d(h[j]) = h[(j+N/2)%N] + h[(j-N/2)%N] = 2*h[(j+N/2)%N] since (j+N/2)%N == (j-N/2)%N mod N
    
    So d(overlap_k)/d(h[j]) = -d(autocorr_k)/d(h[j])
    
    soft_max = max_ov + (1/beta)*log(sum(exp(beta*(ov_k - max_ov))))
    d(soft_max)/d(h[j]) = sum_k p_k * d(ov_k)/d(h[j])
    where p_k = exp(beta*(ov_k - max_ov)) / sum(exp(beta*(ov_k - max_ov)))
    """
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    overlaps = N / 2.0 - autocorr
    
    max_ov = np.max(overlaps)
    exp_shifted = np.exp(beta * (overlaps - max_ov))
    weights = exp_shifted / np.sum(exp_shifted)
    
    # For each shift k, compute gradient contribution
    # d(overlap_k)/d(h[j]) = -(h[(j+k)%N] + h[(j-k)%N]) for k not 0 or N/2
    # For k=0: -2*h[j]
    # For k=N/2: -2*h[(j+N/2)%N]
    
    grad = np.zeros(N)
    for k in range(N):
        w = weights[k]
        if w < 1e-15:
            continue
        if k == 0:
            grad -= w * 2.0 * h
        elif k == N // 2:
            grad -= w * 2.0 * np.roll(h, N // 2)
        else:
            grad -= w * (np.roll(h, k) + np.roll(h, -k))
    
    return grad


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


def _smooth_optimize(h0, beta=50.0, time_limit=10.0):
    """
    Use SLSQP with smooth objective to optimize h.
    """
    start_time = time.time()
    
    def obj(x):
        return _smooth_max_overlap(x, beta=beta)
    
    def grad(x):
        return _smooth_max_overlap_grad(x, beta=beta)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0.0, 1.0)] * N
    
    try:
        result = minimize(
            obj, h0.copy(),
            jac=grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10, 'time_limit': time_limit}
        )
        if result.success and time.time() - start_time < time_limit:
            return _project_to_feasible(result.x)
    except Exception:
        pass
    
    return h0


def _local_search(h0, max_iters=2000, time_limit=20.0):
    """Local search with adaptive step sizes and multi-coordinate moves."""
    h = h0.copy()
    best_score = _max_overlap_fast(h)
    start_time = time.time()
    step_sizes = [0.15, 0.1, 0.05, 0.02, 0.01]
    current_step_idx = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if time.time() - start_time > time_limit * 0.7 and current_step_idx < len(step_sizes) - 1:
            current_step_idx += 1
        step = step_sizes[current_step_idx]
        
        # Try single coordinate
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
        
        # Try multi-coordinate move occasionally
        if not improved and it % 50 == 0:
            n_coords = np.random.choice([2, 3, 5])
            indices = np.random.choice(N, size=n_coords, replace=False)
            for _ in range(3):
                new_h = h.copy()
                deltas = np.random.uniform(-step, step, n_coords)
                for idx, d in zip(indices, deltas):
                    new_h[idx] = np.clip(new_h[idx] + d, 0, 1)
                new_h = _project_to_feasible(new_h)
                score = _max_overlap_fast(new_h)
                if score < best_score - 1e-10:
                    h = new_h
                    best_score = score
                    improved = True
                    break
        
        if not improved and it % 200 == 0:
            # Larger perturbation
            for _ in range(5):
                new_h = h + np.random.uniform(-0.2, 0.2, N)
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
    
    # Single sine waves
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
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 50:
            break
        score = _max_overlap_fast(h)
        scored_candidates.append((score, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top candidates for optimization
    top_candidates = [h for _, h in scored_candidates[:12]]
    
    # Phase 1: Smooth optimization from top candidates
    for h0 in top_candidates:
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time() - 15
        if remaining < 5:
            break
        h_smooth = _smooth_optimize(h0, beta=50.0, time_limit=min(8.0, remaining))
        score_smooth = _max_overlap_fast(h_smooth)
        if score_smooth < best_score:
            best_score = score_smooth
            best_h = h_smooth
    
    # Phase 2: Local search from best and top candidates
    all_starts = [best_h] + top_candidates[:5]
    for h0 in all_starts:
        if time.time() > deadline - 15:
            break
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        h_opt, score_opt = _local_search(
            h0, max_iters=3000,
            time_limit=min(12.0, remaining)
        )
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Phase 3: Final smooth polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            h_final = _smooth_optimize(best_h, beta=100.0, time_limit=min(10.0, remaining))
            score_final = _max_overlap_fast(h_final)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
    
    # Ensure the result is valid
    best_h = _project_to_feasible(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
