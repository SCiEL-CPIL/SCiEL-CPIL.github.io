"""Bootstrap portfolio search for Erdos minimum overlap."""
import time
import numpy as np

N = 100
DEADLINE = 150.0

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _score(h):
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _fast_score(h):
    """Faster score using FFT-based cross-correlation."""
    # np.correlate(h, 1-h, 'full')[k] = sum_i h[i+k] * (1-h)[i] for k in -(N-1)..N-1
    # We want max over all lags.
    # Use FFT: cross-correlation via IFFT(FFT(h) * conj(FFT(1-h)))
    # But we need the full correlation, not just circular.
    # For N=100, direct correlate is fast enough. Let's just use np.correlate.
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True


def _fix_sum(h):
    target = N / 2
    current = np.sum(h)
    diff = target - current
    if abs(diff) < 1e-12:
        return h.copy()
    
    h = h.copy()
    for _ in range(100):
        if abs(diff) < 1e-10:
            break
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask):
                break
            idx = np.where(mask)[0]
            # Distribute evenly among those with room
            room = (1.0 - h[idx]).sum()
            if room < 1e-10:
                break
            add_per = min(diff / len(idx), 1.0)
            # Actually just add diff/len(idx) to each, then clip
            h[idx] += diff / len(idx)
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            idx = np.where(mask)[0]
            h[idx] += diff / len(idx)
        
        # Clip
        h = np.clip(h, 0, 1)
        diff = target - np.sum(h)
    
    # Final exact fix if needed
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            # Add to element with most room
            idx = np.argmin(h)
            h[idx] += diff
        else:
            idx = np.argmax(h)
            h[idx] += diff
        h = np.clip(h, 0, 1)
    
    return h


def _random_binary_candidates(count=200, seed=42):
    rng = np.random.RandomState(seed)
    candidates = []
    for _ in range(count):
        # Random binary with sum ~ N/2
        h = rng.randint(0, 2, N).astype(float)
        # Adjust sum
        target = N / 2
        diff = target - np.sum(h)
        if diff > 0:
            # Need to add diff ones
            zeros = np.where(h == 0)[0]
            if len(zeros) >= int(diff):
                flip = rng.choice(zeros, size=int(diff), replace=False)
                h[flip] = 1
        elif diff < 0:
            ones = np.where(h == 1)[0]
            if len(ones) >= int(-diff):
                flip = rng.choice(ones, size=int(-diff), replace=False)
                h[flip] = 0
        h = _fix_sum(h)
        if _is_valid(h):
            candidates.append(h)
    return candidates


def _random_block_candidates(count=100, seed=42):
    rng = np.random.RandomState(seed + 1)
    candidates = []
    for _ in range(count):
        # Create block structure: alternating segments
        num_blocks = rng.randint(3, 11)
        block_sizes = rng.multinomial(N, np.ones(num_blocks) / num_blocks)
        h = np.zeros(N)
        pos = 0
        for i, bs in enumerate(block_sizes):
            val = 1.0 if i % 2 == 0 else 0.0
            h[pos:pos + bs] = val
            pos += bs
        # Smooth transitions and adjust
        h = 0.5 * (h + h[::-1])  # Symmetrize
        h = np.clip(h, 0, 1)
        h = _fix_sum(h)
        if _is_valid(h):
            candidates.append(h)
    return candidates


def _random_continuous_candidates(count=100, seed=42):
    rng = np.random.RandomState(seed + 2)
    candidates = []
    for _ in range(count):
        # Smooth random via low-pass
        raw = rng.uniform(0, 1, N)
        # Apply smoothing
        kernel = np.ones(5) / 5
        h = np.convolve(raw, kernel, mode='same')
        h = np.clip(h, 0, 1)
        h = _fix_sum(h)
        if _is_valid(h):
            candidates.append(h)
    return candidates


def _sinusoidal_candidates():
    candidates = []
    for P in [10, 20, 25, 50]:
        for amp in [0.3, 0.4, 0.5, 0.45]:
            for phase in [0, np.pi / 2, np.pi]:
                h = 0.5 + amp * np.cos(2 * np.pi * np.arange(N) / P + phase)
                h = np.clip(h, 0, 1)
                h = _fix_sum(h)
                if _is_valid(h):
                    candidates.append(h)
    return candidates


def _optimize_with_scipy(h0, deadline_check):
    """Use scipy minimize to optimize the score."""
    h0 = h0.copy()
    
    def objective(x):
        h = np.clip(x, 0, 1)
        h = _fix_sum(h)
        return _score(h)
    
    def objective_grad(x):
        # Numerical gradient (expensive but okay for small N)
        eps = 1e-6
        grad = np.zeros_like(x)
        base = objective(x)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            grad[i] = (objective(xp) - base) / eps
        return grad
    
    try:
        result = minimize(
            objective, h0, method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        if deadline_check():
            return h0
        h_opt = result.x
        h_opt = _fix_sum(h_opt)
        return h_opt
    except Exception:
        return h0


def _hill_climb(h0, max_iters=3000, step_size=0.05, deadline_check=None):
    h = h0.copy()
    current_val = _score(h)
    best_val = current_val
    best_h = h.copy()
    rng = np.random.RandomState(123)
    
    for i in range(max_iters):
        if deadline_check and deadline_check():
            break
        
        num_perturb = rng.randint(1, 4)
        indices = rng.choice(N, size=num_perturb, replace=False)
        
        h_new = h.copy()
        for idx in indices:
            h_new[idx] += rng.uniform(-step_size, step_size)
        
        h_new = np.clip(h_new, 0, 1)
        h_new = _fix_sum(h_new)
        
        if not _is_valid(h_new):
            continue
        
        new_val = _score(h_new)
        if new_val < current_val:
            current_val = new_val
            h = h_new
            if new_val < best_val:
                best_val = new_val
                best_h = h_new.copy()
    
    return best_h, best_val


def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE - 10.0
    
    # Generate diverse initial candidates
    all_candidates = [np.full(N, 0.5)]
    
    # Sinusoidal
    all_candidates.extend(_sinusoidal_candidates())
    
    # Binary
    all_candidates.extend(_random_binary_candidates(count=100, seed=42))
    
    # Block
    all_candidates.extend(_random_block_candidates(count=50, seed=42))
    
    # Continuous smooth
    all_candidates.extend(_random_continuous_candidates(count=50, seed=42))
    
    # Evaluate all candidates
    scored = []
    for h in all_candidates:
        if deadline_check():
            break
        if _is_valid(h):
            s = _score(h)
            scored.append((s, h))
    
    scored.sort(key=lambda x: x[0])
    
    best_h = scored[0][1] if scored else np.full(N, 0.5)
    best_score = scored[0][0] if scored else _score(best_h)
    
    # Take top candidates and refine
    top_k = min(10, len(scored))
    for i in range(top_k):
        if deadline_check():
            break
        h0 = scored[i][1]
        
        # First try scipy optimization
        h_opt = _optimize_with_scipy(h0, deadline_check)
        if _is_valid(h_opt):
            s_opt = _score(h_opt)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        
        # Then hill climb from the result
        if not deadline_check():
            h_climbed, s_climbed = _hill_climb(h_opt if _is_valid(h_opt) else h0, 
                                                max_iters=2000, step_size=0.03,
                                                deadline_check=deadline_check)
            if s_climbed < best_score:
                best_score = s_climbed
                best_h = h_climbed.copy()
    
    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
