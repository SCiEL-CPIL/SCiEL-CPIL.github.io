import numpy as np
import time

N = 100

def _time_remaining(start, deadline):
    return deadline - time.time()

def _compute_all_corrs_fft(h):
    """Compute all cyclic cross-correlations C[k] = sum_i h[i] * h[(i+k) % N] using FFT."""
    H = np.fft.fft(h)
    C = np.fft.ifft(np.abs(H)**2).real
    return C

def _max_overlap(h):
    """Compute the maximum overlap value, which is max_k (N/2 - C[k])."""
    C = _compute_all_corrs_fft(h)
    return N/2.0 - np.min(C)

def _project_to_box_simplex(v, lower=0.0, upper=1.0, target_sum=None):
    """Project v onto the box [lower, upper]^N with sum = target_sum."""
    if target_sum is None:
        target_sum = N/2.0
    
    v_clipped = np.clip(v, lower, upper)
    
    if np.allclose(np.sum(v_clipped), target_sum, atol=1e-12):
        return v_clipped
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(v - lam, lower, upper))
    
    lo = lower - np.max(v) - 1.0
    hi = upper - np.min(v) + 1.0
    
    for _ in range(100):
        mid = (lo + hi) / 2.0
        s = sum_at_lambda(mid)
        if np.abs(s - target_sum) < 1e-12:
            break
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2.0
    result = np.clip(v - lam, lower, upper)
    
    diff = target_sum - np.sum(result)
    interior = (result > lower + 1e-12) & (result < upper - 1e-12)
    if np.any(interior):
        result[interior] += diff / np.sum(interior)
        result = np.clip(result, lower, upper)
        diff2 = target_sum - np.sum(result)
        if np.abs(diff2) > 1e-10:
            mid_idx = np.argmin(np.abs(result - 0.5))
            result[mid_idx] += diff2
            result = np.clip(result, lower, upper)
    
    return result

def _simulated_annealing(h_init, start_temp, end_temp, n_steps, deadline):
    """Run simulated annealing to minimize max_overlap."""
    h = h_init.copy()
    current_score = _max_overlap(h)
    best_h = h.copy()
    best_score = current_score
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule (geometric cooling)
        progress = step / max(n_steps - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Generate a perturbation: pick random indices and adjust their values
        n_perturb = np.random.randint(2, 10)  # perturb 2-9 elements
        indices = np.random.choice(N, n_perturb, replace=False)
        deltas = np.random.uniform(-0.3, 0.3, n_perturb)
        
        h_trial = h.copy()
        h_trial[indices] += deltas
        h_trial = _project_to_box_simplex(h_trial, 0, 1, N/2.0)
        
        trial_score = _max_overlap(h_trial)
        delta_score = trial_score - current_score
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta_score < 0 or np.random.random() < np.exp(-delta_score / max(temp, 1e-10)):
            h = h_trial
            current_score = trial_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
    
    return best_h, best_score

def _pairwise_swap_refine(h, deadline):
    """Refine h by trying pairwise swaps of values between two indices."""
    current_score = _max_overlap(h)
    best_h = h.copy()
    best_score = current_score
    
    # Try random pairs of indices and swap/blend their values
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        i, j = np.random.choice(N, 2, replace=False)
        
        # Try blending: move some value from one to the other
        for blend in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            h_trial = h.copy()
            h_trial[i] = h[i] * (1 - blend) + h[j] * blend
            h_trial[j] = h[j] * (1 - blend) + h[i] * blend
            h_trial = _project_to_box_simplex(h_trial, 0, 1, N/2.0)
            
            trial_score = _max_overlap(h_trial)
            if trial_score < current_score - 1e-12:
                h = h_trial
                current_score = trial_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
                break
    
    return best_h, best_score

def _generate_diverse_inits(n_inits):
    """Generate diverse initializations."""
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Bang-bang patterns
    for _ in range(min(15, n_inits)):
        h_bb = np.zeros(N)
        idx = np.random.permutation(N)[:N//2]
        h_bb[idx] = 1.0
        inits.append(h_bb)
    
    # 3. Block patterns
    for block_size in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        if np.sum(h_block) != N/2:
            current_sum = np.sum(h_block)
            if current_sum > N/2:
                ones = np.where(h_block == 1)[0]
                to_flip = np.random.choice(ones, int(current_sum - N/2), replace=False)
                h_block[to_flip] = 0
            else:
                zeros = np.where(h_block == 0)[0]
                to_flip = np.random.choice(zeros, int(N/2 - current_sum), replace=False)
                h_block[to_flip] = 1
        inits.append(h_block)
    
    # 4. Random smooth patterns
    for _ in range(min(10, n_inits)):
        coeffs = np.random.randn(N) * 0.3
        coeffs[0] = 0
        h_rand = np.real(np.fft.ifft(coeffs * np.exp(2j * np.pi * np.random.randn(N) * 0.5)))
        h_rand = 0.5 + h_rand
        h_rand = _project_to_box_simplex(h_rand, 0, 1, N/2.0)
        inits.append(h_rand)
    
    # 5. Fourier-based
    for _ in range(min(10, n_inits)):
        P = np.abs(np.random.randn(N) + 1j * np.random.randn(N))**2
        h_fourier = np.fft.ifft(P).real
        h_fourier = _project_to_box_simplex(h_fourier, 0, 1, N/2.0)
        inits.append(h_fourier)
    
    # 6. Random with structure: alternating high/low with noise
    for _ in range(min(10, n_inits)):
        h_alt = np.where(np.arange(N) % 2 == 0, 0.7, 0.3)
        h_alt += np.random.randn(N) * 0.1
        h_alt = _project_to_box_simplex(h_alt, 0, 1, N/2.0)
        inits.append(h_alt)
    
    return inits[:n_inits]

def construct():
    start_time = time.time()
    deadline = 147  # 3s margin from 150s limit
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initializations
    n_inits = 40
    inits = _generate_diverse_inits(n_inits)
    
    # Phase 1: Simulated annealing from each initialization
    # Use shorter runs for more restarts
    sa_configs = [
        (0.1, 0.001, 2000),  # (start_temp, end_temp, n_steps)
        (0.05, 0.0005, 3000),
        (0.2, 0.002, 1500),
    ]
    
    for init_h in inits:
        if time.time() > deadline - 5.0:
            break
        
        # Pick a SA config based on which init we're on
        config_idx = inits.index(init_h) % len(sa_configs)
        start_temp, end_temp, n_steps = sa_configs[config_idx]
        
        # Limit steps based on remaining time
        time_per_step = 0.002  # rough estimate
        max_steps_by_time = int((deadline - 5.0 - time.time()) / time_per_step)
        n_steps = min(n_steps, max(100, max_steps_by_time))
        
        h_sa, score_sa = _simulated_annealing(init_h, start_temp, end_temp, n_steps, deadline - 5.0)
        
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()
    
    # Phase 2: Pairwise swap refinement on the best solution
    h_refined, score_refined = _pairwise_swap_refine(best_h, deadline - 2.0)
    if score_refined < best_score:
        best_score = score_refined
        best_h = h_refined.copy()
    
    # Phase 3: Final short SA polish on the refined solution
    h_polish, score_polish = _simulated_annealing(best_h, 0.01, 0.0001, 1000, deadline - 1.0)
    if score_polish < best_score:
        best_score = score_polish
        best_h = h_polish.copy()
    
    # Final: ensure exact sum
    best_h = _project_to_box_simplex(best_h, 0, 1, N/2.0)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    C = _compute_all_corrs_fft(h)
    min_C = np.min(C)
    max_overlap = N/2.0 - min_C
    print(f"Min C: {min_C}, Max Overlap: {max_overlap}")
    print(f"Sum h: {np.sum(h)}")
    print(f"Min h: {np.min(h)}, Max h: {np.max(h)}")
    print(f"Score (value): {max_overlap}")
