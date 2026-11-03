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


def _spectral_to_h(mags, phases):
    """
    Convert spectral magnitudes and phases to time-domain signal.
    Enforces Hermitian symmetry for real output.
    """
    H = np.zeros(N, dtype=complex)
    H[0] = mags[0]
    H[N//2] = mags[N//2]
    for k in range(1, N//2):
        H[k] = mags[k] * np.exp(1j * phases[k])
        H[N-k] = mags[k] * np.exp(-1j * phases[k])
    
    h = np.real(np.fft.ifft(H))
    return h


def _compute_min_autocorr(mags, phases):
    """
    Compute the minimum autocorrelation value for given spectral parameters.
    """
    h = _spectral_to_h(mags, phases)
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    return np.min(autocorr)


def _optimize_spectral(mags0, phases0, time_limit=10.0):
    """
    Optimize spectral phases to maximize min autocorrelation.
    This is equivalent to minimizing max overlap.
    """
    start_time = time.time()
    
    def neg_min_autocorr(phases):
        if time.time() - start_time > time_limit:
            return -1e10
        min_ac = _compute_min_autocorr(mags0, phases)
        return -min_ac
    
    # Gradient approximation via finite differences
    def grad_neg_min_autocorr(phases):
        eps = 1e-4
        grad = np.zeros_like(phases)
        base_val = neg_min_autocorr(phases)
        for i in range(len(phases)):
            phases_plus = phases.copy()
            phases_minus = phases.copy()
            phases_plus[i] += eps
            phases_minus[i] -= eps
            val_plus = neg_min_autocorr(phases_plus)
            val_minus = neg_min_autocorr(phases_minus)
            grad[i] = (val_plus - val_minus) / (2 * eps)
        return grad
    
    # Use Nelder-Mead for robustness (no gradient required)
    try:
        result = minimize(
            neg_min_autocorr, phases0.copy(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        if time.time() - start_time < time_limit:
            return result.x
    except Exception:
        pass
    
    return phases0


def _optimize_spectral_grad(mags0, phases0, time_limit=10.0, lr=0.05):
    """
    Gradient ascent on min autocorrelation using spectral phase optimization.
    """
    start_time = time.time()
    phases = phases0.copy()
    best_phases = phases.copy()
    best_val = _compute_min_autocorr(mags0, phases)
    
    # Precompute gradient using FFT-based approach
    def compute_grad(phases):
        h = _spectral_to_h(mags0, phases)
        H = np.fft.fft(h)
        autocorr = np.real(np.fft.ifft(H * np.conj(H)))
        
        # Find indices of min autocorrelation
        min_idx = np.argmin(autocorr)
        
        # Gradient of autocorr_k with respect to phases
        # This is complex; use finite difference for stability
        eps = 1e-4
        grad = np.zeros_like(phases)
        for i in range(len(phases)):
            p_plus = phases.copy()
            p_minus = phases.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            
            h_plus = _spectral_to_h(mags0, p_plus)
            H_plus = np.fft.fft(h_plus)
            ac_plus = np.real(np.fft.ifft(H_plus * np.conj(H_plus)))
            
            h_minus = _spectral_to_h(mags0, p_minus)
            H_minus = np.fft.fft(h_minus)
            ac_minus = np.real(np.fft.ifft(H_minus * np.conj(H_minus)))
            
            grad[i] = (np.min(ac_plus) - np.min(ac_minus)) / (2 * eps)
        
        return grad
    
    for step in range(200):
        if time.time() - start_time > time_limit:
            break
        
        grad = compute_grad(phases)
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Adaptive learning rate
        new_phases = phases + lr * grad
        new_val = _compute_min_autocorr(mags0, new_phases)
        
        if new_val > best_val:
            best_val = new_val
            best_phases = new_phases.copy()
            phases = new_phases
            lr = min(lr * 1.1, 0.5)
        else:
            lr = max(lr * 0.7, 0.01)
            # Try smaller step
            new_phases = phases + lr * grad
            new_val = _compute_min_autocorr(mags0, new_phases)
            if new_val > best_val:
                best_val = new_val
                best_phases = new_phases.copy()
                phases = new_phases
    
    return best_phases


def _construct_spectral_candidates():
    """
    Generate diverse spectral magnitude profiles.
    """
    candidates = []
    
    # 1. Flat spectrum (white noise-like)
    mags_flat = np.ones(N) * 0.5
    mags_flat[0] = N / 2.0  # DC component for mean 0.5
    candidates.append(mags_flat)
    
    # 2. Low-pass spectrum
    for cutoff in [0.1, 0.2, 0.3, 0.5]:
        mags_lp = np.zeros(N)
        for k in range(N):
            freq = min(k, N-k) / N
            if freq <= cutoff:
                mags_lp[k] = 1.0
        mags_lp[0] = N / 2.0
        candidates.append(mags_lp)
    
    # 3. High-pass spectrum
    for cutoff in [0.1, 0.2, 0.3]:
        mags_hp = np.zeros(N)
        for k in range(N):
            freq = min(k, N-k) / N
            if freq >= cutoff:
                mags_hp[k] = 1.0
        mags_hp[0] = N / 2.0
        candidates.append(mags_hp)
    
    # 4. Band-pass spectrum
    for lo, hi in [(0.1, 0.3), (0.2, 0.5), (0.3, 0.7)]:
        mags_bp = np.zeros(N)
        for k in range(N):
            freq = min(k, N-k) / N
            if lo <= freq <= hi:
                mags_bp[k] = 1.0
        mags_bp[0] = N / 2.0
        candidates.append(mags_bp)
    
    # 5. Cosine-tapered spectrum
    for beta in [0.5, 1.0, 2.0, 5.0]:
        mags_cos = np.zeros(N)
        for k in range(N):
            freq = min(k, N-k) / N
            mags_cos[k] = np.cos(np.pi * beta * freq) ** 2
        mags_cos = np.clip(mags_cos, 0, 1)
        mags_cos[0] = N / 2.0
        candidates.append(mags_cos)
    
    # 6. Random smooth spectra
    rng = np.random.RandomState(42)
    for _ in range(20):
        mags_rand = np.abs(rng.normal(0, 1, N))
        # Smooth in frequency domain
        M = np.fft.fft(mags_rand)
        M[np.abs(np.fft.fftfreq(N)) > 0.2] *= 0.1
        mags_rand = np.real(np.fft.ifft(M))
        mags_rand = np.clip(mags_rand, 0, 1)
        mags_rand[0] = N / 2.0
        candidates.append(mags_rand)
    
    return candidates


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
    
    # Generate spectral candidates
    spectral_candidates = _construct_spectral_candidates()
    
    # Evaluate spectral candidates and optimize phases
    spectral_results = []
    for mags in spectral_candidates:
        if time.time() > deadline - 40:
            break
        
        # Initial phases: random
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        phases0 = rng.uniform(0, 2 * np.pi, N//2)
        
        # Optimize phases using gradient ascent
        remaining = deadline - time.time() - 20
        if remaining < 5:
            break
        
        phases_opt = _optimize_spectral_grad(mags, phases0, time_limit=min(8.0, remaining))
        
        # Convert to time domain and project
        h = _spectral_to_h(mags, phases_opt)
        h = _project_to_feasible(h)
        
        score = _max_overlap_fast(h)
        spectral_results.append((score, h))
        
        if score < best_score:
            best_score = score
            best_h = h
    
    # Sort by score and take top candidates
    spectral_results.sort(key=lambda x: x[0])
    top_spectral = [h for _, h in spectral_results[:5]]
    
    # Also generate some traditional candidates for diversity
    traditional_candidates = []
    
    # Sine waves
    for freq in [1, 2, 3, 5, 7, 10, 13, 17, 20, 25, 30, 40, 50]:
        for amp in [0.3, 0.4, 0.45, 0.48]:
            x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
            h = 0.5 + amp * np.sin(x)
            h = _project_to_feasible(h)
            traditional_candidates.append(h)
    
    # Binary candidates
    for seed in range(20):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(N)
        h = np.zeros(N)
        h[idx[:N//2]] = 1.0
        traditional_candidates.append(h)
    
    # Evaluate traditional candidates
    for h in traditional_candidates:
        if time.time() > deadline - 20:
            break
        score = _max_overlap_fast(h)
        if score < best_score:
            best_score = score
            best_h = h
    
    # Local search from best candidates
    all_starts = [best_h] + top_spectral[:3]
    for h0 in all_starts:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        h_opt, score_opt = _local_search(
            h0, max_iters=3000,
            time_limit=min(10.0, remaining)
        )
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            # Try spectral optimization again from best
            H_best = np.fft.fft(best_h)
            mags_best = np.abs(H_best)
            phases_best = np.angle(H_best[:N//2])
            
            phases_opt2 = _optimize_spectral_grad(mags_best, phases_best, time_limit=min(5.0, remaining))
            h_final = _spectral_to_h(mags_best, phases_opt2)
            h_final = _project_to_feasible(h_final)
            score_final = _max_overlap_fast(h_final)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
    
    # Ensure the result is valid
    best_h = _project_to_feasible(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
