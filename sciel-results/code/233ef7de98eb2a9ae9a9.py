"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr(x):
    """Compute autocorrelation of x using FFT."""
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf

def _score_from_acf(acf):
    """Upper bound on C3: peak sidelobe of autocorrelation."""
    # Exclude lag 0
    sidelobes = np.concatenate([acf[1:], acf[-1:][::-1]]) if len(acf) > 1 else np.array([0.0])
    # Actually for linear autocorrelation, lags 1..n-1 and their negatives
    # acf[k] for k=1..n-1, and acf[n-k] gives negative lags
    # Peak sidelobe = max(|acf[k]|) for k != 0
    return np.max(np.abs(acf[1:]))

def _continuous_seed(time_budget=10.0):
    """Use gradient descent on continuous relaxation to get good initial sign pattern."""
    deadline = time.time() + time_budget
    rng = np.random.default_rng(42)
    
    # Start with random continuous values in [-1, 1]
    x = rng.uniform(-1, 1, N)
    
    lr = 0.01
    best_x = x.copy()
    best_score = float('inf')
    
    while time.time() < deadline:
        acf = _autocorr(x)
        # Objective: sum of squares of sidelobes (L2 norm squared)
        # This encourages flat spectrum
        obj = np.sum(acf[1:]**2)
        
        if obj < best_score:
            best_score = obj
            best_x = x.copy()
        
        # Gradient of sum(acf[k]^2) w.r.t. x
        # acf = ifft(fft(x)^2).real
        # Use FFT-based gradient
        f = np.fft.fft(x, n=2*N)
        # d(acf)/dx involves circular correlation
        # Gradient of sum_k acf[k]^2 = 2 * acf * (d acf / dx)
        # d acf / dx is the circular autocorrelation matrix, which in frequency domain
        # means: grad = 2 * real(ifft(2 * conj(f) * f)) -- actually let's use a simpler approach
        
        # Simpler: use the fact that sum_k acf[k]^2 = (1/N) * sum |F[k]|^4 (Parseval-like)
        # Actually for the L2 norm of autocorrelation:
        # sum_k acf[k]^2 = (1/(2N)) * sum_k |F[k]|^4 where F is DFT of zero-padded x
        # Let's just compute gradient numerically-free via FFT
        
        # The gradient of sum_{k=1}^{N-1} acf[k]^2 with respect to x[j]
        # acf[k] = sum_i x[i] x[i+k] (for linear autocorrelation)
        # Using FFT: acf = ifft(fft(x, 2N) * conj(fft(x, 2N))).real[:N]
        
        f_full = np.fft.fft(x, n=2*N)
        # acf = ifft(|f_full|^2).real[:N]
        # d(acf[k])/dx[j] = delta-related, but in freq domain:
        # grad_x = 2 * real(ifft(f_full * conj(acf_freq)))
        # where acf_freq is the FFT of the sidelobe-weighted acf
        
        # Let's use a cleaner derivation:
        # Let S = sum_{k=1}^{N-1} acf[k]^2
        # acf = real(ifft(P)) where P = |F|^2, F = fft(x, 2N)
        # S = sum_{k=1}^{N-1} [real(ifft(P))_k]^2
        # This is getting complex. Let's use a practical gradient.
        
        # Practical approach: the gradient of the L2 norm of autocorrelation sidelobes
        # can be computed as: grad = 2 * (acf * x) circularly, but for linear it's different.
        
        # Let's just use: minimize sum |F[k]|^4 which is equivalent up to constants
        # d/dx sum |F[k]|^4 = 4 * real(ifft(conj(F) * |F|^2 * F)) = 4 * real(ifft(|F|^2 * conj(F)))
        # Wait: d/dx_j |F[k]|^4 = 4 |F[k]|^2 * d|F[k]|^2/dx_j
        # d|F[k]|^2/dx_j = 2 * real(conj(F[k]) * dF[k]/dx_j) = 2 * real(conj(F[k]) * exp(-2pi i k j / M))
        # So grad_j = sum_k 4 |F[k]|^2 * 2 * real(conj(F[k]) * exp(-2pi i k j / M))
        # = 8 * real(ifft(4 |F|^2 * conj(F)))... 
        # Actually: sum_k |F[k]|^2 * conj(F[k]) * exp(-2pi i k j/M) = ifft(|F|^2 * conj(F))_j
        # So grad = 8 * real(ifft(|F|^2 * conj(F)))
        
        M = 2*N
        P = np.abs(f_full)**2
        grad_freq = P * np.conj(f_full)
        grad = 8.0 * np.real(np.fft.ifft(grad_freq))[:N]
        
        # Gradient step
        x = x - lr * grad
        
        # Project to [-1, 1]
        x = np.clip(x, -1, 1)
        
        # Occasionally sharpen toward ±1
        if np.random.random() < 0.1:
            x = np.sign(x) * np.clip(np.abs(x) * 1.1, 0.5, 1.0)
    
    # Convert to signs
    signs = np.sign(best_x)
    signs[signs == 0] = 1
    return signs

def _local_search(signs, time_budget=30.0):
    """Local search by flipping individual bits to minimize peak sidelobe."""
    deadline = time.time() + time_budget
    best_signs = signs.copy()
    best_score = _score_from_acf(_autocorr(best_signs))
    
    rng = np.random.default_rng(123)
    n_iter = 0
    
    while time.time() < deadline:
        n_iter += 1
        # Try random flips
        idx = rng.integers(0, N)
        trial = best_signs.copy()
        trial[idx] = -trial[idx]
        score = _score_from_acf(_autocorr(trial))
        
        if score < best_score:
            best_signs = trial
            best_score = score
            # Try more flips in same direction
            for _ in range(5):
                if time.time() >= deadline:
                    break
                idx2 = rng.integers(0, N)
                trial2 = best_signs.copy()
                trial2[idx2] = -trial2[idx2]
                score2 = _score_from_acf(_autocorr(trial2))
                if score2 < best_score:
                    best_signs = trial2
                    best_score = score2
        elif n_iter % 1000 == 0:
            # Occasionally try multi-bit flips
            k = rng.integers(2, 6)
            indices = rng.choice(N, size=k, replace=False)
            trial = best_signs.copy()
            trial[indices] = -trial[indices]
            score = _score_from_acf(_autocorr(trial))
            if score < best_score:
                best_signs = trial
                best_score = score
    
    return best_signs, best_score

def _anneal(signs, time_budget=30.0):
    """Simulated annealing to escape local minima."""
    deadline = time.time() + time_budget
    current = signs.copy()
    current_score = _score_from_acf(_autocorr(current))
    best = current.copy()
    best_score = current_score
    
    T = 0.1
    T_min = 0.001
    rng = np.random.default_rng(456)
    
    while time.time() < deadline:
        idx = rng.integers(0, N)
        trial = current.copy()
        trial[idx] = -trial[idx]
        trial_score = _score_from_acf(_autocorr(trial))
        
        delta = trial_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = trial
            current_score = trial_score
            if current_score < best_score:
                best = trial.copy()
                best_score = current_score
        
        # Cool down
        T *= 0.999
        if T < T_min:
            T = T_min
    
    return best, best_score

def construct():
    start_time = time.time()
    total_deadline = start_time + 140.0  # Safety margin before 150s
    
    # Phase 1: Continuous seed
    seed_signs = _continuous_seed(time_budget=15.0)
    
    if time.time() > total_deadline:
        return seed_signs
    
    # Phase 2: Local search
    ls_signs, ls_score = _local_search(seed_signs, time_budget=40.0)
    
    if time.time() > total_deadline:
        return ls_signs
    
    # Phase 3: Annealing from best
    ann_signs, ann_score = _anneal(ls_signs, time_budget=40.0)
    
    if time.time() > total_deadline:
        return ann_signs
    
    # Phase 4: Final local search polish
    final_signs, final_score = _local_search(ann_signs, time_budget=30.0)
    
    if time.time() > total_deadline:
        return final_signs
    
    # Choose the best
    candidates = [
        (seed_signs, _score_from_acf(_autocorr(seed_signs))),
        (ls_signs, ls_score),
        (ann_signs, ann_score),
        (final_signs, final_score),
    ]
    
    best = min(candidates, key=lambda x: x[1])
    return best[0]

# EVOLVE-BLOCK-END
