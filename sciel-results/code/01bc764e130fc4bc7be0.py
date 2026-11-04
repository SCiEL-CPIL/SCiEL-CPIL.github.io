"""
Rewritten program: C3' minimization via spectral-domain optimization.
Key structural change: work in the frequency domain where the autoconvolution
becomes a pointwise product, enabling a differentiable surrogate objective.
"""
import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _fft_size(n):
    return 1 << (2 * n - 1).bit_length()


def _score_offdiag(x):
    """max(|autoconv[k]| for k!=0) / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e12
    size = _fft_size(len(x))
    H = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(H * H, n=size)[:2 * len(x) - 1]
    if len(ac) <= 1:
        return 0.0
    return np.max(np.abs(ac[1:])) / (s * s)


def _score_full(x):
    """max(|autoconv[k]| for all k) / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e12
    size = _fft_size(len(x))
    H = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(H * H, n=size)[:2 * len(x) - 1]
    return np.max(np.abs(ac)) / (s * s)


def _soft_max_offdiag(x, beta=40.0):
    """Smoothed off-diagonal metric for gradient-based optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    size = _fft_size(len(x))
    H = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(H * H, n=size)[:2 * len(x) - 1]
    if len(ac) <= 1:
        return 0.0
    abs_c = np.abs(ac[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)


def _param_to_signal(params):
    """
    Convert spectral parameters to time-domain signal.
    params: [c0, c1, d1, c2, d2, ..., cM, dM]
    x[n] = c0 + sum_{k=1}^{M} c_k*cos(2*pi*k*n/N) + d_k*sin(2*pi*k*n/N)
    """
    M = len(params) // 2
    c0 = params[0]
    x = np.full(N, c0)
    n = np.arange(N)
    for k in range(1, M + 1):
        c_k = params[2 * k - 1]
        d_k = params[2 * k]
        x += c_k * np.cos(2 * np.pi * k * n / N) + d_k * np.sin(2 * np.pi * k * n / N)
    return x


def _signal_to_params(x):
    """Extract spectral coefficients from a signal."""
    n = np.arange(N)
    c0 = np.mean(x)
    M = 50
    params = np.zeros(2 * M + 1)
    params[0] = c0
    for k in range(1, M + 1):
        params[2 * k - 1] = np.dot(x, np.cos(2 * np.pi * k * n / N)) / (N / 2)
        params[2 * k] = np.dot(x, np.sin(2 * np.pi * k * n / N)) / (N / 2)
    return params


def _spectral_de(x_init, deadline, rng):
    """
    Differential evolution in spectral parameter space.
    The soft-max surrogate is smooth and amenable to DE.
    """
    M = 40
    n_params = 2 * M + 1
    params_init = _signal_to_params(x_init)
    
    bounds = [(-30, 30)] * n_params
    
    def objective(params):
        x = _param_to_signal(params)
        return _soft_max_offdiag(x, beta=35.0)
    
    try:
        result = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
            maxiter=80,
            popsize=15,
            tol=1e-7,
            seed=int(rng.integers(0, 2**31)),
            polish=True,
            updating='immediate',
            init='sobol'
        )
        if result.fun < 1e9:
            x_opt = _param_to_signal(result.x)
            return x_opt, _score_offdiag(x_opt)
    except Exception:
        pass
    return x_init, _score_offdiag(x_init)


def _spectral_l_bfgs(x_init, deadline, rng):
    """L-BFGS-B refinement in spectral space."""
    M = 50
    n_params = 2 * M + 1
    params = _signal_to_params(x_init)
    
    def objective(params):
        x = _param_to_signal(params)
        return _soft_max_offdiag(x, beta=50.0)
    
    try:
        result = scipy.optimize.minimize(
            objective,
            params,
            method='L-BFGS-B',
            bounds=[(-30, 30)] * n_params,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.fun < 1e9:
            x_opt = _param_to_signal(result.x)
            return x_opt, _score_offdiag(x_opt)
    except Exception:
        pass
    return x_init, _score_offdiag(x_init)


def _bit_flip_local_search(x, current_score, best_score, best_h, deadline, rng, n):
    """
    Greedy single-bit flip search with peak-targeting.
    Works on the discrete ±1 signal.
    """
    size = _fft_size(n)
    improved = True
    passes = 0
    max_passes = 30
    
    while improved and time.time() < deadline and passes < max_passes:
        improved = False
        passes += 1
        
        # Compute current autoconvolution to find peak
        H = np.fft.rfft(x, size)
        ac = np.fft.irfft(H * H, size)[:2 * n - 1]
        if len(ac) > 1:
            peak_idx = np.argmax(np.abs(ac[1:])) + 1
        else:
            peak_idx = 0
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            x[i] = -x[i]
            
            s = np.sum(x)
            if abs(s) < 1e-12:
                new_score = np.inf
            else:
                H = np.fft.rfft(x, size)
                ac_new = np.fft.irfft(H * H, size)[:2 * n - 1]
                if len(ac_new) > 1:
                    max_abs = np.max(np.abs(ac_new[1:]))
                else:
                    max_abs = np.max(np.abs(ac_new))
                new_score = max_abs / (s * s)
            
            if new_score <= current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_h = x.copy()
            else:
                x[i] = -x[i]
    
    return x, current_score, best_score, best_h


def _chunk_flip_search(x, current_score, best_score, best_h, deadline, rng, n):
    """
    Chunk-based local search: flip contiguous segments.
    """
    size = _fft_size(n)
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Compute peak for targeting
        H = np.fft.rfft(x, size)
        ac = np.fft.irfft(H * H, size)[:2 * n - 1]
        if len(ac) > 1:
            peak_idx = np.argmax(np.abs(ac[1:])) + 1
        else:
            peak_idx = 0
        
        # Choose chunk size
        r = rng.random()
        if r < 0.3:
            chunk_sz = 1
        elif r < 0.5:
            chunk_sz = rng.choice([2, 3, 4])
        elif r < 0.7:
            chunk_sz = rng.choice([5, 8, 12])
        else:
            chunk_sz = rng.choice([16, 24, 32, 48])
        
        # Targeted or random position
        if rng.random() < 0.5:
            target = peak_idx % n
            spread = max(chunk_sz, n // 10)
            start = int(rng.integers(max(0, target - spread), min(n - chunk_sz, target + spread) + 1))
        else:
            start = int(rng.integers(0, max(1, n - chunk_sz + 1)))
        
        start = max(0, min(start, n - chunk_sz))
        end = start + chunk_sz
        
        old = x[start:end].copy()
        x[start:end] = -old
        
        s = np.sum(x)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(x, size)
            ac_new = np.fft.irfft(H * H, size)[:2 * n - 1]
            if len(ac_new) > 1:
                max_abs = np.max(np.abs(ac_new[1:]))
            else:
                max_abs = np.max(np.abs(ac_new))
            new_score = max_abs / (s * s)
        
        if new_score <= current_score:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_h = x.copy()
        else:
            x[start:end] = old
            no_improve += 1
    
    return x, current_score, best_score, best_h


def _generate_candidates(rng):
    """Generate diverse initial candidates."""
    candidates = []
    
    # Windows
    candidates.append(np.hanning(N))
    candidates.append(np.hamming(N))
    candidates.append(np.bartlett(N))
    candidates.append(np.blackman(N))
    
    # M-sequences
    for k in [6, 7, 8]:
        candidates.append(_m_sequence(N, k))
    
    # Block patterns
    for bs in [4, 8, 16, 32, 64]:
        for nf in [0.0, 0.25, 0.5]:
            candidates.append(_block_pattern(N, bs, nf))
    
    # Random ±1
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Sinusoidal
    for freq in [1, 2, 3, 5, 7]:
        candidates.append(np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N)))
    
    # Walsh-Hadamard
    for level in range(5):
        h = np.ones(N)
        for i in range(N):
            val = 1.0
            for bit in range(level + 1):
                if (i >> bit) & 1:
                    val *= -1.0
            h[i] = val
        candidates.append(h)
    
    # Gaussian
    x = np.arange(N)
    sigma = N / 4
    candidates.append(np.exp(-0.5 * ((x - N / 2) / sigma) ** 2))
    
    # Sinc-like
    candidates.append(np.sinc(np.arange(N) / 10.0 - N / 20.0))
    
    return candidates


def _m_sequence(n, k):
    """Generate m-sequence."""
    polys = {
        4: [3, 0],
        5: [4, 3, 0],
        6: [5, 3, 0],
        7: [6, 5, 3, 0],
        8: [7, 5, 3, 2, 0],
    }
    if k not in polys:
        return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
    
    taps = polys[k]
    state = [0] * k
    state[0] = 1
    seq = []
    for _ in range(n):
        fb = 0
        for t in taps:
            fb ^= state[t]
        for i in range(k - 1, 0, -1):
            state[i] = state[i - 1]
        state[0] = fb
        seq.append(1.0 if state[0] else -1.0)
    return np.array(seq, dtype=np.float64)


def _block_pattern(n, block_size, neg_frac):
    """Alternating block pattern."""
    h = np.zeros(n, dtype=np.float64)
    for i in range(n):
        block_idx = i // block_size
        pos_in_block = i % block_size
        n_neg = int(neg_frac * block_size)
        if block_idx % 2 == 0:
            h[i] = -1.0 if pos_in_block < n_neg else 1.0
        else:
            h[i] = 1.0 if pos_in_block < n_neg else -1.0
    return h


def _multi_restart(x_best, score_best, deadline, rng, n):
    """Multiple restarts from perturbed best solution."""
    current_best = score_best
    current_best_h = x_best.copy()
    
    remaining = deadline - time.time()
    if remaining < 5:
        return current_best_h, current_best
    
    num_restarts = max(1, int(remaining / 8))
    per_restart = remaining / num_restarts
    
    for _ in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        h = current_best_h.copy()
        flip_frac = rng.choice([0.05, 0.1, 0.15, 0.2, 0.3])
        n_flips = int(flip_frac * n)
        flip_indices = rng.choice(n, size=n_flips, replace=False)
        h[flip_indices] = -h[flip_indices]
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        size = _fft_size(n)
        H = np.fft.rfft(h, size)
        ac = np.fft.irfft(H * H, size)[:2 * n - 1]
        if len(ac) > 1:
            max_abs = np.max(np.abs(ac[1:]))
        else:
            max_abs = np.max(np.abs(ac))
        score = max_abs / (s * s)
        
        if score > current_best:
            h = current_best_h.copy()
            score = current_best
        
        restart_deadline = min(time.time() + per_restart * 0.8, deadline - 2)
        h, score, r_best, r_best_h = _chunk_flip_search(
            h, score, current_best, current_best_h.copy(), restart_deadline, rng, n
        )
        
        if r_best < current_best:
            current_best = r_best
            current_best_h = r_best_h
    
    return current_best_h, current_best


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill
    rng = np.random.default_rng(12345)
    n = N
    
    # Phase 0: Generate and evaluate diverse candidates
    candidates = _generate_candidates(rng)
    best_score = np.inf
    best_h = np.ones(N)
    
    for h in candidates:
        if time.time() > deadline - 80:
            break
        s = _score_offdiag(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Spectral DE optimization
    if time.time() < deadline - 60:
        spec_deadline = min(time.time() + 35, deadline - 50)
        best_h, best_score = _spectral_de(best_h, spec_deadline, rng)
    
    # Phase 2: L-BFGS refinement in spectral space
    if time.time() < deadline - 40:
        lbfgs_deadline = min(time.time() + 20, deadline - 30)
        best_h, best_score = _spectral_l_bfgs(best_h, lbfgs_deadline, rng)
    
    # Phase 3: Discrete chunk-flip search
    if time.time() < deadline - 25:
        chunk_deadline = min(time.time() + 25, deadline - 15)
        best_h, best_score, best_score, best_h = _chunk_flip_search(
            best_h.copy(), best_score, best_score, best_h.copy(), chunk_deadline, rng, n
        )
    
    # Phase 4: Multi-restart
    if time.time() < deadline - 15:
        best_h, best_score = _multi_restart(best_h, best_score, deadline - 8, rng, n)
    
    # Phase 5: Final bit-flip refinement
    if time.time() < deadline - 5:
        best_h, best_score, best_score, best_h = _bit_flip_local_search(
            best_h.copy(), best_score, best_score, best_h.copy(), deadline - 2, rng, n
        )
    
    # Final safety check
    if time.time() > deadline - 1:
        pass  # Just return what we have
    
    return best_h

# EVOLVE-BLOCK-END
