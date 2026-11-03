import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT


def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _autocorr(h):
    return _autocorr_fast(h)


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score(h, beta):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.sum(np.exp(z))) / beta


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n])


def _analytic_flip_estimate(h, ac, i):
    """
    Estimate the new autocorrelation after flipping bit i without full FFT.
    """
    ac_new = ac.copy()
    if i < N:
        hi = h[i]
        k_vals = np.arange(1, N)
        
        valid_plus = (i + k_vals) < N
        h_plus = np.zeros(N - 1)
        if np.any(valid_plus):
            h_plus[valid_plus] = h[i + k_vals[valid_plus]]
            
        valid_minus = (i - k_vals) >= 0
        h_minus = np.zeros(N - 1)
        if np.any(valid_minus):
            h_minus[valid_minus] = h[i - k_vals[valid_minus]]
            
        delta = -2.0 * hi * (h_plus + h_minus)
        ac_new[1:] += delta
        
    return ac_new

def _analytic_block_flip_estimate(h, ac, flip_indices):
    """
    Estimate the new autocorrelation after flipping a set of bits.
    """
    ac_new = ac.copy()
    if len(flip_indices) == 0:
        return ac_new
        
    k_vals = np.arange(1, N)
    
    for i in flip_indices:
        i = int(i)
        hi = h[i]
        
        valid_plus = k_vals < (N - i)
        h_plus = np.zeros(N - 1)
        if np.any(valid_plus):
            h_plus[valid_plus] = h[i + k_vals[valid_plus]]
            
        valid_minus = k_vals <= i
        h_minus = np.zeros(N - 1)
        if np.any(valid_minus):
            h_minus[valid_minus] = h[i - k_vals[valid_minus]]
            
        delta = -2.0 * hi * (h_plus + h_minus)
        ac_new[1:] += delta
        
    return ac_new

def _spectral_gradient_step(h, ac, lr):
    """
    Spectral gradient descent step.
    Minimizes the peak magnitude of the autocorrelation sidelobes.
    This corresponds to flattening the spectral peaks.
    """
    # F = FFT(h)
    F = np.fft.rfft(h, n=_NFFT)
    
    # Gradient of max(|ac|) w.r.t. h
    # ac = ifft(|F|^2)
    # d ac / d h = ifft(2 * F * conj(F) * d/dh log(F) ... ) 
    # Simpler: use subgradient of max norm.
    # Let S = |ac[1:]| / ac[0]
    # We want to minimize max(S).
    # Gradient of max(S) is concentrated on the index of the max.
    # But we can use a smoothed version.
    
    # Let's try a simple heuristic:
    # If ac[k] is large positive, we want to reduce it.
    # Flipping h[i] changes ac by -2 h[i] (h[i+k] + h[i-k]).
    # This is essentially what the greedy search does.
    # However, spectral view:
    # |F(e^jw)|^2 is the PSD.
    # We want to minimize max_w |F(e^jw)|^2 subject to h[i] in {-1, 1}.
    # This is the "Golay sequence" or "low autocorrelation" problem.
    
    # A known gradient for this is related to the "Welch bound" or similar.
    # Let's just use the analytic flip estimate to find the best single flip
    # that reduces the spectral peak (which is the same as autocorrelation peak).
    
    # Actually, the current program already does this effectively in Phase 1.
    # The "spectral" idea might suggest looking at the FFT directly to identify
    # which frequencies are dominant and then flipping bits to cancel them.
    
    # Let's identify the dominant frequency bin in the FFT.
    # |F|^2 = PSD.
    # We want to minimize max(PSD).
    # This is equivalent to minimizing max(ac) if we consider the peak of the autocorrelation.
    # Wait, the peak of the autocorrelation is at lag 0 (dc).
    # The sidelobes are the rest.
    # Minimizing max(|ac[1:]|) is the goal.
    
    # Let's stick to the time-domain greedy which is proven to work well for this.
    # The "spectral" insight might be that we should prioritize flipping bits
    # that have large amplitude in the dominant frequency bins.
    
    # For now, return None to indicate no specific spectral step, 
    # or implement a simple one if time permits.
    # Given the constraints, the analytic flip is likely more robust.
    
    return None

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0
        rng = np.random.default_rng(20240517)

        idx = np.arange(N)
        inits = [
            _thue_morse(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr(h)
        dc = ac[0]

        # Phase 1: Greedy single-bit search
        max_iterations_phase1 = 50000
        for _iter in range(max_iterations_phase1):
            if time.time() > deadline - 30.0:
                break
            
            if _iter % 50 == 0:
                ac = _autocorr(h)
                dc = ac[0]
                if dc <= 1e-30:
                    break
                cur_hard = float(np.max(np.abs(ac)) / dc)
            
            sidelobes = np.abs(ac[1:])
            k_star = int(np.argmax(sidelobes)) + 1
            c_star = ac[k_star]
            
            max_i = N - k_star
            indices = np.arange(max_i)
            products = h[indices] * h[indices + k_star]
            
            if c_star > 0:
                candidates = indices[products > 0]
            else:
                candidates = indices[products < 0]
            
            if len(candidates) == 0:
                i = int(rng.integers(0, N))
                ac_new = _analytic_flip_estimate(h, ac, i)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
                if new_hard < cur_hard:
                    h[i] *= -1.0
                    ac = ac_new
                    cur_hard = new_hard
                    if cur_hard < best_hard:
                        best_hard = cur_hard
                        best_h = h.copy()
                continue

            num_tests = min(50, len(candidates))
            if len(candidates) > num_tests:
                test_indices = rng.choice(candidates, num_tests, replace=False)
            else:
                test_indices = candidates
            
            best_candidate = None
            best_candidate_score = cur_hard
            best_candidate_ac = None
            
            for i in test_indices:
                ac_new = _analytic_flip_estimate(h, ac, int(i))
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
                if new_hard < best_candidate_score:
                    best_candidate_score = new_hard
                    best_candidate = int(i)
                    best_candidate_ac = ac_new
            
            if best_candidate is not None and best_candidate_score < cur_hard:
                h[best_candidate] *= -1.0
                ac = best_candidate_ac
                cur_hard = best_candidate_score
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
            else:
                if rng.random() < 0.3:
                    i = int(rng.integers(0, N))
                    ac_new = _analytic_flip_estimate(h, ac, i)
                    new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
                    if new_hard < cur_hard:
                        h[i] *= -1.0
                        ac = ac_new
                        cur_hard = new_hard
                        if cur_hard < best_hard:
                            best_hard = cur_hard
                            best_h = h.copy()

        # Phase 2: Simulated Annealing with block flips
        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr(h)
        
        T_start = 0.5
        T_end = 0.01
        cooling_rate = 0.995
        max_sa_iterations = 20000
        
        T = T_start
        for _iter in range(max_sa_iterations):
            if time.time() > deadline - 10.0:
                break
            
            block_size = int(2 ** rng.integers(0, 5))
            if block_size > N:
                block_size = N
            
            flip_indices = rng.choice(N, size=block_size, replace=False)
            
            ac_new = _analytic_block_flip_estimate(h, ac, flip_indices)
            new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
            
            if new_hard < cur_hard or rng.random() < np.exp(-(new_hard - cur_hard) / T):
                h[flip_indices] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
            
            T *= cooling_rate
            if T < T_end:
                T = T_end

        # Phase 3: High-precision polish
        h = best_h.copy()
        for _pass in range(200):
            if time.time() > deadline - 0.5:
                break
            any_improve = False
            ac = _autocorr(h)
            dc = ac[0]
            if dc <= 1e-30:
                break
            for i in range(N):
                ac_new = _analytic_flip_estimate(h, ac, i)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
                if new_hard < best_hard - 1e-12:
                    h[i] *= -1.0
                    ac = ac_new
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
            if not any_improve:
                break

        final_hard = _hard_score(best_h)
        if final_hard < best_hard:
            best_hard = final_hard

        return best_h
    except Exception:
        return np.ones(N)
