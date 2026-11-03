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
    return m + float(np.log(np.sum(np.exp(z)))) / beta


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
        # This is fast and finds a good local minimum quickly
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

            # Vectorized evaluation of all candidates
            # To keep memory and time bounded, we limit the number of candidates to test
            max_vec_tests = 100
            if len(candidates) > max_vec_tests:
                test_indices = rng.choice(candidates, max_vec_tests, replace=False)
            else:
                test_indices = candidates
            
            if len(test_indices) == 0:
                continue

            h_i = h[test_indices].reshape(-1, 1)
            k_vals = np.arange(1, N).reshape(1, -1)
            
            idx_plus = test_indices[:, None] + k_vals
            idx_minus = test_indices[:, None] - k_vals
            
            h_plus_val = np.zeros((len(test_indices), N - 1))
            valid_plus = (idx_plus < N)
            flat_idx_plus = idx_plus[valid_plus]
            h_plus_val[valid_plus] = h[flat_idx_plus]
            
            h_minus_val = np.zeros((len(test_indices), N - 1))
            valid_minus = (idx_minus >= 0)
            flat_idx_minus = idx_minus[valid_minus]
            h_minus_val[valid_minus] = h[flat_idx_minus]
            
            delta = -2.0 * h_i * (h_plus_val + h_minus_val)
            ac_sidelobes = ac[1:].reshape(1, -1)
            new_ac_sidelobes = ac_sidelobes + delta
            
            new_dc = ac[0] - 2.0
            if new_dc <= 1e-30:
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

            max_sidelobes = np.max(np.abs(new_ac_sidelobes), axis=1)
            scores = max_sidelobes / new_dc
            
            best_local_idx = int(np.argmin(scores))
            best_local_score = scores[best_local_idx]
            
            if best_local_score < cur_hard:
                best_candidate = int(test_indices[best_local_idx])
                ac_new = _analytic_flip_estimate(h, ac, best_candidate)
                h[best_candidate] *= -1.0
                ac = ac_new
                cur_hard = best_local_score
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
        # We combine the speed of analytic estimation (from current) with
        # the block flip structure (from inspiration 1) to escape local minima.
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
            
            # Use varied block sizes
            block_size = int(2 ** rng.integers(0, 5))  # 1, 2, 4, 8, 16
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
        # This is a "final high-precision polish pass" as suggested by the directive.
        # We use exact FFT recomputation to ensure no drift, and iterate until convergence.
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
