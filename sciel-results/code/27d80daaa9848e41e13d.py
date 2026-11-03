import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT

def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    # h is expected to be +1 or -1
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac

def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # We want to minimize max(|ac[k]| for k>0) / ac[0]
    # Note: ac[0] is sum h[i]^2 = N for binary +/-1
    return float(np.max(np.abs(ac[1:])) / dc)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n], dtype=np.float64)

def _analytic_flip_estimate(h, ac, i):
    """
    Estimate the new autocorrelation after flipping bit i without full FFT.
    ac is the current autocorrelation.
    """
    ac_new = ac.copy()
    if i < N:
        hi = h[i]
        # ac_new[k] = ac[k] - 2 * h[i] * (h[i+k] + h[i-k]) for valid k
        # We only care about k=1..N-1 for the score, but we update all to keep ac consistent
        # Actually, we can just compute the delta for the sidelobes if we want to be fast,
        # but for simplicity and correctness in the SA loop, we update the whole array.
        
        # Vectorized update for ac_new[1:]
        k_vals = np.arange(1, N)
        
        # h_plus[k] = h[i+k] if i+k < N else 0
        # h_minus[k] = h[i-k] if i-k >= 0 else 0
        
        h_plus = np.zeros(N - 1)
        valid_plus = k_vals < (N - i)
        if np.any(valid_plus):
            h_plus[valid_plus] = h[i + k_vals[valid_plus]]
            
        h_minus = np.zeros(N - 1)
        valid_minus = k_vals <= i
        if np.any(valid_minus):
            h_minus[valid_minus] = h[i - k_vals[valid_minus]]
            
        delta = -2.0 * hi * (h_plus + h_minus)
        ac_new[1:] += delta
        
        # ac_new[0] is always N, so it doesn't change.
    return ac_new

def _analytic_block_flip_estimate(h, ac, flip_indices):
    """
    Estimate the new autocorrelation after flipping a set of bits.
    This is less efficient than single bit, so we use it sparingly or for small blocks.
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
        deadline = t0 + 115.0  # Safety margin for 150s limit
        rng = np.random.default_rng(42)

        idx = np.arange(N)
        
        # 1. Initialization
        # Try several structured and random starts
        inits = [
            _thue_morse(N),
            np.ones(N),
            -np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N) * 10 > 0 # Just a deterministic pattern
        ]
        # Add some random chirps
        inits.append(np.sign(np.cos(2 * np.pi * idx**2 / (2*N))))
        inits.append(np.sign(np.cos(2 * np.pi * idx**2 / (N))))
        
        best_h = None
        best_hard = 1e9
        for h in inits:
            h = h.astype(np.float64)
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr_fast(h)

        # 2. Greedy Phase: Exhaustive single-bit search
        # We iterate until no improvement is found or time runs out.
        # Each pass is O(N^2). N=400, so 160k ops per pass. Very fast.
        max_greedy_passes = 100
        for p in range(max_greedy_passes):
            if time.time() > deadline - 60.0: # Reserve time for SA
                break
                
            improved = False
            # Evaluate all possible single-bit flips
            best_i = -1
            best_score_pass = cur_hard
            best_ac_pass = None
            
            # To speed this up, we can vectorize the evaluation of all i at once?
            # ac_new_i = ac - 2*h[i]*(h[i+k]+h[i-k])
            # This is tricky to vectorize fully for the max operation.
            # So we loop i, but use analytic update.
            
            for i in range(N):
                ac_new = _analytic_flip_estimate(h, ac, i)
                # Score is max(abs(ac_new[1:])) / ac_new[0]
                # ac_new[0] is constant N
                max_side = np.max(np.abs(ac_new[1:]))
                score = max_side / N # dc is N
                
                if score < best_score_pass - 1e-12:
                    best_score_pass = score
                    best_i = i
                    best_ac_pass = ac_new
            
            if best_i != -1:
                h[best_i] *= -1.0
                ac = best_ac_pass
                cur_hard = best_score_pass
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
                improved = True
                
            if not improved:
                break

        # 3. Simulated Annealing Phase
        # Use block flips to escape local minima.
        # We will use analytic updates for speed, but recompute FFT periodically to avoid drift.
        
        T_start = 0.1
        T_end = 0.001
        cooling_rate = 0.99
        max_sa_iterations = 10000
        
        T = T_start
        ac = _autocorr_fast(h) # Reset ac to exact value before SA
        
        for _iter in range(max_sa_iterations):
            if time.time() > deadline - 10.0:
                break
                
            # Choose block size
            # Smaller blocks are more likely to be accepted but move slowly.
            # Larger blocks escape faster but are less likely to improve.
            # Let's mix them.
            r = rng.random()
            if r < 0.5:
                block_size = 1
            elif r < 0.8:
                block_size = 2
            elif r < 0.95:
                block_size = 4
            else:
                block_size = 8
            
            flip_indices = rng.choice(N, size=block_size, replace=False)
            
            ac_new = _analytic_block_flip_estimate(h, ac, flip_indices)
            new_hard = float(np.max(np.abs(ac_new[1:])) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
            
            # Acceptance criterion
            if new_hard < cur_hard or (T > 0 and rng.random() < np.exp(-(new_hard - cur_hard) / T)):
                h[flip_indices] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
                
            T *= cooling_rate
            if T < T_end:
                T = T_end
            
            # Periodic drift correction
            if _iter % 100 == 0:
                ac = _autocorr_fast(h)
                cur_hard = float(np.max(np.abs(ac[1:])) / ac[0])

        # 4. Final Polish
        # Run the greedy phase again on the best solution found.
        h = best_h.copy()
        ac = _autocorr_fast(h)
        cur_hard = _hard_score(h)
        
        for p in range(50):
            if time.time() > deadline - 1.0:
                break
            improved = False
            best_i = -1
            best_score_pass = cur_hard
            best_ac_pass = None
            
            for i in range(N):
                ac_new = _analytic_flip_estimate(h, ac, i)
                max_side = np.max(np.abs(ac_new[1:]))
                score = max_side / N
                
                if score < best_score_pass - 1e-12:
                    best_score_pass = score
                    best_i = i
                    best_ac_pass = ac_new
            
            if best_i != -1:
                h[best_i] *= -1.0
                ac = best_ac_pass
                cur_hard = best_score_pass
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
                improved = True
            else:
                break

        return best_h

    except Exception:
        return np.ones(N)
