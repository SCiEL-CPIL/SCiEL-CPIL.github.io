import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac

def _hard_score(h):
    """The actual metric: peak autoconvolution concentration (lower is better)."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac[1:])) / dc)

def _smooth_score(h, beta):
    """Smooth (log-sum-exp) surrogate of the peak sidelobe energy."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # Use absolute values of sidelobes
    y = np.abs(ac[1:]) / dc
    m = float(np.max(y))
    if m < 1e-30:
        return 0.0
    # Log-sum-exp trick for stability
    # sum(exp(beta * y)) is large, so we use m + (1/beta) * log(sum(exp(beta * (y-m))))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta

def _generate_chirp(seed_val):
    idx = np.arange(N)
    # Quadratic phase chirp
    h = np.cos(2 * np.pi * idx ** 2 / (2 * N))
    h[h == 0] = 1.0
    return np.sign(h)

def _continuous_relax_seed(rng):
    """
    Minimize the L2 norm of the sidelobes of a continuous signal, then threshold.
    """
    x = rng.standard_normal(N)
    mask = np.ones(L // 2 + 1)
    mask[0] = 0.0  # Zero out DC
    
    lr = 0.01
    # Gradient descent to minimize sum_{k=1}^{N-1} ac[k]^2
    # This is equivalent to minimizing sum_{f!=0} |X_f|^4 / L
    # Gradient of |X_f|^4 is 4 |X_f|^2 X_f
    for step in range(200):
        F = np.fft.rfft(x, n=L)
        P = F * np.conj(F)
        # Gradient in frequency domain
        grad_freq = 4.0 * P * F * mask
        grad_time = np.fft.irfft(grad_freq, n=L)[:N]
        
        x -= lr * grad_time
        
        # Normalize
        norm = np.linalg.norm(x)
        if norm > 0:
            x /= norm
            
    h = np.sign(x)
    h[h == 0] = 1.0
    return h

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0  # Safety margin
        rng = np.random.default_rng(_SEED)

        # 1. Initialization
        inits = [
            _generate_chirp(_SEED),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            _continuous_relax_seed(rng),
            np.ones(N),
            np.cos(2 * np.pi * np.arange(N) ** 2 / (N + 1)),
        ]
        
        best_h = None
        best_hard = 1e9
        for h in inits:
            h = np.sign(h)
            h[h == 0] = 1.0
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        # 2. Simulated Annealing
        h = best_h.copy()
        cur_hard = best_hard
        beta = 50.0  # Sharpness of LSE
        T = 0.05     # Initial temperature
        T_min = 1e-5
        decay = 0.9995
        
        # Block sizes to flip
        block_sizes = [1, 2, 3, 4, 5, 8, 16, 32, 64]
        
        no_improve_count = 0
        restart_threshold = 5000 # Restart if stuck for this many steps
        
        while time.time() < deadline - 20.0:
            # Adaptive beta: increase beta as we cool to focus on max peak
            # beta = 10.0 + (1.0 - T) * 40.0 
            beta = 50.0
            
            # Choose block size
            # Weight smaller blocks more heavily for polishing
            size = int(rng.choice(block_sizes, p=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))
            start = int(rng.integers(0, N - size + 1))
            sl = slice(start, start + size)
            
            old_block = h[sl].copy()
            h[sl] *= -1.0
            
            new_hard = _hard_score(h)
            
            # Calculate smooth score for acceptance
            # Only compute smooth score if hard score didn't improve, to save time?
            # Actually, we need smooth score for the Metropolis criterion if hard score worsens.
            if new_hard < cur_hard:
                cur_hard = new_hard
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
                    no_improve_count = 0
            else:
                # Hard score worsened or same. Check smooth score.
                cur_smooth = _smooth_score(h, beta)
                # We need the previous smooth score. To avoid recalculating, we could store it, 
                # but it's expensive. Let's approximate: if hard score is close, accept with prob exp(-dS/T).
                # If hard score is much worse, reject.
                # Simple heuristic: Accept if new_hard < cur_hard * 1.1 (allow some worsening)
                # Better: Use the smooth score difference.
                # Since we don't have prev_smooth cached easily in this loop structure without overhead,
                # let's use a heuristic based on hard score difference.
                # Delta hard = new_hard - cur_hard
                delta_h = new_hard - cur_hard
                # If delta_h is small, accept with probability exp(-delta_h / T)
                # Note: T is in units of the score.
                if rng.random() < np.exp(-delta_h / T):
                    cur_hard = new_hard
                else:
                    h[sl] = old_block
                    # No improvement
                    no_improve_count += 1
            
            T = max(T_min, T * decay)
            
            if no_improve_count > restart_threshold:
                # Restart from best
                h = best_h.copy()
                cur_hard = best_hard
                T = 0.05
                no_improve_count = 0
                # Randomize slightly to escape local min?
                # Or just restart SA loop.

        # 3. Intensive Local Search (Polish)
        h = best_h.copy()
        cur_hard = best_hard
        
        # Single bit polish
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                old_val = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < cur_hard - 1e-12:
                    cur_hard = new_hard
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val
        
        # Block polish (size 2, 4)
        for size in [2, 4, 8]:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 2.0:
                        break
                    sl = slice(start, start + size)
                    old_block = h[sl].copy()
                    h[sl] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < cur_hard - 1e-12:
                        cur_hard = new_hard
                        if new_hard < best_hard:
                            best_hard = new_hard
                            best_h = h.copy()
                        improved = True
                    else:
                        h[sl] = old_block
                        
        return best_h
    except Exception:
        return np.ones(N)
