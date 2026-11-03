"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=_FFT_SIZE)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_FFT_SIZE)
    return np.real(acf[:N])


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _continuous_seed(rng, time_budget):
    """
    Generate a high-quality binary seed using continuous gradient descent
    on the L2 norm of the autocorrelation (flat spectrum optimization),
    followed by rounding.
    """
    # Start with random continuous values in [-1, 1]
    x = rng.uniform(-1, 1, size=N)
    
    # We want to minimize sum_k |A[k]|^2 for k=1..N-1
    # Gradient descent on x with projection to [-1, 1] or just let it go and round at the end.
    # Actually, for signed step functions, we usually fix magnitude to 1.
    # Let's optimize x in R^N to minimize ||A[1:]||_2^2, then round to signs.
    
    start = time.time()
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(x)
    
    while time.time() - start < time_budget:
        # Compute autocorrelation
        A = _calc_acf(x)
        # Gradient of sum(A[k]^2) w.r.t x
        # A[k] = sum_i x[i] x[i+k]
        # d/dx[j] A[k] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
        # Sum over k=1..N-1 A[k] * d/dx[j] A[k] * 2
        
        grad = np.zeros_like(x)
        for k in range(1, N):
            val = 2.0 * A[k]
            # x[j+k] term: j = 0..N-1-k -> grad[j] += val * x[j+k]
            # x[j-k] term: j = k..N-1 -> grad[j] += val * x[j-k]
            
            # Vectorized update
            # grad[0:N-k] += val * x[k:N]
            grad[:N-k] += val * x[k:]
            # grad[k:N] += val * x[0:N-k]
            grad[k:] += val * x[:N-k]
        
        # Update with momentum
        v = momentum * v - lr * grad
        x = x + v
        
        # Optional: project to keep values bounded, though rounding at end handles it.
        # Clipping helps prevent divergence.
        x = np.clip(x, -2, 2)
        
        if int(time.time() * 10) % 10 == 0: # Check every ~100ms roughly
             if time.time() - start > time_budget: break

    # Round to +/- 1
    h = np.sign(x)
    # Ensure no zeros (sign(0) is 0)
    h[h == 0] = 1.0
    
    return h


def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    # Use a better initial seed if possible, but _anneal is generic.
    # We will call _continuous_seed separately and pass the result as init if we modify _anneal, 
    # or just use _anneal starting from random for diversity, and use a dedicated loop for the continuous seed.
    
    # For now, keep _anneal simple, starting from random.
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-6
    start = time.time()
    
    while time.time() - start < time_budget:
        # Block flip move
        L = int(rng.integers(1, 12))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99995)
        
    return best_h, best


def _greedy_polish(h, time_budget, start):
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        # Single flip search
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i < 0:
            break
        h[best_i] *= -1.0
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def construct():
    try:
        hard_deadline = 135.0
        start = time.time()
        
        # 1. Generate a high-quality continuous seed
        rng_seed = np.random.default_rng(42)
        seed_budget = 10.0
        h_cont = _continuous_seed(rng_seed, seed_budget)
        
        # Evaluate continuous seed
        val_cont = _peak_sidelobe(h_cont)
        best_h = h_cont.copy()
        best_val = val_cont
        
        # 2. Anneal from this continuous seed
        # We need to modify _anneal or create a wrapper that starts from a given h.
        # Let's create a local anneal function that takes initial h.
        
        def _anneal_from(h_init, seed, time_budget):
            rng = np.random.default_rng(seed)
            h = h_init.copy()
            cur = _peak_sidelobe(h)
            local_best_h = h.copy()
            local_best = cur
            T = 1.0 # Start lower since we have a good init
            T_min = 1e-6
            start_a = time.time()
            
            while time.time() - start_a < time_budget:
                L = int(rng.integers(1, 12))
                i = int(rng.integers(0, N - L + 1))
                h[i:i + L] *= -1.0
                new = _peak_sidelobe(h)
                d = new - cur
                if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                    cur = new
                    if cur < local_best:
                        local_best = cur
                        local_best_h = h.copy()
                else:
                    h[i:i + L] *= -1.0
                T = max(T_min, T * 0.9999)
                
            return local_best_h, local_best

        # Anneal from continuous seed
        remaining = hard_deadline - (time.time() - start) - 5.0
        if remaining > 5.0:
            h_a, val_a = _anneal_from(h_cont, 12345, min(30.0, remaining))
            if val_a < best_val:
                best_val = val_a
                best_h = h_a

        # 3. Multiple restarts from random for diversity
        restart_id = 1
        while time.time() - start < hard_deadline - 5.0:
            sub = min(10.0, hard_deadline - (time.time() - start) - 5.0)
            if sub < 1.0:
                break
            # Use standard anneal from random
            h_r, val_r = _anneal(100000 + restart_id, sub)
            if val_r < best_val:
                best_val = val_r
                best_h = h_r
            restart_id += 1

        # 4. Final greedy polish
        remaining = hard_deadline - (time.time() - start)
        if remaining > 1.0:
            h_p, val_p = _greedy_polish(best_h, remaining, time.time())
            if val_p < best_val:
                best_val = val_p
                best_h = h_p

        return best_h

    except Exception:
        # Fallback to all ones if something fails
        return np.ones(N)
# EVOLVE-BLOCK-END
