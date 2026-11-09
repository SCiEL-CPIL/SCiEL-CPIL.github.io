import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    def compute_sidelobe_norm(seq):
        """Compute L2 norm of off-center autocorrelation sidelobes."""
        n = len(seq)
        size = 1
        while size < 2*n:
            size *= 2
        f = np.zeros(size, dtype=np.complex128)
        f[:n] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac = np.fft.ifft(F * np.conj(F)).real[:n]
        # L2 norm of lags 1..n-1
        return np.linalg.norm(ac[1:])
    
    def max_autocorr(seq):
        """Compute max absolute autocorrelation for final scoring."""
        n = len(seq)
        size = 1
        while size < 2*n:
            size *= 2
        f = np.zeros(size, dtype=np.complex128)
        f[:n] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac = np.fft.ifft(F * np.conj(F)).real[:n]
        return np.max(np.abs(ac[1:]))
    
    rng = np.random.default_rng(42)
    
    # Strategy: Projected gradient descent on continuous signal, then threshold to ±1
    # Initialize with smooth windowed sinusoid for low-sidelobe starting point
    t = np.linspace(0, 1, N, endpoint=False)
    # Use a few different frequency initializations
    best_a = None
    best_score = np.inf
    
    for freq in [1, 2, 3, 5, 7]:
        if time.time() > deadline:
            break
        
        # Initialize continuous signal with windowed sinusoid
        window = np.hanning(N)
        x = window * np.sin(2 * np.pi * freq * t)
        # Add small random perturbation
        x += 0.1 * rng.standard_normal(N)
        
        # Projected gradient descent to minimize L2 norm of sidelobes
        # The objective is ||ac[1:]||^2 where ac is the autocorrelation
        # Gradient with respect to x can be computed via FFT
        
        lr = 0.1
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Compute autocorrelation
            size = 1
            while size < 2*N:
                size *= 2
            f = np.zeros(size, dtype=np.complex128)
            f[:N] = x
            F = np.fft.fft(f)
            ac = np.fft.ifft(F * np.conj(F)).real[:N]
            
            # Objective: sum of squares of ac[1:]
            # Gradient of (1/2)||ac[1:]||^2 with respect to x
            # ac[k] = sum_i x[i]x[i+k], so d(ac[k])/dx[j] = x[j+k] + x[j-k] (with wraparound for periodic)
            # For non-periodic, it's more complex. Let's use the fact that:
            # ac = ifft(fft(x) * conj(fft(x)))
            # d(ac)/dx = 2 * ifft(fft(x) * conj(fft(x))) ... no, that's not right.
            
            # Actually, let's use a simpler approach: the gradient of ||ac[1:]||^2 
            # can be approximated by correlating the sidelobes with shifted versions of x.
            
            # Alternative: use the fact that for the periodic case, 
            # the gradient of (1/2) sum_k |ac[k]|^2 is related to the inverse FFT of |F|^2 * F
            # But we want non-periodic. Let's just do coordinate descent or a simpler method.
            
            # Simpler: try flipping individual elements based on gradient-like heuristic
            # For each element j, the change in sidelobe norm when flipping x[j]
            # is approximately related to how x[j] correlates with the sidelobes.
            
            # Let's try a different approach: use the current x to compute a "pressure" 
            # for each position, then update x[j] -= lr * pressure[j], then project to [-1, 1]
            
            # Pressure: for each j, compute sum over k of ac[k] * (x[j+k] + x[j-k]) 
            # This is the gradient of the autocorrelation energy with respect to x[j]
            pressure = np.zeros(N)
            for k in range(1, N):
                # ac[k] = sum_i x[i]*x[i+k]
                # d(ac[k])/dx[j] = x[j+k] + x[j-k] (with appropriate bounds)
                # For non-periodic, we need to be careful
                idx_pos = np.arange(0, N-k)
                idx_neg = np.arange(k, N)
                # Contribution to pressure from positive lag k
                pressure[idx_pos] += ac[k] * x[idx_neg]
                # Contribution from negative lag -k (same as positive due to symmetry)
                pressure[idx_neg] += ac[k] * x[idx_pos]
            
            # Update x
            x -= lr * pressure
            # Project to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
        
        # Threshold to get ±1 sequence
        a = np.sign(x)
        a[a == 0] = 1  # handle zeros
        
        # Evaluate
        score = max_autocorr(a)
        if score < best_score:
            best_score = score
            best_a = a.copy()
        
        # Local search refinement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                best_a[idx] *= -1
                new_score = max_autocorr(best_a)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_a[idx] *= -1
    
    # If we haven't found a good solution, fall back to random + local search
    if best_a is None or time.time() > deadline:
        a = rng.choice([-1, 1], size=N)
        best_a = a.copy()
        best_score = max_autocorr(a)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                best_a[idx] *= -1
                new_score = max_autocorr(best_a)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_a[idx] *= -1
    
    return best_a
