import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    N = 100
    target_sum = N / 2.0
    
    # Precompute FFT size for zero-padding to avoid circular correlation artifacts
    # We need linear correlation, so pad to at least 2N
    fft_size = 1 << (2 * N - 1).bit_length()
    
    def get_correlations_fft(h):
        """
        Compute the linear correlation of g=1-h and h at all non-zero lags.
        Returns array of length 2(N-1) containing c_k for k != 0.
        """
        g = 1.0 - h
        # Zero pad
        H = np.fft.fft(h, n=fft_size)
        G = np.fft.fft(g, n=fft_size)
        # Correlation is IFFT( G * conj(H) ) ? 
        # np.correlate(g, h) computes sum g[n] h[n+k].
        # In frequency domain: IFFT( G * H^* ) gives circular correlation.
        # With zero padding, it gives linear correlation.
        # Let's verify indices.
        # c = np.fft.ifft(G * np.conj(H)).real
        # The peak is at index N-1 (lag 0) if we consider standard alignment?
        # Actually, for zero-padded signals of length N, the correlation 
        # is centered.
        
        c_full = np.fft.ifft(G * np.conj(H)).real
        # The linear correlation of two length N signals has length 2N-1.
        # The "full" mode in np.correlate puts lag 0 at index N-1.
        # FFT method: 
        # c_full[k] corresponds to lag k - (N-1)?
        # Let's just use np.correlate for N=100 as it's very fast and safe.
        # 100*100 = 10k ops. Even 10k iterations is 100M ops, which is fine in 120s.
        # Let's stick to np.correlate to avoid index errors, unless speed is critical.
        # N=100 is small.
        
        g_corr = np.correlate(g, h, mode='full')
        # g_corr indices:
        # 0 to N-2: negative lags
        # N-1: lag 0
        # N to 2N-2: positive lags
        c_nonzero = np.concatenate([g_corr[:N-1], g_corr[N:]])
        return c_nonzero

    def max_overlap(h):
        c = get_correlations_fft(h)
        return np.max(c)
    
    def smooth_max(c, temp):
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))
    
    def project(h):
        h = np.clip(h, 0, 1)
        for _ in range(50):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0, 1)
        return h
    
    def objective_lse(h, temp):
        c = get_correlations_fft(h)
        val = smooth_max(c, temp)
        sum_err = np.sum(h) - target_sum
        return val + 10.0 * sum_err**2

    # Generate initial candidates
    candidates = []
    
    # 1. Sturmian / Quasi-periodic
    def generate_sturmian(alpha, N):
        # Beatty sequence: floor(n*alpha) - floor((n-1)*alpha)
        n = np.arange(1, N + 1)
        # Use conjugate alpha to ensure density 0.5?
        # If we want sum N/2, we need a sequence with density 0.5.
        # Sturmian words have density alpha.
        # So we should use alpha = 0.5? That's trivial (101010).
        # We want a sequence that minimizes max overlap.
        # Let's try various irrationals and project.
        # Actually, the "step heights" can be continuous.
        # Let's initialize with a binary sequence derived from a rotation.
        # h_i = 1 if {i * alpha} < 0.5 else 0.
        # This gives density 0.5 exactly if we handle boundaries carefully, 
        # but for finite N, it might be N/2 or N/2 +/- 1.
        # We will project it.
        h = np.zeros(N)
        for i in range(N):
            if (i * alpha) % 1.0 < 0.5:
                h[i] = 1.0
            else:
                h[i] = 0.0
        return h

    alphas = [
        (1.0 - np.sqrt(5.0)) / 2.0 + 1.0,  # Golden ratio conjugate ~ 0.618
        np.sqrt(2.0) - 1.0,
        np.sqrt(3.0) - 1.5,
        0.5 + 0.1,
        0.5 - 0.1,
        np.pi / 10.0,
        0.381966 # 1 - phi
    ]
    
    for a in alphas:
        h_init = generate_sturmian(a, N)
        h_init = project(h_init)
        candidates.append(h_init)
        
    # 2. Random Binary
    rng = np.random.default_rng(42)
    for _ in range(5):
        h_rand = rng.choice([0.0, 1.0], size=N)
        h_rand = project(h_rand)
        candidates.append(h_rand)
        
    # 3. Sinusoidal
    for freq in [1, 3, 7]:
        h_sine = 0.5 + 0.3 * np.sin(2.0 * np.pi * np.arange(N) / N * freq)
        h_sine = project(h_sine)
        candidates.append(h_sine)
        
    # Evaluate candidates and pick top 3
    scored_candidates = []
    for h in candidates:
        score = max_overlap(h)
        scored_candidates.append((score, h))
        
    scored_candidates.sort(key=lambda x: x[0])
    best_starts = [h for _, h in scored_candidates[:3]]
    
    best_h = best_starts[0]
    best_score = max_overlap(best_h)
    
    # Optimization Loop
    # Use L-BFGS-B with decreasing temperature
    temps = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    for h_start in best_starts:
        h_curr = h_start.copy()
        
        for temp in temps:
            if time.time() - start_time > time_limit - 10:
                break
                
            def obj_func(x, t=temp):
                return objective_lse(x, t)
            
            try:
                res = minimize(obj_func, h_curr, method='L-BFGS-B', 
                               bounds=[(0, 1)] * N,
                               options={'maxiter': 2000, 'maxfun': 20000, 'gtol': 1e-8, 'ftol': 1e-10})
                h_opt = project(res.x)
                score_opt = max_overlap(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                h_curr = h_opt
            except Exception:
                pass
        
        # Check time
        if time.time() - start_time > time_limit - 10:
            break

    # Final Polish: Coordinate Descent / Small Steps
    if time.time() - start_time < time_limit - 5:
        h_polish = best_h.copy()
        # Try small perturbations
        steps = [0.01, -0.01, 0.02, -0.02]
        for _ in range(1000):
            if time.time() - start_time > time_limit - 2:
                break
            improved = False
            cur_score = max_overlap(h_polish)
            for i in range(N):
                for step in steps:
                    h_try = h_polish.copy()
                    h_try[i] += step
                    h_try = project(h_try)
                    new_score = max_overlap(h_try)
                    if new_score < cur_score - 1e-9:
                        h_polish = h_try
                        cur_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
                
        if max_overlap(h_polish) < best_score:
            best_h = h_polish
            best_score = max_overlap(best_h)

    best_h = project(best_h)
    return best_h
