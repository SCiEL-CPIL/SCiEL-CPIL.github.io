"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Generate diverse initial points (random, structured, bump profiles).
    2. Use smooth surrogate (log-sum-exp) + L-BFGS-B with analytic gradient.
    3. Final exact evaluation and polish.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def smooth_objective(x):
        """Smooth surrogate using log-sum-exp of conv entries."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        beta = 50.0
        shifted = conv - max_c
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val
    
    def smooth_grad(x):
        """Analytic gradient of the smooth surrogate."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        beta = 50.0
        shifted = conv - max_c
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        # log_sum_exp = max_c + (1/beta) * log(sum(exp(beta*(conv - max_c))))
        # d(log_sum_exp)/d(conv_k) = weights[k]
        # 
        # conv = np.convolve(h, h) has length 2N-1
        # conv[k] = sum_j h[j] * h[k-j] where 0<=j<N and 0<=k-j<N
        # d(conv[k])/d(h[i]) = sum_j [delta_{j,i}*h[k-j] + h[j]*delta_{k-j,i}]
        #                    = h[k-i] (if 0<=k-i<N) + h[k-i] (if 0<=k-i<N)
        #                    = 2*h[k-i] if 0<=k-i<N, else 0
        # But wait: conv[k] = sum_{j=0}^{N-1} h[j]*h[k-j] where h is zero-padded
        # d(conv[k])/d(h[i]) = h[k-i] + h[k-i] = 2*h[k-i] if 0<=k-i<N
        # Actually: conv[k] = sum_j h[j] h[k-j]
        # d/dh[i]: sum_j [delta_{ji} h[k-j] + h[j] delta_{k-j,i}]
        #         = h[k-i] + h[k-i] if both terms valid
        #         = h[k-i] if only one valid
        # For i in [0,N-1], k-i in [0,N-1]: both h[j] at j=i and h[k-j] at k-j=i contribute
        # So d(conv[k])/d(h[i]) = 2*h[k-i] if 0<=k-i<N, else 0
        
        # d(log_sum_exp)/d(h[i]) = sum_k weights[k] * d(conv[k])/d(h[i])
        #                        = 2 * sum_k weights[k] * h[k-i] (where 0<=k-i<N)
        # Let m = k-i, so k = m+i, m in [0,N-1], k in [i, i+N-1]
        # k ranges over [0, 2N-2]
        # So m+i in [0, 2N-2] and m in [0, N-1]
        
        grad_lse = np.zeros(N)
        # For each i, grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[m+i] * h[m]
        # This is a cross-correlation-like operation
        # weights has length 2N-1, h has length N
        # We need for each i: sum_{m=0}^{N-1} weights[m+i] * h[m]
        # This is like a sliding dot product
        
        # Efficient computation using FFT or direct
        # weights[i:i+N] dot h for each i
        # Use np.correlate or manual
        
        # grad_lse[i] = 2 * dot(weights[i:i+N], h)
        # This can be computed as a correlation
        # Let's compute it directly since N=400 is small
        
        for i in range(N):
            grad_lse[i] = 2.0 * np.dot(weights[i:i+N], h)
        
        # Now value = abs(2*N*lse / s^2)
        # Let f = 2*N*lse / s^2
        # value = abs(f)
        # If f > 0: d(value)/d(h[i]) = d(f)/d(h[i])
        # If f < 0: d(value)/d(h[i]) = -d(f)/d(h[i])
        
        lse = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        f = 2.0 * N * lse / (s * s)
        
        # d(f)/d(h[i]) = 2*N * [d(lse)/d(h[i]) * s^2 - lse * 2*s * 1] / s^4
        #              = 2*N * [grad_lse[i] / s^2 - 2*lse / s^3]
        #              = 2*N * grad_lse[i] / s^2 - 4*N*lse / s^3
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return grad
    
    rng = np.random.RandomState(42)
    
    # --- Generate diverse initial points ---
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(15):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal: mostly zero with random spikes
    for _ in range(15):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))
    
    # Parameterized bump profiles (shown to work well)
    i_arr = np.arange(N)
    for center in [N//4, N//2, 3*N//4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            # Double bump
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_double)
            # Bump with offset
            h_off = np.exp(-((i_arr - center)**2) / (2 * width**2)) + 0.3 * np.ones(N)
            initial_points.append(h_off)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(5):
            freqs = rng.randint(1, 30, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)
    
    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Try to optimize
        try:
            result = minimize(
                smooth_objective,
                init_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-5}
            )
            
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass
        
        # Also try exact evaluation of the initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()
    
    # --- Local polish using coordinate descent on the best solution ---
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        lr = 0.05
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for delta in [lr, -lr, lr*0.5, -lr*0.5, lr*0.25, -lr*0.25]:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try
                        h_curr = h_try
                        improved = True
                        break
            if not improved:
                lr *= 0.7
    
    return best_h
# EVOLVE-BLOCK-END
