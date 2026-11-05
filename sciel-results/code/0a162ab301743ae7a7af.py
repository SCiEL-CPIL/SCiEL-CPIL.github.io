import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 24
    M = 2**12 # 4096 FFT size for fast eval
    sqrt_N = np.sqrt(N)
    
    # Precompute DFT matrix for fast evaluation? 
    # No, FFT is faster for M=4096.
    
    def fast_eval(c):
        """Evaluate sup norm using FFT on a grid of M points."""
        c_pad = np.zeros(M, dtype=complex)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        return np.max(np.abs(F))
        
    def accurate_eval(c):
        """Evaluate sup norm with refinement of peaks."""
        # First, get a coarse max
        c_pad = np.zeros(M, dtype=complex)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        
        # Find top 5 peaks
        top_indices = np.argsort(abs_F)[-5:]
        max_val = np.max(abs_F)
        
        k = np.arange(N)
        
        for idx in top_indices:
            theta_c = 2 * np.pi * idx / M
            delta_theta = 2 * np.pi / M
            # Sample 200 points around the peak
            thetas = np.linspace(theta_c - delta_theta, theta_c + delta_theta, 200)
            Z = np.exp(1j * np.outer(thetas, k))
            P_vals = Z @ c
            local_max = np.max(np.abs(P_vals))
            if local_max > max_val:
                max_val = local_max
                
        return max_val

    def get_score(c, accurate=False):
        if accurate:
            return accurate_eval(c) / sqrt_N
        else:
            return fast_eval(c) / sqrt_N

    rng = np.random.RandomState(42)
    
    # Generate Initializations
    initializations = []
    
    # 1. Random
    for _ in range(20):
        initializations.append(np.random.choice([-1.0, 1.0], N))
        
    # 2. Golay-Rudin-Shapiro based
    # Generate GRS of length 32
    def gen_grs(length):
        # length must be power of 2
        # g_0 = [1]
        # g_{k+1} = [g_k, (-1)^i g_k] ? 
        # Standard: G_{k+1}(z) = G_k(z) + z^{2^k} G_k(-z)
        # Coeffs: 
        g = [1]
        while len(g) < length:
            # g_k(-z) coeffs are (-1)^i g_k[i]
            g_neg = [(-1)**i * x for i, x in enumerate(g)]
            # New g is g + z^{len(g)} g_neg
            # This means we append g_neg to g
            g = g + g_neg
        return g[:length]
    
    g32 = gen_grs(32)
    # Truncate to 24
    initializations.append(np.array(g32[:24], dtype=float))
    # Flip signs
    initializations.append(-np.array(g32[:24], dtype=float))
    
    # Try shifting?
    for shift in range(8):
        g_shifted = np.roll(g32, shift)
        initializations.append(np.array(g_shifted[:24], dtype=float))
        
    # 3. Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    initializations.append(alt)
    initializations.append(-alt)
    
    # 4. All 1s, All -1s
    initializations.append(np.ones(N))
    initializations.append(-np.ones(N))
    
    # Evaluate all initializations with fast_eval
    init_scores = []
    for c in initializations:
        score = get_score(c, accurate=False)
        init_scores.append(score)
        
    # Sort by score
    sorted_indices = np.argsort(init_scores)
    
    best_score = float('inf')
    best_c = None
    
    # Select top 10 for Simulated Annealing
    top_k = 10
    sa_time_per_run = 8.0
    
    for rank in range(top_k):
        if time.time() > deadline - 10.0: # Leave 10s for final polish
            break
            
        idx = sorted_indices[rank]
        c = initializations[idx].copy()
        current_score = get_score(c, accurate=False)
        
        # Simulated Annealing
        T = 0.05
        alpha = 0.999
        T_min = 1e-6
        sa_start = time.time()
        
        while time.time() - sa_start < sa_time_per_run and T > T_min:
            if rng.rand() < 0.7:
                idx_flip = rng.randint(N)
                c[idx_flip] *= -1
                new_score = get_score(c, accurate=False)
                delta = new_score - current_score
                if delta < 0 or rng.rand() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    c[idx_flip] *= -1 # Revert
            else:
                idx1, idx2 = rng.choice(N, 2, replace=False)
                c[idx1] *= -1
                c[idx2] *= -1
                new_score = get_score(c, accurate=False)
                delta = new_score - current_score
                if delta < 0 or rng.rand() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    c[idx1] *= -1
                    c[idx2] *= -1 # Revert
            
            T *= alpha
            
        # After SA, do a local search with accurate eval?
        # Or just keep the best fast_eval candidate and polish later.
        # Let's do a quick local search with accurate eval on the SA result
        # to ensure we don't miss a good peak.
        
        # Local Search (1-flip) with accurate eval
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            best_delta = 0
            best_idx = -1
            current_acc_score = get_score(c, accurate=True)
            
            for idx in range(N):
                c[idx] *= -1
                new_acc_score = get_score(c, accurate=True)
                delta = new_acc_score - current_acc_score
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx
                c[idx] *= -1 # Revert for next check
            
            if best_idx != -1:
                c[best_idx] *= -1
                improved = True
                
        # Update global best
        final_score = get_score(c, accurate=True)
        if final_score < best_score:
            best_score = final_score
            best_c = c.copy()

    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
