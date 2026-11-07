"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def objective(h):
    """
    Compute the evaluation value:
    value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    """
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def construct():
    """
    Uses Projected Subgradient Descent with Momentum.
    
    Strategy:
    1. The objective V(h) = max_k C_k(h) is the maximum of differentiable functions.
    2. We use the subgradient: grad = mean of gradients of active constraints.
    3. We update h using momentum: v = mu*v - lr*grad; h = h + v.
    4. We project h onto the feasible set: sum(h) = N/2, 0 <= h_i <= 1.
    5. We use multiple restarts with different initializations.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    def get_corrs(h):
        return np.correlate(h, 1.0 - h, mode="full")

    def project(h):
        """
        Project h onto {x in [0,1]^N : sum(x) = N/2}
        """
        h = np.clip(h, 0.0, 1.0)
        # Iterative repair for sum
        for _ in range(50):
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            rem = N / 2.0 - current_sum
            if abs(rem) < 1e-9:
                break
            if rem > 0:
                mask = h < 1.0
            else:
                mask = h > 0.0
            if np.sum(mask) == 0:
                break
            h[mask] += rem / np.sum(mask)
        return h

    def compute_subgradient(h):
        """
        Compute the subgradient of V(h) = max_k C_k(h).
        C_k(h) = sum_i h_{i+k} - sum_i h_i h_{i+k} (with appropriate indices for full correlate)
        
        Let's derive the gradient of C_k with respect to h_j.
        np.correlate(h, 1-h, 'full')[m] = sum_i h[i] * (1-h[i+m-N+1])? 
        Actually, np.correlate(a, v, 'full')[k] = sum_n a[n] * v[n-k] ? No.
        Definition: c_k = sum_n h[n] * (1-h)[n-k] for k=0..N-1?
        Let's check np.correlate documentation.
        c[k] = sum_n a[n+k] * v[n] for 'full'?
        Actually, it's easier to use finite differences or analytic derivation.
        
        Let's use the property:
        C_k = sum_i h_i (1 - h_{i+k}) for k >= 0?
        In 'full' mode, index m corresponds to lag m - (N-1).
        Let lag l = m - (N-1).
        If l >= 0: C_l = sum_{i=0}^{N-1-l} h_i (1 - h_{i+l})
        If l < 0: C_l = sum_{i=0}^{N-1+l} h_{i-l} (1 - h_i) = sum_j h_j (1 - h_{j-l})
        
        Gradient of C_l with respect to h_j:
        Case l >= 0:
        Term h_j appears as h_i if j <= N-1-l. Contrib: 1 - h_{j+l}
        Term h_j appears as h_{i+l} if j-l >= 0 and j-l <= N-1-l => j >= l and j <= N-1.
        Contrib: -h_{j-l}
        So grad[j] += (1 - h_{j+l}) - h_{j-l} (with bounds checks)
        
        Case l < 0: Let l' = -l.
        C_{-l'} = sum_j h_j (1 - h_{j-l'})
        Grad[j]: (1 - h_{j-l'}) - h_{j+l'}
        
        Basically, for a lag l, the gradient is:
        g_l[j] = (1 - h_{j+l}) - h_{j-l} (where h_out_of_bounds = 0)
        
        We sum g_l for active lags and average.
        """
        corrs = get_corrs(h)
        max_corr = np.max(corrs)
        # Find active lags (with small tolerance to avoid numerical issues)
        active_mask = corrs > max_corr - 1e-6
        num_active = np.sum(active_mask)
        if num_active == 0:
            return np.zeros(N)
            
        grad = np.zeros(N)
        # Indices in 'full' correlate range from 0 to 2N-2.
        # Lag l = idx - (N - 1)
        for idx in np.where(active_mask)[0]:
            l = idx - (N - 1)
            # Compute gradient for this lag
            g_l = np.zeros(N)
            if l >= 0:
                # h_i (1 - h_{i+l})
                # d/dh_j:
                # 1. h_j is h_i: i=j. Valid if j <= N-1-l. Term: 1 - h_{j+l}
                # 2. h_j is h_{i+l}: i=j-l. Valid if j-l >= 0 and j-l <= N-1-l => j>=l, j<=N-1. Term: -h_{j-l}
                
                # Part 1: i=j
                valid1 = np.arange(N) <= (N - 1 - l)
                g_l[valid1] += 1.0 - h[valid1 + l] # h[j+l] is valid for j <= N-1-l
                
                # Part 2: i=j-l
                valid2 = (np.arange(N) >= l) & (np.arange(N) <= N - 1)
                # j-l is the index in h.
                # h[j-l] is valid.
                g_l[valid2] -= h[np.arange(N)[valid2] - l]
            else:
                # l < 0. Let l' = -l.
                # C_{-l'} = sum_j h_j (1 - h_{j-l'})
                # d/dh_j:
                # 1. h_j is h_j. Term: 1 - h_{j-l'}
                # 2. h_j is h_{j-l'} => j-l' = k => j = k+l'. Term: -h_k
                # So for index j:
                # Term 1: 1 - h[j-l'] (if j-l' in [0, N-1])
                # Term 2: -h[j+l'] (if j+l' in [0, N-1])
                
                lp = -l
                # Part 1
                valid1 = np.arange(N) >= lp
                g_l[valid1] += 1.0 - h[np.arange(N)[valid1] - lp]
                
                # Part 2
                valid2 = (np.arange(N) + lp) < N
                g_l[valid2] -= h[np.arange(N)[valid2] + lp]
            
            grad += g_l
        
        grad /= num_active
        return grad

    def run_subgradient(h_init, lr_schedule, max_iter):
        h = h_init.copy()
        h = project(h)
        v = np.zeros(N)
        mu = 0.9
        best_val = objective(h)
        best_h = h.copy()
        
        for it in range(max_iter):
            # Check time
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Adaptive learning rate
            lr = lr_schedule(it)
            
            grad = compute_subgradient(h)
            v = mu * v - lr * grad
            h_new = h + v
            h_new = project(h_new)
            
            val = objective(h_new)
            if val < best_val:
                best_val = val
                best_h = h_new.copy()
            
            h = h_new
            
        return best_h, best_val

    best_h = np.full(N, 0.5)
    best_val = objective(best_h)

    # Initializations
    inits = []
    inits.append(np.full(N, 0.5))
    
    # Random
    for _ in range(10):
        h_r = np.random.uniform(0, 1, N)
        inits.append(project(h_r))
        
    # Structured: Alternating
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    inits.append(project(h_alt))
    
    # Structured: Blocks
    h_block = np.zeros(N)
    h_block[0:50] = 1.0
    inits.append(project(h_block))
    
    # Structured: Sine wave
    x = np.linspace(0, 4*np.pi, N)
    h_sin = 0.5 + 0.5 * np.sin(x)
    inits.append(project(h_sin))

    # Run optimization for each init
    for h0 in inits:
        if time.time() - start_time > time_limit - 10.0:
            break
        
        # Decreasing learning rate
        def lr_schedule(it, total=500):
            return 0.1 / (1 + it * 0.01)
        
        h_opt, val_opt = run_subgradient(h0, lr_schedule, 500)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()

    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
