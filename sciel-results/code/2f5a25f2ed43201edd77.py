"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    # FFT-based autocorrelation for speed
    def compute_ratio(h):
        h = np.asarray(h, dtype=np.float64)
        if np.sum(h) < 1e-15:
            return 0.0
        # Use FFT for O(N log N) autocorrelation
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        ac = np.fft.irfft(H * np.conj(H), n=n_fft)
        sup_norm = np.max(np.abs(ac))
        if sup_norm < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        return l2_sq / (l1 * sup_norm)

    # Piecewise-constant representation:
    # Define k segments with transition points and amplitudes
    # Parameters: [t1, t2, ..., t_{k-1}, a1, a2, ..., ak]
    # where ti are transition points in [0, N) and ai are amplitudes >= 0
    
    def params_to_heights(params, k):
        """Convert parameters to height vector."""
        # First k-1 params are transition points, remaining k are amplitudes
        transitions = params[:k-1]
        amps = params[k-1:]
        
        # Sort transitions to ensure valid segmentation
        sorted_transitions = np.sort(transitions)
        
        # Create segment boundaries
        boundaries = np.concatenate([[0], sorted_transitions, [N]])
        
        # Create height vector
        h = np.zeros(N)
        for i in range(k):
            start = int(np.floor(boundaries[i]))
            end = int(np.floor(boundaries[i+1]))
            if start < end:
                h[start:end] = max(0, amps[i])
        
        # Ensure at least one non-zero
        if np.sum(h) < 1e-15:
            h[0] = 1.0
        
        return h

    def neg_ratio_from_params(params, k):
        """Negative ratio for minimization."""
        try:
            h = params_to_heights(params, k)
            return -compute_ratio(h)
        except:
            return 1e10

    # Strategy 1: Evaluate several structured patterns
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)

    patterns = [
        np.ones(N),
        np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)]),
        np.array([1.0 - abs(i - N//2)/(N//2) for i in range(N)]),
        np.array([1.0/np.sqrt(i+1) for i in range(N)]),
        np.array([np.exp(-i/10) for i in range(N)]),
        np.array([np.cos(2*np.pi*i/10) + 1.5 for i in range(N)]),
        np.array([np.sin(np.pi*i/(N-1)) for i in range(N)]),
        # Sparse patterns
        np.array([1.0 if i in [0, 10, 20, 30, 40] else 0.0 for i in range(N)]),
        np.array([1.0 if i in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45] else 0.0 for i in range(N)]),
    ]
    
    for p in patterns:
        p = np.maximum(0, p)
        if np.sum(p) < 1e-15:
            continue
        s = compute_ratio(p)
        if s > best_score:
            best_score = s
            best_h = p.copy()

    # Strategy 2: Nelder-Mead on piecewise-constant representation
    # Try different numbers of segments
    for k in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Multiple random starts
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + k)
            
            # Random initial parameters
            transitions = np.sort(np.random.uniform(0, N, k-1))
            amps = np.random.exponential(1.0, k)
            params0 = np.concatenate([transitions, amps])
            
            result = minimize(
                neg_ratio_from_params,
                params0,
                args=(k,),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if time.time() > deadline:
                break
            
            if result.fun < -1e10:  # Invalid
                continue
                
            h_opt = params_to_heights(result.x, k)
            s = compute_ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()

    # Strategy 3: Differential evolution for global exploration
    from scipy.optimize import differential_evolution
    
    if time.time() < deadline - 30:
        # Define bounds for full 50-dim optimization
        bounds = [(0, 5)] * N
        
        def neg_ratio_full(h):
            h = np.maximum(0, h)
            return -compute_ratio(h)
        
        # Multiple random structured starts
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            
            # Create diverse initial population
            if seed == 0:
                # Sparse
                init = np.zeros(N)
                idx = np.random.choice(N, 5, replace=False)
                init[idx] = np.random.exponential(1.0, 5)
            elif seed == 1:
                # Ramp
                init = np.linspace(0, 1, N)
            else:
                # Random dense
                init = np.random.exponential(1.0, N)
            
            result = differential_evolution(
                neg_ratio_full,
                bounds,
                seed=seed,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol'
            )
            
            if time.time() > deadline - 10:
                break
            
            if result.fun < -1e10:
                continue
                
            h_opt = np.maximum(0, result.x)
            s = compute_ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()

    # Strategy 4: Coordinate descent polish from best
    h = best_h.copy()
    for it in range(200):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try several perturbations
            for delta in [-0.1, -0.01, 0.01, 0.1, 0.5]:
                new_val = max(0.0, old_val + delta)
                h[i] = new_val
                s = compute_ratio(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val
        if not improved:
            break

    # Final check
    return list(best_h)
# EVOLVE-BLOCK-END
