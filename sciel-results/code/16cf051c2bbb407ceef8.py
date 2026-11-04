"""
Combined approach:
1. Global search via Simulated Annealing to escape local minima.
2. Local refinement via L-BFGS-B with LogSumExp approximation.
3. Diverse initializations including low-autocorrelation sequences.
"""
import numpy as np
import time
import scipy.optimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    target_sum = N / 2.0
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    def get_correlations(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts."""
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude center (shift 0)
        c1 = c[:N-1]
        c2 = c[N:]
        return np.concatenate([c1, c2])

    def true_score(h):
        """Calculate the raw score (max overlap)."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def project_to_feasible(h):
        """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1."""
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        
        if abs(diff) > 1e-12:
            for _ in range(20):
                if abs(diff) < 1e-12:
                    break
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                diff = target_sum - current_sum
        return h

    def make_objective(beta, lambda_sum):
        def obj(h):
            c_all = get_correlations(h)
            max_c = np.max(c_all)
            if beta > 0:
                exp_c = np.exp(beta * (c_all - max_c))
                lse = max_c + np.log(np.sum(exp_c)) / beta
            else:
                lse = max_c
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return lse + sum_pen
        return obj

    # Initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Random
    for _ in range(10):
        h_rand = rng.uniform(0.1, 0.9, N)
        candidates.append(h_rand)
        
    # Sine waves
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 10]:
        candidates.append(0.5 + 0.4 * np.sin(2 * np.pi * freq * t / N))
        candidates.append(0.5 + 0.4 * np.cos(2 * np.pi * freq * t / N))
        
    # Thue-Morse like (low autocorrelation)
    def thue_morse(n):
        tm = [0]
        for i in range(n-1):
            tm.append(1 - tm[-1])
        # Generate longer sequence
        res = []
        for i in range(n):
            res.append(tm[i % n])
        return res
        
    # Actually, let's use a simple binary pattern converted to continuous
    h_bin = np.zeros(N)
    h_bin[:N//2] = 1.0
    # Shuffle to break periodicity
    rng.shuffle(h_bin)
    candidates.append(h_bin)
    
    # Van der Corput
    def van_der_corput(n, base=2):
        vdc = [0.0]
        for i in range(1, n):
            vdc.append(1.0 - (vdc[-1] + 1.0 / (base ** (i - 1))) % 1.0)
        return vdc
        
    h_vdc = van_der_corput(N)
    # Scale to [0,1] and adjust sum
    h_vdc = (h_vdc - np.min(h_vdc)) / (np.max(h_vdc) - np.min(h_vdc) + 1e-9)
    candidates.append(h_vdc)

    best_h = None
    best_score = float('inf')
    
    # Phase 1: Evaluate candidates
    for h_cand in candidates:
        h_proj = project_to_feasible(h_cand)
        score = true_score(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj

    # Phase 2: Multi-start L-BFGS-B
    # Replace SA with more direct optimization from diverse starts
    num_starts = 15
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate diverse start
        if start_idx == 0:
            h_curr = best_h.copy()
        elif start_idx % 3 == 0:
            # Random uniform
            h_curr = rng.uniform(0.2, 0.8, N)
        elif start_idx % 3 == 1:
            # Sine wave with random parameters
            freq = rng.integers(1, 10)
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arange(N)
            h_curr = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t / N + phase)
        else:
            # Random walk
            h_curr = np.cumsum(rng.normal(0, 0.1, N))
            h_curr = (h_curr - np.min(h_curr)) / (np.max(h_curr) - np.min(h_curr) + 1e-9)
            h_curr = 0.2 + 0.6 * h_curr
            
        h_curr = project_to_feasible(h_curr)
        
        # Use L-BFGS-B with moderate beta for global-ish search
        obj_func = make_objective(30.0, 100.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                h_curr, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = res.x
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
                
        except Exception:
            pass

    # Phase 3: Final high-precision polish on the best found
    if time.time() < deadline - 5.0:
        # Use a very high beta to approximate max closely
        obj_func = make_objective(200.0, 1000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = res.x
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass
            
        # One more polish with even tighter tolerance if time permits
        if time.time() < deadline - 2.0:
            try:
                res = scipy.optimize.minimize(
                    obj_func, 
                    best_h, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_opt = res.x
                h_opt = project_to_feasible(h_opt)
                score_opt = true_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
            except Exception:
                pass

    return best_h
