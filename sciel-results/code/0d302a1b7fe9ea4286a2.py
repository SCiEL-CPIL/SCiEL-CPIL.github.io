import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    N = 100
    target_sum = N / 2.0
    
    def get_correlations_linear(h):
        g = 1.0 - h
        c_full = np.correlate(g, h, mode='full')
        c_nonzero = np.concatenate([c_full[:N-1], c_full[N:]])
        return c_nonzero

    def max_overlap(h):
        c = get_correlations_linear(h)
        return np.max(c)
    
    def smooth_max(c, temp=0.05):
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))
    
    def project(h):
        h = np.clip(h, 0, 1)
        for _ in range(30):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-13:
                break
            h += diff / N
            h = np.clip(h, 0, 1)
        return h
    
    def objective_smooth(h, temp=0.05):
        c = get_correlations_linear(h)
        val = smooth_max(c, temp)
        sum_err = np.sum(h) - target_sum
        return val + 10.0 * sum_err**2

    # Epigraph formulation: variables are [h_0, ..., h_{N-1}, t]
    # Minimize t subject to c_i <= t + soft_penalty, 0<=h<=1, sum(h)=N/2
    def epigraph_objective(x, temp=0.05):
        h = x[:N]
        t = x[N]
        c = get_correlations_linear(h)
        # Smooth penalty for c_i > t: sum of softplus(c_i - t)
        viol = c - t
        # softplus(v) = log(1 + exp(v)) but stabilized
        max_v = np.max(viol)
        if max_v > 50:
            penalty = np.sum(np.where(viol > 0, viol, 0))
        else:
            penalty = np.sum(np.logaddexp(0, viol))
        sum_err = np.sum(h) - target_sum
        return t + 5.0 * penalty + 10.0 * sum_err**2

    # Initialize with Sturmian/quasi-periodic pattern
    rng = np.random.default_rng(42)
    
    # Use golden ratio conjugate for quasi-periodic sequence
    alpha = (1.0 - np.sqrt(5.0)) / 2.0  # ~ -0.618
    # Generate a Sturmian-like sequence
    n = np.arange(N)
    h_sturm = np.where(np.floor(n * 0.5 + n * alpha) - np.floor((n - 1) * 0.5 + (n - 1) * alpha) == 1, 1.0, 0.0)
    # Smooth it a bit for continuous optimization
    h_curr = project(h_sturm * 0.7 + 0.15)
    
    # Also try a sinusoidal initialization
    h_sine = 0.5 + 0.3 * np.sin(2.0 * np.pi * np.arange(N) / N * 3.0)
    h_sine = project(h_sine)
    
    # Pick the better initial point
    if max_overlap(h_sine) < max_overlap(h_curr):
        h_curr = h_sine
    
    best_h = h_curr.copy()
    best_score = max_overlap(best_h)
    
    # Simulated annealing
    T_init = 0.1
    T_final = 0.001
    n_iterations = 5000
    
    def find_clusters(h, threshold=0.05):
        clusters = []
        i = 0
        while i < N:
            j = i
            base_val = h[i]
            while j < N and abs(h[j] - base_val) < threshold:
                j += 1
            if j > i:
                clusters.append((i, j))
            i = j
        return clusters
    
    def cluster_move(h, clusters, rng):
        if not clusters:
            return h.copy()
        idx = rng.integers(0, len(clusters))
        i, j = clusters[idx]
        h_new = h.copy()
        shift = rng.uniform(-0.2, 0.2)
        h_new[i:j] += shift
        h_new = project(h_new)
        return h_new
    
    def cluster_flip(h, clusters, rng):
        if not clusters:
            return h.copy()
        idx = rng.integers(0, len(clusters))
        i, j = clusters[idx]
        h_new = h.copy()
        h_new[i:j] = 1.0 - h[i:j]
        h_new = project(h_new)
        return h_new

    for it in range(n_iterations):
        if time.time() - start_time > time_limit - 25:
            break
        
        progress = it / n_iterations
        T = T_init * (T_final / T_init) ** progress
        
        clusters = find_clusters(h_curr, threshold=0.05)
        
        move_type = rng.random()
        if move_type < 0.4:
            h_prop = cluster_move(h_curr, clusters, rng)
        elif move_type < 0.7:
            h_prop = cluster_flip(h_curr, clusters, rng)
        else:
            h_prop = h_curr.copy()
            i = rng.integers(0, N)
            h_prop[i] += rng.uniform(-0.1, 0.1)
            h_prop = project(h_prop)
        
        score_curr = max_overlap(h_curr)
        score_prop = max_overlap(h_prop)
        
        delta = score_prop - score_curr
        if delta < 0 or rng.random() < np.exp(-delta / T):
            h_curr = h_prop
        
        curr_score = max_overlap(h_curr)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h_curr.copy()
    
    # High-precision polish with epigraph formulation
    if time.time() - start_time < time_limit - 10:
        # First pass: LSE smooth max
        h_init = best_h.copy()
        bounds = [(0, 1)] * N
        
        for temp in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if time.time() - start_time > time_limit - 5:
                break
            
            def obj_lse(x, t=temp):
                return objective_smooth(x, t)
            
            try:
                res = minimize(obj_lse, h_init, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 3000, 'maxfun': 30000, 'gtol': 1e-10, 'ftol': 1e-12})
                h_opt = res.x
                h_opt = project(h_opt)
                score_opt = max_overlap(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    h_init = h_opt.copy()
            except Exception:
                pass
        
        # Second pass: epigraph formulation with auxiliary t
        if time.time() - start_time < time_limit - 3:
            # Build initial x = [h, t] where t = max_overlap(h)
            t_init = max_overlap(h_init)
            x_init = np.concatenate([h_init, [t_init]])
            bounds_ep = [(0, 1)] * N + [(0, 2.0)]
            
            for temp in [0.05, 0.02, 0.01]:
                if time.time() - start_time > time_limit - 2:
                    break
                
                def obj_epi(x, t=temp):
                    return epigraph_objective(x, t)
                
                try:
                    res = minimize(obj_epi, x_init, method='L-BFGS-B', bounds=bounds_ep,
                                   options={'maxiter': 3000, 'maxfun': 30000, 'gtol': 1e-10, 'ftol': 1e-12})
                    h_opt = res.x[:N]
                    h_opt = project(h_opt)
                    score_opt = max_overlap(h_opt)
                    
                    if score_opt < best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
                        h_init = h_opt.copy()
                        t_init = score_opt
                        x_init = np.concatenate([h_opt, [t_init]])
                except Exception:
                    pass

    # Binary swap local search
    if time.time() - start_time < time_limit - 2:
        h_polish = best_h.copy()
        improved = True
        max_rounds = 50
        for rnd in range(max_rounds):
            if time.time() - start_time > time_limit - 1:
                break
            if not improved:
                break
            improved = False
            cur_score = max_overlap(h_polish)
            for i in range(N):
                for j in range(i+1, N):
                    if time.time() - start_time > time_limit - 1:
                        break
                    # Try swapping values at i and j
                    h_try = h_polish.copy()
                    h_try[i], h_try[j] = h_try[j], h_try[i]
                    h_try = project(h_try)
                    new_score = max_overlap(h_try)
                    if new_score < cur_score - 1e-10:
                        h_polish = h_try
                        cur_score = new_score
                        improved = True
                        break
                if time.time() - start_time > time_limit - 1:
                    break
            if improved and cur_score < best_score:
                best_score = cur_score
                best_h = h_polish.copy()

    # Final coordinate descent polish
    if time.time() - start_time < time_limit - 1:
        h_polish = best_h.copy()
        for _ in range(500):
            if time.time() - start_time > time_limit - 1:
                break
            improved = False
            cur_score = max_overlap(h_polish)
            for i in range(N):
                for step in [-0.02, -0.01, 0.01, 0.02]:
                    h_try = h_polish.copy()
                    h_try[i] += step
                    h_try = project(h_try)
                    if max_overlap(h_try) < cur_score - 1e-12:
                        h_polish = h_try
                        cur_score = max_overlap(h_polish)
                        improved = True
                        break
            if not improved:
                break
        if max_overlap(h_polish) < best_score:
            best_h = h_polish
            best_score = max_overlap(best_h)

    best_h = project(best_h)
    return best_h
