import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Uses linear (non-circular) correlation: C(k) = sum_i h[i]*(1-h[i+k])
    Optimized via log-sum-exp smooth max + simulated annealing + gradient polish.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 100
    target_sum = N / 2.0
    
    def get_linear_correlations(h):
        """C(k) = sum_i h[i] * (1 - h[i+k]) for k=1..N-1 (linear, not circular)."""
        g = 1.0 - h
        # np.correlate(h, g, 'full') gives cross-correlation
        # c[0] = h[N-1]*g[0], ..., c[N-1] = sum h[i]*g[i] (k=0), ...
        # For positive shifts k: c[N-1+k] = sum_i h[i+k]*g[i] = sum_i h[i+k]*(1-h[i])
        # But we want sum_i h[i]*(1-h[i+k]) = sum_i h[i]*g[i+k]
        # That's c[N-1-k] for k>=0... let me just compute directly for clarity
        # Actually np.correlate(h, g, 'full')[N-1-k] = sum_i h[i+k]*g[i] for k>=0
        # And np.correlate(h, g, 'full')[N-1+k] = sum_i h[i]*g[i+k] for k>=0
        c = np.correlate(h, g, mode='full')
        # Positive shifts k=1..N-1: c[N-1+k] for k=1..N-1 -> indices N..2N-2
        # Wait, let me verify: np.correlate(a, v, 'full')[k] = sum_n a[n] * v[n+k-(len(v)-1)]
        # For a=h, v=g, len(v)=N: c[k] = sum_n h[n] * g[n+k-(N-1)]
        # For k = N-1: c[N-1] = sum_n h[n]*g[n] (shift 0)
        # For k = N-1+j: c[N-1+j] = sum_n h[n]*g[n+j] = sum_n h[n]*(1-h[n+j]) for j>=0
        # So positive shifts j=1..N-1 are at indices N..2N-2
        # For k = N-1-j: c[N-1-j] = sum_n h[n]*g[n-j] = sum_n h[n]*(1-h[n-j]) for j>=0
        # Negative shifts are at indices 0..N-2
        # We want all non-zero shifts: indices 0..N-2 and N..2N-2
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def max_overlap(h):
        c = get_linear_correlations(h)
        return np.max(c)
    
    def smooth_max(c, temp=0.03):
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))
    
    def objective_smooth(h, temp=0.03):
        c = get_linear_correlations(h)
        return smooth_max(c, temp)
    
    def project(h):
        """Project to 0<=h<=1, sum(h)=N/2."""
        h = np.clip(h, 0, 1)
        for _ in range(30):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            # Only adjust non-saturated entries
            mask = (h > 0) & (h < 1)
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
        return h
    
    # Initialize
    rng = np.random.default_rng(42)
    h_curr = np.full(N, 0.5)
    best_h = h_curr.copy()
    best_score = max_overlap(best_h)
    
    # Stage 1: Simulated annealing with diverse moves
    T_init = 0.15
    T_final = 0.0005
    n_sa = 4000
    
    def find_clusters(h, threshold=0.08):
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
    
    for it in range(n_sa):
        if time.time() > deadline - 30:
            break
        
        progress = it / n_sa
        T = T_init * (T_final / T_init) ** progress
        
        move_type = rng.random()
        if move_type < 0.35:
            # Cluster shift
            clusters = find_clusters(h_curr, threshold=0.06)
            if clusters:
                idx = rng.integers(0, len(clusters))
                i, j = clusters[idx]
                h_prop = h_curr.copy()
                shift = rng.uniform(-0.15, 0.15)
                h_prop[i:j] += shift
                h_prop = project(h_prop)
            else:
                h_prop = h_curr.copy()
                i = rng.integers(0, N)
                h_prop[i] += rng.uniform(-0.1, 0.1)
                h_prop = project(h_prop)
        elif move_type < 0.6:
            # Antiferromagnetic flip on a block
            i = rng.integers(0, N - 5)
            length = rng.integers(3, 15)
            j = min(i + length, N)
            h_prop = h_curr.copy()
            h_prop[i:j] = 1.0 - h_curr[i:j]
            h_prop = project(h_prop)
        elif move_type < 0.85:
            # Pair swap
            i, j = rng.integers(0, N, size=2)
            h_prop = h_curr.copy()
            delta = rng.uniform(-0.08, 0.08)
            h_prop[i] += delta
            h_prop[j] -= delta
            h_prop = project(h_prop)
        else:
            # Random restart of a segment
            i = rng.integers(0, N - 10)
            length = rng.integers(5, 20)
            j = min(i + length, N)
            h_prop = h_curr.copy()
            h_prop[i:j] = rng.uniform(0.2, 0.8, size=j-i)
            h_prop = project(h_prop)
        
        score_curr = max_overlap(h_curr)
        score_prop = max_overlap(h_prop)
        
        delta = score_prop - score_curr
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            h_curr = h_prop
        
        curr_score = max_overlap(h_curr)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h_curr.copy()
    
    # Stage 2: Gradient-based refinement using smooth objective
    if time.time() < deadline - 15:
        h = best_h.copy()
        # Use L-BFGS-B with the smooth objective
        def obj_and_grad(x):
            x = np.clip(x, 0, 1)
            c = get_linear_correlations(x)
            temp = 0.02
            val = smooth_max(c, temp)
            # Gradient of smooth max w.r.t. c
            scaled = c / temp
            m = np.max(scaled)
            weights = np.exp(scaled - m) / np.sum(np.exp(scaled - m))
            # dc/dh for each shift k
            # C(k) = sum_i h[i]*(1-h[i+k])
            # dC(k)/d h[j] = (1-h[j+k]) - h[j-k] (with boundary handling)
            # For linear correlation, this is more complex. Use finite difference.
            eps = 1e-4
            grad = np.zeros(N)
            for i in range(N):
                h_plus = x.copy()
                h_plus[i] = min(1.0, x[i] + eps)
                h_minus = x.copy()
                h_minus[i] = max(0.0, x[i] - eps)
                c_plus = get_linear_correlations(h_plus)
                c_minus = get_linear_correlations(h_minus)
                val_plus = smooth_max(c_plus, temp)
                val_minus = smooth_max(c_minus, temp)
                grad[i] = (val_plus - val_minus) / (2 * eps)
            return val, grad
        
        # Use scipy minimize with the smooth objective
        # We'll do a few iterations of projected gradient
        lr = 0.003
        v = np.zeros(N)
        momentum = 0.9
        temp_schedule = [0.05, 0.03, 0.02, 0.01, 0.005]
        
        for temp in temp_schedule:
            if time.time() > deadline - 10:
                break
            for it in range(300):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient via finite differences (batched)
                eps = 1e-4
                grad = np.zeros(N)
                c_base = get_linear_correlations(h)
                base_val = smooth_max(c_base, temp)
                
                # Vectorized finite differences
                h_plus = h.copy()
                h_minus = h.copy()
                h_plus += eps
                h_minus -= eps
                h_plus = np.clip(h_plus, 0, 1)
                h_minus = np.clip(h_minus, 0, 1)
                
                # For each element, compute the change in smooth max
                # This is expensive, so do it in batches
                batch_size = 20
                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    for i in range(start, end):
                        hp = h.copy()
                        hp[i] = min(1.0, h[i] + eps)
                        hm = h.copy()
                        hm[i] = max(0.0, h[i] - eps)
                        cp = get_linear_correlations(hp)
                        cm = get_linear_correlations(hm)
                        vp = smooth_max(cp, temp)
                        vm = smooth_max(cm, temp)
                        grad[i] = (vp - vm) / (2 * eps)
                
                v = momentum * v - lr * grad
                h = h + v
                h = project(h)
                
                curr_score = max_overlap(h)
                if curr_score < best_score:
                    best_score = curr_score
                    best_h = h.copy()
    
    # Final: ensure validity
    best_h = project(best_h)
    
    # Verify sum constraint
    diff = target_sum - best_h.sum()
    if abs(diff) > 1e-10:
        mask = (best_h > 0) & (best_h < 1)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
