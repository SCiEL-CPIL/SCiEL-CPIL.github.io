import numpy as np
import time

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Uses linear (non-circular) correlation: C(k) = sum_i h[i]*(1-h[i+k])
    Optimized via log-sum-exp smooth max + simulated annealing + analytic gradient polish.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 100
    target_sum = N / 2.0
    
    def get_linear_correlations(h):
        """
        Returns array of length 2*(N-1) with:
          indices 0..N-2: C_neg(k) = sum_i h[i]*(1-h[i-k]) for k=1..N-1 (negative shifts)
          indices N-1..2N-3: C_pos(k) = sum_i h[i]*(1-h[i+k]) for k=1..N-1 (positive shifts)
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c has length 2N-1
        # c[N-1] = sum h[i]*g[i] (shift 0)
        # c[N-1+j] = sum h[i]*g[i+j] = sum h[i]*(1-h[i+j]) for j=0..N-1 (positive shifts)
        # c[N-1-j] = sum h[i]*g[i-j] = sum h[i]*(1-h[i-j]) for j=0..N-1 (negative shifts)
        # Positive shifts j=1..N-1: indices N..2N-2
        # Negative shifts j=1..N-1: indices 0..N-2
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
    
    def smooth_max_grad(c, temp=0.03):
        """Returns (value, softmax_weights) for log-sum-exp."""
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:
            val = m * temp
            weights = np.zeros_like(c)
            weights[np.argmax(c)] = 1.0
            return val, weights
        exp_scaled = np.exp(scaled - m)
        sum_exp = np.sum(exp_scaled)
        val = temp * (m + np.log(sum_exp))
        weights = exp_scaled / sum_exp
        return val, weights
    
    def analytic_gradient(h, temp=0.03):
        """
        Compute gradient of smooth_max(C(h), temp) w.r.t. h.
        
        C_pos(k) = sum_i h[i]*(1-h[i+k]) for i=0..N-1-k, k=1..N-1
        C_neg(k) = sum_i h[i]*(1-h[i-k]) for i=k..N-1, k=1..N-1
        
        dC_pos(k)/dh[j] = (1-h[j+k]) for j=0..N-1-k  (j is first index)
                         - h[j-k] for j=k..N-1      (j is second index, h[j-k] is first)
        
        More carefully:
        C_pos(k) = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
        dC_pos(k)/dh[j]:
          - If j appears as first index: h[j]*(1-h[j+k]), contributes (1-h[j+k]) if j+k < N
          - If j appears as second index: h[j-k]*(1-h[j]), contributes -h[j-k] if j-k >= 0
        
        C_neg(k) = sum_{i=k}^{N-1} h[i]*(1-h[i-k])
        dC_neg(k)/dh[j]:
          - If j appears as first index: h[j]*(1-h[j-k]), contributes (1-h[j-k]) if j-k >= 0
          - If j appears as second index: h[j+k]*(1-h[j]), contributes -h[j+k] if j+k < N
        """
        c = get_linear_correlations(h)
        val, weights = smooth_max_grad(c, temp)
        
        # weights[0..N-2] correspond to C_neg(1)..C_neg(N-1)
        # weights[N-1..2N-3] correspond to C_pos(1)..C_pos(N-1)
        w_neg = weights[:N-1]  # for k=1..N-1
        w_pos = weights[N-1:]  # for k=1..N-1
        
        grad = np.zeros(N)
        
        # Contribution from C_pos(k) = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
        # For each k (1-indexed, stored at index k-1 in w_pos):
        #   dC_pos(k)/dh[j] = (1-h[j+k]) if j <= N-1-k (j is first index)
        #                     -h[j-k] if j >= k (j is second index)
        for k_idx in range(N-1):
            k = k_idx + 1
            w = w_pos[k_idx]
            if w < 1e-15:
                continue
            # First index contribution: j=0..N-1-k, term = (1-h[j+k])
            first_indices = np.arange(0, N-k)
            grad[first_indices] += w * (1.0 - h[first_indices + k])
            # Second index contribution: j=k..N-1, term = -h[j-k]
            second_indices = np.arange(k, N)
            grad[second_indices] += w * (-h[second_indices - k])
        
        # Contribution from C_neg(k) = sum_{i=k}^{N-1} h[i]*(1-h[i-k])
        # For each k (1-indexed, stored at index k-1 in w_neg):
        #   dC_neg(k)/dh[j] = (1-h[j-k]) if j >= k (j is first index)
        #                     -h[j+k] if j <= N-1-k (j is second index)
        for k_idx in range(N-1):
            k = k_idx + 1
            w = w_neg[k_idx]
            if w < 1e-15:
                continue
            # First index contribution: j=k..N-1, term = (1-h[j-k])
            first_indices = np.arange(k, N)
            grad[first_indices] += w * (1.0 - h[first_indices - k])
            # Second index contribution: j=0..N-1-k, term = -h[j+k]
            second_indices = np.arange(0, N-k)
            grad[second_indices] += w * (-h[second_indices + k])
        
        return val, grad
    
    def project(h):
        """Project to 0<=h<=1, sum(h)=N/2 using bisection on Lagrange multiplier."""
        h = np.clip(h, 0, 1)
        # Use bisection to find lambda such that sum(clip(h - lambda, 0, 1)) = target_sum
        # f(lambda) = sum(clip(h - lambda, 0, 1)) is decreasing in lambda
        lo, hi = -1.0, 2.0
        for _ in range(60):
            mid = (lo + hi) / 2
            s = np.sum(np.clip(h - mid, 0, 1))
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        h = np.clip(h - (lo + hi) / 2, 0, 1)
        # Final adjustment for any residual
        diff = target_sum - h.sum()
        if abs(diff) > 1e-12:
            mask = (h > 1e-15) & (h < 1 - 1e-15)
            if np.any(mask):
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
    
    # Stage 2: Gradient-based refinement using analytic gradient
    if time.time() < deadline - 15:
        h = best_h.copy()
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
                
                val, grad = analytic_gradient(h, temp)
                
                v = momentum * v - lr * grad
                h = h + v
                h = project(h)
                
                curr_score = max_overlap(h)
                if curr_score < best_score:
                    best_score = curr_score
                    best_h = h.copy()
    
    # Stage 3: High-precision polish with very small temperature
    if time.time() < deadline - 5:
        h = best_h.copy()
        for temp in [0.003, 0.001, 0.0005]:
            if time.time() > deadline - 3:
                break
            for it in range(200):
                if time.time() > deadline - 2:
                    break
                val, grad = analytic_gradient(h, temp)
                h = h - lr * grad
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
