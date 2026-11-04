import numpy as np
import time

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Uses linear (non-circular) correlation: C(k) = sum_i h[i]*(1-h[i+k])
    Optimized via epigraph formulation (auxiliary t) + log-sum-exp smooth max
    + simulated annealing + analytic gradient polish with diverse restarts.
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
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def max_overlap(h):
        c = get_linear_correlations(h)
        return np.max(c)
    
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
        """
        c = get_linear_correlations(h)
        val, weights = smooth_max_grad(c, temp)
        
        w_neg = weights[:N-1]
        w_pos = weights[N-1:]
        
        grad = np.zeros(N)
        
        # C_pos(k) = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
        for k_idx in range(N-1):
            k = k_idx + 1
            w = w_pos[k_idx]
            if w < 1e-15:
                continue
            first_indices = np.arange(0, N-k)
            grad[first_indices] += w * (1.0 - h[first_indices + k])
            second_indices = np.arange(k, N)
            grad[second_indices] += w * (-h[second_indices - k])
        
        # C_neg(k) = sum_{i=k}^{N-1} h[i]*(1-h[i-k])
        for k_idx in range(N-1):
            k = k_idx + 1
            w = w_neg[k_idx]
            if w < 1e-15:
                continue
            first_indices = np.arange(k, N)
            grad[first_indices] += w * (1.0 - h[first_indices - k])
            second_indices = np.arange(0, N-k)
            grad[second_indices] += w * (-h[second_indices + k])
        
        return val, grad
    
    def project(h):
        """Project to 0<=h<=1, sum(h)=N/2 using bisection on Lagrange multiplier."""
        h = np.clip(h, 0, 1)
        lo, hi = -1.0, 2.0
        for _ in range(60):
            mid = (lo + hi) / 2
            s = np.sum(np.clip(h - mid, 0, 1))
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        h = np.clip(h - (lo + hi) / 2, 0, 1)
        diff = target_sum - h.sum()
        if abs(diff) > 1e-12:
            mask = (h > 1e-15) & (h < 1 - 1e-15)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        return h
    
    def epigraph_objective_and_grad(h, t, temp=0.03):
        """
        Epigraph formulation: minimize t subject to C(k) <= t for all k.
        Smoothed: L = t + sum_k max(0, C(k) - t) / temp * log-sum-exp style
        Actually, use: L = t + (1/temp) * log(sum_k exp(temp*(C(k)-t)))
        dL/dt = 1 - sum_k exp(temp*(C(k)-t)) / sum_k exp(temp*(C(k)-t)) = 0 at optimum
        dL/dh = (1/temp) * sum_k [temp * dC(k)/dh * exp(temp*(C(k)-t))] / sum_k exp(temp*(C(k)-t))
              = sum_k w_k * dC(k)/dh
        where w_k = exp(temp*(C(k)-t)) / sum_k exp(temp*(C(k)-t))
        """
        c = get_linear_correlations(h)
        scaled = temp * (c - t)
        m = np.max(scaled)
        if m > 700:
            val = t
            weights = np.zeros_like(c)
            weights[np.argmax(c)] = 1.0
        else:
            exp_scaled = np.exp(scaled - m)
            sum_exp = np.sum(exp_scaled)
            val = t + (1.0 / temp) * (m + np.log(sum_exp))
            weights = exp_scaled / sum_exp
        
        w_neg = weights[:N-1]
        w_pos = weights[N-1:]
        
        grad_h = np.zeros(N)
        for k_idx in range(N-1):
            k = k_idx + 1
            w = w_pos[k_idx]
            if w < 1e-15:
                continue
            first_indices = np.arange(0, N-k)
            grad_h[first_indices] += w * (1.0 - h[first_indices + k])
            second_indices = np.arange(k, N)
            grad_h[second_indices] += w * (-h[second_indices - k])
        
        for k_idx in range(N-1):
            k = k_idx + 1
            w = w_neg[k_idx]
            if w < 1e-15:
                continue
            first_indices = np.arange(k, N)
            grad_h[first_indices] += w * (1.0 - h[first_indices - k])
            second_indices = np.arange(0, N-k)
            grad_h[second_indices] += w * (-h[second_indices + k])
        
        # grad_t = 1 - sum(weights) = 0 by construction of softmax
        # But for the epigraph, we want to push t down when all C(k) < t
        # The gradient of the smoothed max w.r.t. t is 1 - sum(w_k) = 0
        # So we just use the smooth max value as the objective
        return val, grad_h
    
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
    
    def gradient_refine(h_init, temp_schedule, max_iters_per_temp, lr_init=0.003, momentum=0.9):
        """Gradient-based refinement with adaptive learning rate."""
        h = h_init.copy()
        v = np.zeros(N)
        best_local = h.copy()
        best_local_score = max_overlap(h)
        lr = lr_init
        
        for temp in temp_schedule:
            for it in range(max_iters_per_temp):
                if time.time() > deadline - 5:
                    return best_local, best_local_score
                
                val, grad = analytic_gradient(h, temp)
                
                # Adaptive LR: reduce if gradient norm is large
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1.0:
                    lr_eff = lr / (1.0 + grad_norm)
                else:
                    lr_eff = lr
                
                v = momentum * v - lr_eff * grad
                h_new = h + v
                h_new = project(h_new)
                
                curr_score = max_overlap(h_new)
                if curr_score < best_local_score:
                    best_local_score = curr_score
                    best_local = h_new.copy()
                
                # Accept if improved or with small noise
                if curr_score <= max_overlap(h) + 1e-8:
                    h = h_new
                else:
                    v *= 0.5
                
                # Decay learning rate
                lr *= 0.995
        
        return best_local, best_local_score
    
    # Initialize with diverse starting points
    rng = np.random.default_rng(42)
    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)
    
    # Multiple diverse initializations
    initializations = []
    # Uniform
    initializations.append(np.full(N, 0.5))
    # Sine wave
    x = np.linspace(0, 2*np.pi, N)
    initializations.append(0.5 + 0.3 * np.sin(x))
    # Cosine wave
    initializations.append(0.5 + 0.3 * np.cos(x))
    # Random
    initializations.append(rng.uniform(0.1, 0.9, N))
    # Block pattern
    h_block = np.zeros(N)
    for i in range(N):
        h_block[i] = 0.8 if (i % 4 < 2) else 0.2
    initializations.append(h_block)
    
    for h_init in initializations:
        if time.time() > deadline - 40:
            break
        h_init = project(h_init)
        score_init = max_overlap(h_init)
        if score_init < best_score:
            best_score = score_init
            best_h = h_init.copy()
    
    # Stage 1: Simulated annealing from best initialization
    h_curr = best_h.copy()
    T_init = 0.12
    T_final = 0.0003
    n_sa = 3500
    
    for it in range(n_sa):
        if time.time() > deadline - 35:
            break
        
        progress = it / n_sa
        T = T_init * (T_final / T_init) ** progress
        
        move_type = rng.random()
        if move_type < 0.30:
            # Cluster shift
            clusters = find_clusters(h_curr, threshold=0.06)
            if clusters:
                idx = rng.integers(0, len(clusters))
                i, j = clusters[idx]
                h_prop = h_curr.copy()
                shift = rng.uniform(-0.12, 0.12)
                h_prop[i:j] += shift
                h_prop = project(h_prop)
            else:
                h_prop = h_curr.copy()
                i = rng.integers(0, N)
                h_prop[i] += rng.uniform(-0.08, 0.08)
                h_prop = project(h_prop)
        elif move_type < 0.50:
            # Antiferromagnetic flip on a block
            i = rng.integers(0, N - 3)
            length = rng.integers(2, 12)
            j = min(i + length, N)
            h_prop = h_curr.copy()
            h_prop[i:j] = 1.0 - h_curr[i:j]
            h_prop = project(h_prop)
        elif move_type < 0.75:
            # Pair swap
            i, j = rng.integers(0, N, size=2)
            if i == j:
                j = (i + 1) % N
            h_prop = h_curr.copy()
            delta = rng.uniform(-0.06, 0.06)
            h_prop[i] += delta
            h_prop[j] -= delta
            h_prop = project(h_prop)
        elif move_type < 0.90:
            # Random restart of a segment
            i = rng.integers(0, N - 5)
            length = rng.integers(4, 15)
            j = min(i + length, N)
            h_prop = h_curr.copy()
            h_prop[i:j] = rng.uniform(0.15, 0.85, size=j-i)
            h_prop = project(h_prop)
        else:
            # Large-scale perturbation
            h_prop = h_curr.copy()
            n_pert = rng.integers(5, 20)
            indices = rng.choice(N, size=n_pert, replace=False)
            h_prop[indices] = rng.uniform(0.1, 0.9, size=n_pert)
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
    
    # Stage 2: Gradient-based refinement with epigraph-inspired smooth max
    if time.time() < deadline - 20:
        h_refined, score_refined = gradient_refine(
            best_h,
            temp_schedule=[0.05, 0.03, 0.02, 0.01, 0.005, 0.003],
            max_iters_per_temp=250,
            lr_init=0.004,
            momentum=0.9
        )
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined.copy()
    
    # Stage 3: Additional restart from perturbed best
    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            h_pert = best_h.copy()
            n_pert = rng.integers(3, 10)
            indices = rng.choice(N, size=n_pert, replace=False)
            h_pert[indices] = rng.uniform(0.1, 0.9, size=n_pert)
            h_pert = project(h_pert)
            
            h_refined, score_refined = gradient_refine(
                h_pert,
                temp_schedule=[0.03, 0.015, 0.008, 0.004, 0.002],
                max_iters_per_temp=200,
                lr_init=0.003,
                momentum=0.9
            )
            if score_refined < best_score:
                best_score = score_refined
                best_h = h_refined.copy()
    
    # Stage 4: High-precision polish
    if time.time() < deadline - 5:
        h_refined, score_refined = gradient_refine(
            best_h,
            temp_schedule=[0.002, 0.001, 0.0005, 0.0003, 0.0002],
            max_iters_per_temp=150,
            lr_init=0.001,
            momentum=0.95
        )
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined.copy()
    
    # Final: ensure validity
    best_h = project(best_h)
    
    diff = target_sum - best_h.sum()
    if abs(diff) > 1e-10:
        mask = (best_h > 0) & (best_h < 1)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
