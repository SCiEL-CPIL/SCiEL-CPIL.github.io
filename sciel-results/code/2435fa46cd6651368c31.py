import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def autoconv(h):
        return np.convolve(h, h)

    def calculate_ratio(h):
        s = np.sum(h)
        if s == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        c = autoconv(h)
        peak = np.max(c)
        if peak == 0:
            return 0.0
        return l2_sq / (s * peak)

    best_h = np.ones(N) / N
    best_ratio = calculate_ratio(best_h)

    def find_sidon_set(N, target_size, seed=0):
        rng = np.random.RandomState(seed)
        S = [0]
        sums = set([0])
        candidates = list(range(1, N))
        rng.shuffle(candidates)
        for x in candidates:
            if len(S) >= target_size:
                break
            valid = True
            new_sums = set()
            if 2 * x in sums:
                valid = False
            else:
                new_sums.add(2 * x)
                for s in S:
                    sm = x + s
                    if sm in sums:
                        valid = False
                        break
            if valid:
                S.append(x)
                sums.update(new_sums)
        return S

    def optimize_weights_on_support(support, n_iter=200, lr=0.1):
        K = len(support)
        if K == 0:
            return np.zeros(N)
        
        # Initialize with exponential decay
        w = np.ones(K)
        for i in range(1, K):
            w[i] = 0.8 ** i
        w = w / np.sum(w)
        
        best_w = w.copy()
        best_val = -np.inf
        
        for it in range(n_iter):
            h = np.zeros(N)
            for i, s in enumerate(support):
                h[s] = w[i]
            
            c = autoconv(h)
            peak = np.max(c)
            l2_sq = np.sum(h**2)
            val = l2_sq / (peak)  # Since sum(w)=1, l1=1
            
            if val > best_val:
                best_val = val
                best_w = w.copy()
            
            # Gradient ascent on w (unconstrained, will project)
            # Objective: sum(w_i^2) / max_k c_k
            # Let M = max_k c_k
            # d/dw_i [sum(w_j^2) / M] = (2*w_i*M - sum(w_j^2)*dM/dw_i) / M^2
            # dM/dw_i: if the max is unique at index k*, dM/dw_i = 2*w_{k*-i} (if valid)
            # But we can use a smoother approach: softmax over top few peaks
            
            # For simplicity and robustness, use subgradient
            # Maximize f(w) = sum(w_i^2) / M(w)
            # grad f = (2w*M - (sum w^2) * grad M) / M^2
            
            # Find indices of max
            max_idx = np.where(c == peak)[0]
            if len(max_idx) == 0:
                break
            
            # Average gradient of M over max indices for stability
            grad_M = np.zeros(K)
            for k in max_idx:
                for i in range(K):
                    s_i = support[i]
                    j = k - s_i
                    # Find if j is in support
                    for m in range(K):
                        if support[m] == j:
                            grad_M[i] += 2.0 * w[m]
                            break
            
            grad_M /= len(max_idx)
            
            sum_w2 = np.sum(w**2)
            grad_f = (2.0 * w * peak - sum_w2 * grad_M) / (peak * peak)
            
            # Update
            w += lr * grad_f
            
            # Project onto simplex (sum=1, w>=0)
            # Simple projection: normalize after clipping
            w = np.clip(w, 0, None)
            s = np.sum(w)
            if s > 0:
                w = w / s
            else:
                w = np.ones(K) / K
        
        h_best = np.zeros(N)
        for i, s in enumerate(support):
            h_best[s] = best_w[i]
        return h_best

    # Try various Sidon set sizes and seeds
    for target_size in range(3, 12):
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            support = find_sidon_set(N, target_size, seed)
            if len(support) < 2:
                continue
            
            h_cand = optimize_weights_on_support(support, n_iter=100, lr=0.05)
            ratio = calculate_ratio(h_cand)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_cand.copy()

    # Also try some random sparse supports
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        
        K = np.random.randint(3, 10)
        support = np.random.choice(N, K, replace=False)
        h_cand = optimize_weights_on_support(support, n_iter=50, lr=0.05)
        ratio = calculate_ratio(h_cand)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h_cand.copy()

    # Final polish: try to improve best_h with small perturbations
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        
        h_pert = best_h.copy()
        h_pert += np.random.normal(0, 0.01, N)
        h_pert = np.clip(h_pert, 0, None)
        s = np.sum(h_pert)
        if s > 0:
            h_pert = h_pert / s
        
        ratio = calculate_ratio(h_pert)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h_pert.copy()

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
