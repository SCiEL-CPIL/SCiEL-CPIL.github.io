import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def score(h):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def soft_max(values, beta):
        """Numerically stable soft-max."""
        shifted = values - np.max(values)
        return np.max(values) + np.log(np.exp(beta * shifted).sum()) / beta

    def score_smooth(h, beta=10.0):
        """Smooth approximation of the score using soft-max for the supremum."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        # Use soft-max to approximate max(conv) smoothly
        max_conv_smooth = soft_max(conv, beta)
        if max_conv_smooth < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv_smooth)

    def grad_score_smooth(h, beta=10.0):
        """
        Compute gradient of the smooth score with respect to h.
        score = ||h||_2^2 / (||h||_1 * max(conv))
        
        Let f = ||h||_2^2, g = ||h||_1, m = max(conv) (smoothed)
        score = f / (g * m)
        
        d(score)/dh = (f' * g * m - f * (g' * m + g * m')) / (g * m)^2
        f' = 2h
        g' = 1 (vector of ones)
        m' = gradient of soft-max of conv w.r.t. h
        """
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return np.zeros(N)
        
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        
        # Soft-max weights
        shifted = conv - np.max(conv)
        weights = np.exp(beta * shifted)
        weights /= weights.sum()
        m_smooth = np.max(conv) + np.log(weights.sum()) / beta  # This is wrong, let's redo
        
        # Correct soft-max
        shifted = conv - np.max(conv)
        exp_vals = np.exp(beta * shifted)
        log_sum = np.log(exp_vals.sum())
        m_smooth = np.max(conv) + log_sum / beta
        
        # Gradient of m_smooth w.r.t. conv is: beta * exp_vals / exp_vals.sum()
        # Actually, d(soft_max)/dx_i = exp(beta * (x_i - max)) / sum(exp(beta*(x_j - max)))
        # = exp_vals[i] / exp_vals.sum()
        grad_conv = exp_vals / exp_vals.sum()
        
        # conv = h * h (auto-convolution)
        # d(conv_k)/dh_i = h_{k-i} + h_{k-i} (if i != k-i) or 2*h_i if i = k-i
        # More precisely, conv_k = sum_j h_j * h_{k-j}
        # d(conv_k)/dh_i = h_{k-i} + h_{k-i} = 2*h_{k-i} (since convolution is symmetric)
        # Wait, conv_k = sum_j h_j h_{k-j}. 
        # d/dh_i (conv_k) = h_{k-i} + h_{k-i} = 2 h_{k-i} if we consider both terms.
        # Actually, if k-j = i, then j = k-i. So one term is h_i * h_{k-i}.
        # The derivative is h_{k-i} (from varying h_i in the first factor) + h_{k-i} (from varying h_i in the second factor, i.e., h_j where j=k-i, so if i=j then h_i^2, derivative 2h_i).
        # Generally, d(conv_k)/dh_i = h_{k-i} + h_{k-i} = 2 h_{k-i} is not quite right if indices overlap.
        # Let's use the standard formula: d(conv_k)/dh_i = h_{k-i} + h_{k-i} is wrong.
        # conv_k = sum_j h_j h_{k-j}.
        # Term 1: j=i -> h_i h_{k-i}. Derivative w.r.t h_i is h_{k-i}.
        # Term 2: k-j=i -> j=k-i -> h_{k-i} h_i. Derivative w.r.t h_i is h_{k-i}.
        # So if i != k-i, we have two terms: h_{k-i} + h_{k-i} = 2 h_{k-i}.
        # If i = k-i (i.e., 2i=k), we have one term h_i^2. Derivative is 2 h_i.
        # In both cases, the contribution is h_{k-i} + h_{k-i} = 2 h_{k-i} (with h_{k-i} = h_i when 2i=k).
        # So d(conv_k)/dh_i = h_{k-i} + h_{k-i} is correct if we define h_x = 0 for x outside [0, N-1].
        
        # grad_m[i] = sum_k grad_conv[k] * d(conv_k)/dh_i
        # grad_m[i] = sum_k grad_conv[k] * (h_{k-i} + h_{k-i}) = 2 * sum_k grad_conv[k] * h_{k-i}
        # This is 2 * (grad_conv * h)[i] where * is convolution? No, it's a cross-correlation or convolution.
        # Let's just compute it directly for simplicity, N=50 is small.
        grad_m = np.zeros(N)
        for i in range(N):
            for k in range(N):  # k is the index in conv, but conv has size 2N-1
                # Actually conv index goes from 0 to 2N-2.
                pass
        
        # Let's do it more efficiently using numpy.
        # grad_m = 2 * (grad_conv_full * h) where * is convolution?
        # d(conv_k)/dh_i = h_{k-i} + h_{k-i} = 2 h_{k-i}
        # grad_m[i] = sum_{k=0}^{2N-2} grad_conv[k] * 2 * h_{k-i}
        # This is 2 * sum_k grad_conv[k] h_{k-i}
        # Let v = h. Then sum_k grad_conv[k] v_{k-i}.
        # This is the cross-correlation of grad_conv and v, or convolution with flipped v.
        # Let's just loop.
        grad_m = np.zeros(N)
        for i in range(N):
            s = 0.0
            for k in range(2 * N - 1):
                idx = k - i
                if 0 <= idx < N:
                    s += grad_conv[k] * h[idx]
            grad_m[i] = 2.0 * s
        
        # Now compute gradient of score
        # score = f / (g * m)
        # f = l2_sq, g = l1, m = m_smooth
        # f' = 2h
        # g' = 1
        # m' = grad_m
        
        num = l2_sq
        den = l1 * m_smooth
        
        # d(score) = (f' * den - num * d(den)) / den^2
        # d(den) = g' * m + g * m' = m + l1 * grad_m (element-wise)
        
        df = 2.0 * h
        dg = np.ones(N)
        dm = grad_m
        
        d_den = dg * m_smooth + l1 * dm
        
        grad = (df * (l1 * m_smooth) - num * d_den) / (den * den)
        
        return grad

    def optimize_amplitudes_smooth(S, beta=10.0, max_iter=1000):
        """
        Optimize amplitudes for a given support S using gradient ascent on the smooth score.
        """
        if not S:
            return np.zeros(N), 0.0
        support = list(S)
        k = len(support)
        
        # Initial guess: uniform
        x0 = np.ones(k) / k
        x = x0.copy()
        
        lr = 0.1
        best_x = x0.copy()
        best_val = score_smooth(x0, beta)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Build full h
            h_full = np.zeros(N)
            h_full[support] = x
            
            # Compute gradient
            grad_full = grad_score_smooth(h_full, beta)
            grad_x = grad_full[support]
            
            # Gradient ascent
            x_new = x + lr * grad_x
            x_new = np.maximum(x_new, 1e-8)
            
            # Adaptive learning rate: if value decreases, reduce lr
            val_new = score_smooth(x_new, beta)
            if val_new < best_val - 1e-6:
                lr *= 0.5
                x_new = x  # Revert
            else:
                x = x_new
                if val_new > best_val:
                    best_val = val_new
                    best_x = x.copy()
                    lr *= 1.05  # Slightly increase lr if improving
                    lr = min(lr, 1.0)
            
            # Check convergence
            if np.max(np.abs(grad_x)) < 1e-8:
                break
        
        h_opt = np.zeros(N)
        h_opt[support] = best_x
        h_opt = np.maximum(h_opt, 0)
        s_opt = score(h_opt)
        return h_opt, s_opt

    def local_search_support(S_init, max_iters=3000):
        """
        Local search to minimize M_max (max multiplicity of sums) for a fixed size set.
        """
        S = set(S_init)
        k = len(S)
        
        def get_max_mult(S):
            if not S:
                return 0
            sums = {}
            S_list = list(S)
            for i in S_list:
                for j in S_list:
                    s = i + j
                    sums[s] = sums.get(s, 0) + 1
            return max(sums.values()) if sums else 0
        
        current_m = get_max_mult(S)
        
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            candidates = [x for x in range(N) if x not in S]
            random.shuffle(candidates)
            
            for i in S_list:
                for j in candidates:
                    S_trial = S.copy()
                    S_trial.remove(i)
                    S_trial.add(j)
                    m_trial = get_max_mult(S_trial)
                    if m_trial < current_m:
                        S = S_trial
                        current_m = m_trial
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                if random.random() < 0.1:
                    i = random.choice(list(S))
                    j = random.choice(candidates)
                    S.remove(i)
                    S.add(j)
                    current_m = get_max_mult(S)
                else:
                    break
        return S

    best_h = np.zeros(N)
    best_score = 0.0
    
    np.random.seed(42)
    random.seed(42)
    
    candidate_supports = []
    
    # Structured seeds
    candidate_supports.append([0, 1, 4, 10, 18, 23, 30])
    candidate_supports.append([0, 1, 4, 10, 18, 23, 30, 38])
    candidate_supports.append(list(range(5)))
    
    # Random sets of various sizes
    for k in range(5, 21):
        for _ in range(20):
            S = np.random.choice(N, k, replace=False)
            candidate_supports.append(list(S))
    
    processed = []
    for S_init in candidate_supports:
        if time.time() > deadline - 30:
            break
        S = set(S_init)
        h_uniform = np.zeros(N)
        for i in S:
            h_uniform[i] = 1.0 / len(S)
        s_uniform = score(h_uniform)
        
        S_improved = local_search_support(S, max_iters=2000)
        
        h_uniform_improved = np.zeros(N)
        for i in S_improved:
            h_uniform_improved[i] = 1.0 / len(S_improved)
        s_uniform_improved = score(h_uniform_improved)
        
        processed.append((s_uniform_improved, S_improved, h_uniform_improved))
    
    processed.sort(key=lambda x: x[0], reverse=True)
    
    top_supports = processed[:15]
    
    for s_proxy, S, h_proxy in top_supports:
        if time.time() > deadline - 15:
            break
        h_opt, s_opt = optimize_amplitudes_smooth(S, beta=10.0, max_iter=500)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Also try with different beta values for the top few
        if s_proxy > 1.05:
            for beta in [5.0, 20.0]:
                if time.time() > deadline - 10:
                    break
                h_opt2, s_opt2 = optimize_amplitudes_smooth(S, beta=beta, max_iter=300)
                if s_opt2 > best_score:
                    best_score = s_opt2
                    best_h = h_opt2.copy()
    
    if best_score <= 1.0:
        h = [0.0] * N
        h[N // 2] = 1.0
        return h
        
    return list(best_h)
