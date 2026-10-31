import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def generate_sidon_sets():
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(42)
    for _ in range(10000):
        k = np.random.randint(5, 12)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 80:
                    break
    
    return sidon_sets

def frank_wolfe_optimize(support, max_iter=3000, tol=1e-8, time_limit=None):
    """
    Frank-Wolfe (conditional gradient) method to minimize max of autoconvolution
    over the probability simplex on the given support.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initialize uniformly
    h_local = np.ones(k) / k
    h_full = np.zeros(N)
    h_full[support] = h_local
    
    t0 = time.time() if time_limit else 0
    deadline = t0 + time_limit if time_limit else float('inf')
    
    # Step size parameters
    step_size = 2.0  # alpha_t = 2/(t+2)
    
    for t in range(max_iter):
        if time_limit and time.time() > deadline:
            break
            
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        
        # Compute gradient: grad_j = sum_i h_i * (conv[j-i] for valid i)
        # In the full vector representation, grad[h_full[m]] = 2 * sum_{n} h_full[n] * conv[m-n+N-1] (with proper indexing)
        # More precisely, the derivative of (h*h)[m] wrt h[n] is h[m-n] + h[n-m] (if in range)
        # Since conv is symmetric, and we're maximizing sum h^2 / max(conv), the gradient of max(conv) wrt h_full
        # can be approximated. Let's compute the gradient of the objective max(conv).
        # grad_full[m] = sum_n h_full[n] * (derivative of conv[m] wrt h_full[n])
        # conv[m] = sum_i h_full[i] * h_full[m-i] (with appropriate bounds)
        # d(conv[m])/d(h_full[n]) = h_full[m-n] if m-n is in [0,N-1], else 0
        #                     + h_full[n-m] if n-m is in [0,N-1], else 0 (but since conv is symmetric, these are the same term if we define it properly)
        # Actually, conv_full = h_full * h_full. The gradient wrt h_full is 2 * (h_full * conv_full) but with proper alignment?
        # No, the gradient of (h*h)[m] wrt h[n] is h[m-n] (when indices are valid).
        # So grad_full[m] = sum_n h_full[n] * h_full[m-n] (valid) = conv_full[m]
        # Wait, that's just the convolution value itself.
        # Let's re-derive. f(h) = max_m (h*h)[m]. The subgradient wrt h[n] is sum_m I(m=argmax) * h[m-n].
        # If we approximate the max with a soft-max or just take the argmax, the gradient is sparse.
        # For Frank-Wolfe, we need the gradient at the current point.
        
        # Let's use the fact that for a fixed argmax m*, grad[n] = h[m*-n] (if valid).
        # But max can have multiple argmaxes. Let's use all argmaxes with equal weight for stability.
        argmaxes = np.where(conv >= max_c - 1e-12)[0]
        
        grad_local = np.zeros(k)
        for m_star in argmaxes:
            for idx, s in enumerate(support):
                # contribution from h[s] to conv[m_star] is h[m_star - s] if m_star-s is in support
                offset = m_star - s
                if 0 <= offset < N:
                    # find if offset is in support
                    if offset in support:
                        pos = support.index(offset)
                        grad_local[idx] += h_local[pos]
        
        # The gradient of the objective (which is max_c, we want to minimize it) is grad_local
        # But we want to maximize h2_sq / max_c. Since h2_sq is fixed if we just move on simplex? No, h2_sq changes.
        # Actually, the objective is R = ||h||_2^2 / max(h*h).
        # Let's just minimize max(h*h) for now, assuming h2_sq is roughly constant or we optimize jointly?
        # The prompt suggests transforming to quadratic form. But Frank-Wolfe is for linear objectives.
        # Let's try to maximize the ratio directly? No, that's non-convex.
        # Let's stick to minimizing the denominator max(h*h) while keeping sum h = 1.
        # This might not increase the numerator enough.
        
        # Let's try a different approach: Optimize the weights to maximize the ratio.
        # Since R = sum(h^2) / max(conv), and sum h = 1.
        # This is hard.
        
        # Let's go back to the log-sum-exp soft-max surrogate for the max in the denominator,
        # as suggested in the directive, and use gradient ascent on the log-likelihood or similar?
        # Or just use the existing L-BFGS-B but with a better initialization from FW?
        
        # Actually, the directive says: "Model the height vector as the squared magnitude of a complex signal g, i.e., h = |g|^2... exploit linearity of gradient with respect to h".
        # This implies we should optimize h directly using FW on the simplex, minimizing a proxy for max(conv).
        
        # Let's implement the FW step:
        # 1. Compute gradient g = grad(max(conv)) at h.
        # 2. Find s = argmin_{j in support} g_j (the vertex of the simplex with lowest gradient).
        # 3. Update h = h + alpha * (e_s - h), where e_s is the unit vector at s.
        
        if len(argmaxes) == 0:
            break
            
        # Normalize gradient by number of argmaxes? Or just sum?
        # The subgradient is sum of gradients of active constraints.
        # Let's just use the average gradient for stability.
        grad_avg = grad_local / len(argmaxes)
        
        # Find the vertex that minimizes the linear approximation
        min_grad_idx = np.argmin(grad_avg)
        
        # Step size alpha_t = 2/(t+2)
        alpha = 2.0 / (t + 2)
        
        # Update
        h_new = h_local.copy()
        h_new -= alpha * h_local
        h_new[min_grad_idx] += alpha
        
        # Check for improvement
        h_full_new = np.zeros(N)
        h_full_new[support] = h_new
        sc_new = compute_ratio(h_full_new)
        sc_old = compute_ratio(h_full)
        
        if sc_new > sc_old + 1e-10:
            h_local = h_new
            h_full = h_full_new
        else:
            # Try smaller step size
            alpha_half = alpha / 2
            h_new2 = h_local.copy()
            h_new2 -= alpha_half * h_local
            h_new2[min_grad_idx] += alpha_half
            h_full_new2 = np.zeros(N)
            h_full_new2[support] = h_new2
            sc_new2 = compute_ratio(h_full_new2)
            if sc_new2 > sc_old + 1e-10:
                h_local = h_new2
                h_full = h_full_new2
            else:
                # No improvement, maybe converged
                break
                
    sc_final = compute_ratio(h_full)
    return h_full, sc_final

def optimize_weights_for_support(support, beta=100.0, maxiter=2000, ftol=1e-14):
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    x0 = np.zeros(k)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x):
        h = softmax(x)
        h_full = np.zeros(N)
        h_full[support] = h
        
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        obj_val = max_c + (1.0 / beta) * np.log(sum_exp)
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h_full[m]
            if hm > 1e-15:
                end = min(m + N, 2 * N - 1)
                grad_h[:N] += 2.0 * hm * weights[m:end]
        
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h)
        grad_x = h * (grad_h_support - dot_prod)
        
        return obj_val, grad_x
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-12})
        x_opt = res.x
        h_opt_local = softmax(x_opt)
        
        h_full = np.zeros(N)
        h_full[support] = h_opt_local
        sc = compute_ratio(h_full)
        return h_full, sc
    except Exception:
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        sc = compute_ratio(h_full)
        return h_full, sc

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    best_support = [0]
    
    sidon_sets = generate_sidon_sets()
    sidon_sets.sort(key=lambda x: -len(x))
    
    # Phase 1: Try Sidon sets with both L-BFGS and Frank-Wolfe
    for s in sidon_sets:
        if time.time() > deadline - 30.0:
            break
        
        # L-BFGS optimization
        for beta in [30.0, 60.0, 120.0]:
            if time.time() > deadline - 30.0:
                break
            h_opt, sc = optimize_weights_for_support(s, beta=beta, maxiter=1500)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
        
        # Frank-Wolfe optimization on the same support
        if time.time() > deadline - 30.0:
            break
        h_fw, sc_fw = frank_wolfe_optimize(s, max_iter=500, time_limit=2.0)
        if sc_fw > best_score:
            best_score = sc_fw
            best_h = h_fw.copy()
            best_support = list(s)
    
    # Phase 2: Local search with swaps, using both optimizers
    if time.time() < deadline - 20.0:
        improved = True
        while improved and time.time() < deadline - 20.0:
            improved = False
            for old_idx in best_support:
                for new_idx in range(N):
                    if new_idx in best_support:
                        continue
                    new_support = [x for x in best_support if x != old_idx]
                    new_support.append(new_idx)
                    new_support.sort()
                    
                    if not is_sidon_set(new_support):
                        continue
                    
                    # Try L-BFGS
                    h_new, sc_new = optimize_weights_for_support(new_support, beta=80.0, maxiter=500)
                    if sc_new > best_score + 1e-8:
                        best_h = h_new
                        best_score = sc_new
                        best_support = new_support
                        improved = True
                        break
                    
                    # Try Frank-Wolfe
                    if time.time() > deadline - 20.0:
                        break
                    h_fw, sc_fw = frank_wolfe_optimize(new_support, max_iter=300, time_limit=1.0)
                    if sc_fw > best_score + 1e-8:
                        best_h = h_fw
                        best_score = sc_fw
                        best_support = new_support
                        improved = True
                        break
                if improved:
                    break
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 10.0:
        for beta in [200.0, 400.0, 800.0]:
            if time.time() > deadline - 10.0:
                break
            h_polish, sc_polish = optimize_weights_for_support(best_support, beta=beta, maxiter=3000, ftol=1e-15)
            if sc_polish > best_score:
                best_score = sc_polish
                best_h = h_polish.copy()
        
        # Final FW polish
        if time.time() < deadline - 5.0:
            h_fw_final, sc_fw_final = frank_wolfe_optimize(best_support, max_iter=1000, time_limit=deadline - time.time() - 5.0)
            if sc_fw_final > best_score:
                best_score = sc_fw_final
                best_h = h_fw_final.copy()
    
    return list(best_h)
