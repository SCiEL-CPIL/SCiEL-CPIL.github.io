import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * Sup(AutoConv))."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _objective_and_grad(h):
    """
    Compute the objective (negative of normalized score) and its gradient.
    
    We maximize: S(h) = L2^2 / (L1 * AutoConv_max)
    Subject to: L1 = 1, h >= 0
    
    With L1 = 1: S(h) = L2^2 / AutoConv_max
    
    Let m = argmax(conv(h,h)), and let M = conv(h,h)[m].
    Then AutoConv_max = DELTA * M.
    
    S(h) = (sum h_i^2) / (DELTA * M)
    
    dS/dh_j = [2*h_j * M - sum(h_i^2) * dM/dh_j] / (DELTA * M^2)
    
    Now, dM/dh_j where M = conv(h,h)[m].
    conv(h,h)[k] = sum_i h_i * h_{k-i} (with appropriate indexing)
    
    For the full convolution of length 2N-1, index k (0-based) corresponds to:
    conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i] * h[k-i]
    
    d(conv[k])/dh[j] = h[k-j] (if 0 <= k-j < N) + h[j-k] (if 0 <= j-k < N)
    But we need to be careful about the indexing.
    
    Actually, np.convolve(h, h, mode='full') gives:
    result[k] = sum_{i} h[i] * h[k-i] where the sum is over valid i.
    
    More precisely, for full convolution of two arrays of length N:
    result has length 2N-1.
    result[k] = sum_{i=max(0, k-N+1)}^{min(N-1, k)} h[i] * h[k-i]
    
    d(result[k])/dh[j] = h[k-j] (if 0 <= k-j < N) + h[j-k] (if 0 <= j-k < N)
    Wait, that's not quite right. Let me think again.
    
    result[k] = sum_i h[i] * h[k-i]
    d(result[k])/dh[j] = sum_i [delta_{i,j} * h[k-i] + h[i] * delta_{k-i,j}]
                       = h[k-j] + h[k-j]  (if both terms are valid)
                       = 2*h[k-j] (if 0 <= k-j < N)
    
    Wait, no. The second term: h[i] * delta_{k-i,j} is nonzero when i = k-j.
    So d(result[k])/dh[j] = h[k-j] (from first term, i=j) + h[k-j] (from second term, i=k-j)
    But only if both indices are valid.
    
    Actually, let's be more careful.
    result[k] = sum_{i=0}^{N-1} h[i] * h[k-i] (with h[x] = 0 if x < 0 or x >= N)
    
    d(result[k])/dh[j] = sum_{i=0}^{N-1} [delta_{i,j} * h[k-i] + h[i] * delta_{k-i,j}]
                       = h[k-j] + h[k-j]  (if 0 <= k-j < N)
                       = 2 * h[k-j]
    
    But wait, this assumes that when we differentiate h[i] with respect to h[j], we get delta_{i,j},
    and when we differentiate h[k-i] with respect to h[j], we get delta_{k-i,j} = delta_{i,k-j}.
    
    So d(result[k])/dh[j] = h[k-j] + h[k-j] = 2*h[k-j] (if 0 <= k-j < N), else 0.
    
    Hmm, but this doesn't seem right either. Let me test with a small example.
    
    h = [a, b, c] (N=3)
    conv = [a^2, ab, ac+b^2, ab, c^2]
    
    d(conv[0])/da = 2a, d(conv[0])/db = 0, d(conv[0])/dc = 0
    d(conv[1])/da = b, d(conv[1])/db = a, d(conv[1])/dc = 0
    d(conv[2])/da = c, d(conv[2])/db = 2b, d(conv[2])/dc = a
    d(conv[3])/da = 0, d(conv[3])/db = c, d(conv[3])/dc = b
    d(conv[4])/da = 0, d(conv[4])/db = 0, d(conv[4])/dc = 2c
    
    So for k=1, j=0: d(conv[1])/da = b = h[1-0] = h[1]. And 2*h[1-0] = 2b. That's wrong.
    
    Let me re-derive.
    result[k] = sum_{i=0}^{N-1} h[i] * h[k-i]
    
    d(result[k])/dh[j] = sum_{i=0}^{N-1} [delta_{i,j} * h[k-i] + h[i] * delta_{k-i,j}]
    
    First term: i=j, so h[k-j] (if 0 <= k-j < N)
    Second term: k-i=j, so i=k-j, so h[k-j] (if 0 <= k-j < N)
    
    So d(result[k])/dh[j] = h[k-j] + h[k-j] = 2*h[k-j] (if 0 <= k-j < N)
    
    But from the example: d(conv[1])/da = b, not 2b.
    
    Ah, I see the issue. When k-j is valid, we get h[k-j] from the first term AND h[k-j] from the second term.
    But in the example, k=1, j=0: k-j=1, which is valid. So we should get 2*h[1] = 2b.
    But the actual derivative is b.
    
    Let me recompute the example more carefully.
    h = [a, b, c]
    conv[1] = h[0]*h[1] + h[1]*h[0] = a*b + b*a = 2ab
    
    Oh! I made an error in the example. conv[1] = 2ab, not ab.
    So d(conv[1])/da = 2b, which matches 2*h[1].
    
    Let me redo the full convolution:
    conv[0] = h[0]*h[0] = a^2
    conv[1] = h[0]*h[1] + h[1]*h[0] = 2ab
    conv[2] = h[0]*h[2] + h[1]*h[1] + h[2]*h[0] = 2ac + b^2
    conv[3] = h[1]*h[2] + h[2]*h[1] = 2bc
    conv[4] = h[2]*h[2] = c^2
    
    So:
    d(conv[0])/da = 2a = 2*h[0] (k-j=0, valid)
    d(conv[1])/da = 2b = 2*h[1] (k-j=1, valid)
    d(conv[2])/da = 2c = 2*h[2] (k-j=2, valid)
    d(conv[2])/db = 2b = 2*h[1] (k-j=0, valid)
    d(conv[3])/db = 2c = 2*h[2] (k-j=1, valid)
    
    Great, so the formula is: d(conv[k])/dh[j] = 2*h[k-j] if 0 <= k-j < N, else 0.
    
    But wait, what about the case where j-k is valid but k-j is not?
    For example, k=0, j=1: k-j=-1, not valid. But j-k=1, valid.
    d(conv[0])/db = 0 (from example above).
    So the formula only uses k-j, not j-k.
    
    This makes sense because conv[k] = sum_i h[i]*h[k-i], and when we differentiate with respect to h[j],
    we get contributions from i=j (giving h[k-j]) and from i=k-j (giving h[k-j]).
    But i=k-j is only valid if 0 <= k-j < N, which is the same condition as for the first term.
    
    So the formula is correct: d(conv[k])/dh[j] = 2*h[k-j] if 0 <= k-j < N, else 0.
    
    Now, back to the objective:
    S(h) = L2^2 / (DELTA * M) where M = conv(h,h)[m]
    
    dS/dh_j = [2*h_j * M - L2^2 * dM/dh_j] / (DELTA * M^2)
    
    dM/dh_j = 2*h[m-j] if 0 <= m-j < N, else 0
    
    So:
    dS/dh_j = [2*h_j * M - L2^2 * 2*h[m-j] * I(0<=m-j<N)] / (DELTA * M^2)
    
    where I(condition) is 1 if condition is true, 0 otherwise.
    
    We want to minimize -S(h), so the gradient of the objective is -dS/dh_j.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0, np.zeros_like(h)
    
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    m = np.argmax(h_conv)
    M = h_conv[m]
    
    if M == 0:
        return 0.0, np.zeros_like(h)
    
    # S = l2_sq / (DELTA * M)
    # dS/dh_j = [2*h_j * M - l2_sq * dM/dh_j] / (DELTA * M^2)
    # dM/dh_j = 2*h[m-j] if 0 <= m-j < N, else 0
    
    grad_S = np.zeros_like(h)
    for j in range(N):
        dM_dhj = 0.0
        idx = m - j
        if 0 <= idx < N:
            dM_dhj = 2.0 * h[idx]
        grad_S[j] = (2.0 * h[j] * M - l2_sq * dM_dhj) / (DELTA * M**2)
    
    # We minimize -S, so gradient of objective = -grad_S
    obj = -l2_sq / (DELTA * M)
    grad = -grad_S
    
    return obj, grad

def _coordinate_descent(h, max_iters=500, tol=1e-12):
    """
    Coordinate descent to refine h, keeping L1=1 and h>=0.
    For each coordinate j, optimize h_j while keeping others fixed.
    """
    h = h.copy()
    for iteration in range(max_iters):
        improved = False
        for j in range(N):
            # Optimize h_j
            # Fix all other h_i
            # Let s = sum_{i!=j} h_i, so h_j = 1 - s
            # But we also need h_j >= 0, so s <= 1
            s = np.sum(h) - h[j]
            if s >= 1.0:
                # h_j can only be 0
                h[j] = 0.0
                continue
            
            # Try a few values of h_j
            # S(h_j) = (h_j^2 + sum_{i!=j} h_i^2) / (DELTA * max_conv(h_j))
            # max_conv depends on h_j in a complex way.
            # Let's just do a simple line search.
            best_hj = h[j]
            best_score = _score_raw(h)
            
            # Try h_j = 0, 1-s, and some intermediate values
            candidates = [0.0, 1.0 - s]
            # Add some points in between
            if s < 1.0:
                candidates.extend(np.linspace(0, 1-s, 20))
            
            for hj in candidates:
                h[j] = hj
                s_candidate = _score_raw(h)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_hj = hj
            
            h[j] = best_hj
            if abs(h[j] - best_hj) > tol:
                improved = True
        
        if not improved:
            break
    
    return h

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (150s limit)
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    # Parametric search for good initializations
    x_idx = np.arange(N)
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(0.5, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(0.5, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 4. Random sparse initialization
    for _ in range(200):
        if time.time() > deadline - 30: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(3, 20)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = np.random.rand(num_peaks)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using SLSQP with analytic gradients
    # Normalize to L1=1
    h_start = best_h.copy()
    h_start = h_start / np.sum(h_start)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)},
        {'type': 'ineq', 'fun': lambda h: h, 'jac': lambda h: np.eye(N)}
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    def neg_obj_h(h):
        obj, _ = _objective_and_grad(h)
        return obj
    
    def neg_obj_grad_h(h):
        _, grad = _objective_and_grad(h)
        return grad
    
    # First optimization
    try:
        res = minimize(neg_obj_h, h_start, jac=neg_obj_grad_h, method='SLSQP', 
                      bounds=bounds, constraints=constraints, 
                      options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            h_candidate = res.x
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
    except Exception:
        pass
    
    # Coordinate descent refinement
    if time.time() < deadline - 20:
        h_cd = _coordinate_descent(best_h / np.sum(best_h), max_iters=200)
        s_cd = _score_raw(h_cd)
        if s_cd > best_score:
            best_score = s_cd
            best_h = h_cd
    
    # Generate diverse initializations and optimize each
    inits = []
    
    # 1. Best found so far
    inits.append(best_h / np.sum(best_h))
    
    # 2. Uniform
    inits.append(np.ones(N) / N)
    
    # 3. Gaussians (more diverse)
    for c in np.linspace(1, N-1, 10):
        for w in np.linspace(0.5, N/3, 10):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 4. Triangles
    for c in np.linspace(1, N-1, 10):
        for w in np.linspace(0.5, N/2, 10):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random sparse
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 20)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = np.random.rand(k)
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 6. Random smooth
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(7)/7
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 0:
            h_r /= np.sum(h_r)
            inits.append(h_r)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline - 15: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(neg_obj_h, h_init, jac=neg_obj_grad_h, method='SLSQP', 
                          bounds=bounds, constraints=constraints, 
                          options={'maxiter': 300, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # Final polish with coordinate descent
    if time.time() < deadline - 5:
        h_final = _coordinate_descent(best_h / np.sum(best_h), max_iters=500)
        s_final = _score_raw(h_final)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, jac=neg_obj_grad_h, method='SLSQP', 
                          bounds=bounds, constraints=constraints, 
                          options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)

# EVOLVE-BLOCK-END
