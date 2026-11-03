import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse structured (Sidon-like with structure)
    np.random.seed(42)
    for trial in range(30):
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(3, N-3, 6):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Structured sparse: pairs symmetric around center
    for k in range(3, 10):
        for offset in range(1, 8):
            h_pair = np.zeros(N)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                h_pair /= np.sum(h_pair)
                inits.append(h_pair)

    # --- Phase 2: Epigraph Formulation Optimization (focused on top peaks) ---
    # Instead of constraining all 99 convolution points, focus on top K peaks
    # This reduces constraint count and allows SLSQP to converge faster

    def run_epigraph_focused(h_init, maxiter=300, ftol=1e-12, top_k=5):
        """Minimize M where M >= max of top_k convolution peaks."""
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        
        # Find top peaks in initial convolution
        conv_init = np.convolve(h_init, h_init, mode='full')
        top_indices = np.argsort(conv_init)[-top_k:]
        M_init = np.max(conv_init) + 1e-6
        
        # Variables: x = [h_0, ..., h_{N-1}, M]
        def objective(x):
            return x[N]
            
        def objective_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad
            
        def constraint_eq_fun(x):
            return np.sum(x[:N]) - 1.0
            
        def constraint_eq_jac(x):
            jac = np.zeros(N + 1)
            jac[:N] = 1.0
            return jac
            
        def constraint_ineq_fun(x):
            h = x[:N]
            M = x[N]
            conv = np.convolve(h, h, mode='full')
            # Only constrain the top_k peaks (by index, not value)
            return M - conv[top_indices]
            
        def constraint_ineq_jac(x):
            h = x[:N]
            num_constr = len(top_indices)
            jac = np.zeros((num_constr, N + 1))
            jac[:, N] = 1.0
            for idx_i, k in enumerate(top_indices):
                for j in range(N):
                    # d(conv_k)/dh_j = h_{k-j} if 0 <= k-j < N else 0
                    # But since conv = h*h, d(conv_k)/dh_j = h_{k-j} + h_{k-j} = 2*h_{k-j} if symmetric
                    # Actually conv_k = sum_i h_i * h_{k-i}, so d(conv_k)/dh_j = h_{k-j} (if valid)
                    # Wait, that's not right. Let me reconsider.
                    # conv_k = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h_i * h_{k-i}
                    # d(conv_k)/dh_j = h_{k-j} if 0 <= k-j < N, else 0
                    # But since we're differentiating with respect to h_j, and h appears twice...
                    # Actually no: conv_k is a function of all h_i. 
                    # d(conv_k)/dh_j = sum over terms where h_j appears.
                    # h_j appears as h_i when i=j, and as h_{k-i} when k-i=j i.e. i=k-j.
                    # So d(conv_k)/dh_j = h_{k-j} (from first term) + h_j (from second term if k-j != j)
                    # Wait, this is getting confusing. Let me just compute numerically or use the correct formula.
                    # 
                    # Actually: conv_k = sum_i h_i * h_{k-i}
                    # d(conv_k)/dh_j: 
                    #   Term where i=j: h_{k-j} (if 0 <= k-j < N)
                    #   Term where k-i=j i.e. i=k-j: h_j (if 0 <= k-j < N and k-j != j)
                    #   If k-j == j (i.e., k=2j), then it's h_j^2, derivative is 2*h_j
                    #
                    # So: d(conv_k)/dh_j = h_{k-j} + (h_j if k-j != j else 0) if 0 <= k-j < N
                    # But wait, that's not right either. Let me think again.
                    #
                    # conv_k = sum_{i} h_i * h_{k-i}
                    # The derivative with respect to h_j:
                    #   For each term h_i * h_{k-i}, it contributes if i=j or k-i=j
                    #   If i=j: contribution is h_{k-j} (if valid)
                    #   If k-i=j i.e. i=k-j: contribution is h_j (if valid)
                    #   If both (i=j and i=k-j, i.e., j=k-j, k=2j): the term is h_j^2, derivative is 2*h_j
                    #
                    # So: d(conv_k)/dh_j = [h_{k-j} if 0<=k-j<N else 0] + [h_j if 0<=k-j<N and k-j!=j else 0]
                    #                              + [h_j if k==2j else 0]  # for the double count
                    #
                    # Actually simpler: d(conv_k)/dh_j = h_{k-j} + h_{k-j} if we consider both appearances?
                    # No. Let me just use the fact that conv is quadratic.
                    # 
                    # Let me just compute the Jacobian numerically for safety, or use the correct analytical form.
                    # For now, let's use a simpler approach: only constrain M >= conv_k for the single max k.
                    pass
            # Simplified: just constrain M >= conv[k_star] where k_star is the argmax
            # This is a single inequality constraint
            return jac
        
        # Let's use a simpler formulation: minimize M subject to M >= conv[argmax(conv)]
        # But argmax changes... Instead, let's just use the original all-constraints but with better init.
        
        # Alternative: use only the single peak constraint
        def constraint_ineq_single_fun(x):
            h = x[:N]
            M = x[N]
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            return M - conv[k_star]
            
        def constraint_ineq_single_jac(x):
            h = x[:N]
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            jac = np.zeros(N + 1)
            jac[N] = 1.0
            # d(M - conv[k_star])/dh_j
            # d(conv[k_star])/dh_j = h_{k_star - j} if 0 <= k_star - j < N
            # Plus if k_star - j == j, there's an additional h_j term? No, as discussed above.
            # Actually, the correct derivative:
            # conv[k_star] = sum_i h_i * h_{k_star - i}
            # d/dh_j conv[k_star] = h_{k_star - j} (if valid) + h_j (if k_star - j valid and != j)
            # But this is wrong. Let me just do:
            # d/dh_j [sum_i h_i * h_{k_star-i}] = 
            #   sum over i where i=j: h_{k_star-j}
            #   sum over i where k_star-i=j: h_j
            # So = h_{k_star-j} (if 0<=k_star-j<N) + h_j (if 0<=k_star-j<N and k_star-j != j)
            # If k_star-j == j, then the term is h_j^2, derivative 2*h_j, which matches the formula above.
            for j in range(N):
                k_j = k_star - j
                if 0 <= k_j < N:
                    jac[j] = -h[k_j]  # from first term
                    if k_j != j:
                        jac[j] -= h[j]  # from second term
                    else:
                        jac[j] -= h[j]  # additional h_j for the square term
            return jac
        
        constraints = [
            {
                'type': 'eq',
                'fun': constraint_eq_fun,
                'jac': constraint_eq_jac
            },
            {
                'type': 'ineq',
                'fun': constraint_ineq_single_fun,
                'jac': constraint_ineq_single_jac
            }
        ]
        
        bounds = [(0.0, 1.0)] * N + [(0.0, None)]
        
        x0 = np.concatenate((h_init, [M_init]))
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        return None
    
    # Run focused epigraph from multiple inits
    for h_init in inits:
        if time.time() > deadline - 30:
            break
        h_candidate = run_epigraph_focused(h_init, maxiter=300, ftol=1e-12)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 3: Efficient Block-Coordinate Refinement ---
    def block_coordinate_refine(h_start, max_iter=50):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 10:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            
            # For each position i, try to optimize h[i] and h[k_star - i] (if valid)
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                    
                hi = h[i]
                hj = h[j]
                
                # t range: [-hi, hj] to keep both non-negative
                t_min = -hi
                t_max = hj
                
                if t_max - t_min < 1e-15:
                    continue
                    
                # Evaluate score at several t values (reduced from 10 to 6 for speed)
                best_t = 0.0
                best_local_score = _score_raw(h)
                
                for t in np.linspace(t_min * 0.8, t_max * 0.8, 6):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-15):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    s_test = _score_raw(h_test)
                    if s_test > best_local_score:
                        best_local_score = s_test
                        best_t = t
                        
                if abs(best_t) > 1e-10:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    if np.sum(h) > 1e-15:
                        h /= np.sum(h)
                    improved = True
                    
        return h
    
    # Apply block-coordinate refinement to best solution
    if time.time() < deadline - 15:
        h_refined = block_coordinate_refine(best_h, max_iter=30)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # --- Phase 4: Final Polish with Epigraph (high precision) ---
    if time.time() < deadline - 5:
        h_candidate = run_epigraph_focused(best_h, maxiter=500, ftol=1e-14)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
                
        # Another round of block-coordinate
        h_refined = block_coordinate_refine(best_h, max_iter=20)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    return list(best_h)

# EVOLVE-BLOCK-END
