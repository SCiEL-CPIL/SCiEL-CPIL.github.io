import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def find_sidon_set(N, K, attempts=1000):
    """
    Attempts to find a Sidon set of size K in [0, N).
    Returns the set if found, None otherwise.
    """
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == K:
                break
            
            conflict = False
            new_sums = set()
            
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums:
                        conflict = True
                        break
                    if sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == K:
            return S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def calculate_ratio(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h)
        peak = np.max(c)
        if peak == 0:
            return 0.0
        return l2_sq / (l1 * peak)

    def optimize_on_support(h0, max_iter=1000, ftol=1e-12):
        """
        Optimize weights on a fixed support using SLSQP.
        The constraint is M >= max(conv(h)), which is handled via epigraph.
        We minimize M subject to sum(h)=1, h>=0, M >= conv_k(h) for all k.
        """
        support = np.where(h0 > 1e-12)[0]
        if len(support) == 0:
            return h0.copy()
        
        K = len(support)
        # Variables: h on support (K vars) + M (1 var)
        # Total K+1 variables
        
        h_init = np.zeros(N)
        h_init[support] = h0[support]
        
        x0 = np.concatenate([h0[support], [np.max(np.convolve(h0, h0))]])
        
        def obj(x):
            return x[-1]
        
        def obj_grad(x):
            g = np.zeros(K + 1)
            g[-1] = 1.0
            return g
        
        # Constraint: sum(h) = 1
        def c_sum(x):
            return np.sum(x[:K]) - 1.0
        
        def c_sum_grad(x):
            g = np.zeros(K + 1)
            g[:K] = 1.0
            return g
        
        # Constraints: M - conv_k(h) >= 0 for all k
        # conv_k(h) = sum_{i+j=k} h_i h_j
        # This is a quadratic constraint.
        
        # Precompute indices for each lag k
        # For each k, list of pairs (i, j) such that i+j=k, i,j in support
        # Actually, it's easier to just compute the gradient on the fly or precompute the structure.
        # Since K is small (< 20 usually), we can afford some overhead.
        
        def c_autoconv(x):
            h = x[:K]
            M = x[-1]
            # Map h back to full vector for convolution? No, just compute directly.
            # But np.convolve is easy.
            h_full = np.zeros(N)
            h_full[support] = h
            c = np.convolve(h_full, h_full)
            return M - c
        
        def c_autoconv_grad(x):
            h = x[:K]
            # grad w.r.t h_i is -2 * sum_{j: i+j=k} h_j ... wait.
            # Constraint: M - sum_{i,j: i+j=k} h_i h_j >= 0
            # d/dh_i (M - sum h_i h_j) = - sum_{j: i+j=k} h_j - sum_{j: i+j=k} h_j?
            # No, sum_{i,j} h_i h_j. Derivative w.r.t h_p is sum_{j: p+j=k} h_j + sum_{j: j+p=k} h_j.
            # Since sum is symmetric, it's 2 * sum_{j: p+j=k} h_j if we consider i != j, 
            # but for i=j it's just 2*h_i?
            # Let's be precise. f(h) = sum_{i,j} h_i h_j subject to i+j=k.
            # df/dh_p = sum_{j: p+j=k} h_j + sum_{j: j+p=k} h_j.
            # If p != j, these are distinct terms. If p=j, it's the same term counted twice in the sum?
            # No, the sum is over pairs.
            # Let's use the full vector derivative.
            # d/dh_full_i (sum_{a,b: a+b=k} h_a h_b) = 2 * h_full_{k-i} if k-i is in range, else 0.
            # Wait, if i = k-i, i.e., 2i=k, then it's 2*h_i.
            # If i != k-i, then the term h_i h_{k-i} appears twice in the sum? No.
            # The sum is sum_{a} h_a h_{k-a}.
            # Derivative w.r.t h_p: 
            # Term a=p: h_p h_{k-p} -> deriv h_{k-p}
            # Term a=k-p: h_{k-p} h_p -> deriv h_p
            # So if p != k-p, deriv is h_{k-p} + h_p.
            # If p == k-p, deriv is 2 h_p.
            # In both cases, it's 2 h_{k-p} (since if p=k-p, 2 h_{k-p} = 2 h_p).
            # So d/dh_p (conv_k) = 2 h_{k-p} if 0 <= k-p < N, else 0.
            
            g = np.zeros((2*N - 1, K + 1))
            # We only need the rows for the constraints.
            # But scipy minimize expects a single vector for 'ineq' constraints if not using LinearConstraint.
            # Wait, scipy minimize with 'ineq' constraints expects a vector of values.
            # The gradient should be a matrix of shape (num_constraints, num_vars).
            
            # Let's compute the full gradient matrix.
            for k in range(2*N - 1):
                # Derivative of conv_k w.r.t h_full
                d_conv_k = np.zeros(N)
                for p in support:
                    j = k - p
                    if 0 <= j < N:
                        # Check if j is in support? No, h_full is zero outside support.
                        # So we just need the value of h_full at j.
                        # h_full[j] is h[idx] if j in support, else 0.
                        if j in support:
                            d_conv_k[p] += 2.0 * h_full[j] # Wait, h_full is not updated in this function.
                            # We need to use x[:K] mapped to support.
                            # Let's recompute h_full inside the loop or pass it.
                
                # Let's optimize this.
                # We have h (K values) on support.
                # For each k, we need d/dh_i (conv_k).
                # d/dh_i (conv_k) = 2 * h_{k - support[i]} if (k - support[i]) is in support.
                
                # Let's precompute a lookup table for support indices.
                # support_to_idx = {val: idx for idx, val in enumerate(support)}
                
                # For each k, for each i in 0..K-1:
                # p = support[i]
                # j = k - p
                # if j in support_to_idx:
                #    j_idx = support_to_idx[j]
                #    d_conv_k[i] = 2.0 * h[j_idx]
                
                # The constraint is M - conv_k >= 0.
                # So d/dx_i (M - conv_k) = - d_conv_k[i] for i < K.
                # d/dx_K (M - conv_k) = 1.0.
                
                pass # Placeholder, will implement properly below
        
        # Let's implement the gradient properly using precomputation.
        support_set = set(support)
        support_to_idx = {v: i for i, v in enumerate(support)}
        
        # Precompute for each k, the list of (i, j_idx) such that support[i] + support[j_idx] = k?
        # No, for each k, and each i, we need to know if k - support[i] is in support.
        
        # To be efficient, we can just loop over k and i.
        # K is small, N is 50, 2N-1 is 99.
        # 99 * K operations. If K=15, 1485 ops. Very fast.
        
        def c_autoconv_grad_impl(x):
            h = x[:K]
            g = np.zeros((2*N - 1, K + 1))
            for k in range(2*N - 1):
                # d/dM is 1
                g[k, -1] = 1.0
                for i in range(K):
                    p = support[i]
                    j = k - p
                    if 0 <= j < N and j in support_to_idx:
                        j_idx = support_to_idx[j]
                        # Derivative of conv_k w.r.t h_i is 2 * h_{j_idx}
                        # Constraint is M - conv_k >= 0
                        # So derivative is -2 * h_{j_idx}
                        g[k, i] = -2.0 * h[j_idx]
            return g
        
        bounds = [(0, None)] * K + [(0, None)]
        
        cons = [
            {'type': 'eq', 'fun': c_sum, 'jac': c_sum_grad},
            {'type': 'ineq', 'fun': c_autoconv, 'jac': c_autoconv_grad_impl}
        ]
        
        try:
            res = minimize(
                obj, 
                x0, 
                method='SLSQP', 
                jac=obj_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )
            
            if res.success or res.fun < x0[-1]:
                h_res = np.zeros(N)
                h_res[support] = res.x[:K]
                # Clip and normalize
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res /= s
                return h_res
            else:
                return h0.copy()
        except Exception:
            return h0.copy()

    # 1. Initial Best: Uniform
    best_h = np.ones(N) / N
    best_ratio = calculate_ratio(best_h)

    # 2. Find best Sidon set (Sparse structure)
    best_sidon_h = None
    best_sidon_k = 0
    for K in range(12, 4, -1):
        if time.time() > deadline - 40.0:
            break
        attempts = 200 if K > 8 else 500
        S = find_sidon_set(N, K, attempts)
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            best_sidon_h = h
            best_sidon_k = K
            break

    # Update best with Sidon result if found
    if best_sidon_h is not None:
        r_sidon = calculate_ratio(best_sidon_h)
        if r_sidon > best_ratio:
            best_ratio = r_sidon
            best_h = best_sidon_h.copy()

    # 3. Optimization Phase
    # Generate starting points
    starts = []
    
    # Add current best
    starts.append(best_h.copy())
    
    # Add Uniform
    starts.append(np.ones(N) / N)
    
    # Add Sidon set if found
    if best_sidon_h is not None:
        starts.append(best_sidon_h.copy())
    
    # Add random sparse supports
    # Try different sparsities
    for K_rand in [5, 6, 7, 8, 9, 10, 12]:
        if time.time() > deadline - 20.0:
            break
        for _ in range(3):
            if time.time() > deadline - 15.0:
                break
            support = random.sample(range(N), K_rand)
            h_rand = np.zeros(N)
            h_rand[support] = 1.0 / K_rand
            starts.append(h_rand)
    
    # Run optimization on each start
    for h0 in starts:
        if time.time() > deadline - 10.0:
            break
        h_opt = optimize_on_support(h0)
        r = calculate_ratio(h_opt)
        if r > best_ratio:
            best_ratio = r
            best_h = h_opt.copy()

    # 4. Local Refinement via Coordinate Perturbations
    # This helps escape local minima or polish the solution.
    h = best_h.copy()
    for _ in range(50):
        if time.time() > deadline - 2.0:
            break
        for _ in range(100):
            h_trial = h.copy()
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            delta = random.uniform(-0.05, 0.05)
            h_trial[i] += delta
            h_trial[j] -= delta
            h_trial = np.clip(h_trial, 0, None)
            s = np.sum(h_trial)
            if s < 1e-10:
                continue
            h_trial = h_trial / s
            r_trial = calculate_ratio(h_trial)
            if r_trial > best_ratio:
                best_ratio = r_trial
                best_h = h_trial.copy()
                h = h_trial.copy()
                break

    # Final cleanup
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
