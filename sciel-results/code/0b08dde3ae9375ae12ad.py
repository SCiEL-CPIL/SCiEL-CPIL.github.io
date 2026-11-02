import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Function to compute the ratio
    def score(h):
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

    # Helper to get max multiplicity of differences in a set
    def get_max_mult(S):
        if len(S) <= 1:
            return 1
        diffs = []
        S = np.sort(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                diffs.append(S[j] - S[i])
        diffs = np.array(diffs)
        vals, counts = np.unique(diffs, return_counts=True)
        return np.max(counts)

    # Greedy construction of a Sidon-like set
    def greedy_build(target_size, seed_idx=0):
        S = [seed_idx]
        current_diffs = []
        for _ in range(target_size - 1):
            best_candidate = -1
            best_max_mult = float('inf')
            # Try candidates from 0 to N-1
            for cand in range(N):
                if cand in S:
                    continue
                # Check new differences
                new_diffs = []
                for s in S:
                    d = abs(cand - s)
                    new_diffs.append(d)
                all_diffs = current_diffs + new_diffs
                vals, counts = np.unique(all_diffs, return_counts=True)
                max_mult = np.max(counts)
                if max_mult < best_max_mult:
                    best_max_mult = max_mult
                    best_candidate = cand
                    # Early break if we achieve Sidon property (max_mult == 1)
                    if best_max_mult == 1:
                        break
            
            if best_candidate != -1:
                S.append(best_candidate)
                # Update current_diffs
                for s in S[:-1]:
                    current_diffs.append(abs(S[-1] - s))
            else:
                break
        return S

    # Objective function for optimization: minimize negative log ratio
    # Ratio = sum(h^2) / (sum(h) * max_conv)
    # We want to maximize this.
    # Let's fix sum(h) = 1 to remove scale invariance.
    # Then Ratio = sum(h^2) / max_conv.
    # Maximize f(h) = sum(h^2) / max_conv subject to sum(h)=1, h>=0.
    
    def neg_log_ratio(h):
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        l2_sq = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 1e10
        return -np.log(l2_sq / max_conv)

    def jac_neg_log_ratio(h):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(h)
        f0 = neg_log_ratio(h)
        for i in range(len(h)):
            h_old = h[i]
            h[i] += eps
            f1 = neg_log_ratio(h)
            h[i] = h_old
            grad[i] = (f1 - f0) / eps
        return grad

    best_h = np.zeros(N)
    best_score = 0.0

    # Search over different support sizes and sets
    np.random.seed(42)
    
    # 1. Try Greedy Sidon sets of various sizes
    for k in range(2, 12):
        if time.time() > deadline - 10.0:
            break
        S = greedy_build(k)
        if len(S) < k:
            continue
        
        # Set up optimization
        # Variables are amplitudes on S
        # Constraint: sum(amplitudes) = 1
        # We can use SLSQP
        
        x0 = np.ones(len(S)) / len(S)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(len(x))}
        ]
        bounds = [(0, 1)] * len(S)
        
        def obj_full(x):
            h_full = np.zeros(N)
            h_full[S] = x
            return neg_log_ratio(h_full)
            
        def jac_full(x):
            h_full = np.zeros(N)
            h_full[S] = x
            grad_full = jac_neg_log_ratio(h_full)
            return grad_full[S]

        try:
            res = minimize(obj_full, x0, method='SLSQP', jac=jac_full, bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-9})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[S] = res.x
                s = score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass

    # 2. Random search for better supports if time permits
    # We can also try to perturb the best set found
    if best_score < 1.13 and time.time() < deadline - 20.0:
        # Try random sets with optimization
        for _ in range(2000):
            if time.time() > deadline - 10.0:
                break
            k = np.random.randint(3, 8)
            S = np.random.choice(N, k, replace=False)
            
            x0 = np.ones(k) / k
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(len(x))}
            ]
            bounds = [(0, 1)] * k
            
            def obj_full(x):
                h_full = np.zeros(N)
                h_full[S] = x
                return neg_log_ratio(h_full)
                
            def jac_full(x):
                h_full = np.zeros(N)
                h_full[S] = x
                grad_full = jac_neg_log_ratio(h_full)
                return grad_full[S]

            try:
                res = minimize(obj_full, x0, method='SLSQP', jac=jac_full, bounds=bounds, constraints=constraints, options={'maxiter': 50, 'ftol': 1e-8})
                if res.success:
                    h_opt = np.zeros(N)
                    h_opt[S] = res.x
                    s = score(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
            except Exception:
                pass

    # 3. Polish the best result found with more iterations
    if best_score > 0 and time.time() < deadline - 5.0:
        # Identify support of best_h
        support_idx = np.where(best_h > 1e-6)[0]
        if len(support_idx) > 0:
            S = support_idx
            x0 = best_h[S]
            # Normalize
            x0 = x0 / np.sum(x0)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(len(x))}
            ]
            bounds = [(0, 1)] * len(S)
            
            def obj_full(x):
                h_full = np.zeros(N)
                h_full[S] = x
                return neg_log_ratio(h_full)
                
            def jac_full(x):
                h_full = np.zeros(N)
                h_full[S] = x
                grad_full = jac_neg_log_ratio(h_full)
                return grad_full[S]

            try:
                res = minimize(obj_full, x0, method='SLSQP', jac=jac_full, bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
                if res.success:
                    h_opt = np.zeros(N)
                    h_opt[S] = res.x
                    s = score(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
            except Exception:
                pass

    return list(best_h)
