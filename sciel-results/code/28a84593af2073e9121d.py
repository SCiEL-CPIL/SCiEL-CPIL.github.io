import random
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit

    def score_raw(h):
        """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
        h = np.asarray(h, dtype=float)
        if len(h) == 0:
            return 0.0
        h1 = np.sum(h)
        if h1 <= 0:
            return 0.0
        h2 = np.sum(h * h)
        c = np.convolve(h, h)
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        return h2 / (h1 * cmax)

    def soft_max(c, tau=1.0):
        """Smooth approximation of max(c) using log-sum-exp."""
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        # Stable log-sum-exp
        shifted = c / tau
        shifted = shifted - np.max(shifted)
        lse = np.max(c) + tau * np.log(np.sum(np.exp(shifted)))
        return lse

    def optimize_smooth(h_init, max_iter=1000, ftol=1e-10, tau=0.01):
        """
        Optimize h using smooth epigraph with log-sum-exp surrogate.
        Variables: x = [h_0, ..., h_{N-1}]
        Minimize soft_max(conv(h,h)) subject to sum(h)=1, h >= 0
        """
        def obj(x):
            h = np.clip(x, 1e-12, None)
            s = np.sum(h)
            if s <= 0:
                return 1e6
            h = h / s
            c = np.convolve(h, h)
            return soft_max(c, tau)

        def obj_grad(x):
            h = np.clip(x, 1e-12, None)
            s = np.sum(h)
            if s <= 0:
                return np.zeros(N)
            h = h / s
            c = np.convolve(h, h)
            cmax = np.max(c)
            if cmax <= 0:
                return np.zeros(N)
            
            # Gradient of soft_max w.r.t. c
            shifted = c / tau
            shifted = shifted - np.max(shifted)
            weights = np.exp(shifted) / np.sum(np.exp(shifted))
            
            # Gradient of c_k w.r.t. h_i
            grad_c = np.zeros((2 * N - 1, N))
            for k in range(2 * N - 1):
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        grad_c[k, i] = 2.0 * h[j]
            
            # Chain rule: grad_x = sum_k weights[k] * grad_c[k] / s - (obj/s) * ones
            grad_h = weights @ grad_c / s
            grad_x = grad_h - (obj(x) / s) * np.ones(N)
            
            # Project back (accounting for normalization)
            return grad_x

        h_init = np.asarray(h_init, dtype=float)
        s = np.sum(h_init)
        if s > 0:
            h_init = h_init / s
        else:
            h_init = np.ones(N) / N
        h_init = np.clip(h_init, 1e-12, None)
        s = np.sum(h_init)
        h_init = h_init / s

        bounds = [(1e-12, None)] * N

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
             'jac': lambda x: np.ones(N)}
        ]

        try:
            res = minimize(
                obj,
                h_init,
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            if res.success or res.fun < 1e6:
                h_res = np.clip(res.x, 1e-12, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0

                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    def optimize_epigraph(h_init, max_iter=500, ftol=1e-8):
        """
        Optimize h using SLSQP with epigraph formulation.
        Variables: x = [h_0, ..., h_{N-1}, M]
        Minimize M subject to sum(h)=1, M >= c_k for all k, h >= 0
        """
        def obj(x):
            return x[N]

        def obj_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad

        def constraint_sum(x):
            return np.sum(x[:N]) - 1.0

        def constraint_sum_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad

        def constraints_autoconv(x):
            h = x[:N]
            M = x[N]
            c = np.convolve(h, h)
            return M - c

        def constraints_autoconv_grad(x):
            h = x[:N]
            grad = np.zeros((2 * N - 1, N + 1))
            for k in range(2 * N - 1):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * h[j]
                grad[k, :N] = -derivs
                grad[k, N] = 1.0
            return grad

        h_init = np.asarray(h_init, dtype=float)
        c_init = np.convolve(h_init, h_init)
        M_init = np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])

        bounds = [(0.0, None)] * N + [(0.0, None)]

        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
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

            if res.success or res.fun < np.max(c_init):
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0

                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    def find_sidon_support(target_size):
        """
        Greedy construction of a near-Sidon set to minimize pairwise sum collisions.
        Returns a list of indices.
        """
        if target_size <= 0:
            return []

        support = [0]
        used_sums = {0: 1}

        for _ in range(target_size - 1):
            best_candidate = None
            best_score = -1

            for candidate in range(N):
                if candidate in support:
                    continue

                # Count new collisions
                new_collisions = 0
                for s in support:
                    pair_sum = candidate + s
                    if pair_sum in used_sums:
                        new_collisions += 1

                # We want to minimize collisions, so maximize (target - collisions)
                score = target_size - new_collisions
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate is not None:
                support.append(best_candidate)
                for s in support:
                    pair_sum = best_candidate + s
                    used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
            else:
                break

        return support

    def find_sidon_support_randomized(target_size, num_trials=50):
        """
        Find the best Sidon-like support among multiple random trials.
        """
        best_support = []
        best_collisions = float('inf')

        for _ in range(num_trials):
            if time.time() > deadline - 10:
                break

            support = [0]
            used_sums = {0: 1}
            collisions = 0

            # Randomize the order of candidates
            candidates = list(range(1, N))
            random.shuffle(candidates)

            for candidate in candidates:
                if len(support) >= target_size:
                    break

                # Count new collisions
                new_collisions = 0
                for s in support:
                    pair_sum = candidate + s
                    if pair_sum in used_sums:
                        new_collisions += 1

                # Heuristic: accept if collisions are low
                # We prioritize adding elements with fewer collisions
                if new_collisions < 2:  # Threshold can be tuned
                    support.append(candidate)
                    for s in support:
                        pair_sum = candidate + s
                        used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
                        if used_sums[pair_sum] > 1:
                            collisions += 1

            if len(support) >= 3:
                if collisions < best_collisions:
                    best_collisions = collisions
                    best_support = support[:]

        if not best_support:
            # Fallback to deterministic greedy
            best_support = find_sidon_support(target_size)

        return best_support

    def find_best_support_by_score(target_size, num_trials=100):
        """
        Find the best support by directly evaluating the ratio for uniform weights.
        """
        best_support = []
        best_ratio = 0.0

        for _ in range(num_trials):
            if time.time() > deadline - 10:
                break

            # Random subset
            support = sorted(random.sample(range(N), target_size))
            h = np.zeros(N)
            h[support] = 1.0 / target_size
            ratio = score_raw(h)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_support = support[:]

        # Also try greedy Sidon
        support = find_sidon_support(target_size)
        if len(support) >= 3:
            h = np.zeros(N)
            h[support] = 1.0 / len(support)
            ratio = score_raw(h)
            if ratio > best_ratio:
                best_ratio = ratio
                best_support = support[:]

        return best_support, best_ratio

    best_h = None
    best_score = 0.0

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = optimize_epigraph(h_uniform)
    if h_opt is not None and s_opt > best_score:
        best_score = s_opt
        best_h = h_opt.copy()
    
    # Also try smooth optimization from uniform
    h_opt, s_opt = optimize_smooth(h_uniform, max_iter=500, tau=0.05)
    if h_opt is not None and s_opt > best_score:
        best_score = s_opt
        best_h = h_opt.copy()

    # 2. Try Sidon-like structured supports
    # The structural change: focus on finding supports that minimize sum collisions
    for target_k in [5, 6, 7, 8, 9, 10, 12]:
        if time.time() > deadline - 25:
            break

        # Use randomized search for better Sidon sets
        support = find_sidon_support_randomized(target_k, num_trials=30)
        if len(support) >= 3:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0 / len(support)
            h_opt, s_opt = optimize_epigraph(h_sidon)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            
            # Also try smooth
            h_opt, s_opt = optimize_smooth(h_sidon, max_iter=500, tau=0.05)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 2b. Try best support by direct score evaluation
    for target_k in [5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 20:
            break
        
        support, ratio = find_best_support_by_score(target_k, num_trials=50)
        if len(support) >= 3 and ratio > best_score:
            h_best = np.zeros(N)
            h_best[support] = 1.0 / len(support)
            h_opt, s_opt = optimize_epigraph(h_best)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            
            h_opt, s_opt = optimize_smooth(h_best, max_iter=500, tau=0.05)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 3. Try random sparse starts with varying support sizes
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break

        K = random.randint(3, 12)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K

        h_opt, s_opt = optimize_epigraph(h_rand)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        h_opt, s_opt = optimize_smooth(h_rand, max_iter=300, tau=0.05)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 4. Try geometric progressions
    for base in [2, 3]:
        if time.time() > deadline - 10:
            break
        support = []
        val = 0
        while val < N:
            support.append(val)
            val = val * base + 1
        if len(support) > 2:
            h_geom = np.zeros(N)
            h_geom[support] = 1.0 / len(support)
            h_opt, s_opt = optimize_epigraph(h_geom)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            
            h_opt, s_opt = optimize_smooth(h_geom, max_iter=300, tau=0.05)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 5. More random restarts if time allows
    while time.time() < deadline - 10:
        K = random.randint(2, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K

        h_opt, s_opt = optimize_epigraph(h_rand, max_iter=300)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Smooth optimization for some restarts
        if random.random() < 0.5:
            h_opt, s_opt = optimize_smooth(h_rand, max_iter=300, tau=0.05)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 6. Final polish pass on the best solution found
    if best_h is not None and time.time() < deadline - 5:
        h_polished, s_polished = optimize_epigraph(best_h, max_iter=2000, ftol=1e-10)
        if h_polished is not None and s_polished > best_score:
            best_score = s_polished
            best_h = h_polished.copy()
    
    # 7. High-precision smooth polish
    if best_h is not None and time.time() < deadline - 3:
        h_smooth_polished, s_smooth = optimize_smooth(best_h, max_iter=3000, ftol=1e-12, tau=0.001)
        if h_smooth_polished is not None and s_smooth > best_score:
            best_score = s_smooth
            best_h = h_smooth_polished.copy()
        
        # One more epigraph polish after smooth
        if time.time() < deadline - 1:
            h_final, s_final = optimize_epigraph(best_h, max_iter=1000, ftol=1e-12)
            if h_final is not None and s_final > best_score:
                best_score = s_final
                best_h = h_final.copy()

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_score = score_raw(best_h)

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
# EVOLVE-BLOCK-END
