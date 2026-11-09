import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        d = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = np.zeros(d)
        best_f = np.inf
        all_points = [] # Store (f, x) to select top-K for polishing

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            call_count += 1
            if call_count > budget:
                raise ValueError("Budget exceeded")
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            all_points.append((val, np.array(x, dtype=float)))
            return val

        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def _primes(n):
            primes = []
            candidate = 2
            while len(primes) < n:
                is_prime = True
                for p in primes:
                    if p * p > candidate:
                        break
                    if candidate % p == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(candidate)
                candidate += 1
            return primes

        # Phase 1: Latin Hypercube for stratified initial coverage
        # 50 points is a good balance for 5D
        n_lh = 50
        if budget >= n_lh:
            rng = np.random.default_rng(42)
            lh = np.zeros((n_lh, d))
            for i in range(d):
                perm = rng.permutation(n_lh)
                # Add jitter to avoid hitting exact grid lines
                lh[:, i] = (perm + rng.uniform(size=n_lh)) / n_lh
            # Map to [-5, 5]
            lh = lh * 10.0 - 5.0
            
            for i in range(n_lh):
                if time.time() > deadline or call_count >= budget:
                    break
                x = np.clip(lh[i], -5, 5)
                try:
                    counted_func(x)
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    continue

        # Phase 2: Halton sequence for dense systematic coverage
        # Target: Use up to 60% of budget here, or until we have 300 points
        target_global_calls = int(budget * 0.6)
        primes = self._primes(d)
        
        # Start Halton from index 1 (index 0 is usually origin, which is a local min)
        halton_idx = 1
        while call_count < target_global_calls and time.time() < deadline:
            try:
                x = np.array([halton(halton_idx, p) for p in primes])
                x = x * 10.0 - 5.0
                x = np.clip(x, -5, 5)
                counted_func(x)
                halton_idx += 1
            except (TimeoutError, ValueError):
                break
            except Exception:
                # If an error occurs, skip to next index
                halton_idx += 1
                continue

        # Phase 3: Local Refinement using Nelder-Mead on Top-K candidates
        # Select top 3 candidates from global search
        if best_f < np.inf and call_count < budget - 10:
            # Sort by function value
            all_points.sort(key=lambda x: x[0])
            top_k = min(3, len(all_points))
            candidates = [all_points[i][1] for i in range(top_k)]
            
            remaining_budget = budget - call_count
            budget_per_candidate = remaining_budget // top_k
            
            for cand in candidates:
                if call_count >= budget - 5 or time.time() > deadline - 10:
                    break
                self._nelder_mead_polish(counted_func, cand, budget_per_candidate, deadline)

        # Fallback: If something went wrong and we didn't make enough calls, do a random check
        if call_count < 10:
             try:
                 counted_func(np.zeros(d))
             except:
                 pass

        return best_x, best_f

    def _primes(self, n):
        """Return first n prime numbers."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def _nelder_mead_polish(self, func, x0, budget_left, deadline):
        """
        Nelder-Mead local search with strict budget and time guards.
        Updates global best_x/best_f via the func wrapper.
        """
        d = self.dim
        x = np.array(x0, dtype=float)
        x = np.clip(x, -5, 5)
        
        # Check if we have enough budget for a simplex
        if budget_left < d + 2:
            return

        fx = func(x)
        
        # Initial simplex
        simplex = [x.copy()]
        simplex_f = [fx]

        delta = 0.5
        for i in range(d):
            if budget_left <= 0 or time.time() > deadline:
                return
            xi = x.copy()
            xi[i] += delta
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            simplex_f.append(func(xi))
            budget_left -= 1

        # Nelder-Mead iterations
        # We can do at most (budget_left // (d+1)) iterations roughly, 
        # but NM can use 1, 2, or d+1 calls per iteration.
        # Conservative estimate: 1 call per iter on average for reflection/contraction logic.
        max_iter = max(1, budget_left // 2)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if budget_left <= 0:
                break

            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            
            if budget_left <= 0: break
            fr = func(xr)
            budget_left -= 1

            if fr < simplex_f[0]:
                # Expansion
                if budget_left <= 0:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    break
                gamma = 2.0
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                budget_left -= 1
                
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if budget_left <= 0:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    break
                beta = 0.5
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + beta * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - beta * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                budget_left -= 1
                
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    # This is expensive (d calls). Only do if we have budget.
                    if budget_left < d:
                        break
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        budget_left -= 1
                        if budget_left <= 0:
                            break
                    if budget_left <= 0:
                        break
        return
