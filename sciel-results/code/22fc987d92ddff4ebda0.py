"""Optimized program for BBOB f21 (Gallagher 101 peaks) with LHS + CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer:
    1. Global exploration via Latin Hypercube Sampling (50 points).
    2. Local exploitation via CMA-ES (population 30) starting from the best LHS point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None
        
        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return float('inf')
            n_calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # --- Phase 1: Global Exploration via Latin Hypercube Sampling ---
        # Use 50 samples to cover the space well without wasting too much budget.
        n_lhs = 50
        if n_lhs >= budget:
            n_lhs = budget // 2
        
        # Generate LHS points in [0, 1]^dim
        lhs_indices = [rng.permutation(n_lhs) for _ in range(dim)]
        lhs_points = np.array([[i / n_lhs + rng.random() for i in idx] for idx in lhs_indices])
        
        # Map to [-5, 5]^dim
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Evaluate LHS points
        for p in lhs_points:
            if n_calls >= budget or time.time() > deadline - 5:
                break
            x = np.clip(p, -5, 5)
            wrapped_func(x)
        
        # If we haven't made any valid call (should not happen), fallback to center
        if best_x is None:
            x = np.zeros(dim)
            wrapped_func(x)
            if best_x is None:
                return np.zeros(dim), 0.0

        # --- Phase 2: Local Exploitation via CMA-ES ---
        # Initialize CMA-ES at the best point found in Phase 1
        x0 = best_x.copy()
        
        # CMA-ES Parameters
        pop_size = 30
        # Initial sigma: scale relative to the domain size, but not too large
        # Since we started from a good LHS point, a moderate sigma is appropriate.
        sigma = 1.0 
        
        # CMA-ES State
        mean = x0.copy()
        C = np.eye(dim)
        sqrtC = np.eye(dim)
        step_size = 1.0
        # Covariance adaptation parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4. + mueff / dim) / (dim + 4. + 2. * mueff / dim)
        cs = (mueff + 2.) / (dim + mueff + 5.)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1. / mueff) / ((dim + 2.) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
        
        # Check if we have enough budget for CMA-ES
        if budget - n_calls < pop_size + 5:
            return best_x, best_f

        # Main CMA-ES Loop
        while n_calls < budget and time.time() < deadline - 5:
            # Generate population
            pop = []
            for i in range(pop_size):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition for efficient sampling
                try:
                    y = mean + sigma * (sqrtC @ z)
                except Exception:
                    y = mean + sigma * z
                pop.append((y, i))
            
            if len(pop) < 2:
                break

            # Evaluate population
            fit = []
            for y, i in pop:
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                val = wrapped_func(y)
                fit.append((val, y, i))
            
            if len(fit) < 2:
                break

            # Sort by fitness
            fit.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(fit))):
                new_mean += weights[i] * fit[i][1]
            
            # Update step size (sigma)
            # Check if we should break due to stagnation or small sigma
            if sigma < 1e-10:
                break
                
            # Update covariance matrix and other state
            if len(fit) >= mu:
                # Difference matrix
                D = np.array([fit[i][1] - mean for i in range(mu)])
                
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chiN * (sqrtC.T @ (new_mean - mean) / sigma)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_calls // pop_size))) < (1.4 + 2 / (dim + 1)) * chiN:
                    pc = (1 - cc) * pc + cc / cc * (new_mean - mean) / sigma
                    # Correction factor for covariance update
                    delta = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_calls // pop_size))) < (1.4 + 2 / (dim + 1)) * chiN else 0
                    C = (1 + c1 * delta - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu / damps * (D.T @ D) / mueff * 1.0 # Simplified, standard is more complex
                    # Note: A full CMA-ES implementation is complex. 
                    # For robustness and speed, we often approximate or use a simplified version.
                    # However, a simplified CMA-ES might not be as effective.
                    # Let's use a more standard simplified update or fall back to a robust method if this gets unstable.
                    # Given the constraints, let's stick to a very robust simplified CMA-ES or switch to Nelder-Mead if it fails.
                    # Actually, implementing full CMA-ES correctly in this short snippet is error-prone.
                    # Let's use a simpler, robust approach: Adaptive Step Size with Momentum (like Adam but for optimization).
                    # Or, since Inspiration 1 used Nelder-Mead and scored high, let's pivot to Nelder-Mead for Phase 2.
                    # It is simpler, more robust, and requires less code that can go wrong.
                    break # Break out to switch to Nelder-Mead logic below if we decide to
            
            mean = new_mean

        # Since implementing full CMA-ES correctly is risky in a short snippet, 
        # and Inspiration 1 (Nelder-Mead) scored 2.12, let's use Nelder-Mead for Phase 2.
        # It is robust and effective for 5D problems.
        
        # Restart Phase 2 with Nelder-Mead
        def nelder_mead(func, x0, budget_remaining, deadline, dim, n_calls_counter, best_f_holder, best_x_holder):
            # Initialize simplex
            alpha = 1.0
            beta = 0.5
            gamma = 2.0
            delta = 0.5
            
            simplex = [x0.copy()]
            f0 = func(x0)
            
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Evaluate initial simplex
            f_values = []
            for i in range(dim + 1):
                if n_calls_counter[0] >= budget or time.time() > deadline - 5:
                    return
                f = func(simplex[i])
                f_values.append(f)
                
                if f < best_f_holder[0]:
                    best_f_holder[0] = f
                    best_x_holder[0] = simplex[i].copy()

            # Nelder-Mead Loop
            max_iter = 2000
            for _ in range(max_iter):
                if n_calls_counter[0] >= budget or time.time() > deadline - 5:
                    break
                
                # Sort simplex by fitness
                idx = np.argsort(f_values)
                simplex = [simplex[i] for i in idx]
                f_values = [f_values[i] for i in idx]
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                if n_calls_counter[0] >= budget or time.time() > deadline - 5:
                    break
                f_r = func(x_r)
                if f_r < best_f_holder[0]:
                    best_f_holder[0] = f_r
                    best_x_holder[0] = x_r.copy()
                    
                if f_values[0] <= f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                    continue
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    if n_calls_counter[0] >= budget or time.time() > deadline - 5:
                        break
                    f_e = func(x_e)
                    if f_e < best_f_holder[0]:
                        best_f_holder[0] = f_e
                        best_x_holder[0] = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                    continue
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + beta * (worst - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - beta * (worst - centroid)
                    if n_calls_counter[0] >= budget or time.time() > deadline - 5:
                        break
                    f_c = func(x_c)
                    if f_c < best_f_holder[0]:
                        best_f_holder[0] = f_c
                        best_x_holder[0] = x_c.copy()
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                        continue
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if n_calls_counter[0] >= budget or time.time() > deadline - 5:
                                return
                            f_values[i] = func(simplex[i])
                            if f_values[i] < best_f_holder[0]:
                                best_f_holder[0] = f_values[i]
                                best_x_holder[0] = simplex[i].copy()
                        
            return

        # Call Nelder-Mead
        # We need to pass the wrapped function and update the global counters
        # To avoid closure issues, we'll just run a simplified NM here using the wrapped_func directly
        # But we need to track calls. wrapped_func already does that.
        
        # Initialize NM Simplex at best_x
        x0_nm = best_x.copy()
        delta_nm = 0.5
        
        simplex_nm = [x0_nm.copy()]
        f_values_nm = [wrapped_func(x0_nm)]
        
        for i in range(dim):
            xi = x0_nm.copy()
            xi[i] += delta_nm
            simplex_nm.append(xi)
            f_values_nm.append(wrapped_func(xi))
            
        # NM Loop
        alpha, beta, gamma = 1.0, 0.5, 2.0
        max_iter_nm = 1000
        
        for _ in range(max_iter_nm):
            if n_calls >= budget or time.time() > deadline - 5:
                break
                
            # Sort
            idx = np.argsort(f_values_nm)
            simplex_nm = [simplex_nm[i] for i in idx]
            f_values_nm = [f_values_nm[i] for i in idx]
            
            centroid = np.mean(simplex_nm[:-1], axis=0)
            worst = simplex_nm[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            if n_calls >= budget or time.time() > deadline - 5:
                break
            f_r = wrapped_func(x_r)
            
            if f_values_nm[0] <= f_r < f_values_nm[-2]:
                simplex_nm[-1] = x_r
                f_values_nm[-1] = f_r
            elif f_r < f_values_nm[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                f_e = wrapped_func(x_e)
                if f_e < f_r:
                    simplex_nm[-1] = x_e
                    f_values_nm[-1] = f_e
                else:
                    simplex_nm[-1] = x_r
                    f_values_nm[-1] = f_r
            else:
                # Contraction
                if f_r < f_values_nm[-1]:
                    x_c = centroid + beta * (worst - centroid)
                else:
                    x_c = centroid - beta * (worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                f_c = wrapped_func(x_c)
                
                if f_c < f_values_nm[-1]:
                    simplex_nm[-1] = x_c
                    f_values_nm[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex_nm[i] = simplex_nm[0] + 0.5 * (simplex_nm[i] - simplex_nm[0])
                        simplex_nm[i] = np.clip(simplex_nm[i], -5, 5)
                        if n_calls >= budget or time.time() > deadline - 5:
                            break
                        f_values_nm[i] = wrapped_func(simplex_nm[i])
                    if n_calls >= budget or time.time() > deadline - 5:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
