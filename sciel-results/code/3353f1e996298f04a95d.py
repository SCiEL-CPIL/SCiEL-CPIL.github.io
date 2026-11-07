import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        # 330s limit, use 290s for safety
        deadline = start_time + 290.0

        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            
            calls_made += 1
            # Clip to box
            x = np.clip(x, -5, 5)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            return fx

        # Halton sequence for deterministic global exploration
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim, bounds=(-5, 5)):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(n):
                for j in range(dim):
                    base = primes[j % len(primes)]
                    idx = i + 1
                    val = halton(idx, base)
                    points[i, j] = bounds[0] + val * (bounds[1] - bounds[0])
            return points

        # Phase 1: Global Exploration with Halton
        num_global = max(10, min(50, total_budget // 5))
        if calls_made + num_global > total_budget:
            num_global = total_budget - calls_made
            
        halton_points = halton_sequence(num_global, dim)
        
        candidates = []
        for i in range(num_global):
            if time.time() > deadline or calls_made >= total_budget:
                break
            x = halton_points[i]
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except:
                return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: CMA-ES-lite with coordinate-wise adaptive step sizes
        # This is a lightweight version of CMA-ES that adapts step sizes per dimension
        # based on success rate of recent moves
        
        remaining = total_budget - calls_made
        if remaining > 20:
            # Use top candidate as starting point
            start_x = candidates[0][1].copy()
            
            # Initialize step sizes
            sigma = np.ones(dim) * 1.0
            
            # CMA-ES-like parameters
            pop_size = max(4, min(10, dim + 3))
            c_sigma = (pop_size + 2) / ((pop_size + 1) ** 2 + dim)
            c_c = 4 / (pop_size + 4)
            c_1 = 2 / ((pop_size + 1.3) ** 2 + dim)
            c_mu = min(1 - c_1, 2 * (mu_w - 1) / ((pop_size - 2 + mu_w) ** 2 - mu_w))
            
            # Weights for intermediate recombination
            weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
            weights /= weights.sum()
            mu = int(np.ceil(pop_size / 2))
            mu_w = 1.0 / weights[:mu].sum()
            
            # Memory for success rate
            success_history = [0.0] * dim
            success_window = 10
            
            # Run CMA-ES-lite
            x_mean = start_x.copy()
            best_local_f = best_f
            best_local_x = best_x.copy()
            
            while calls_made < total_budget and time.time() < deadline:
                # Generate population
                if calls_made + pop_size > total_budget:
                    pop_size = total_budget - calls_made
                    if pop_size < 2:
                        break
                
                population = []
                for i in range(pop_size):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    z = np.random.randn(dim)
                    # Apply adaptive step sizes
                    x_i = x_mean + sigma * z
                    try:
                        f_i = wrapper(x_i)
                    except TimeoutError:
                        break
                    population.append((f_i, x_i.copy()))
                
                if len(population) < 2:
                    break
                
                # Sort by fitness
                population.sort(key=lambda x: x[0])
                
                # Update best
                if population[0][0] < best_local_f:
                    best_local_f = population[0][0]
                    best_local_x = population[0][1].copy()
                    if best_local_f < best_f:
                        best_f = best_local_f
                        best_x = best_local_x.copy()
                
                # Update success rate for each dimension
                # A move is successful if it improved the mean
                old_mean = x_mean.copy()
                
                # Intermediate recombination
                x_new = np.zeros(dim)
                for i in range(min(mu, len(population))):
                    x_new += weights[i] * population[i][1]
                
                # Update mean
                x_mean = x_new
                
                # Check success per dimension
                for d in range(dim):
                    if abs(x_mean[d] - old_mean[d]) > 1e-10:
                        # Move was made in this dimension
                        if population[0][0] < best_local_f:
                            success_history[d] = 0.7 * success_history[d] + 0.3 * 1.0
                        else:
                            success_history[d] = 0.7 * success_history[d] + 0.3 * 0.0
                    
                    # Adapt step size based on success rate
                    if success_history[d] > 0.5:
                        sigma[d] *= 1.05  # Increase step size
                    else:
                        sigma[d] *= 0.95  # Decrease step size
                    
                    # Clip step size
                    sigma[d] = np.clip(sigma[d], 1e-4, 10.0)
                
                # Check for convergence
                if np.all(sigma < 1e-3) or calls_made >= total_budget:
                    break
                
                # Periodic local polish with finite-difference gradient
                if calls_made % (pop_size * 5) == 0 and calls_made + 2 * dim < total_budget:
                    # Estimate gradient using finite differences
                    grad = np.zeros(dim)
                    h = 1e-4
                    f0 = best_local_f
                    for d in range(dim):
                        if calls_made >= total_budget:
                            break
                        x_plus = best_local_x.copy()
                        x_plus[d] += h
                        try:
                            f_plus = wrapper(x_plus)
                        except TimeoutError:
                            break
                        grad[d] = (f_plus - f0) / h
                    
                    # Take a gradient descent step
                    if np.all(np.isfinite(grad)) and np.linalg.norm(grad) > 1e-6:
                        step_size = 0.01
                        x_new = best_local_x - step_size * grad
                        x_new = np.clip(x_new, -5, 5)
                        try:
                            f_new = wrapper(x_new)
                        except TimeoutError:
                            break
                        if f_new < best_local_f:
                            best_local_f = f_new
                            best_local_x = x_new.copy()
                            if best_local_f < best_f:
                                best_f = best_local_f
                                best_x = best_local_x.copy()
                            x_mean = best_local_x.copy()

        # Phase 3: Final Nelder-Mead polish on the best solution
        remaining = total_budget - calls_made
        if remaining > 10 and best_x is not None:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapper, best_x, method='Nelder-Mead',
                               options={
                                   'maxfev': remaining,
                                   'xatol': 1e-8,
                                   'fatol': 1e-8,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_f
