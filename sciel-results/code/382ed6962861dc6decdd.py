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
        deadline = start_time + 285.0

        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= total_budget:
                raise TimeoutError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        def halton_sequence(i, b):
            f = 1.0
            r = 0.0
            while i > 0:
                f /= b
                r += f * (i % b)
                i //= b
            return r

        def halton_point(index, dim):
            # Use first dim primes
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point = np.zeros(dim)
            for d in range(dim):
                # offset index to avoid 0
                point[d] = halton_sequence(index + d + 1, primes[d])
            return point

        # Phase 1: Initial Global Search with Halton + Random
        # Use a mix of Halton for coverage and random for diversity
        n_global = min(30, total_budget)
        
        # First, evaluate origin and a few deterministic points
        deterministic_points = [
            np.zeros(dim),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
        ]
        
        for x_det in deterministic_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            try:
                fx = wrapper(x_det)
                if fx < best_f:
                    best_f = fx
                    best_x = x_det.copy()
            except TimeoutError:
                break

        # Halton sequence for structured coverage
        # Map [0,1]^dim to [-5,5]^dim
        n_halton = min(20, max(0, total_budget - calls_made))
        for i in range(n_halton):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x_unit = halton_point(i + len(deterministic_points) + 1, dim)
            x = x_unit * 10.0 - 5.0
            try:
                fx = wrapper(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except TimeoutError:
                break

        # Random sampling for the rest of the global phase
        remaining_global = n_global - calls_made
        for _ in range(remaining_global):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            try:
                fx = wrapper(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except TimeoutError:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                best_x = np.zeros(dim)
                best_f = float('inf')
                return best_x, best_f

        # Phase 2: Restartable (1+1)-ES with 1/5 success rule
        # This is the core local optimizer
        # Parameters for (1+1)-ES
        # Step size sigma, population size lambda
        # 1/5 success rule: increase sigma if success rate > 0.2, decrease if < 0.2
        
        sigma = 1.0  # Initial step size
        lambda_ = 10  # Number of offspring per iteration
        success_rate = 0.0
        successes = 0
        total_trials = 0
        
        # We will run this until budget or time is exhausted
        # Each iteration uses lambda_ calls
        # We restart if no improvement for a while or if sigma becomes too small/large
        
        no_improve_count = 0
        max_no_improve = 50  # Restart after 50 iterations without improvement
        
        while calls_made < total_budget and time.time() < deadline:
            # Check if we should restart
            if no_improve_count > max_no_improve:
                # Restart with a new random point or perturb best_x
                # Try a few random restarts
                for _ in range(5):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    x_new = rng.uniform(-5.0, 5.0, dim)
                    try:
                        fx_new = wrapper(x_new)
                        if fx_new < best_f:
                            best_f = fx_new
                            best_x = x_new.copy()
                            no_improve_count = 0
                            sigma = 1.0
                            successes = 0
                            total_trials = 0
                    except TimeoutError:
                        break
                
                if no_improve_count > max_no_improve:
                    # If still no improvement, just stop or continue with smaller sigma?
                    # Let's just break to save time for potential other strategies if any
                    # But for now, break the ES loop
                    break

            # Generate lambda_ offspring
            improved = False
            for _ in range(lambda_):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                # Gaussian perturbation
                x_trial = best_x + sigma * rng.standard_normal(dim)
                try:
                    fx_trial = wrapper(x_trial)
                except TimeoutError:
                    break
                
                total_trials += 1
                if fx_trial < best_f:
                    best_f = fx_trial
                    best_x = x_trial.copy()
                    successes += 1
                    improved = True
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update sigma using 1/5 success rule
            if total_trials >= 20:  # Need enough samples
                current_success_rate = successes / total_trials
                # Increase sigma if success rate is high, decrease if low
                # Standard rule: sigma = sigma * exp( (1/5 - success_rate) / (5*lambda) )
                # Or simpler: if success_rate > 0.2, sigma *= 1.5, else sigma /= 1.5
                if current_success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma /= 1.2
                
                # Reset counters
                successes = 0
                total_trials = 0
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

        # Phase 3: Final Polish
        # Use a few calls to refine the best point with very small steps
        if calls_made < total_budget and time.time() < deadline:
            polish_sigma = max(1e-5, sigma * 0.1)
            for _ in range(min(20, total_budget - calls_made)):
                if time.time() > deadline:
                    break
                x_trial = best_x + polish_sigma * rng.standard_normal(dim)
                try:
                    fx_trial = wrapper(x_trial)
                    if fx_trial < best_f:
                        best_f = fx_trial
                        best_x = x_trial.copy()
                except TimeoutError:
                    break

        return best_x, best_f
