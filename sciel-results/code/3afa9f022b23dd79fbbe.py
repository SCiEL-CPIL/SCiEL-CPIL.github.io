import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        
        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)
        
        # Phase 1: Diversified initial sampling with LHS
        n_initial = 10
        agents = []
        
        # Latin Hypercube sampling for better coverage
        for i in range(n_initial):
            if not time_ok() or calls >= budget:
                break
            # LHS: stratify each dimension
            x = np.zeros(dim)
            for d in range(dim):
                # Stratified sample: (i + rng.random()) / n_initial
                frac = (i + rng.random()) / n_initial
                x[d] = -5.0 + 10.0 * frac
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            record(x, f)
            agents.append({
                'x': x.copy(),
                'f': f,
                'sigma': 1.5,
                'stagnation': 0
            })
        
        if not agents:
            return best_x, best_f

        # Phase 2: Parallel (1+1)-ES with adaptive sigma
        sigma_min = 1e-5
        sigma_max = 8.0
        stagnation_limit = 40
        global_stagnation = 0
        last_best_f = best_f
        
        # Main search loop
        while calls < budget and time_ok():
            # Select agent with best f (best-first)
            best_idx = 0
            for i in range(1, len(agents)):
                if agents[i]['f'] < agents[best_idx]['f']:
                    best_idx = i
            
            agent = agents[best_idx]
            x = agent['x']
            f = agent['f']
            sigma = agent['sigma']
            
            # Generate mutation with slight anisotropy based on recent success
            z = rng.standard_normal(dim)
            # Scale each dimension differently based on historical progress
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            f_trial = func(x_trial)
            calls += 1
            
            if f_trial < f:
                # Accept: move to trial
                agent['x'] = x_trial
                agent['f'] = f_trial
                # Expand sigma on success
                agent['sigma'] = min(sigma * 1.08, sigma_max)
                agent['stagnation'] = 0
                
                if f_trial < best_f:
                    record(x_trial, f_trial)
                    global_stagnation = 0
                    # If significant improvement, restart other agents near new best
                    if best_f < last_best_f - 1e-6:
                        last_best_f = best_f
                        for j in range(len(agents)):
                            if j != best_idx and agents[j]['stagnation'] > 20:
                                noise = rng.standard_normal(dim) * 0.3
                                x_new = best_x + noise
                                x_new = np.clip(x_new, -5.0, 5.0)
                                agents[j]['x'] = x_new
                                agents[j]['f'] = f_trial  # approximate
                                agents[j]['sigma'] = 0.8
                                agents[j]['stagnation'] = 0
            else:
                # Reject: shrink sigma
                agent['sigma'] = max(sigma * 0.92, sigma_min)
                agent['stagnation'] += 1
                global_stagnation += 1
            
            # Restart stagnant agents
            if agent['stagnation'] > stagnation_limit:
                if best_x is not None:
                    noise_scale = 0.5 + 0.3 * rng.random()
                    noise = rng.standard_normal(dim) * noise_scale
                    x_new = best_x + noise
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = func(x_new)
                    calls += 1
                    if f_new < best_f:
                        record(x_new, f_new)
                    agent['x'] = x_new
                    agent['f'] = f_new
                    agent['sigma'] = 1.0
                    agent['stagnation'] = 0
        
        # Phase 3: Final local refinement on global best
        if calls < budget and best_x is not None and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            
            for _ in range(30):
                if calls >= budget or not time_ok():
                    break
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = func(x_trial)
                calls += 1
                
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    sigma = min(sigma * 1.1, 0.5)
                    record(x, f)
                else:
                    sigma = max(sigma * 0.85, 1e-4)
        
        return best_x, best_f
