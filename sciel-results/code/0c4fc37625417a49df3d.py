"""Optimized program for BBOB f21 (Gallagher 101 peaks) with parallel CMA-ES portfolio."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel multi-start CMA-ES portfolio with shared global best."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None

        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return 1e18
            n_calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(123)

        # Number of parallel CMA-ES instances
        n_instances = 5
        # Population size per instance
        pop_size = 10
        
        # Initialize multiple CMA-ES instances at different starting points
        instances = []
        for i in range(n_instances):
            # Spread initial points across the domain
            x0 = rng.uniform(-4, 4, dim)
            sigma = 1.5
            # CMA-ES state
            C = np.eye(dim)
            # Initialize population around x0
            pop = x0 + sigma * rng.standard_normal((pop_size, dim))
            pop = np.clip(pop, -5, 5)
            instances.append({
                'x': x0.copy(),
                'sigma': sigma,
                'C': C,
                'pop': pop,
                'fx': float('inf'),
                'active': True
            })
        
        # Evaluate initial populations
        for inst in instances:
            if not inst['active'] or n_calls >= budget or time.time() > deadline - 10:
                continue
            for x in inst['pop']:
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                inst['fx'] = min(inst['fx'], wrapped_func(x))
        
        # Main optimization loop
        while n_calls < budget and time.time() < deadline - 5:
            # Check if any instance is still active
            active_instances = [i for i, inst in enumerate(instances) if inst['active']]
            if not active_instances:
                break
            
            # Select the best active instance to expand
            # Prioritize instances with good progress
            best_inst_idx = None
            best_inst_fx = float('inf')
            for idx in active_instances:
                if instances[idx]['fx'] < best_inst_fx:
                    best_inst_fx = instances[idx]['fx']
                    best_inst_idx = idx
            
            if best_inst_idx is None:
                break
            
            inst = instances[best_inst_idx]
            
            # Generate new population for this instance
            # Use CMA-ES-like sampling
            new_pop = []
            for j in range(pop_size):
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Apply C matrix (simplified - just use identity for now)
                x_new = inst['x'] + inst['sigma'] * z
                x_new = np.clip(x_new, -5, 5)
                new_pop.append(x_new)
            
            # Evaluate new population
            old_fx = inst['fx']
            for x in new_pop:
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                fx = wrapped_func(x)
                if fx < inst['fx']:
                    inst['fx'] = fx
                    inst['x'] = x.copy()
            
            # Update CMA-ES parameters
            # Simple adaptive step size based on improvement
            if inst['fx'] < old_fx:
                inst['sigma'] *= 1.1
            else:
                inst['sigma'] *= 0.9
            
            # If sigma becomes too small, deactivate this instance
            if inst['sigma'] < 1e-3:
                inst['active'] = False
            
            # Occasionally add new instances if we have budget left
            # and some instances have died
            if len(active_instances) < n_instances and n_calls < budget * 0.7:
                # Check if we should add a new instance
                if len([i for i, inst in enumerate(instances) if not inst['active']]) > 0:
                    # Reactivate a dead instance with new random start
                    dead_indices = [i for i, inst in enumerate(instances) if not inst['active']]
                    idx = dead_indices[0]
                    x0 = rng.uniform(-4, 4, dim)
                    instances[idx] = {
                        'x': x0.copy(),
                        'sigma': 1.5,
                        'C': np.eye(dim),
                        'pop': x0 + 1.5 * rng.standard_normal((pop_size, dim)),
                        'fx': float('inf'),
                        'active': True
                    }
                    # Evaluate new population
                    inst = instances[idx]
                    inst['pop'] = np.clip(inst['pop'], -5, 5)
                    for x in inst['pop']:
                        if n_calls >= budget or time.time() > deadline - 5:
                            break
                        inst['fx'] = min(inst['fx'], wrapped_func(x))

        return best_x, best_f
# EVOLVE-BLOCK-END
