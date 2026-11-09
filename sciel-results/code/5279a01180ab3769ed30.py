"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


class Optimizer:
    """(1+lambda)-ES with adaptive step-size (1/5 rule), time guard, and restart logic."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        # Phase 1: Initial random sampling to find a good starting point
        # Use ~10% of budget for initial exploration
        init_budget = min(50, budget // 10)
        for _ in range(init_budget):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no evaluations were made
            x = np.zeros(dim)
            fx = func(x)
            calls = 1
            best_f = fx
            best_x = x.copy()

        # Phase 2: (1+lambda)-ES with adaptive sigma
        # lambda = 4 offspring per generation
        lam = 4
        max_restarts = 5
        stall_threshold = 10  # No improvement in 10 generations -> restart
        
        for restart in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            if restart == 0:
                x = best_x.copy()
                sigma = 1.0
            else:
                # Perturb best point for restart
                x = best_x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)
                sigma = 1.0
            
            # Evaluate current x if not already best
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            stall_counter = 0
            prev_best = best_f
            
            while calls < budget and time.time() < deadline:
                # Generate lambda offspring
                offspring = []
                for _ in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    offspring.append((fy, y))
                
                if not offspring:
                    break
                
                # Select the best offspring
                offspring.sort(key=lambda t: t[0])
                best_offspring_f, best_offspring_x = offspring[0]
                
                # Adapt sigma using 1/5 rule
                # Count successes among the offspring
                successes = sum(1 for f, _ in offspring if f < fx)
                if successes > lam / 5.0:
                    sigma *= 1.25
                elif successes < lam / 5.0:
                    sigma *= 0.8
                
                # Update parent if best offspring is better
                if best_offspring_f < fx:
                    x, fx = best_offspring_x, best_offspring_f
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        stall_counter = 0
                    else:
                        stall_counter += 1
                else:
                    stall_counter += 1
                
                if stall_counter >= stall_threshold:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
