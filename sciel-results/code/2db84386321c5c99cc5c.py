"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization with diversity maintenance."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Parameters
        NP = 20  # Population size
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0
        v_max = 1.5 # 15% of the range (10)
        reset_interval = 5 # Generations without improvement before reset
        
        # Track evaluations
        eval_count = 0
        best_f_global = np.inf
        best_x_global = np.zeros(self.dim)
        
        def count_func(x):
            nonlocal eval_count, best_f_global, best_x_global
            eval_count += 1
            f_val = func(x)
            if f_val < best_f_global:
                best_f_global = f_val
                best_x_global = x.copy()
            return f_val
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(NP, self.dim))
        vel = rng.uniform(-v_max, v_max, size=(NP, self.dim))
        
        # Personal bests
        pbest_f = np.empty(NP)
        pbest_x = np.empty((NP, self.dim))
        pbest_gen = np.zeros(NP, dtype=int) # Generation last improved
        
        # Evaluate initial population
        for i in range(NP):
            if eval_count >= self.budget:
                break
            pbest_f[i] = count_func(pop[i])
            pbest_x[i] = pop[i].copy()
            pbest_gen[i] = 0
            if pbest_f[i] < best_f_global:
                best_f_global = pbest_f[i]
                best_x_global = pop[i].copy()
        
        if eval_count >= self.budget:
            return best_x_global, best_f_global

        max_gens = (self.budget - eval_count) // NP
        
        for g in range(1, max_gens + 1):
            w = w_start + (w_end - w_start) * ((g - 1) / max(max_gens - 1, 1))
            
            for i in range(NP):
                if eval_count >= self.budget:
                    break
                
                # Diversity maintenance: Reset if no improvement for reset_interval generations
                if g - pbest_gen[i] > reset_interval:
                    pop[i] = rng.uniform(-5, 5, size=self.dim)
                    vel[i] = rng.uniform(-v_max, v_max, size=self.dim)
                
                # Update velocity
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                
                vel[i] = w * vel[i] + c1 * r1 * (pbest_x[i] - pop[i]) + c2 * r2 * (best_x_global - pop[i])
                
                # Clip velocity
                vel[i] = np.clip(vel[i], -v_max, v_max)
                
                # Update position
                pop[i] = pop[i] + vel[i]
                
                # Clip position
                pop[i] = np.clip(pop[i], -5, 5)
                
                # Evaluate
                f_val = count_func(pop[i])
                
                # Update personal best
                if f_val < pbest_f[i]:
                    pbest_f[i] = f_val
                    pbest_x[i] = pop[i].copy()
                    pbest_gen[i] = g
                    
                    # Update global best
                    if f_val < best_f_global:
                        best_f_global = f_val
                        best_x_global = pop[i].copy()

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
