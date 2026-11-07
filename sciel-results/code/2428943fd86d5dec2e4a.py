"""Rewritten program using Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for the BBOB-style f16 (Weierstrass) task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Bounds
        lb = -5.0
        ub = 5.0

        # Population size: small to allow many generations
        # With budget 500, pop_size 10 allows 50 generations
        pop_size = 10
        
        # Initialize population
        # Start with the center to ensure good initial baseline
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls_made = 1

        # Generate rest of the population randomly
        pop = [x0.copy()]
        pop_f = [f0]
        
        for i in range(pop_size - 1):
            if calls_made >= budget:
                break
            x = rng.uniform(lb, ub, dim)
            fx = func(x)
            pop.append(x)
            pop_f.append(fx)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # DE Parameters
        F = 0.5  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # If we have enough budget for at least 1 generation
        while calls_made < budget:
            # Check if we can afford a full generation
            # We need pop_size evaluations
            if calls_made + pop_size > budget:
                # If not enough for full generation, do individual evaluations?
                # Standard DE needs a full generation for proper selection.
                # We can break or do partial updates. Breaking is safer for simplicity.
                # However, to use all budget, we can evaluate one by one and update if better.
                # But DE logic relies on comparing to the parent in the population.
                # Let's just break if we can't do a full generation to maintain DE integrity,
                # or implement a "greedy" local search for remaining budget.
                # Given the budget is 500 and pop_size 10, 500 % 10 == 0, so this might not be an issue.
                # But if budget is not divisible, we might waste some.
                # Let's adjust pop_size to divide budget? No, budget is fixed 500.
                # 500 / 10 = 50. It divides evenly.
                break

            new_pop = []
            new_pop_f = []
            
            # Generate offspring
            for i in range(pop_size):
                # Select 3 distinct random individuals a, b, c
                # Ensure they are distinct from i if possible, but not strictly required for DE/rand/1
                # Standard: a, b, c distinct from each other.
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != b and b != c and a != c:
                        break
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover: Binomial crossover
                # Choose a random index j0
                j0 = rng.integers(0, dim)
                r = rng.random(dim)
                
                u = np.copy(pop[i])
                for j in range(dim):
                    if r[j] <= CR or j == j0:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Clip to bounds
                u = np.clip(u, lb, ub)
                
                # Evaluate
                fu = func(u)
                calls_made += 1
                
                # Selection: Greedy
                if fu <= pop_f[i]:
                    new_pop.append(u)
                    new_pop_f.append(fu)
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                else:
                    new_pop.append(pop[i])
                    new_pop_f.append(pop_f[i])

            pop = new_pop
            pop_f = new_pop_f

        return best_x, best_f
# EVOLVE-BLOCK-END
