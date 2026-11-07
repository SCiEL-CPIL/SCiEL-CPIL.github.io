"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 20
        
        # Initialize population
        # Use a mix of random and heuristic initialization for ZDT4
        # x1 uniform, others near 0
        pop = np.empty((pop_size, n))
        for i in range(pop_size):
            pop[i, 0] = np.random.uniform(lo[0], hi[0])
            for j in range(1, n):
                pop[i, j] = np.random.normal(0, 0.5)
                pop[i, j] = np.clip(pop[i, j], lo[j], hi[j])
        
        # Evaluate initial population
        f_pop = np.empty((pop_size, self.n_obj))
        evals = 0
        for i in range(pop_size):
            f_pop[i] = problem(pop[i])
            evals += 1
            
        # Archive to store non-dominated points
        # List of (x, f)
        archive = []
        
        def is_dominated(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 != f2
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(x, f):
            # Check if x is dominated by any in archive
            for ax, af in archive:
                if is_dominated(af, f):
                    return False
            # Remove any points in archive dominated by x
            new_archive = []
            for ax, af in archive:
                if not is_dominated(f, af):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive
            return True

        # Initialize archive with initial population
        for i in range(pop_size):
            update_archive(pop[i], f_pop[i])

        # Differential Evolution parameters
        F = 0.8
        # CR is effectively 1.0 in the vector operation below
        
        # Main loop
        while evals < budget:
            # Pick a random individual to mutate
            i = np.random.randint(0, pop_size)
            
            # Pick two distinct random individuals
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
                
            # Create trial vector
            v = pop[i] + F * (pop[r1] - pop[r2])
            v = np.clip(v, lo, hi)
            
            # Evaluate trial
            f_v = problem(v)
            evals += 1
            
            # Update archive
            is_new = update_archive(v, f_v)
            
            # Update population
            # If v is non-dominated (is_new is True), it is valuable.
            # If v is dominated by archive, it is less valuable.
            if is_new:
                # v is non-dominated by archive. It's a good candidate.
                # Replace pop[i] with v to converge towards non-dominated front.
                pop[i] = v
                f_pop[i] = f_v
            else:
                # v is dominated by archive.
                # Check if v is better than current pop[i]?
                # If pop[i] is also dominated by archive, we might keep v if it's "less dominated"?
                # Simple strategy: If v dominates pop[i], replace.
                if is_dominated(f_v, f_pop[i]):
                    pop[i] = v
                    f_pop[i] = f_v
                # Otherwise, keep pop[i] to maintain diversity/convergence state.

        return None

# EVOLVE-BLOCK-END
