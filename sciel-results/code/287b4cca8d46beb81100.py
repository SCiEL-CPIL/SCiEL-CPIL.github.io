"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

class Optimizer:
    """Multi-objective optimizer: Differential Evolution with Archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Archive of non-dominated points
        archive = []
        used = 0
        
        def dominates(a, b):
            """True if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)
            
        def add_to_archive(fval):
            nonlocal archive
            # Check if dominated by existing
            for f in archive:
                if dominates(f, fval):
                    return False
            # Remove existing points dominated by new one
            new_archive = []
            for f in archive:
                if not dominates(fval, f):
                    new_archive.append(f)
            new_archive.append(fval)
            archive = new_archive
            return True

        def is_deadline_reached():
            return time.time() > deadline or used >= self.budget

        # 1. Initialization: Latin Hypercube Sampling (LHS) for diversity
        # We want a good initial spread. Let's use about 20-30% of budget for init.
        # 4D space. 20 points is a reasonable start for LHS in 4D.
        init_budget = max(10, self.budget // 5)
        
        # Generate LHS points
        # Simple LHS: divide each dimension into N bins, pick one point per bin, permute.
        # For 4D, we can just generate random points but ensure stratification.
        # A simple stratified sampling approach:
        n_init = init_budget
        # To make it simple and effective, we can just use random points but 
        # ensure we cover the corners and center.
        # Actually, true LHS is better.
        # Let's implement a simple stratified sampler.
        # For each dimension, divide [0,1] into n_init intervals.
        # For each interval, pick a random point.
        # Then permute the assignments for each dimension independently.
        
        # However, implementing full LHS correctly for arbitrary N in 4D is a bit of code.
        # A simpler "stratified" approach:
        # Divide space into 2^4 = 16 hypercubes. Evaluate center of each. (16 points)
        # Then fill remaining init_budget - 16 with random points.
        
        init_points = []
        # 1. Centers of 2^4 sub-cubes
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        x = np.array([
                            lo[0] + (hi[0]-lo[0]) * (i + 0.5) / 2.0,
                            lo[1] + (hi[1]-lo[1]) * (j + 0.5) / 2.0,
                            lo[2] + (hi[2]-lo[2]) * (k + 0.5) / 2.0,
                            lo[3] + (hi[3]-lo[3]) * (l + 0.5) / 2.0
                        ])
                        init_points.append(x)
        
        # 2. Add random points to fill init_budget
        while len(init_points) < init_budget:
            x = lo + (hi - lo) * np.random.rand(dim)
            init_points.append(x)
            
        # Evaluate init points
        for x in init_points:
            if is_deadline_reached():
                break
            f = problem(x)
            used += 1
            add_to_archive(f)
            
        if not archive:
            return None

        # Maintain a population for DE.
        # Population size: small, e.g., 20.
        pop_size = 20
        # Initialize population with random points or from archive?
        # Random is better for diversity.
        population = []
        for _ in range(pop_size):
            x = lo + (hi - lo) * np.random.rand(dim)
            f = problem(x)
            used += 1
            # Don't add to archive yet, just keep in population
            population.append((x, f))
            add_to_archive(f) # Add to archive immediately to update front
            
        if is_deadline_reached():
            return None

        # DE Parameters
        # CR: crossover rate, F: differential weight
        # Adaptive F and CR can help, but fixed often works well.
        # Let's use F=0.8, CR=0.9 (standard for DE/rand/1)
        # Or DE/best/1?
        # For multi-objective, we need a selection mechanism.
        # Standard MOEA/D or NSGA-II style.
        # Let's use a simple "Archive + Population" approach.
        # At each step:
        # 1. Pick 3 distinct parents from population (or archive?).
        #    Using population is standard DE.
        # 2. Generate trial vector: v = x_r1 + F * (x_r2 - x_r3)
        # 3. Clip to bounds.
        # 4. Evaluate.
        # 5. Selection: If trial is non-dominated, add to archive.
        #    Replace a member in population?
        #    Standard DE replaces the target if trial is better.
        #    In multi-objective, "better" is ambiguous.
        #    Strategy: 
        #    - Keep a fixed size population.
        #    - For each target x_i in population:
        #      - Generate trial v_i.
        #      - If v_i is non-dominated by archive, add to archive.
        #      - Decide whether to replace x_i with v_i.
        #      - A common heuristic: Replace if v_i dominates x_i, OR if v_i is non-dominated and x_i is dominated (by archive or others), OR randomly if both non-dominated.
        #      - Simpler: Just keep the population diverse. If v_i is non-dominated, replace a random member of population that is dominated or worst.
        
        # Let's refine the loop:
        # We have `used` evaluations. We have `budget` total.
        # We will loop until budget is exhausted.
        
        # To speed up, we can do multiple DE steps per evaluation? No, 1 eval per step.
        
        # Let's use a "Crowding Distance" or simple random selection for replacement.
        # Given the small budget (200), a simple strategy is:
        # 1. Generate trial.
        # 2. If trial is non-dominated, it goes to archive.
        # 3. For population update: 
        #    - If trial dominates target, replace.
        #    - If target dominates trial, keep target.
        #    - If non-dominated: 
        #      - If trial is closer to ideal point (sum of normalized objectives), replace?
        #      - Or just keep population as is to maintain diversity, unless trial is clearly better.
        
        # Actually, a very effective simple strategy for small budgets:
        # 1. Maintain a population of size P.
        # 2. Each iteration:
        #    a. Select 3 distinct random indices i, j, k from population.
        #    b. Create trial vector v.
        #    c. Evaluate v.
        #    d. Add v to archive if non-dominated.
        #    e. Select a victim from population. 
        #       - If v dominates victim, replace victim.
        #       - If victim dominates v, do nothing.
        #       - If non-dominated, replace victim with some probability (e.g., 0.5) or if v has better "spread" metric.
        #       - To ensure progress, let's replace the "worst" point in population based on sum of objectives (normalized) if v is non-dominated.
        
        # Let's implement:
        # 1. Normalize objectives based on current archive min/max.
        # 2. Score each point by sum of normalized objectives.
        # 3. Replace the point with highest score (worst) if trial is non-dominated.
        
        def get_norm_bounds():
            if not archive:
                return np.zeros(n_obj), np.ones(n_obj)
            f_arr = np.array(archive)
            f_min = f_arr.min(axis=0)
            f_max = f_arr.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range < 1e-12] = 1.0
            return f_min, f_range

        while not is_deadline_reached():
            # Select parents
            # Ensure we have enough population members
            if len(population) < 3:
                # Add random points
                x = lo + (hi - lo) * np.random.rand(dim)
                f = problem(x)
                used += 1
                population.append((x, f))
                add_to_archive(f)
                continue

            # Randomly pick 3 distinct indices
            idxs = np.random.choice(len(population), 3, replace=False)
            x1 = population[idxs[0]][0]
            x2 = population[idxs[1]][0]
            x3 = population[idxs[2]][0]
            
            # Differential Evolution Mutation
            F = 0.8
            v = x1 + F * (x2 - x3)
            
            # Clip to bounds
            v = np.clip(v, lo, hi)
            
            # Evaluate
            f_v = problem(v)
            used += 1
            
            # Check dominance
            is_dominated = False
            for f_a in archive:
                if dominates(f_a, f_v):
                    is_dominated = True
                    break
            
            if not is_dominated:
                add_to_archive(f_v)
                
                # Update Population
                # Find the "worst" point in population to potentially replace
                # Use sum of normalized objectives as a proxy for quality
                f_min, f_range = get_norm_bounds()
                scores = []
                for i, (x_p, f_p) in enumerate(population):
                    norm_f = (f_p - f_min) / f_range
                    scores.append(np.sum(norm_f))
                
                worst_idx = np.argmax(scores)
                worst_f = population[worst_idx][1]
                
                # Replace if v is better than worst, or randomly if non-dominated
                # Simple rule: If v is non-dominated, replace the worst point.
                # This biases towards improving the population quality.
                population[worst_idx] = (v, f_v)
                
            # If dominated, we don't add to archive, and we don't necessarily update population
            # to maintain diversity, we might still replace if it's very close?
            # For simplicity, only replace if non-dominated.
            
        return None

# EVOLVE-BLOCK-END
