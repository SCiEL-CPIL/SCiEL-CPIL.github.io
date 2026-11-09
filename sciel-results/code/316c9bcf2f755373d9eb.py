"""Cooperative Coevolution Optimizer for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Cooperative Coevolution (CC) strategy for BBOB f4.
    
    Strategy:
    - Decompose 5D search space into 5 islands (one per dimension).
    - Each island maintains a population of candidates that differ only in their specific dimension.
    - A 'cross' operator is used: to generate a candidate for island i, we take the current global best,
      modify dimension i using the island's local evolutionary step, and evaluate.
    - This tests if combining global context (the rest of the vector) with local dimension-specific search
      (the cross) leads to better progress than monolithic CMA-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        # State
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Initialize with a random point
        rng = np.random.default_rng(42)
        x_init = rng.uniform(-5, 5, dim)
        f_init = func(x_init)
        calls_made += 1
        best_x = x_init.copy()
        best_f = f_init
        
        # Cooperative Coevolution Setup
        # Each island i corresponds to dimension i
        # Island state: current best value for that dimension (in the context of best_x)
        # We maintain a population for each island to explore variations in that dimension.
        
        # For simplicity and robustness, we use a (1+1)-ES for each island, 
        # but the "cross" is implicit: the candidate is formed by replacing dim i in best_x.
        
        # Parameters for local (1+1)-ES on each dimension
        # We use a shared step size strategy but independent directions per dimension
        sigma = 1.0  # Initial step size for dimension modifications
        
        # To make it more robust, we maintain a small population per island (e.g., 3 candidates)
        # and use a simple selection.
        
        pop_size = 3
        # Pop[i] is a list of (value_for_dim_i, f_value)
        # Actually, to keep it simple and effective within budget:
        # We will iterate through dimensions cyclically.
        
        # Let's refine: 
        # The "cross" operation: 
        # Candidate x_c is formed by taking best_x, and changing dimension i to a new value v.
        # v is sampled from a distribution centered at best_x[i] with scale sigma.
        # If f(x_c) < f(best_x), we accept and update best_x.
        # This is essentially a coordinate descent with random perturbations.
        # To make it "Cooperative Coevolution", we allow each dimension to have its own 
        # "memory" of good values.
        
        # Let's implement a simple CC where each island has a population of 5 values for its dimension.
        # The fitness of a value v for dimension i is f(best_x with dim i set to v).
        # But this requires many function calls.
        
        # Alternative CC: 
        # Each island i maintains a mean m_i and step size s_i.
        # In each iteration, pick an island i (round-robin or best-improvement).
        # Generate a candidate: x_new = best_x; x_new[i] = m_i + s_i * N(0,1).
        # Evaluate f(x_new).
        # Update island i's statistics based on success/failure.
        # If successful, update best_x.
        
        # This is a structured random search with memory per dimension.
        
        # Initialize island states
        m = best_x.copy()  # Mean for each dimension
        s = np.full(dim, 1.0)  # Step size for each dimension
        
        # We will use a budget-aware loop
        while calls_made < total_budget:
            if time.time() >= deadline:
                break
            
            # Select dimension to explore
            # Strategy: Explore the dimension that has not been improved recently, 
            # or just cycle through them. Cycling is simple and ensures coverage.
            # Let's cycle: 0, 1, 2, 3, 4, 0, ...
            # But to be smarter, we can prioritize dimensions where the current best is "stuck".
            # For now, simple cycling with a small population per dimension.
            
            # Actually, let's implement a proper island model.
            # Island i has a population of 5 candidates for dimension i.
            # The "cross" is: take the global best, replace dim i with a candidate from island i.
            
            # To save complexity, let's use a (1+1) ES per dimension, but with 
            # adaptive step sizes and a "restart" if no improvement for a while.
            
            # Let's try a different angle: 
            # The "cross" operator in CC usually combines sub-solutions.
            # Here, the sub-solution is the value of a single dimension.
            # So, we maintain for each dimension i a set of "good" values found so far.
            # We sample from these sets to create new candidates.
            
            # Implementation:
            # history[i] = list of values for dimension i that have been part of a best_x or near-best.
            # Start with best_x[i] in history[i].
            
            history = [[] for _ in range(dim)]
            for i in range(dim):
                history[i].append(best_x[i])
            
            # Step sizes for each dimension
            sigma_dim = np.full(dim, 1.0)
            
            # No-improvement counter per dimension
            no_improve = np.zeros(dim, dtype=int)
            
            while calls_made < total_budget:
                if time.time() >= deadline:
                    break
                
                # Select dimension: pick the one with the longest no_improve streak
                i = np.argmax(no_improve)
                
                # Generate candidate using "cross":
                # Take best_x, modify dimension i.
                # Sample new value for dimension i.
                # Mix of: 
                # 1. Local search around best_x[i]
                # 2. Resampling from history[i] (if available)
                # 3. Random restart in [-5, 5] if stuck too long
                
                r = rng.random()
                if r < 0.5:
                    # Local search
                    new_val = best_x[i] + rng.normal(0, sigma_dim[i])
                elif r < 0.8 and len(history[i]) > 1:
                    # Sample from history
                    new_val = rng.choice(history[i])
                    # Add small noise
                    new_val += rng.normal(0, sigma_dim[i] * 0.5)
                else:
                    # Random restart
                    new_val = rng.uniform(-5, 5)
                
                new_val = np.clip(new_val, -5, 5)
                
                # Construct candidate
                x_cand = best_x.copy()
                x_cand[i] = new_val
                
                f_cand = func(x_cand)
                calls_made += 1
                
                if f_cand < best_f:
                    # Improvement!
                    best_f = f_cand
                    best_x = x_cand.copy()
                    
                    # Update history for dimension i
                    history[i].append(new_val)
                    # Keep history short to avoid bias
                    if len(history[i]) > 5:
                        history[i] = history[i][-5:]
                    
                    # Reset no_improve for dimension i
                    no_improve[i] = 0
                    # Reset no_improve for others? No, let them accumulate.
                    
                    # Adjust step size: increase slightly to explore more
                    sigma_dim[i] *= 1.1
                else:
                    # No improvement
                    no_improve[i] += 1
                    # Decrease step size to refine
                    sigma_dim[i] *= 0.9
                    
                    # If stuck for too long, force a larger restart
                    if no_improve[i] > 20:
                        sigma_dim[i] = 2.0
                        no_improve[i] = 0
            
            # If we broke out of the inner loop due to budget or time, exit outer loop
            if calls_made >= total_budget or time.time() >= deadline:
                break
            
            # If we haven't improved the global best for a long time, 
            # we might need a global restart. 
            # But the per-dimension no_improve handles this locally.
            
            # To prevent infinite loops if no dimension improves, 
            # we add a global counter.
            # Actually, the while loop condition handles budget.
            
            # Let's add a global stagnation check.
            # If no global improvement for 50 calls, do a random restart of best_x?
            # No, that's risky. Let's stick to the CC structure.
            
            # Break the outer loop if we've used the budget
            if calls_made >= total_budget:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
