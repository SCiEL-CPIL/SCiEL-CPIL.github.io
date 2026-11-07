"""Simulated Annealing Optimizer with Multi-Start Reheating."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Simulated Annealing Optimizer for BBOB f21.
    
    Strategy:
    1. Initialize at a random point (or center).
    2. Perform SA with geometric cooling.
    3. If the algorithm stagnates (no improvement for N steps) or temperature 
       drops too low, perform a 'reheat' by resetting the temperature to a 
       high value and potentially jumping to a new random location or 
       staying at the current best to explore the neighborhood more broadly.
    4. Continue until budget or time limit is reached.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s, we aim for 300s max execution
        deadline = start_time + 295.0 
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # SA Parameters
        # Initial temperature high enough to escape local minima in [-5,5]^5
        # Assuming function values are in a reasonable range, e.g., [-100, 100]
        # We need T to be large enough that exp(-delta/T) is not 0 for small deltas.
        T_init = 10.0 
        T_min = 1e-6
        cooling_rate = 0.995  # Geometric cooling
        max_iter_no_improve = 50  # Stagnation limit
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        # Initial point: Center of the box is often a good heuristic for BBOB
        current_x = np.zeros(dim)
        current_f = func(current_x)
        calls_made += 1
        best_x = current_x.copy()
        best_f = current_f
        
        T = T_init
        no_improve_count = 0
        
        # Adaptive Step Size
        # Start with a step size that covers a significant portion of the box
        step_size = 2.0 

        while calls_made < budget and time.time() < deadline:
            # Generate candidate
            # Use Gaussian perturbation scaled by step_size
            # We clip to the box [-5, 5]
            candidate_x = current_x + step_size * rng.standard_normal(dim)
            candidate_x = np.clip(candidate_x, -5, 5)
            
            candidate_f = func(candidate_x)
            calls_made += 1
            
            delta = candidate_f - current_f
            
            # Acceptance Criteria
            if delta < 0:
                # Always accept improvement
                current_x = candidate_x
                current_f = candidate_f
                no_improve_count = 0
                
                # Update global best
                if current_f < best_f:
                    best_f = current_f
                    best_x = current_x.copy()
            else:
                # Accept with probability exp(-delta / T)
                # Prevent overflow/underflow in exp
                if T > 1e-10:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                
                if rng.random() < prob:
                    current_x = candidate_x
                    current_f = candidate_f
                    # Note: We don't reset no_improve_count here for acceptance of worse solutions?
                    # Usually, no_improve tracks the *best* solution found.
                    # Let's stick to tracking best solution improvement for restart logic.
                else:
                    # Rejection
                    pass
            
            # Track stagnation relative to GLOBAL BEST
            if current_f >= best_f:
                 # This logic is slightly tricky. 
                 # Let's define stagnation as: best_f has not improved for 'max_iter_no_improve' steps.
                 pass 
            
            # Better stagnation tracking:
            # We need to know when best_f was last updated.
            # Let's maintain a counter 'steps_since_best'
            # Actually, simpler: if we don't update best_x in the loop above, increment a counter.
            # But 'current_f' can be better than 'best_f' only if we just updated it.
            # Let's refactor the update logic for clarity.
            
            # Re-evaluating the loop logic for clarity and correctness
            
            # Cooling
            T *= cooling_rate
            
            # Restart / Reheat Logic
            # If T is too low, or if we've been stuck for a long time, reheat.
            # "Stuck" means best_f hasn't improved for a while.
            # Let's track 'stagnation_count' which increments every time we DON'T improve best_f.
            
            # Wait, the loop above doesn't track stagnation_count explicitly.
            # Let's rewrite the loop body to be cleaner.
            
            pass # Placeholder, will rewrite below

        # The logic above was getting messy. Let's write a clean version.
        
        return best_x, best_f

# REWRITE FOR CLEANLINESS AND CORRECTNESS

class Optimizer:
    """
    Simulated Annealing Optimizer with Multi-Start Reheating.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0 
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # SA Parameters
        T_init = 10.0 
        T_min = 1e-5
        cooling_rate = 0.995 
        stagnation_limit = 100  # Steps without improving global best before reheat
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        # Initial point
        current_x = np.zeros(dim)
        current_f = func(current_x)
        calls_made += 1
        best_x = current_x.copy()
        best_f = current_f
        
        T = T_init
        steps_since_best = 0
        step_size = 2.0  # Initial step size
        
        # We will loop until budget or time is exhausted
        while calls_made < budget and time.time() < deadline:
            # Generate candidate
            # Use Gaussian perturbation
            # Adaptive step size: shrink if we are improving frequently, expand if stuck?
            # For simplicity, keep step_size constant or slowly decaying.
            # Let's try a constant step size of 1.0 or 2.0. 2.0 is good for global search.
            
            candidate_x = current_x + step_size * rng.standard_normal(dim)
            candidate_x = np.clip(candidate_x, -5, 5)
            
            candidate_f = func(candidate_x)
            calls_made += 1
            
            delta = candidate_f - current_f
            
            # Acceptance
            if delta < 0:
                current_x = candidate_x
                current_f = candidate_f
            else:
                if T > 1e-10:
                    # Avoid overflow in exp if delta/T is large positive
                    # exp(-large) is 0
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                
                if rng.random() < prob:
                    current_x = candidate_x
                    current_f = candidate_f
            
            # Update Global Best and Stagnation Counter
            if current_f < best_f:
                best_f = current_f
                best_x = current_x.copy()
                steps_since_best = 0
            else:
                steps_since_best += 1
            
            # Cooling
            T *= cooling_rate
            
            # Reheat / Restart Logic
            # If T is very low, the algorithm becomes a local search.
            # If we are stuck (steps_since_best > limit) OR T is very low, reheat.
            # Reheating: Reset T to a high value.
            # Additionally, we can "kick" the current position to a random point 
            # to escape the local basin entirely. This is the "Multi-Start" aspect.
            
            should_reheat = (T < T_min) or (steps_since_best > stagnation_limit)
            
            if should_reheat:
                T = T_init
                steps_since_best = 0
                
                # Optional: Jump to a new random location to explore other basins
                # This makes it a true multi-start SA.
                # With budget 500, we can afford a few restarts.
                # If we just reheat T, we stay in the same basin but explore more broadly.
                # If we jump, we might find a better global basin.
                # Let's jump with a probability, or always jump if stuck for a long time.
                
                # Strategy: If stuck for a very long time (e.g., 2*stagnation_limit), jump.
                # Otherwise, just reheat T.
                if steps_since_best > 2 * stagnation_limit:
                    current_x = rng.uniform(-5, 5, dim)
                    current_f = func(current_x)
                    calls_made += 1
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                        steps_since_best = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
