"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Pattern Search based optimizer.
    
    Strategy:
    1. Initialize at origin (common heuristic for BBOB, though shifts exist).
    2. Use a fixed set of search directions: 
       - 2 * dim orthogonal axes (+/- e_i)
       - 2 * dim diagonal directions (+/- 1/sqrt(d) * ones) 
       This creates a "star" pattern.
    3. For each direction, perform a binary search on the step size to find 
       a point that improves the current best. If found, update current best 
       and expand the step size for that direction (or globally). 
       If no improvement in a full cycle, shrink the step size.
    4. This deterministic local refinement is robust to ill-conditioning 
       because it doesn't rely on gradient estimates which can be noisy 
       or misleading in Schaffers F7. It guarantees progress if a direction 
       is good enough.
    5. To handle multiple local minima (Schaffers F7 has many), we use 
       random restarts. We allocate a portion of the budget to the first 
       run, then restart from random points if stagnation is detected 
       (no improvement for X function calls) or if step size becomes 
       too small.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 300s to be safe.
        deadline = start_time + 300.0
        
        best_x_global = None
        best_f_global = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def make_call(x):
            nonlocal call_count, best_f_global, best_x_global
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = np.array(x)
            return fx

        # Pre-compute search directions
        # 1. Orthogonal axes
        dirs = []
        for i in range(self.dim):
            v = np.zeros(self.dim)
            v[i] = 1.0
            dirs.append(v)
            dirs.append(-v)
        
        # 2. Diagonal directions (all + and all -)
        v_diag = np.ones(self.dim) / np.sqrt(self.dim)
        dirs.append(v_diag)
        dirs.append(-v_diag)
        
        # 3. Additional random directions to break symmetry if needed?
        # Let's stick to the structured ones first.
        
        num_dirs = len(dirs)
        
        # Pattern Search Parameters
        # Initial step size. Box is [-5, 5].
        # A step of 1.0 is reasonable to start.
        # We will adapt this.
        delta = 1.0
        delta_max = 10.0
        delta_min = 1e-6
        expansion_factor = 2.0
        contraction_factor = 0.5
        
        # Binary search parameters for step size within a direction
        # We want to find the largest step that improves? Or just any step?
        # Standard pattern search: try delta. If success, expand. If fail, shrink.
        # But "any step" via binary search is more robust for AOCC?
        # Actually, standard adaptive step size is better for convergence.
        # Let's use the standard approach: 
        # For each direction d:
        #   Try x + delta * d. If better, accept, set delta = delta * expansion.
        #   Try x - delta * d. If better, accept, set delta = delta * expansion.
        #   If neither works, after all directions, set delta = delta * contraction.
        
        # However, Schaffers F7 is ill-conditioned. Large steps might jump over minima.
        # Small steps might get stuck.
        # Let's add a "restart" mechanism.
        
        # Phase 1: Initial exploration
        # Start at 0
        current_x = np.zeros(self.dim)
        current_f = make_call(current_x)
        if current_f is None:
            return best_x_global, best_f_global

        # Restart logic
        # If we don't improve for `stagnation_limit` calls, restart.
        # With 500 calls, we can afford a few restarts.
        # Let's say if no improvement in 50 calls, restart.
        stagnation_limit = 50
        last_improve_call = 0
        
        # Main loop
        while call_count < self.budget:
            if time.time() > deadline:
                break
            
            # Check for stagnation
            if call_count - last_improve_call > stagnation_limit:
                # Restart from a random point
                # Choose a random point in [-4, 4] to stay within likely minimizer region
                current_x = rng.uniform(-4, 4, self.dim)
                current_f = make_call(current_x)
                if current_f is None:
                    break
                if current_f < best_f_global:
                    last_improve_call = call_count
                # Reset step size for new search
                delta = 1.0
                # Maybe keep the global best? Yes, best_x_global is tracked.
                # But current_x is now the new starting point for local search.
                # We should probably compare against best_f_global to see if we improved.
                # The loop continues.
            
            # Pattern Search Step
            # Try each direction
            improved_in_cycle = False
            for d in dirs:
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Try positive direction
                new_x = current_x + delta * d
                new_x = np.clip(new_x, -5, 5)
                new_f = make_call(new_x)
                if new_f is None:
                    break
                
                if new_f < current_f:
                    current_x = new_x
                    current_f = new_f
                    improved_in_cycle = True
                    last_improve_call = call_count
                    # Expand step size for next direction? 
                    # Standard PS often expands delta for the *next* iteration of the cycle.
                    # Or just for this direction? 
                    # Let's expand delta globally for the next direction to explore further.
                    delta = min(delta * expansion_factor, delta_max)
                    continue # Move to next direction with expanded delta? 
                    # Or should we stick to the same delta for this cycle?
                    # Standard PS: delta is updated after a direction succeeds/fails.
                    # If success, delta *= expansion.
                    # Let's do that.
                
                # Try negative direction
                new_x = current_x - delta * d
                new_x = np.clip(new_x, -5, 5)
                new_f = make_call(new_x)
                if new_f is None:
                    break
                
                if new_f < current_f:
                    current_x = new_x
                    current_f = new_f
                    improved_in_cycle = True
                    last_improve_call = call_count
                    delta = min(delta * expansion_factor, delta_max)
                    continue

            # If no direction improved, shrink step size
            if not improved_in_cycle:
                delta = max(delta * contraction_factor, delta_min)
                
                # If step size is too small, it's likely stuck in a local minimum
                # or at the optimum.
                # If delta < delta_min, we force a restart to escape.
                if delta <= delta_min * 1.1:
                    # Force restart by setting stagnation check to trigger
                    last_improve_call = 0 
                    # Or just restart immediately?
                    # Let's let the stagnation check handle it, but 
                    # setting last_improve_call = 0 ensures next check triggers restart.
                    # Actually, if we are stuck, we should restart sooner.
                    # Let's just restart if delta is very small and we haven't improved recently.
                    if call_count - last_improve_call > 10:
                         current_x = rng.uniform(-4, 4, self.dim)
                         current_f = make_call(current_x)
                         if current_f is None:
                             break
                         if current_f < best_f_global:
                             last_improve_call = call_count
                         delta = 1.0

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
