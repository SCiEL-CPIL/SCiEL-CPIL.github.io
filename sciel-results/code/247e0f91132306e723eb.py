"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead with adaptive global restarts for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, leave 10s
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        def time_ok():
            return time.time() < deadline and calls_made < budget
        
        def nelder_mead(func, x0, budget_left, calls_counter, best_state):
            """
            Run Nelder-Mead on a local copy of x0.
            Returns (best_x_local, best_f_local, calls_used)
            """
            nonlocal calls_made
            dim_local = self.dim
            
            # Initialize simplex
            simplex = np.zeros((dim_local + 1, dim_local))
            simplex[0] = x0.copy()
            for i in range(1, dim_local + 1):
                simplex[i] = x0.copy()
                simplex[i, i-1] += 0.1  # Small perturbation
            
            # Evaluate initial simplex
            fvals = np.zeros(dim_local + 1)
            for i in range(dim_local + 1):
                if not time_ok():
                    break
                x = np.clip(simplex[i], -5, 5)
                f = func(x)
                calls_made += 1
                fvals[i] = f
                
                if f < best_state[1]:
                    best_state[0] = x.copy()
                    best_state[1] = f
            
            if calls_made >= budget or time.time() >= deadline:
                return best_state[0], best_state[1], 0
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            max_iter = 100  # Limit iterations per restart to avoid getting stuck
            iter_count = 0
            
            while iter_count < max_iter and time_ok():
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1
                if f_ref < best_state[1]:
                    best_state[0] = x_ref.copy()
                    best_state[1] = f_ref
                
                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1
                    if f_exp < best_state[1]:
                        best_state[0] = x_exp.copy()
                        best_state[1] = f_exp
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        calls_made += 1
                        if f_con < best_state[1]:
                            best_state[0] = x_con.copy()
                            best_state[1] = f_con
                        
                        if f_con <= f_ref:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim_local + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                x_shrink = np.clip(simplex[i], -5, 5)
                                f_shrink = func(x_shrink)
                                calls_made += 1
                                if f_shrink < best_state[1]:
                                    best_state[0] = x_shrink.copy()
                                    best_state[1] = f_shrink
                                simplex[i] = x_shrink
                                fvals[i] = f_shrink
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        calls_made += 1
                        if f_con < best_state[1]:
                            best_state[0] = x_con.copy()
                            best_state[1] = f_con
                        
                        if f_con < fvals[-1]:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim_local + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                x_shrink = np.clip(simplex[i], -5, 5)
                                f_shrink = func(x_shrink)
                                calls_made += 1
                                if f_shrink < best_state[1]:
                                    best_state[0] = x_shrink.copy()
                                    best_state[1] = f_shrink
                                simplex[i] = x_shrink
                                fvals[i] = f_shrink
                
                iter_count += 1
                if not time_ok():
                    break
                    
            return best_state[0], best_state[1], 0
        
        # Main loop: Multi-start strategy
        # We will alternate between global exploration (random restarts) and local exploitation (Nelder-Mead)
        
        # Initial phase: Coarse grid or random points to find basins
        # Given 500 budget and dim 5, we can afford ~50 restarts with ~10 calls each for NM setup + some steps
        # Or fewer restarts with more steps.
        
        # Strategy:
        # 1. Pick a random point.
        # 2. Run Nelder-Mead for a fixed number of calls (e.g., 20-30).
        # 3. If improvement is slow or simplex is stuck, restart from a new random point.
        # 4. Keep the global best.
        
        # Let's structure it as:
        # While budget remains:
        #   If we have a current best, try to refine it with NM.
        #   If NM stagnates (no improvement in last K calls), pick a new random start.
        
        # Simpler robust approach:
        # Divide budget into chunks.
        # Each chunk:
        #   1. Random point x0.
        #   2. Run NM starting from x0 for chunk_size calls.
        #   3. Update global best.
        
        # How many chunks?
        # If chunk_size is 10, we have 50 chunks.
        # If chunk_size is 20, we have 25 chunks.
        
        # Let's use a dynamic approach:
        # Start with a random point.
        # Run NM.
        # Track calls since last improvement.
        # If calls_since_improvement > threshold (e.g., 10), restart.
        
        calls_since_improvement = 0
        last_best_f = float('inf')
        
        while time_ok():
            # Choose starting point
            if calls_made == 0:
                # First start at origin or random?
                # Origin is often good for centered distributions, but shift is hidden.
                # Let's start with a few random points to spread out.
                x0 = rng.uniform(-4, 4, size=dim)
            else:
                # If we are restarting, pick a new random point
                # Sometimes perturb the current best? No, random is better for global search in step functions.
                x0 = rng.uniform(-4, 4, size=dim)
                
                # Occasionally, try a point near the current best to refine?
                # For now, pure random restarts are robust.
                
            # Run Nelder-Mead for a limited number of calls
            # We will run NM until it finishes its internal max_iter or we run out of budget.
            # But we want to be able to interrupt if it's not improving.
            
            # Modify NM to return early if no improvement for a while?
            # For simplicity, let's run a fixed number of NM steps per restart.
            # Let's say 15 steps of NM (each step ~1-2 calls). So ~30 calls per restart.
            # With 500 budget, that's ~16 restarts.
            
            # Let's implement a "limited" NM that stops after N calls.
            # We can pass a max_calls parameter to the inner loop.
            
            # Refactor: Create a helper that runs NM for exactly K calls or until convergence.
            
            def run_limited_nm(func, x0, max_calls, calls_counter_ref, best_state):
                nonlocal calls_made
                dim_local = self.dim
                
                simplex = np.zeros((dim_local + 1, dim_local))
                simplex[0] = x0.copy()
                for i in range(1, dim_local + 1):
                    simplex[i] = x0.copy()
                    simplex[i, i-1] += 0.1
                
                fvals = np.zeros(dim_local + 1)
                local_calls = 0
                
                for i in range(dim_local + 1):
                    if local_calls >= max_calls or not time_ok():
                        return
                    x = np.clip(simplex[i], -5, 5)
                    f = func(x)
                    calls_made += 1
                    local_calls += 1
                    fvals[i] = f
                    
                    if f < best_state[1]:
                        best_state[0] = x.copy()
                        best_state[1] = f
                
                if local_calls >= max_calls or not time_ok():
                    return
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                while local_calls < max_calls and time_ok():
                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    if local_calls + 1 > max_calls:
                        break
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = func(x_ref)
                    calls_made += 1
                    local_calls += 1
                    if f_ref < best_state[1]:
                        best_state[0] = x_ref.copy()
                        best_state[1] = f_ref
                    
                    if f_ref < fvals[0]:
                        # Expansion
                        if local_calls + 1 > max_calls:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                            break
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = func(x_exp)
                        calls_made += 1
                        local_calls += 1
                        if f_exp < best_state[1]:
                            best_state[0] = x_exp.copy()
                            best_state[1] = f_exp
                        
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    elif f_ref < fvals[-2]:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                    else:
                        # Contraction
                        if f_ref < fvals[-1]:
                            if local_calls + 1 > max_calls:
                                break
                            x_con = centroid + rho * (x_ref - centroid)
                            x_con = np.clip(x_con, -5, 5)
                            f_con = func(x_con)
                            calls_made += 1
                            local_calls += 1
                            if f_con < best_state[1]:
                                best_state[0] = x_con.copy()
                                best_state[1] = f_con
                            
                            if f_con <= f_ref:
                                simplex[-1] = x_con
                                fvals[-1] = f_con
                            else:
                                # Shrink
                                for i in range(1, dim_local + 1):
                                    if local_calls + 1 > max_calls:
                                        break
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    x_shrink = np.clip(simplex[i], -5, 5)
                                    f_shrink = func(x_shrink)
                                    calls_made += 1
                                    local_calls += 1
                                    if f_shrink < best_state[1]:
                                        best_state[0] = x_shrink.copy()
                                        best_state[1] = f_shrink
                                    simplex[i] = x_shrink
                                    fvals[i] = f_shrink
                        else:
                            if local_calls + 1 > max_calls:
                                break
                            x_con = centroid - rho * (centroid - simplex[-1])
                            x_con = np.clip(x_con, -5, 5)
                            f_con = func(x_con)
                            calls_made += 1
                            local_calls += 1
                            if f_con < best_state[1]:
                                best_state[0] = x_con.copy()
                                best_state[1] = f_con
                            
                            if f_con < fvals[-1]:
                                simplex[-1] = x_con
                                fvals[-1] = f_con
                            else:
                                # Shrink
                                for i in range(1, dim_local + 1):
                                    if local_calls + 1 > max_calls:
                                        break
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    x_shrink = np.clip(simplex[i], -5, 5)
                                    f_shrink = func(x_shrink)
                                    calls_made += 1
                                    local_calls += 1
                                    if f_shrink < best_state[1]:
                                        best_state[0] = x_shrink.copy()
                                        best_state[1] = f_shrink
                                    simplex[i] = x_shrink
                                    fvals[i] = f_shrink

            # Execute limited NM
            # How many calls per restart?
            # Let's start with 15.
            run_limited_nm(func, x0, 15, None, [best_x, best_f])
            
            # Check if we should continue
            if not time_ok():
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
