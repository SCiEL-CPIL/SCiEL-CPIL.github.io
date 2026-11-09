import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_f = np.inf
        history = [] # List of (f, x) for top points

        def record(x, f):
            nonlocal calls, best_x, best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            # Keep a diverse set of good points
            history.append((f, x.copy()))
            if len(history) > 20:
                # Keep top 10 by fitness
                history.sort(key=lambda item: item[0])
                history = history[:10]

        def eval_pt(x):
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # ---- Phase 1: Coarse Scan ----
        # Generate diverse points to find both basins of Lunacek
        candidates = []
        # Center
        candidates.append(np.zeros(n))
        # Axis-aligned offsets
        for i in range(n):
            for s in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0):
                p = np.zeros(n)
                p[i] = s
                candidates.append(p)
        # Diagonals
        for s in (-2.0, -1.0, 1.0, 2.0):
            candidates.append(np.full(n, s))
        # Random
        for _ in range(40):
            candidates.append(rng.uniform(-4.5, 4.5, n))
            
        # Evaluate all candidates
        for x in candidates:
            if calls >= budget or time.time() > deadline:
                break
            f = eval_pt(x)
            record(x, f)
            
        if best_x is None:
            best_x = np.zeros(n)
            best_f = eval_pt(best_x)
            record(best_x, best_f)
            
        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # ---- Phase 2: Coordinate Descent with Line Search and Restarts ----
        
        # Sort history to get best starting points
        history.sort(key=lambda item: item[0])
        start_points = [x for f, x in history[:5]]
        
        # Global parameters
        global_delta = 1.0
        min_delta = 1e-4
        max_steps_per_dim = 5 # Max times to double step in line search
        stagnation_limit = 15 # Cycles without improvement before restart/decay
        
        current_seed_idx = 0
        
        while calls < budget and time.time() < deadline:
            if current_seed_idx >= len(start_points):
                # Generate new random seed
                new_seed = rng.uniform(-4.5, 4.5, n)
                f_new = eval_pt(new_seed)
                record(new_seed, f_new)
                if f_new < best_f:
                    best_x = new_seed.copy()
                    best_f = f_new
                # Add to start points if it's good enough or just use it
                if len(start_points) < 10:
                    start_points.append(new_seed.copy())
                else:
                    start_points[-1] = new_seed.copy() # Replace worst
                current_seed_idx = len(start_points) - 1
                # Sort start points to keep best ones at front? 
                # Actually, let's just cycle through them.
                
            x = start_points[current_seed_idx].copy()
            fx = best_f # We assume we start from a known good point, but we need its f value
            # Wait, we don't have the f value for start_points directly in a dict.
            # Let's re-evaluate the start point to be safe, or store (f, x).
            # Better: store list of (f, x) in start_points.
            
            # Let's restructure start_points to be list of (f, x)
            if current_seed_idx == 0:
                 # Initialize start_points properly if we haven't already
                 pass
            
            # Actually, simpler:
            # Keep a list of active candidates (f, x)
            # We already have 'history'. Let's use the top ones from history.
            
            # Let's restart the loop logic to be cleaner.
            # We will pick a point from history, run CD, and update history.
            
            # Pick a point
            # Strategy: Pick the best point, but if CD fails, pick the second best, etc.
            # To avoid re-processing the same point, we can just always pick the current global best?
            # No, if we get stuck in a local min, we need to try another basin.
            
            # Let's use a "Multi-start" CD.
            # We have 'history' which contains top 10 points.
            # We will iterate through them.
            
            # To simplify, let's just run CD from the current global best.
            # If it stagnates, we will pick a random point from history or generate a new one.
            
            x = best_x.copy()
            fx = best_f
            
            # Reset local CD state
            delta = global_delta
            no_improve_cycles = 0
            
            while calls < budget and time.time() < deadline:
                improved = False
                # Cycle through dimensions
                for i in range(n):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Try moving in dim i
                    # 1. Try +delta
                    x_trial = x.copy()
                    x_trial[i] += delta
                    f_trial = eval_pt(x_trial)
                    record(x_trial, f_trial)
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        improved = True
                        # Line search: try to extend the step
                        step_mult = 1.0
                        for _ in range(max_steps_per_dim):
                            if calls >= budget or time.time() > deadline:
                                break
                            # Try doubling the step in the same direction
                            x_trial = x.copy()
                            x_trial[i] += delta * step_mult
                            f_trial = eval_pt(x_trial)
                            record(x_trial, f_trial)
                            if f_trial < fx:
                                x = x_trial
                                fx = f_trial
                                step_mult *= 2.0
                            else:
                                break
                    
                    # 2. Try -delta (if we didn't just move +, or if + didn't improve)
                    # To save calls, only try - if + didn't improve or if we want to be thorough.
                    # In CD, usually you check both or the one that seems promising.
                    # Let's check -delta if +delta didn't improve.
                    if not improved:
                        x_trial = x.copy()
                        x_trial[i] -= delta
                        f_trial = eval_pt(x_trial)
                        record(x_trial, f_trial)
                        
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True
                            # Line search: try to extend the step in - direction
                            step_mult = 1.0
                            for _ in range(max_steps_per_dim):
                                if calls >= budget or time.time() > deadline:
                                    break
                                x_trial = x.copy()
                                x_trial[i] -= delta * step_mult
                                f_trial = eval_pt(x_trial)
                                record(x_trial, f_trial)
                                if f_trial < fx:
                                    x = x_trial
                                    fx = f_trial
                                    step_mult *= 2.0
                                else:
                                    break

                if improved:
                    no_improve_cycles = 0
                    # Optionally increase delta slightly if we moved far? 
                    # No, keep it stable or decay slowly.
                    # If we used line search, we effectively took a big step. 
                    # Let's decay delta slightly to refine.
                    delta *= 0.95 
                else:
                    no_improve_cycles += 1
                    # Decay delta
                    delta *= 0.8
                    
                # Check for stagnation
                if no_improve_cycles > stagnation_limit:
                    break
                
                # If delta is too small, break to restart
                if delta < min_delta:
                    break
            
            # If we broke out of the inner loop due to stagnation or small delta
            # Update global delta for next restart?
            # Or just pick a new start point.
            
            # Pick next start point from history
            # We need to ensure we don't just pick the same best point over and over if it's stuck.
            # Let's sort history and pick the best one that is NOT the current best_x (if possible)
            # Or just cycle through the top 5.
            
            # For simplicity, let's just pick the global best again but with a fresh random perturbation?
            # No, the directive says "different design".
            # Let's pick the 2nd best point from history.
            
            if len(history) > 1:
                # Find the best point in history that is different from current best_x
                # Or just pick history[1]
                next_f, next_x = history[1]
                # Check if next_x is significantly different from best_x
                if np.linalg.norm(next_x - best_x) > 0.1:
                    best_x = next_x.copy()
                    best_f = next_f
                    # Reset global delta for new region
                    global_delta = 1.0 
                else:
                    # If they are close, just keep best_x
                    pass
            else:
                # Only one point, generate new random
                new_x = rng.uniform(-4.5, 4.5, n)
                new_f = eval_pt(new_x)
                record(new_x, new_f)
                if new_f < best_f:
                    best_x = new_x.copy()
                    best_f = new_f
                else:
                    # Keep best_x
                    pass
                global_delta = 1.0

        return best_x, best_f
