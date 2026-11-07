"""Coordinate Descent with Golden-Section Search for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start Coordinate Descent optimizer with Golden-Section Search for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s, task limit is 300s. 
        # We aim to finish well before 300s to be safe.
        deadline = start_time + 280.0 

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(12345)

        # Phase 1: Coarse Random Search
        # We need to find at least one peak. f22 has 21 peaks.
        # Sampling 20-30% of budget ensures we hit several basins.
        n_random = max(20, budget // 4)
        candidates = []

        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        if not candidates:
            # Fallback if we somehow ran out of budget immediately (unlikely)
            return best_x, best_f

        # Sort by fitness to prioritize the best basins
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Coordinate Descent Refinement
        # Apply CD to the top K candidates.
        # CD is fast: each sweep costs O(dim * gss_iters).
        top_k = min(10, len(candidates))
        
        # Parameters for CD
        # gss_iters: number of golden-section iterations per coordinate per sweep.
        # 10 iterations reduce interval size by factor ~ (0.618)^10 ~ 0.008, 
        # which is precise enough for 1D refinement.
        gss_iters = 10 
        # Initial search window for each coordinate
        init_window = 2.5 
        # Number of sweeps per candidate
        # We have budget - n_random left. 
        # Cost per sweep ~ dim * gss_iters * 1 (evals) + dim (overhead).
        # Let's estimate cost per candidate.
        # If we have 350 evals left, and dim=5, gss=10, cost/sweep ~ 50.
        # So ~7 sweeps per candidate.
        # We will dynamically allocate sweeps based on remaining budget.
        
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break

            x = candidates[i][1].copy()
            remaining = budget - calls
            if remaining < 20: # Need at least one sweep
                break

            # Determine how many sweeps we can afford
            # Rough cost estimate: dim * gss_iters * 1.5 (safety factor)
            cost_per_sweep = dim * gss_iters * 1.5
            max_sweeps = max(1, int(remaining / cost_per_sweep))
            
            # Cap sweeps to avoid over-optimizing one point if others are close
            # If we are on the best candidate, allow more sweeps.
            if i == 0:
                max_sweeps = min(max_sweeps, 15)
            else:
                max_sweeps = min(max_sweeps, 5)

            # Coordinate Descent Loop
            # We use a "window" that adapts. 
            # If we improve, we can keep the window or shrink it slightly.
            # If we don't improve, we might need to expand to find a better peak.
            
            current_f = eval_and_track(x)
            
            for sweep in range(max_sweeps):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Randomize coordinate order to break symmetry/dependencies
                order = rng.permutation(dim)
                
                sweep_improved = False
                
                for axis in order:
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Define search range for this coordinate
                    # Start with a reasonable window around current value
                    # Clip to box [-5, 5]
                    lo = max(-5.0, x[axis] - init_window)
                    hi = min(5.0, x[axis] + init_window)
                    
                    # If the range is too small, expand it slightly to allow hopping
                    if hi - lo < 0.5:
                        center = x[axis]
                        lo = max(-5.0, center - 1.0)
                        hi = min(5.0, center + 1.0)
                    
                    # Perform 1D Golden Section Search
                    x_opt, f_opt_1d = self._golden_section_search(
                        func, x, axis, lo, hi, gss_iters, eval_and_track
                    )
                    
                    # Update x if improved
                    if f_opt_1d < current_f:
                        x = x_opt
                        current_f = f_opt_1d
                        sweep_improved = True
                        # Shrink window for next sweep on this coordinate? 
                        # Or keep it? 
                        # Let's keep it stable for now, CD converges naturally.
                    else:
                        # No improvement in 1D. 
                        # This might indicate we are in a local min of the 1D slice,
                        # or the peak is outside the window.
                        # We don't change x, but we could widen window in next sweep.
                        pass
                
                # End of sweep
                # If no improvement in the whole sweep, we might be stuck in a local min
                # or the window is too small.
                # Strategy: If no improvement, slightly increase the window for next sweep
                # to allow "hopping" to a neighboring peak.
                if not sweep_improved:
                    init_window *= 1.2
                    # Cap the window to avoid going out of bounds or being too wide
                    init_window = min(init_window, 4.0)
                else:
                    # If improved, we can tighten the window for faster convergence
                    init_window *= 0.8
                    init_window = max(init_window, 0.5)

        # Phase 3: Final Polish
        # If we have budget left, do a very fine-grained search around the global best.
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining >= 10:
                x = best_x.copy()
                # Fine window
                fine_window = 0.5
                # Fewer sweeps, but very precise
                for _ in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    order = rng.permutation(dim)
                    for axis in order:
                        if calls >= budget or time.time() > deadline:
                            break
                        lo = max(-5.0, x[axis] - fine_window)
                        hi = min(5.0, x[axis] + fine_window)
                        x_opt, f_opt_1d = self._golden_section_search(
                            func, x, axis, lo, hi, 15, eval_and_track # More iters for precision
                        )
                        if f_opt_1d < best_f:
                            x = x_opt
                            best_f = f_opt_1d
                            best_x = x.copy()
                # Ensure best_x is the final polished version if it improved
                # The eval_and_track updates best_x automatically if f_opt_1d < best_f
                # But we should make sure best_x is consistent with best_f
                # Actually, eval_and_track handles the global best tracking.

        return best_x, best_f

    def _golden_section_search(self, func, x, axis, lo, hi, max_iter, eval_func):
        """
        Perform 1D Golden Section Search on a specific axis.
        Returns (best_x, best_f).
        """
        # Golden ratio
        gr = (1.0 + math.sqrt(5.0)) / 2.0
        inv_gr = 1.0 / gr
        inv_gr2 = 1.0 / (gr * gr)
        
        # Initial points
        x1 = x.copy()
        x1[axis] = lo + inv_gr2 * (hi - lo)
        x2 = x.copy()
        x2[axis] = lo + inv_gr * (hi - lo)
        
        f1 = eval_func(x1)
        f2 = eval_func(x2)
        
        best_x = x1.copy()
        best_f = f1
        if f2 < best_f:
            best_x = x2.copy()
            best_f = f2
            
        for _ in range(max_iter):
            # Check budget/time inside the loop via eval_func's internal checks?
            # eval_func returns best_f if budget exceeded. 
            # But we need to stop the loop if budget is effectively exhausted.
            # We can check if f1 and f2 are both "stale" (equal to previous best_f and no change?)
            # Simpler: just rely on the fact that eval_func won't increment calls if over budget.
            # However, we should break if we detect no progress or budget exhaustion.
            
            # Heuristic: if f1 and f2 are identical to the last best_f and we've done some iters, 
            # it might be stuck. But GSS is robust.
            
            if f1 < f2:
                # Min is in [lo, x2]
                hi = x2[axis]
                x2 = x1.copy()
                f2 = f1
                x1 = x.copy()
                x1[axis] = lo + inv_gr2 * (hi - lo)
                f1 = eval_func(x1)
            else:
                # Min is in [x1, hi]
                lo = x1[axis]
                x1 = x2.copy()
                f1 = f2
                x2 = x.copy()
                x2[axis] = lo + inv_gr * (hi - lo)
                f2 = eval_func(x2)
                
            # Update local best
            if f1 < best_f:
                best_f = f1
                best_x = x1.copy()
            if f2 < best_f:
                best_f = f2
                best_x = x2.copy()
                
            # Optional: Early stop if interval is very small
            if hi - lo < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
