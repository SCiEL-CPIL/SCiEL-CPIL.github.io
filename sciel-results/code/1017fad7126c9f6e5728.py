"""Parallel coordinate descent with golden-section search for f22 (Gallagher 21 peaks)."""
import numpy as np
import time
import math

class Optimizer:
    """Multi-start coordinate descent with golden-section search for 21-peak f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def _call(self, func, x):
        """Count a function call."""
        self.calls += 1
        return float(func(x))

    def _golden_section_1d(self, func, x, dim, lo, hi, budget, rng):
        """
        Golden-section search along one dimension.
        x is the full vector, we optimize along axis 'dim'.
        lo, hi are the bounds for that dimension.
        Returns (new_value, best_f, calls_used)
        """
        if self.calls >= self.budget:
            return x[dim], self.best_f, 0
            
        # Initial points
        inv_phi = (1 - math.sqrt(5)) / 2  # ~ -0.618
        phi = (1 + math.sqrt(5)) / 2  # ~ 1.618
        
        x1 = lo + inv_phi * (hi - lo)
        x2 = hi + inv_phi * (hi - lo)
        
        x[dim] = x1
        f1 = self._call(func, x)
        x[dim] = x2
        f2 = self._call(func, x)
        
        # Restore x to original best if possible? No, we just track local best.
        # But we need to be careful: x is a reference. We should probably work on a copy or restore.
        # Let's work on a copy to be safe, or just restore at the end.
        # Actually, the caller passes x which is the current best. We should probably
        # only update the caller's x if we improve.
        
        # Let's use a local copy for the search to avoid side effects until we know it's better
        # But _call uses func(x). If x is modified, the next call uses the modified x.
        # This is standard for coordinate descent: we move along the axis.
        
        # However, we must ensure we don't exceed budget.
        # We already used 2 calls.
        
        best_local_f = min(f1, f2)
        best_local_val = x1 if f1 < f2 else x2
        
        # Golden section loop
        # We can do a fixed number of iterations or until budget is low.
        # Let's do a few iterations.
        max_iters = 10  # Golden section converges fast. 10 iters gives ~1.618^-10 ~ 0.008 reduction
        for _ in range(max_iters):
            if self.calls >= self.budget:
                break
            if f1 < f2:
                hi = x2
                x2 = x1
                f2 = f1
                x1 = lo + inv_phi * (hi - lo)
                x[dim] = x1
                f1 = self._call(func, x)
            else:
                lo = x1
                x1 = x2
                f1 = f2
                x2 = hi + inv_phi * (hi - lo)
                x[dim] = x2
                f2 = self._call(func, x)
            
            if f1 < best_local_f:
                best_local_f = f1
                best_local_val = x1
            if f2 < best_local_f:
                best_local_f = f2
                best_local_val = x2
            
            # Convergence check
            if abs(hi - lo) < 1e-6:
                break
        
        # Set x[dim] to the best found value
        x[dim] = best_local_val
        # Evaluate one more time to get the exact f for this state?
        # We have f1 and f2 which correspond to x1 and x2.
        # best_local_val is either x1 or x2, so best_local_f is correct.
        
        return best_local_val, best_local_f, 2 + max_iters * 1 # Approximate

    def _coordinate_descent(self, func, x0, budget, rng):
        """
        Perform coordinate descent on a copy of x0.
        Returns (best_x, best_f)
        """
        x = x0.copy()
        # Initial evaluation
        f = self._call(func, x)
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            
        dim = self.dim
        # We will iterate through dimensions multiple times.
        # Each sweep costs dim * (2 + iters) calls approx.
        # Let's estimate cost per sweep: dim * 12 calls (2 init + 10 iter)
        # With budget 500, dim 5, that's ~60 calls per sweep. ~8 sweeps.
        
        # Shuffle order to avoid bias
        order = list(range(dim))
        rng.shuffle(order)
        
        while self.calls < self.budget:
            improved = False
            for d in order:
                if self.calls >= self.budget:
                    break
                
                # Determine bounds for this dimension
                # We can use the current x[d] +/- some delta, or the full box.
                # Full box is safer for escaping local minima but slower.
                # Let's try a local window first, then expand if stuck.
                # For now, let's use a window around current x[d] of size 1.0
                # If no improvement, expand to full box.
                
                center = x[d]
                # Try a local search first
                lo = max(-5.0, center - 1.0)
                hi = min(5.0, center + 1.0)
                
                # Save current value
                old_val = x[d]
                
                # Perform golden section search
                # We need to be careful with the budget inside _golden_section_1d
                # It will stop if budget is exceeded.
                
                new_val, new_f, _ = self._golden_section_1d(func, x, d, lo, hi, self.budget, rng)
                
                if new_f < self.best_f:
                    self.best_f = new_f
                    self.best_x = x.copy()
                    improved = True
                elif new_f < f:
                    # Improved locally, but not globally best yet
                    f = new_f
                    improved = True
                else:
                    # No improvement, revert?
                    # Golden section search ends at the best point found.
                    # If it's worse than 'f', we should revert to old_val?
                    # Actually, 'f' was the value before this dimension sweep.
                    # If new_f >= f, it means we didn't improve this dimension.
                    # We should revert x[d] to old_val to not drift.
                    x[d] = old_val
                    # Note: self.best_f and self.best_x are not updated if no improvement.
                    # But x is modified. We revert it.
                    
            # If no improvement in a full sweep, try expanding the search window
            # or break to save budget for other starts.
            # For simplicity, if no improvement, we can break or try a larger window.
            # Let's try one expansion: full box for each dim.
            if not improved:
                # Try full box search for each dim
                for d in order:
                    if self.calls >= self.budget:
                        break
                    old_val = x[d]
                    new_val, new_f, _ = self._golden_section_1d(func, x, d, -5.0, 5.0, self.budget, rng)
                    if new_f < self.best_f:
                        self.best_f = new_f
                        self.best_x = x.copy()
                        improved = True
                    elif new_f < f:
                        f = new_f
                        improved = True
                    else:
                        x[d] = old_val
                
                if not improved:
                    break
            
            # Shuffle order for next sweep
            rng.shuffle(order)

        return self.best_x, self.best_f

    def __call__(self, func):
        deadline = time.time() + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Multi-start
        # Use a portion of the budget for random starts and coordinate descent.
        # Let's use 10 starts.
        # Each start gets budget // 10 calls for coordinate descent.
        
        num_starts = 10
        per_start_budget = self.budget // num_starts
        
        # We need to be careful: _coordinate_descent uses self.calls which is global.
        # So we just call it and it will stop when self.calls hits self.budget.
        
        # However, we want to ensure we don't spend all budget on one start if it's fast.
        # The _coordinate_descent loop checks self.calls < self.budget.
        # So if it finishes early (converged), it returns.
        
        # Let's do a random scan first to find good starting points.
        # Or just start from random points.
        
        # Better: Start from random points, refine with CD.
        # If CD converges fast, we can start another one.
        
        # Let's try 5 starts.
        num_starts = 5
        
        for i in range(num_starts):
            if time.time() > deadline:
                break
            if self.calls >= self.budget:
                break
            
            # Random start
            x0 = rng.uniform(-4, 4, dim)
            
            # Refine with coordinate descent
            # We pass the global budget, so it will use remaining budget.
            # But we might want to limit it per start to ensure diversity.
            # Let's limit per start to per_start_budget.
            # We can do this by temporarily setting self.budget? No, that's messy.
            # Instead, we can pass a local budget limit to _coordinate_descent.
            
            # Let's modify _coordinate_descent to accept a max_calls parameter.
            # Or just let it run until global budget is exhausted or convergence.
            # Convergence is likely to happen before budget exhaustion for CD.
            
            self._coordinate_descent(func, x0, self.budget, rng)
            
            # If we still have budget and time, we can do more starts.
            # The loop above handles the count.
            
        # Final polish: use remaining budget on the best point found so far.
        if self.calls < self.budget and time.time() < deadline:
            if self.best_x is not None:
                self._coordinate_descent(func, self.best_x, self.budget, rng)
            else:
                # If no best found yet (unlikely), do a random start
                x0 = rng.uniform(-4, 4, dim)
                self._coordinate_descent(func, x0, self.budget, rng)

        self.best_x = self.best_x if self.best_x is not None else np.zeros(dim)
        self.best_f = self.best_f if not np.isinf(self.best_f) else 0.0
        return self.best_x, self.best_f
