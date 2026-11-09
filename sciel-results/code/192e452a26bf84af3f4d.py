import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def _halton(self, index, base):
        """Generate the Halton sequence for a given index and base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i = i // base
        return result

    def _halton_sequence(self, count, dim, bases):
        """Generate a sequence of 'count' points in 'dim' dimensions."""
        points = np.zeros((count, dim))
        for i in range(1, count + 1):
            for j, base in enumerate(bases):
                points[i - 1, j] = self._halton(i, base)
        return points

    def _count_call(self, func, x):
        """Wrapper to count function calls."""
        self.calls += 1
        return func(x)

    def _nelder_mead(self, func, x0, f0, max_calls):
        """
        Perform Nelder-Mead optimization starting from x0 with value f0.
        Uses up to max_calls additional function evaluations.
        Returns (best_x, best_f).
        """
        dim = self.dim
        # Initialize simplex
        delta = 0.5
        simplex = [x0.copy()]
        for i in range(dim):
            v = x0.copy()
            v[i] += delta
            simplex.append(v)
        
        # Evaluate initial simplex
        fvals = [f0]
        for i in range(1, dim + 1):
            if self.calls >= self.budget:
                break
            fx = self._count_call(func, simplex[i])
            fvals.append(fx)
            
        # If we couldn't evaluate the full simplex, we can't run NM properly
        if len(fvals) < dim + 1:
            return x0, f0

        # Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        best_x = x0.copy()
        best_f = f0
        
        # Track iterations without improvement to detect stagnation
        stagnation_count = 0
        max_stagnation = 10
        
        while self.calls < self.budget and (self.budget - self.calls) > max_calls:
            # Sort simplex
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Update global best for this NM run
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count > max_stagnation:
                    break

            x_worst = simplex[-1]
            f_worst = fvals[-1]
            x_c = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_c + alpha * (x_c - x_worst)
            x_r = np.clip(x_r, -5, 5)
            
            if self.calls >= self.budget:
                break
            f_r = self._count_call(func, x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                if self.calls >= self.budget:
                    break
                f_e = self._count_call(func, x_e)
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Reflection accepted
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c_r = x_c + rho * (x_r - x_c)
                else:
                    x_c_r = x_c - rho * (x_worst - x_c)
                
                x_c_r = np.clip(x_c_r, -5, 5)
                if self.calls >= self.budget:
                    break
                f_c = self._count_call(func, x_c_r)
                
                if f_c < f_r:
                    simplex[-1] = x_c_r
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if self.calls >= self.budget:
                            break
                        if self.calls < self.budget:
                            fvals[i] = self._count_call(func, simplex[i])
                        else:
                            break
                    
                    if self.calls >= self.budget:
                        break

        return best_x, best_f

    def __call__(self, func):
        # Define bases for 5 dimensions (first 5 primes)
        bases = [2, 3, 5, 7, 11]
        
        # Phase 1: Exploration with Halton sequence
        # Use 100 points for exploration
        n_explore = min(100, self.budget)
        
        # Generate Halton points in [0, 1]^dim
        halton_points = self._halton_sequence(n_explore, self.dim, bases)
        
        # Map to [-5, 5]^dim
        halton_points = halton_points * 10.0 - 5.0
        
        results = []
        
        for i in range(n_explore):
            x = halton_points[i]
            fx = self._count_call(func, x)
            results.append((fx, x.copy()))
            
        # Sort by function value
        results.sort(key=lambda item: item[0])
        
        best_x = results[0][1]
        best_f = results[0][0]
        
        # Phase 2: Exploitation with Multi-start Nelder-Mead
        # Try local search from the top few candidates
        max_starts = 5
        remaining_budget = self.budget - self.calls
        
        if remaining_budget > 0:
            # Allocate budget to each start
            # We give more budget to the best start
            # Simple strategy: split remaining budget among top starts
            # Or just let them run until budget is gone, prioritizing the best first
            
            for i in range(max_starts):
                if self.calls >= self.budget:
                    break
                
                x_start = results[i][1]
                f_start = results[i][0]
                
                # If the start point is worse than the current global best by a lot, maybe skip?
                # No, it might be in a different basin.
                
                # Limit calls for this specific NM run to avoid wasting budget on a bad start
                # But if it's the first start, give it more.
                # Let's give 1/3 of remaining to first, 1/4 to second, etc.
                # Or simpler: just run until budget is exhausted or convergence.
                # The stagnation check in _nelder_mead will handle early exit.
                
                # We pass a large max_calls, but the internal check uses self.budget
                # and the stagnation check.
                
                local_best_x, local_best_f = self._nelder_mead(func, x_start, f_start, self.budget)
                
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

        return best_x, best_f
