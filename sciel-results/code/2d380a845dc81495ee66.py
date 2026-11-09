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

    def __call__(self, func):
        # Define bases for 5 dimensions (first 5 primes)
        bases = [2, 3, 5, 7, 11]
        
        # Phase 1: Exploration with Halton sequence
        # Use 100 points for exploration to leave budget for local search
        n_explore = min(100, self.budget)
        
        # Generate Halton points in [0, 1]^dim
        halton_points = self._halton_sequence(n_explore, self.dim, bases)
        
        # Map to [-5, 5]^dim
        halton_points = halton_points * 10.0 - 5.0
        
        best_x = None
        best_f = float('inf')
        
        for i in range(n_explore):
            x = halton_points[i]
            fx = self._count_call(func, x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Exploitation with Nelder-Mead
        if self.calls < self.budget:
            # Initialize Nelder-Mead simplex
            # Start with best_x and perturbations
            delta = 0.5
            simplex = [best_x.copy()]
            for i in range(self.dim):
                v = best_x.copy()
                v[i] += delta
                simplex.append(v)
            
            # Evaluate initial simplex
            fvals = []
            for i in range(len(simplex)):
                if self.calls >= self.budget:
                    break
                fx = self._count_call(func, simplex[i])
                fvals.append(fx)
            
            # Nelder-Mead Iteration
            # Parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            while self.calls < self.budget:
                # Sort simplex by function value
                # We need to keep track of which point is which
                # Since we are modifying the list, we sort indices
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                
                # Check convergence (optional, but we stop by budget)
                # If the range of fvals is very small, we might stop early
                # but for AOCC, we want to use all budget to get the best possible value
                # unless we are stuck.
                
                # 1. Reflection
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Centroid of best n points
                x_c = np.mean(simplex[:-1], axis=0)
                
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
                    # Outside contraction if f_r < f_worst, else inside
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
                                # If budget ran out during shrink, we just break
                                # We might have incomplete fvals, but we break the loop
                                break
                        
                        if self.calls >= self.budget:
                            break

                # Update best
                current_best_f = min(fvals)
                if current_best_f < best_f:
                    best_f = current_best_f
                    best_idx = np.argmin(fvals)
                    best_x = simplex[best_idx].copy()

        return best_x, best_f
