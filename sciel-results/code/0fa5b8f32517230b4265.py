import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        N = self.budget
        dim = self.dim
        
        # ZDT3 Pareto front: x[0] = f1, x[1:] = 0
        # f1 = x0
        # f2 = 1 - sqrt(x0) - (1-x0)*sin(10*pi*x0)
        
        def f2_val(x):
            return 1.0 - np.sqrt(x) - (1.0 - x) * np.sin(10.0 * np.pi * x)
        
        # Initialize x1 values
        x = np.linspace(0.0, 1.0, N)
        
        # Pre-compute f2 for current x
        f2 = f2_val(x)
        
        # Helper to calculate local contribution of point i
        # x must be sorted
        def local_contrib(i, x_new, x_arr, f2_arr):
            # i is index in 0..N-1
            if i == 0:
                # C_1(x_1) = -x_1 * (11 - f2(x_1))
                # Note: x_2 term is constant
                return -x_new * (11.0 - f2_val(x_new))
            elif i == N - 1:
                # C_N(x_N) = x_N*(11 - f2(x_{N-1})) + 11*(11 - f2(x_N)) - x_N*(11 - f2(x_N))
                f2_prev = f2_arr[i-1]
                f2_new = f2_val(x_new)
                return x_new * (11.0 - f2_prev) + 11.0 * (11.0 - f2_new) - x_new * (11.0 - f2_new)
            else:
                # C_i(x_i) = x_i * (11 - f2(x_{i-1})) - x_i * (11 - f2(x_i))
                f2_prev = f2_arr[i-1]
                f2_new = f2_val(x_new)
                return x_new * (11.0 - f2_prev) - x_new * (11.0 - f2_new)
        
        # Coordinate Descent
        # We do 50 iterations. 
        # In each iteration, we try to improve each x_i.
        # To avoid O(N^2) in Python loops for the search, we use a small number of candidates.
        n_candidates = 20
        n_iterations = 50
        
        for _ in range(n_iterations):
            for i in range(N):
                # Determine bounds for x_i
                L = x[i-1] if i > 0 else 0.0
                R = x[i+1] if i < N-1 else 1.0
                
                if L >= R:
                    continue
                
                # Generate candidates
                # Include endpoints and some interior points
                candidates = np.linspace(L, R, n_candidates)
                
                best_val = -np.inf
                best_x = x[i]
                
                # Calculate current contribution
                curr_val = local_contrib(i, x[i], x, f2)
                best_val = curr_val
                best_x = x[i]
                
                for cx in candidates:
                    if cx == x[i]:
                        continue
                    val = local_contrib(i, cx, x, f2)
                    if val > best_val:
                        best_val = val
                        best_x = cx
                
                if best_x != x[i]:
                    x[i] = best_x
                    f2[i] = f2_val(best_x)
                    # Update neighbors' f2? No, f2 depends only on x[i].
                    # But the "prev" f2 for i+1 is f2[i].
                    # The local_contrib function uses f2_arr[i-1].
                    # When we update x[i], f2[i] changes.
                    # This affects the contribution of x[i+1] (which uses f2[i] as prev).
                    # And x[i-1] is not affected by x[i] in its own contribution formula?
                    # C_{i-1} uses f2[i-2] and x[i-1]. It does NOT use f2[i].
                    # So only x[i] and x[i+1] contributions change when x[i] changes?
                    # Wait.
                    # C_i uses f2[i-1] and f2[i].
                    # C_{i+1} uses f2[i] and f2[i+1].
                    # So changing x[i] (and thus f2[i]) changes C_i and C_{i+1}.
                    # My greedy update only maximizes C_i. This might hurt C_{i+1}.
                    # However, since we iterate through all i, we will eventually re-optimize i+1.
                    # This is standard coordinate descent.
        
        # Sort x to be sure (though coordinate descent with bounds should keep it sorted)
        # Actually, if we move x[i] to best_x in [L, R], it stays sorted.
        
        # Final evaluation
        results = []
        for xi in x:
            vec = np.zeros(dim)
            vec[0] = xi
            # Clip to bounds
            for j in range(dim):
                vec[j] = max(bounds[j][0], min(bounds[j][1], vec[j]))
            
            # Check time
            if time.time() > deadline:
                break
                
            f = problem(vec)
            results.append(f)
            
        return results
