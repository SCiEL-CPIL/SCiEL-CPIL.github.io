import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # Calibration phase
            # Test a few configurations for x_2..x_dim
            # We assume x_1 is the trade-off variable.
            # Let's test x_2.. = 0.5, 0.0, 1.0, and random.
            test_x1 = 0.5
            candidates = [
                np.full(self.dim - 1, 0.5),
                np.full(self.dim - 1, 0.0),
                np.full(self.dim - 1, 1.0),
                np.random.uniform(lo[1:], hi[1:])
            ]
            
            best_val = float('inf')
            best_tail = None
            
            for tail in candidates:
                x = np.concatenate(([test_x1], tail))
                # Ensure within bounds
                x = np.clip(x, lo, hi)
                f = problem(x)
                # We want to minimize f[1] (assuming f[0] is fixed by x1=0.5)
                # Actually, we want the point closest to the front, which means min f[1] for fixed f[0].
                if f[1] < best_val:
                    best_val = f[1]
                    best_tail = tail
            
            # Now use the best tail for the rest of the budget
            # We have used 4 evaluations.
            remaining = self.budget - 4
            if remaining <= 0:
                return best_tail # Just in case, though return value is ignored
            
            # Generate x1 values
            # Use uniform sampling for good coverage of the front
            x1_vals = np.random.uniform(lo[0], hi[0], remaining)
            
            # Construct X
            X = np.zeros((remaining, self.dim))
            X[:, 0] = x1_vals
            X[:, 1:] = best_tail
            
            # Evaluate all
            # We can't vectorize problem() if it's a black box function taking a single vector.
            # The prompt says "problem(x) ... returns 2 objective values".
            # So we must loop.
            results = []
            for x in X:
                f = problem(x)
                results.append(f)
            
            return np.array(results)
