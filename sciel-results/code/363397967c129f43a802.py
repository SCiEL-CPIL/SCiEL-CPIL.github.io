import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Optimizer for BBOB f8 (Rosenbrock) exploiting the valley structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, use 290s to be safe
        self.deadline = self.start_time + 290.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            # Ensure x is a valid float array
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                 raise ValueError("Dimension mismatch")
            
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # 1. Generate diverse starting points
        # The global minimum of the shifted Rosenbrock is likely near the center of the box
        # or shifted by a hidden vector. We sample broadly.
        rng = np.random.default_rng(42)
        
        starts = []
        
        # Center of the box
        starts.append(np.zeros(self.dim))
        
        # The standard Rosenbrock min is at (1,1,1,1,1). 
        # If the shift is small, this is a good start.
        starts.append(np.ones(self.dim))
        
        # Random points in the box
        for _ in range(20):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        # Points near the likely minimum region (center-ish)
        for _ in range(10):
            starts.append(rng.uniform(-2, 2, self.dim))

        # 2. Perform Valley-Aware Descent from each start
        # The Rosenbrock valley satisfies x_i ~ x_{i-1}^2.
        # We iterate: for each i, try setting x_i to x_{i-1}^2.
        
        best_candidate = None
        best_candidate_f = np.inf

        for x0 in starts:
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
                
            x = np.clip(x0, -5, 5)
            try:
                f = wrapped_func(x)
            except StopIteration:
                break
            except Exception:
                continue

            if f < best_candidate_f:
                best_candidate_f = f
                best_candidate = x.copy()

            # Valley descent iterations
            # We do multiple passes over the dimensions.
            # In each pass, we update x_i based on x_{i-1}.
            # We accept the move if it improves the function value.
            
            improved = True
            max_iter = 100  # Limit iterations per start to avoid getting stuck
            iter_count = 0
            
            while improved and iter_count < max_iter:
                improved = False
                iter_count += 1
                if self.calls_made >= self.budget or time.time() > self.deadline:
                    break

                for i in range(1, self.dim):
                    if self.calls_made >= self.budget or time.time() > self.deadline:
                        break
                    
                    # The valley condition: x_i = x_{i-1}^2
                    # We clip to bounds
                    target_val = x[i-1] ** 2
                    target_val = np.clip(target_val, -5, 5)
                    
                    # Only try if the change is significant
                    if abs(target_val - x[i]) < 1e-6:
                        continue

                    trial = x.copy()
                    trial[i] = target_val
                    trial = np.clip(trial, -5, 5)
                    
                    try:
                        f_trial = wrapped_func(trial)
                    except StopIteration:
                        break
                    except Exception:
                        continue

                    if f_trial < f:
                        x = trial
                        f = f_trial
                        improved = True
                        if f < best_candidate_f:
                            best_candidate_f = f
                            best_candidate = x.copy()
                    
                if self.calls_made >= self.budget or time.time() > self.deadline:
                    break

        if best_candidate is None:
            # Fallback if something went wrong
            best_candidate = np.zeros(self.dim)
            try:
                self.best_f = wrapped_func(best_candidate)
            except:
                pass

        # 3. Polish the best candidate with a local optimizer
        # Nelder-Mead is robust and derivative-free, good for final steps.
        # L-BFGS-B is faster if we are close. Let's try L-BFGS-B first, 
        # and if it fails or doesn't improve much, try Nelder-Mead.
        
        if self.calls_made < self.budget and time.time() < self.deadline:
             try:
                 res = scipy.optimize.minimize(
                     wrapped_func,
                     best_candidate,
                     method='L-BFGS-B',
                     bounds=[(-5, 5)] * self.dim,
                     options={
                         'maxiter': 200,
                         'ftol': 1e-12,
                         'gtol': 1e-10
                     }
                 )
                 if res.fun < self.best_f:
                     self.best_f = res.fun
                     self.best_x = np.clip(res.x, -5, 5)
             except StopIteration:
                 pass
             except Exception:
                 pass

        # If L-BFGS-B didn't use all budget or time, try Nelder-Mead for robustness
        if self.calls_made < self.budget and time.time() < self.deadline:
            # Only if we have significant budget left
            if self.budget - self.calls_made > 50:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func,
                        self.best_x if self.best_x is not None else best_candidate,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 500,
                            'xatol': 1e-8,
                            'fatol': 1e-10
                        }
                    )
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5)
                except StopIteration:
                    pass
                except Exception:
                    pass

        # Ensure we return something valid
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            try:
                self.best_f = wrapped_func(self.best_x)
            except:
                self.best_f = 0.0

        return self.best_x, self.best_f
