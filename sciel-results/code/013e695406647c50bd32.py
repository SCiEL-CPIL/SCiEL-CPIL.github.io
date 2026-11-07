import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, leave 15s for overhead/cleanup
        deadline = start_time + 315.0
        calls = 0
        max_calls = self.budget
        dim = self.dim
        
        best_x = None
        best_f = np.inf
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            x = np.asarray(x, dtype=float)
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        try:
            # Phase 1: Differential Evolution (Global Search)
            # DE is robust to multimodality and doesn't suffer from the 
            # "axis-aligned" bias as much as simple ES in early stages if initialized well.
            
            # Parameters
            pop_size = 20 
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate
            
            # Initialize population uniformly in [-5, 5]^dim
            # Using a fixed seed for reproducibility in case of restarts? 
            # No, random is better for different instances.
            # But we need to be careful with the "hidden seeded shift".
            # Uniform random is safe.
            
            population = np.random.uniform(-5, 5, (pop_size, dim))
            fitness = np.zeros(pop_size)
            
            # Evaluate initial population
            for i in range(pop_size):
                try:
                    fitness[i] = wrapper(population[i])
                except (ValueError, TimeoutError):
                    break
                except Exception:
                    fitness[i] = np.inf
            
            if calls >= max_calls or time.time() > deadline:
                return best_x, best_f

            # DE Loop
            # We will run DE until we have used ~70% of the budget or time is running out.
            # Then switch to local search.
            budget_for_de = int(max_calls * 0.7)
            
            while calls < budget_for_de and time.time() < deadline:
                # Sort by fitness
                order = np.argsort(fitness)
                population = population[order]
                fitness = fitness[order]
                
                # Keep track of best
                # Note: wrapper already updates best_x/best_f
                
                # Generate offspring for each individual
                new_population = population.copy()
                new_fitness = fitness.copy()
                
                for i in range(pop_size):
                    if calls >= budget_for_de or time.time() > deadline:
                        break
                    
                    # Select 3 distinct random individuals different from i
                    # Standard DE/rand/1
                    while True:
                        r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                        if len({r1, r2, r3, i}) == 4:
                            break
                    
                    # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                    v = population[r1] + F * (population[r2] - population[r3])
                    
                    # Clip to box [-5, 5]
                    v = np.clip(v, -5, 5)
                    
                    # Crossover: binomial
                    j_rand = np.random.randint(dim)
                    trial = population[i].copy()
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            trial[j] = v[j]
                    
                    # Evaluate trial
                    try:
                        trial_f = wrapper(trial)
                    except (ValueError, TimeoutError):
                        break
                    except Exception:
                        trial_f = np.inf
                    
                    # Selection
                    if trial_f <= fitness[i]:
                        new_population[i] = trial
                        new_fitness[i] = trial_f
                
                population = new_population
                fitness = new_fitness
                
                # Check for stagnation? 
                # If best hasn't improved in many iterations, we might want to restart or just continue.
                # For Weierstrass, DE usually finds a good basin quickly.
                
        except (ValueError, TimeoutError):
            pass
        except Exception:
            # Fallback if DE crashes
            if best_x is None:
                rng = np.random.default_rng(42)
                for _ in range(10):
                    if calls >= max_calls: break
                    x = rng.uniform(-5, 5, dim)
                    try:
                        wrapper(x)
                    except:
                        break

        # Phase 2: Local Polish (Nelder-Mead)
        # Use the remaining budget to refine the best solution found.
        if best_x is not None and calls < max_calls and time.time() < deadline:
            try:
                # Nelder-Mead parameters
                # We implement a simple version or use scipy if available?
                # SciPy is available.
                from scipy.optimize import minimize
                
                # Define the function for scipy
                # Note: minimize calls func multiple times. We need to ensure we don't exceed budget.
                # We can wrap it to count calls.
                
                # Initial simplex: best_x and small perturbations
                # Simplex size should be small, e.g., 0.1
                x0 = best_x.copy()
                
                # Construct simplex
                simplex = [x0]
                for i in range(dim):
                    x_perturbed = x0.copy()
                    x_perturbed[i] += 0.1
                    x_perturbed = np.clip(x_perturbed, -5, 5)
                    simplex.append(x_perturbed)
                
                # We need to be careful with the call count.
                # SciPy's minimize might make many calls.
                # We will use a custom loop or limit maxiter.
                
                # Let's use a simple manual Nelder-Mead or just call scipy with maxiter limited by remaining budget.
                remaining_calls = max_calls - calls
                if remaining_calls > 10:
                    # Use scipy minimize
                    # method='Nelder-Mead'
                    # options={'maxiter': remaining_calls - 5, 'xatol': 1e-6, 'fatol': 1e-6}
                    
                    # We need a wrapper that stops when budget is exceeded
                    # But scipy doesn't support stopping mid-iteration easily via exception in all versions?
                    # It does stop if the function raises an exception? No, it propagates.
                    # So we must ensure the function doesn't raise, but we can't easily stop scipy from inside.
                    # Alternative: Implement a simple local search or use the fact that Nelder-Mead converges fast.
                    
                    # Let's try a simple coordinate descent or small step local search instead of full NM if SciPy is tricky.
                    # Or just use SciPy and catch the exception.
                    
                    def scipy_func(x):
                        nonlocal calls
                        if calls >= max_calls:
                            raise ValueError("Budget Exceeded")
                        if time.time() > deadline:
                            raise TimeoutError("Time Limit")
                        calls += 1
                        x = np.asarray(x, dtype=float)
                        val = func(x)
                        if val < best_f:
                            best_f = val
                            best_x = x.copy()
                        return val

                    try:
                        result = minimize(scipy_func, x0, method='Nelder-Mead', options={'maxiter': remaining_calls - 10, 'xatol': 1e-8, 'fatol': 1e-8})
                    except (ValueError, TimeoutError):
                        pass
                    except Exception:
                        pass
                        
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            if calls == 0:
                try:
                    wrapper(best_x)
                except:
                    pass

        return best_x, best_f
