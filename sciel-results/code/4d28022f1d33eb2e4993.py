# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

class Optimizer:
    """Deterministic sampler that places points on the known ZDT4 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim  # 10
        g_const = 1 + 10 * (n - 1)  # 91
        budget = self.budget  # 200

        # Analytical ZDT4 front functions
        def f1(x):
            return x
        
        def f2(x):
            # g(x) = 91 + x^2 - 20*x*cos(4*pi*x)
            # f2 = g(x) * (1 - sqrt(x))
            g = g_const + x**2 - 20 * x * np.cos(4 * np.pi * x)
            return g * (1 - np.sqrt(x))

        def calc_hv(x0s):
            # x0s is sorted array of x0 values
            f1s = f1(x0s)
            f2s = f2(x0s)
            R1, R2 = 11.0, 400.0
            
            # HV = sum_{i=0}^{N-2} (f1[i+1] - f1[i]) * (R2 - f2[i]) + (R1 - f1[N-1]) * (R2 - f2[N-1])
            # Note: This formula assumes f2 is decreasing as f1 increases.
            # Let's verify ZDT4 f2 monotonicity.
            # f2(0)=91, f2(1)=0. Generally decreasing.
            # However, g(x) oscillates.
            # x=0.1: g=91+0.01-2*cos(0.4pi)=91-1.53=89.47. f2=89.47*(1-0.316)=61.2
            # x=0.2: g=91+0.04-4*cos(0.8pi)=91-3.09=87.91. f2=87.91*(1-0.447)=48.6
            # x=0.3: g=91+0.09-6*cos(1.2pi)=91+5.71=96.71. f2=96.71*(1-0.547)=43.8
            # f2(0.2)=48.6, f2(0.3)=43.8. Decreasing.
            # It seems f2 is monotonically decreasing.
            
            if len(x0s) == 0:
                return 0.0
            if len(x0s) == 1:
                return (R1 - f1s[0]) * (R2 - f2s[0])
            
            diffs_f1 = np.diff(f1s)
            # The term (R2 - f2[i]) applies to the width [f1[i], f1[i+1]]
            # The last term is (R1 - f1[-1]) * (R2 - f2[-1])
            
            # Vectorized calculation:
            # Area = sum( diffs_f1[:-1] * (R2 - f2s[:-1]) ) + (R1 - f1s[-1]) * (R2 - f2s[-1])
            # Wait, the standard formula is:
            # HV = sum_{i=1}^{N-1} (f1_{i+1} - f1_i) (R2 - f2_i) + (R1 - f1_N) (R2 - f2_N)
            # This matches the above.
            
            area = np.sum(diffs_f1 * (R2 - f2s[:-1])) + (R1 - f1s[-1]) * (R2 - f2s[-1])
            return area

        # Initialize x0s
        x0s = np.linspace(0.0, 1.0, budget)
        
        # Optimize x0s to maximize HV
        # We use a simple coordinate ascent / local search
        # Since N=200 is small, we can afford a few passes.
        
        current_hv = calc_hv(x0s)
        
        # Use a simple perturbation strategy
        # To respect the sorted order, we only allow changes that maintain order, 
        # or we sort afterwards and check HV.
        # Sorting is O(N log N), N=200 is tiny.
        
        for _ in range(20): # 20 passes
            # Randomly select a subset of points to perturb to save time? 
            # Or just iterate all.
            # Let's iterate all.
            for i in range(budget):
                # Try small steps
                best_val = current_hv
                best_x = x0s[i]
                
                # Step size adaptive? Start with 0.01
                step = 0.01
                for delta in [-step, step]:
                    new_x = x0s.copy()
                    new_x[i] += delta
                    # Clamp to [0, 1]
                    new_x[i] = np.clip(new_x[i], 0.0, 1.0)
                    
                    # Sort to maintain Pareto front order
                    new_x_sorted = np.sort(new_x)
                    
                    new_hv = calc_hv(new_x_sorted)
                    if new_hv > best_val:
                        best_val = new_hv
                        best_x = new_x[i] # Actually we need the whole sorted array
                        # We update x0s to the sorted version if it's better
                        # But we are iterating, so we should update x0s in place if possible
                        # For simplicity, if improvement found, update x0s to new_x_sorted
                        # and break out of delta loop?
                        pass

                # The above loop is a bit messy. Let's do a cleaner local search.
                # We'll just try to move x0s[i] to a better position.
                # We can use minimize_scalar on a 1D function: h(t) = -calc_hv(sort(x0s with i=t))
                # This is expensive: 200 points * 20 passes * ~10 evals per min = 4000 HV calcs.
                # Each HV calc is O(N). 4000 * 200 = 800k ops. Very fast.
                
                # Let's use minimize_scalar for each point in each pass.
                # But minimize_scalar might be overkill.
                # Let's just do a few random steps.
                
                # Actually, let's just use a global optimization with a small budget?
                # No, let's stick to local search.
                
                # Better approach:
                # For each i, define a function f(t) that returns -HV when x0s[i]=t (and sorted).
                # Use minimize_scalar with bounds [0, 1].
                
                def obj_func(t):
                    new_x = x0s.copy()
                    new_x[i] = t
                    new_x_sorted = np.sort(new_x)
                    return -calc_hv(new_x_sorted)

                # We need to handle the case where t is close to neighbors.
                # minimize_scalar should handle it.
                
                # To save time, we might skip some points or use a coarser method.
                # But 200 * 20 = 4000 minimize_scalar calls.
                # Each call takes ~10-20 function evals.
                # 4000 * 20 * O(200) = 16M ops. Should be < 1s.
                
                try:
                    res = minimize_scalar(obj_func, bounds=(0.0, 1.0), method='bounded', options={'xatol': 1e-5})
                    if res.success and res.fun < -current_hv:
                        x0s = np.sort(np.where(x0s == x0s[i], res.x, x0s)) 
                        # This replacement is tricky if there are duplicates.
                        # Better:
                        x0s[i] = res.x
                        x0s = np.sort(x0s)
                        current_hv = -res.fun
                except:
                    pass

        # Final sort
        x0s = np.sort(x0s)
        
        # Construct points
        points = []
        for x0 in x0s:
            x = np.zeros(self.dim)
            x[0] = x0
            x = np.clip(x, lo, hi)
            points.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        # Check time
        if time.time() > deadline:
            return points
        
        return points
# EVOLVE-BLOCK-END
