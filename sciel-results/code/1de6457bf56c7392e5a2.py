# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Adaptive Multi-Scale Grid Scan + Aggressive (1+1)-ES.
    
    Strategy:
    1. Rapidly scan the search space using a coarse-to-fine grid to locate
       the approximate location of the minimum (basin of attraction).
       This is much more efficient than random sampling for finding the
       'right region' in 5D.
    2. Once a promising region is found, switch to a (1+1)-Evolution Strategy
       with aggressive step size adaptation to converge to the optimum
       using the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Adaptive Grid Scan
        # We want to cover the space [-5, 5]^5.
        # A full grid of 10^5 points is 100,000 calls, which is too many.
        # We use a stratified random sampling approach that mimics a grid
        # but is more efficient. We will divide the budget for this phase
        # into a few levels of resolution.
        
        # Level 1: Coarse Grid (10 points per dim -> 100k points? No, 10^5 is 100,000)
        # Actually, 10^5 is exactly the budget limit if we used it all.
        # Let's use 5 points per dim -> 5^5 = 3125 points. This is very fast.
        # Then 7 points per dim around the best?
        
        # Let's try a simpler approach: 
        # 1. Sample 1000 random points to get a rough idea.
        # 2. Do a local grid search around the best point.
        
        # Actually, the "Sharp Ridge" (f13) suggests the function is flat in some
        # directions and sharp in others. A grid might miss the ridge if aligned
        # poorly. Random sampling is safer for finding the basin.
        
        # Let's stick to a robust (1+1)-ES but with a better initialization
        # and step size control. The previous program's failure was likely
        # due to wasting too much budget on "portfolio" and "multi-start"
        # overheads and not refining enough.
        
        # New Strategy:
        # 1. Initialize with 50 random points.
        # 2. Pick the best.
        # 3. Run a highly tuned (1+1)-ES for the rest of the budget.
        
        # Initialization
        n_init = 50
        candidates = []
        for _ in range(n_init):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = timed_call(x)
            if fx is not None:
                candidates.append((fx, x))
        
        if not candidates:
             # Fallback if we ran out of budget immediately (unlikely)
             x = np.zeros(self.dim)
             fx = func(x)
             calls_made += 1
             return x, fx

        # Sort by fitness
        candidates.sort(key=lambda item: item[0])
        best_f, best_x = candidates[0]

        # Phase 2: (1+1)-ES
        # Start with a moderate sigma
        sigma = 1.0
        x = best_x.copy()
        fx = best_f
        
        # Success rule: 1/5 rule or similar
        # If success rate is too high, increase sigma.
        # If success rate is too low, decrease sigma.
        
        # We will use a simple adaptive scheme:
        # Count successes in a window.
        window_size = 10
        successes = 0
        
        while calls_made < self.budget:
            if time.time() > deadline - 2:
                break
                
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = timed_call(y)
            
            if fy is None:
                break
                
            if fy < fx:
                x = y
                fx = fy
                best_f = fx
                best_x = x.copy()
                successes += 1
            else:
                successes = 0 # Reset on failure? Or keep count?
                # Standard 1/5 rule keeps a running average.
                # Let's use a simpler exponential adaptation based on recent history.
                # For simplicity and speed, let's use the standard 1/5 rule approximation:
                # If success, sigma *= exp(1/sqrt(dim))
                # If fail, sigma *= exp(-1/(2*sqrt(dim)))
                
            # Adaptation (1/5 rule logic)
            # This is a simplified version. A proper 1/5 rule maintains a success rate.
            # Let's use a direct adaptation:
            # If we just had a success, we are likely in a good direction, so expand.
            # If we failed, we might be too big, so shrink.
            
            # However, for "Sharp Ridge", we need to be careful.
            # Let's use the standard CMA-ES style step size adaptation if possible,
            # but (1+1)-ES with 1/5 rule is robust.
            
            # Let's implement the 1/5 rule properly:
            # We need a counter for successes in the last N steps.
            # To keep it simple and fast, let's just use:
            # sigma = sigma * exp( (success - 0.2) / (window_size * sqrt(dim)) )
            # This requires tracking.
            
            # Alternative: Simple multiplicative update
            # If success: sigma *= 1.1
            # If fail: sigma *= 0.9
            # This is fast and often works well.
            
            # Let's try to be slightly smarter.
            # If fy < fx:
            #     sigma *= 1.1
            # else:
            #     sigma *= 0.95
            
            # Let's use the code below for adaptation, replacing the placeholder logic above.
            # (I will implement the adaptation in the loop below properly)
            
            # Re-doing the loop logic with proper adaptation variables
            pass # The loop above was a draft. I will write the final loop below.

        # The draft loop above had 'pass'. I need to rewrite the loop cleanly.
        
        # Let's restart the Phase 2 implementation cleanly.
        # The variables x, fx, sigma are already set.
        # We need to track success rate for 1/5 rule.
        
        # Actually, the loop above didn't execute because of the 'pass' at the end?
        # No, the 'while' loop was defined, but I put 'pass' at the end of the body?
        # No, I put 'pass' after the 'else' block?
        # Let's look at the code structure.
        # The while loop body had:
        # 1. Check time
        # 2. Generate y
        # 3. Call func
        # 4. Update x/fx if better
        # 5. Adapt sigma? I didn't write the adaptation code in the draft loop!
        # I just put comments.
        # So the draft loop is broken/incomplete.
        
        # Let's rewrite the entire __call__ to be clean and correct.

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Use 100 random points to find a good starting point.
        # This is cheap (100 calls) and ensures we don't start in a bad spot.
        n_init = 100
        for _ in range(n_init):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            timed_call(x)
            
        if best_x is None:
            # Fallback
            x = np.zeros(self.dim)
            fx = func(x)
            calls_made += 1
            return x, fx

        # Phase 2: (1+1)-ES with 1/5 Success Rule
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        
        # 1/5 rule parameters
        # Target success rate = 0.2
        # Adaptation: sigma = sigma * exp( (s - 0.2) / (tau * N) )
        # where s is success rate, N is window size, tau is time constant.
        # Simple implementation:
        # Keep a counter of successes in the last 10 steps.
        
        window_size = 10
        successes_in_window = 0
        total_in_window = 0
        
        # We will use a slightly more aggressive adaptation for faster convergence
        # sigma *= exp(0.25 * (success - 0.2)) per step? No, that's too fast.
        # Standard: sigma *= exp( (success_rate - 0.2) / (window_size * sqrt(dim)) )
        
        while calls_made < self.budget:
            if time.time() > deadline - 2:
                break
                
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = timed_call(y)
            
            if fy is None:
                break
                
            success = fy < fx
            if success:
                x = y
                fx = fy
                best_f = fx
                best_x = x.copy()
                successes_in_window += 1
            else:
                successes_in_window = 0 # Reset? No, 1/5 rule uses average.
                # Actually, standard 1/5 rule uses the success rate over a window.
                # Let's implement a simple moving average or just a counter.
                # To keep it simple:
                # If we fail, we don't add to success count.
                # But we need to decay the success count.
                # Let's use a counter that decays by 0.9 each step?
                # Or just a fixed window.
                
                # Fixed window implementation:
                # We need a queue.
                # For speed, let's use a simpler heuristic:
                # If success: sigma *= 1.05
                # If fail: sigma *= 0.95
                # This is often sufficient and much faster to implement.
                
                pass # Placeholder for adaptation logic

            # Let's use the simple multiplicative update for speed and reliability.
            # It avoids the complexity of window management.
            # Tuning: 
            # Success: sigma *= 1.1 (Expand to explore faster)
            # Fail: sigma *= 0.9 (Shrink to refine)
            
            if success:
                sigma *= 1.1
            else:
                sigma *= 0.9
                
            # Clamp sigma
            sigma = np.clip(sigma, 0.001, 5.0)
            
            # Optional: If sigma gets too small, we might be stuck.
            # But in 5D, 0.001 is very small.
            
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
