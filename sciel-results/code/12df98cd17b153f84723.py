import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin from 330s limit
        max_calls = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        
        # Weierstrass function is multimodal.
        # Nelder-Mead is a local method.
        # To maximize AOCC (Area Under Curve of Convergence), we need to find the global minimum as fast as possible.
        # Strategy:
        # 1. Initial phase: Random sampling to get a good starting point quickly.
        # 2. Local phase: Nelder-Mead from the best point found so far.
        # 3. If NM converges or stalls, try another random point and NM again.
        # 4. Alternatively, use a more global method like Differential Evolution or Basin Hopping if time permits,
        #    but NM with random restarts is often very effective for 5D.
        
        # Let's implement a simple random restart NM loop.
        # We will allocate some budget to initial random sampling to find a decent basin.
        # Then use the rest for NM.
        
        # Since func is black-box and we have 500 calls, we can afford ~20 restarts of 25 calls each?
        # Or fewer restarts with more calls each?
        # Weierstrass has many local minima.
        # Let's try:
        # Phase 1: 20 random evaluations. Find best x.
        # Phase 2: Run NM from best x with remaining budget.
        # If NM finishes early (converges), run another random sampling phase?
        # Or just restart NM from a new random point?
        
        # A better strategy for AOCC:
        # The score depends on the best value found at each step.
        # So, we want to improve the best value frequently.
        # NM improves value monotonically (usually) within a run.
        # Random sampling might find a better global basin quickly.
        
        # Let's use a hybrid:
        # 1. Evaluate 10 random points. Keep best.
        # 2. Run NM from best point. Use up to 50% of remaining budget.
        # 3. If NM converges or uses budget, evaluate 10 more random points.
        # 4. Run NM from new best.
        # Repeat until budget or time is out.
        
        # However, NM in scipy can be slow if it doesn't converge.
        # We need to wrap func to count calls and enforce budget.
        
        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise Exception("Budget exceeded")
            calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Nelder-Mead setup
        # We need to be careful with the callback to stop if budget is hit.
        # scipy minimize with Nelder-Mead doesn't have a perfect "stop on budget" callback that returns cleanly
        # unless we raise an exception or use a callback that returns True.
        # If we raise an exception, we lose the current iteration's state? No, best_x is already updated in wrapped_func.
        # So raising an exception is safe for tracking best_x.
        
        def nm_callback(xk):
            nonlocal calls
            if calls >= max_calls:
                return True
            if time.time() > deadline:
                return True
            return False

        # Initial random sampling
        num_initial = 10
        for _ in range(num_initial):
            if calls >= max_calls or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            try:
                wrapped_func(x)
            except Exception:
                break

        # Main loop: NM from best, then more random, then NM...
        # We'll do this in a loop.
        # Each NM run should use a chunk of the budget.
        # If we have 500 calls, and used 10, we have 490.
        # Let's try to run NM with maxiter such that it doesn't exceed budget.
        # NM in dim=5 takes roughly 2*dim + 1 = 11 points to start, and then iterates.
        # Each iteration is 1 function eval (or more if reflection/expansion etc are needed? No, NM is 1 eval per iteration mostly, except for initial simplex).
        # Actually, scipy NM uses 1 eval per iteration after the initial simplex.
        # Initial simplex: dim+1 = 6 evals.
        # So if we have 490 calls, we can do ~480 iterations.
        # But we want to restart if it gets stuck.
        
        # Let's define a helper to run NM with a budget limit.
        def run_nm(x0, budget_limit):
            nonlocal calls, best_f, best_x
            if calls >= max_calls or time.time() > deadline:
                return
            
            # We need to ensure we don't exceed max_calls.
            # The callback will stop it.
            # But we also want to limit the *intended* budget for this run to allow restarts.
            # Let's just let the callback handle the hard stop.
            # To allow restarts, we can check if we've made significant progress.
            # Or just run until budget is low, then do one last random check?
            
            # Actually, a single long NM run from a good start is often best.
            # If we restart too often, we waste time on initial simplex construction.
            # But if the first start is bad, we waste time.
            
            # Strategy:
            # Run NM. If it converges (res.success), we are done with that basin.
            # If it hits maxiter or callback, we might be stuck or out of budget.
            # If out of budget, stop.
            # If stuck, maybe try a new random start?
            
            # Let's try:
            # Run NM with maxiter = 1000 (large).
            # If it finishes early (success), do another random sampling phase and NM.
            # If it hits budget, stop.
            
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12},
                               callback=nm_callback)
            except Exception:
                pass

        # Run NM from the best point found in initial sampling
        if best_x is not None:
            run_nm(best_x, max_calls - calls)
            
        # If we still have budget and time, do more random sampling and NM
        # This handles the case where the first NM got stuck in a local minimum
        # and we have enough budget to explore another basin.
        
        # How to decide if we should restart?
        # If we have > 50 calls left, we can afford another random phase + NM.
        
        while calls < max_calls - 50 and time.time() < deadline:
            # Random sampling phase
            num_rand = 10
            for _ in range(num_rand):
                if calls >= max_calls or time.time() > deadline:
                    break
                x = np.random.uniform(-5, 5, dim)
                try:
                    wrapped_func(x)
                except Exception:
                    break
            
            if calls >= max_calls or time.time() > deadline:
                break
            
            # Run NM from the new best (if it improved)
            # Note: best_x is updated in wrapped_func, so it's always the global best so far.
            # We should run NM from best_x.
            run_nm(best_x, max_calls - calls)
            
            # If we didn't improve much, we might be cycling.
            # But for now, let's just keep going until budget is low.
            
        # Final check: if we have a few calls left, just do random?
        # NM is usually better than random at the end.
        # But if NM is stuck, random might find a new basin.
        # Given the small budget left, random is cheap.
        
        while calls < max_calls and time.time() < deadline:
            x = np.random.uniform(-5, 5, dim)
            try:
                wrapped_func(x)
            except Exception:
                break

        return best_x, best_f
