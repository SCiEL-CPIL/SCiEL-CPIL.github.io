import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Global Linear Model Fitting
        # We assume f(x) is linear: f(x) = c + g^T x
        # We evaluate at the origin and at +/- 5 along each axis.
        # This gives us 1 + 2*dim = 11 evaluations.
        # This allows us to estimate c and g_i with high precision 
        # (assuming the function is indeed linear, which f5 is).
        
        # Evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        update_best(x0, f0)
        
        if not time_ok():
            return best_x, best_fx
            
        g = np.zeros(dim)
        h = 5.0
        
        # Evaluate axis points to estimate gradient
        # We only need one side to determine the sign and magnitude for a linear function
        # if we assume the slope is constant. 
        # Actually, for a linear function f(x) = c + g^T x, 
        # f(h*e_i) - f(0) = g_i * h.
        # So g_i = (f(h*e_i) - f0) / h.
        # We don't need the -h point if we trust the linearity.
        # Using h=5 puts us on the boundary, which is good for resolving the slope direction clearly.
        
        for i in range(dim):
            if not time_ok():
                break
            x_probe = np.zeros(dim)
            x_probe[i] = h
            f_probe = func(x_probe)
            calls += 1
            update_best(x_probe, f_probe)
            g[i] = (f_probe - f0) / h

        if not time_ok():
            return best_x, best_fx

        # Phase 2: Analytic Optimal Corner Calculation
        # For f(x) = c + g^T x, the minimum on the box [-5, 5]^d is at the corner
        # where x_i = -5 if g_i > 0, and x_i = 5 if g_i < 0.
        # If g_i is exactly 0, the direction doesn't matter, but we pick 0 or arbitrary.
        
        x_opt = np.zeros(dim)
        for i in range(dim):
            if g[i] > 0:
                x_opt[i] = -5.0
            elif g[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0 # Doesn't matter for linear, but 0 is center. 
                               # However, the box is [-5, 5], so 0 is valid.
                               # Wait, if g_i=0, any x_i is optimal. 
                               # But the "true" minimum of the linear slope function 
                               # usually has non-zero gradients. 
                               # Let's stick to the box boundaries for robustness 
                               # in case of tiny numerical noise in g.
                               # If g[i] is very small, the sign is noisy.
                               # But f5 (Linear Slope) typically has significant slopes.
                               # Let's use a small threshold.
                               # If abs(g[i]) < 1e-10, set to 0.
                               # Actually, just use the sign.
        
        # Evaluate the predicted optimal corner
        f_opt_pred = func(x_opt)
        calls += 1
        update_best(x_opt, f_opt_pred)
        
        if not time_ok():
            return best_x, best_fx

        # Phase 3: Robustness Check
        # Since the function is linear, x_opt IS the global minimum.
        # However, if there is any non-linearity (which f5 shouldn't have, but 
        # just in case of implementation quirks or if my assumption of pure linearity 
        # is slightly off due to the "hidden seeded shift" interacting with something),
        # or if the gradient estimation was affected by some edge case,
        # we should verify.
        
        # But wait, the task says "BBOB-style function f5 (Linear slope)".
        # This implies f(x) = f_opt + slope * (x - x_opt)^T * v for some vector v?
        # No, BBOB f5 is typically defined as a linear slope.
        # The "hidden seeded shift, rotation and offset" means:
        # f(x) = c + g^T (R(x - s))
        # This is still a linear function of x: f(x) = (c - g^T R s) + (R^T g)^T x.
        # So it is strictly linear.
        
        # Therefore, the corner x_opt calculated from the gradient g is the exact global minimum.
        # We have found it in 1 + dim + 1 = 7 calls.
        
        # To maximize AOCC, we want to spend the remaining budget 
        # confirming this or just returning. 
        # The score is based on the best value found so far.
        # Since we found the global minimum at call 7, 
        # b_i for i >= 7 will be f_opt.
        # e_i = clip(f_opt - f_opt, ...) = 0? No, clip(..., 1e-8, 100) = 1e-8.
        # n_i = (log10(1e-8) + 8) / 10 = (-8 + 8) / 10 = 0.
        # So for all i >= 7, n_i = 0.
        # AOCC = 1 - mean(n_1, ..., n_500).
        # n_1..n_6 will be > 0. n_7..n_500 will be 0.
        # This is the best possible score achievable for this problem class, 
        # limited only by the initial exploration cost.
        
        # Can we reduce the initial exploration cost?
        # We need to determine the sign of each of the 5 gradient components.
        # We used 5 calls for that + 1 for origin + 1 for corner = 7 calls.
        # Is it possible to do it in fewer?
        # To determine the sign of g_i, we need to compare f at two points 
        # that differ in dimension i.
        # We used f(0) and f(5e_i). 
        # We could use a smaller h, but the magnitude of the difference 
        # must be large enough to distinguish from noise (though noise is 0 here).
        # Actually, since it's deterministic, any h > 0 works.
        # But we also need to find the corner.
        # The corner depends on the signs.
        # So we must know the signs before we can jump to the corner.
        # Minimum calls:
        # 1. f(0)
        # 2-6. f(he_i) for i=1..5. (5 calls)
        # 7. f(x_opt)
        # Total 7 calls.
        
        # This seems optimal for a black-box linear function in 5D.
        
        # We will just return now, or make a few redundant calls if we want to 
        # be safe against any weirdness, but it's likely unnecessary.
        # The problem states "More than 500 calls... makes the whole result invalid".
        # We are at 7 calls. We have plenty of budget.
        # We can just return.
        
        return best_x, best_fx
