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
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Helper to safely call func and update best
        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = func(x)
            except Exception:
                return float('inf')
            
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Solve for the linear function f(x) = w^T x + b
        # We need dim + 1 points to determine the system.
        # Use the origin and unit vectors along each axis.
        
        # Evaluate origin
        x0 = np.zeros(dim)
        f0 = try_eval(x0)
        
        # If the first eval fails or is inf, we can't proceed with linear solve.
        # But the problem guarantees valid floats.
        
        w = np.zeros(dim)
        
        # Evaluate points along axes to find gradient components
        # f(e_i) = w_i + b
        # f(0) = b
        # So w_i = f(e_i) - f(0)
        
        for i in range(dim):
            x_i = np.zeros(dim)
            x_i[i] = 1.0
            f_i = try_eval(x_i)
            
            if np.isfinite(f0) and np.isfinite(f_i):
                w[i] = f_i - f0
            
            # Check time/budget
            if calls >= self.budget or time.time() > deadline:
                break
        
        # 2. Construct the optimal vertex
        # For f(x) = w^T x + b, the minimum over [-5, 5]^d is at the vertex where
        # x_i = 5 if w_i < 0, and x_i = -5 if w_i > 0.
        # If w_i == 0, any x_i works, but we pick 0 for safety (or 5/-5, doesn't matter for value).
        
        optimal_x = np.zeros(dim)
        for i in range(dim):
            if w[i] < 0:
                optimal_x[i] = 5.0
            elif w[i] > 0:
                optimal_x[i] = -5.0
            else:
                optimal_x[i] = 0.0
                
        # 3. Evaluate the predicted optimum
        # This is the (dim + 2)-th call.
        try_eval(optimal_x)
        
        # 4. Safety net: If for some reason the function wasn't perfectly linear 
        #    or there was a numerical issue, we have plenty of budget left.
        #    However, the problem states "Linear slope", implying f(x) = w^T x + b.
        #    If we found the vertex, we are at the global minimum.
        #    We can stop here to save time, but the evaluator might expect us to 
        #    "use" the budget? No, the contract says "More than 500 calls... invalid".
        #    It does NOT say we must use all calls.
        #    AOCC is calculated over the calls made. 
        #    "repeating the last n_i if fewer calls are made."
        #    So making fewer calls is fine and actually better if we find the optimum early.
        #    Wait, let's re-read: "AOCC = 1 - mean(n_1, ..., n_500), repeating the last n_i if fewer calls are made."
        #    This means if I make 7 calls and find the optimum, n_1..n_6 might be high, 
        #    but n_7..n_500 will be low (close to 0 error). 
        #    The mean will be dominated by the early high values?
        #    Let's check the scoring: 
        #    e_i = clip(b_i - f_opt, 1e-8, 100)
        #    n_i = (log10(e_i) + 8) / 10
        #    If I find the optimum at call 7, b_7 = f_opt. 
        #    e_7 = 1e-8 (clipped). log10(1e-8) = -8. n_7 = 0.
        #    n_8..n_500 are also 0.
        #    The first 6 calls might have larger errors.
        #    So the AOCC is determined by the first 6 calls.
        #    To maximize AOCC, we want to minimize the error in the FIRST calls.
        
        # My current strategy:
        # Call 1: x=0. Error = |w^T 0 + b - f_opt|.
        # Call 2: x=e1. Error = |w^T e1 + b - f_opt|.
        # ...
        # Call 6: x=e5.
        # Call 7: Optimal vertex. Error = 0.
        
        # Can we do better?
        # The "Linear slope" function is f(x) = w^T x + b.
        # The minimum is at a vertex.
        # If we could guess the vertex earlier, we'd have a higher AOCC.
        
        # However, we don't know w until we probe.
        # Is there a way to find the vertex with fewer calls?
        # If we assume the distribution of w is symmetric, we can't guess.
        # But we can use the information from the first few calls to update our belief.
        
        # Actually, the current strategy is optimal for *finding* the optimum exactly.
        # The only way to improve AOCC is to have a *better* best_x in the first few calls.
        # Call 1 is always x=0 (or some initial guess).
        # Can we pick a better initial guess?
        # If we pick x=0, and the true optimum is a vertex, the distance is large.
        # What if we pick a random vertex as the first guess?
        # Probability 1/32 that it's the correct vertex.
        # If it is, n_1 = 0.
        # If not, n_1 is high.
        # Expected n_1 for random vertex:
        # Most vertices are far from the true optimum.
        # Expected n_1 for x=0:
        # x=0 is the center. Distance to any vertex is sqrt(5)*5 = 11.18.
        # The value difference is |w^T v - w^T 0| = |w^T v|.
        # Since w is unknown, this is also a gamble.
        
        # However, the linear solve is deterministic.
        # Let's stick to the linear solve. It guarantees the optimum by call 7.
        # The AOCC will be determined by the first 6 calls.
        # Is there a way to make the first 6 calls "better"?
        # Maybe we can evaluate the vertices directly?
        # There are 32 vertices.
        # If we evaluate them one by one, we might find the optimum early.
        # But we don't know which one is best.
        # The linear solve uses 6 calls to *learn* the function, then 1 to evaluate the best.
        # The learning calls (x=0, x=e1, ...) are not necessarily close to the optimum.
        
        # Alternative:
        # Evaluate x=0.
        # Then evaluate the 5 vertices closest to x=0? No, all vertices are equidistant.
        
        # Let's consider the "shifted" linear function.
        # The prompt says: "Instances apply a hidden seeded shift, rotation and offset".
        # "Linear slope" usually means f(x) = w^T (R x + s) + b.
        # This is still linear in x: f(x) = (R^T w)^T x + (w^T s + b).
        # So the linear solve works perfectly.
        
        # The key is: Does the linear solve give the best possible AOCC?
        # AOCC depends on the sequence of best values.
        # b_1 = f(0)
        # b_2 = min(f(0), f(e1))
        # ...
        # b_7 = f_opt
        
        # If we instead evaluated vertices in a random order:
        # b_1 = f(v1)
        # b_2 = min(f(v1), f(v2))
        # ...
        # Expected time to find the optimum among 32 vertices is 16 calls.
        # This is worse than 7 calls.
        
        # So the linear solve is superior in terms of *when* we find the optimum.
        # The only question is the quality of b_1..b_6.
        # In the linear solve, b_i is the min of f(0), f(e1)..f(e_{i-1}).
        # These points are on the axes.
        # In a random vertex search, b_i is the min of f(v1)..f(v_i).
        # The vertices are "farther" from the center than the axis points?
        # No, axis points are at distance 1. Vertices are at distance 5*sqrt(5) ~ 11.
        # The function value at an axis point is closer to f(0) than at a vertex.
        # So the "error" (distance to optimum) might be smaller for axis points?
        # Let's check.
        # f(x) = w^T x + b. f_opt = f(v*).
        # Error at x=0: |w^T 0 - w^T v*| = |w^T v*|.
        # Error at x=e1: |w^T e1 - w^T v*| = |w_1 - w^T v*|.
        # Since v* is a vertex, w^T v* is the maximum (or minimum) of w^T x over the box.
        # So |w^T v*| is likely large.
        # |w_1 - w^T v*| is likely smaller?
        # If w is uniform, w^T v* is sum of |w_i|*5.
        # w_1 is just one component.
        # So the error at the axis points is likely SMALLER than the error at the origin?
        # Wait.
        # f(0) = b.
        # f(e1) = w_1 + b.
        # f(v*) = sum(sign(w_i)*5*w_i) + b = sum(|w_i|*5) + b.
        # Error(0) = |b - (sum(|w_i|*5) + b)| = 5 * sum(|w_i|).
        # Error(e1) = |(w_1 + b) - (sum(|w_i|*5) + b)| = |w_1 - 5*sum(|w_i|)|.
        # Since sum(|w_i|) >= |w_1|, 5*sum(|w_i|) is much larger than |w_1|.
        # So Error(e1) is also large, but slightly smaller than Error(0) if w_1 has the same sign as the dominant term in v*?
        # Actually, v* is chosen such that w^T v* is the minimum.
        # So f(v*) = -5*sum(|w_i|) + b (assuming w components can be negative, we pick signs to make w^T v* negative large).
        # Let's assume w > 0. Then v* = -5 * 1.
        # f(v*) = -5*sum(w_i) + b.
        # f(0) = b.
        # Error(0) = 5*sum(w_i).
        # f(e1) = w_1 + b.
        # Error(e1) = |w_1 - (-5*sum(w_i))| = w_1 + 5*sum(w_i).
        # This is LARGER than Error(0).
        # So the axis points are WORSE than the origin!
        
        # This means my linear solve strategy has a "valley" in the first few calls?
        # No, b_i is the BEST value so far.
        # b_1 = f(0).
        # b_2 = min(f(0), f(e1)).
        # If f(e1) > f(0), then b_2 = f(0).
        # So the best value stays at f(0) until we find a better point.
        # Since the axis points are worse than the origin (for a linear function minimized at a vertex),
        # b_1 = b_2 = ... = b_6 = f(0).
        # b_7 = f_opt.
        
        # So the AOCC is determined by the error at the origin, held constant for 6 calls, then dropping to 0.
        # Is there a strategy that gets a better b_1?
        # If we start with a random vertex, we have a 1/32 chance of getting b_1 = f_opt.
        # If we miss, b_1 is likely worse than f(0)?
        # Error(vertex) = 10 * sum(|w_i|) (approx, if vertex is opposite to optimum).
        # Error(origin) = 5 * sum(|w_i|).
        # So the origin is on average CLOSER to the optimum than a random vertex.
        # So starting at the origin is a good bet.
        
        # Can we improve b_2..b_6?
        # We need to find a point better than the origin.
        # The points on the axes are worse.
        # What about points closer to the vertex?
        # If we guess the vertex direction correctly, we can jump close to the optimum.
        # But we don't know the direction.
        
        # Strategy:
        # 1. Evaluate x=0.
        # 2. Evaluate 5 points: x_i = 4.9 * e_i (close to the positive vertices).
        #    Wait, if the optimum is at -5 e_i, then 4.9 e_i is very bad.
        #    If the optimum is at 5 e_i, then 4.9 e_i is very good.
        #    Since we don't know the sign, this is a gamble.
        
        # Let's stick to the linear solve. It is robust and guarantees the optimum.
        # The AOCC loss in the first 6 calls is unavoidable if we don't guess the vertex.
        # And guessing the vertex is unlikely to beat the origin on average.
        
        # One optimization:
        # Instead of evaluating e1, e2, ..., we can evaluate points that are more likely to be better than the origin?
        # No, for a linear function, any point on the box boundary is either better or worse.
        # The origin is the "safest" point in terms of expected error (it's the center).
        
        # So the current strategy is likely optimal.
        # Just ensure we don't waste time.
        
        return best_x, best_f
