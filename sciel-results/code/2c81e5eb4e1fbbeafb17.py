import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Exploit linearity: The global minimum of a linear function on a box 
       is always at a vertex. We can find the optimal vertex by evaluating 
       each coordinate at its two bounds (-5, 5) relative to a center point, 
       and greedily fixing the coordinate with the largest improvement.
       This takes exactly 10 calls (2 per dim) and finds the global vertex 
       (assuming the center is the origin or close enough that the sign 
       of the gradient is consistent, which is true for BBOB f5 shifts 
       within [-4, 4] since the box is [-5, 5]).
       
       Wait, if the shift is up to 4, and we start at 0, the gradient direction 
       is still determined by the coefficients. 
       f(x) = c + sum(c_i * (x_i - shift_i)). 
       Minimizing this means moving x_i to -5 if c_i > 0 and to 5 if c_i < 0.
       
       To determine the sign of c_i, we can evaluate f at x_i = -5 and x_i = 5 
       while keeping other coordinates at 0 (or their current best).
       Since the function is linear, f(5, 0, ..., 0) - f(-5, 0, ..., 0) = 10 * c_1.
       If this difference is positive, c_1 > 0, so we want x_1 = -5.
       If negative, c_1 < 0, so we want x_1 = 5.
       
       This allows us to construct the optimal vertex in 10 calls.
       
    2. Refinement: After finding the vertex, use CMA-ES or Nelder-Mead for 
       the remaining budget to ensure we hit the exact minimum if there are 
       any numerical quirks or if the "linear" assumption has small non-linear 
       components (though f5 is strictly linear).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            self.calls_made += 1
            try:
                x_arr = np.asarray(x, dtype=float)
                val = float(func(x_arr))
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_arr.copy()
                return val
            except Exception:
                return float('inf')

        # Phase 1: Vertex Hunting (Exploit Linearity)
        # We assume the function is f(x) = offset + dot(coeffs, x - shift).
        # The minimum is at x_i = -5 if coeffs[i] > 0, else x_i = 5.
        # We can determine the sign of coeffs[i] by comparing f at 
        # x_i = 5 and x_i = -5 with other coords at 0.
        
        # Start with x at origin (0,0,0,0,0)
        # Note: The problem states the minimizer is in [-4, 4]^5.
        # The box is [-5, 5]^5.
        # If we evaluate at +/- 5, we are definitely outside the "centered" region 
        # but inside the box.
        
        # Let's evaluate f(0) first to have a baseline? 
        # Actually, we can just compare pairs.
        
        # Strategy: 
        # For each dimension i:
        #   x_plus = zeros, x_plus[i] = 5
        #   x_minus = zeros, x_minus[i] = -5
        #   val_plus = f(x_plus)
        #   val_minus = f(x_minus)
        #   if val_plus < val_minus: coeff[i] < 0 -> optimal x[i] is 5
        #   else: coeff[i] > 0 -> optimal x[i] is -5
        #   (If equal, it's 0, but linear slope usually has non-zero coeffs)
        
        # This takes 2 * dim = 10 calls.
        
        optimal_vertex = np.zeros(self.dim)
        
        for i in range(self.dim):
            if self.calls_made >= self.budget or time.time() > deadline:
                break
                
            x_plus = np.zeros(self.dim)
            x_plus[i] = 5.0
            val_plus = wrapped_func(x_plus)
            
            x_minus = np.zeros(self.dim)
            x_minus[i] = -5.0
            val_minus = wrapped_func(x_minus)
            
            # If val_plus is lower, the function decreases as x_i increases -> coeff is negative -> go to 5
            # If val_minus is lower, the function decreases as x_i decreases -> coeff is positive -> go to -5
            if val_plus < val_minus:
                optimal_vertex[i] = 5.0
            else:
                optimal_vertex[i] = -5.0
        
        # Evaluate the constructed vertex
        if self.calls_made < self.budget and time.time() < deadline:
            v_val = wrapped_func(optimal_vertex)
        
        # Phase 2: Refinement
        # Although the vertex is the global minimum for a linear function, 
        # we might have made an error if the function is not perfectly linear 
        # or if there's noise. Also, the "minimizer" is stated to be in [-4, 4]^5.
        # Wait, if the true minimizer is in [-4, 4]^5, and the function is linear 
        # f(x) = c + sum(c_i x_i), the minimum over the box [-5, 5]^5 is still 
        # at a vertex UNLESS the gradient is zero (unlikely) or the function 
        # is not strictly linear in the box? 
        # BBOB f5 is "Linear slope". It is strictly linear.
        # The constraint "minimizer lies in [-4, 4]^5" likely refers to the 
        # *unconstrained* minimizer of the underlying quadratic or the shift 
        # applied to a standard function? 
        # No, BBOB f5 is defined as f(x) = c + sum(c_i * (x_i - shift_i)).
        # If it's linear, the unconstrained minimizer doesn't exist (it's unbounded) 
        # unless we are talking about the *box* constrained minimizer.
        # The text says "the minimizer lies in [-4, 4]^5". This implies the 
        # *box-constrained* minimizer is NOT at the boundary? 
        # That contradicts "Linear slope" unless the coefficients are such that 
        # the minimum is inside? No, a linear function on a box always has its 
        # minimum at a vertex. 
        # Perhaps "minimizer" refers to the *shift* center? "Instances apply a 
        # hidden seeded shift... the minimizer lies in [-4, 4]^5".
        # If the function is f(x) = ||x - shift||^2 (quadratic), the minimizer is 
        # at the shift. But f5 is Linear.
        # Let's re-read carefully: "BBOB-style function f5 (Linear slope)".
        # In BBOB, f5 is indeed linear. 
        # If the minimizer (of the box-constrained problem) is in [-4, 4]^5, 
        # that means it's NOT at the boundary [-5, 5]^5? 
        # This is impossible for a non-constant linear function. 
        # Unless... the "Linear slope" is actually a *piecewise* linear or 
        # the description implies the *unconstrained* minimum of the *original* 
        # function before clipping is in that range?
        # Or maybe the "minimizer" refers to the location of the *gradient* 
        # zero? No, linear has no gradient zero.
        
        # Let's assume the standard BBOB f5: f(x) = c + sum(c_i * (x_i - z_i)).
        # The minimum on [-5, 5]^d is at x_i = -5 if c_i > 0, 5 if c_i < 0.
        # This vertex is in {-5, 5}^d.
        # The statement "minimizer lies in [-4, 4]^5" is confusing if it refers 
        # to the box-constrained minimizer. 
        # However, if the *shift* z is in [-4, 4]^5, and the function is 
        # actually a *quadratic* centered at z, then the minimizer is z.
        # But the task says "Linear slope".
        
        # Let's look at Inspiration 3: "Since it's linear... converge quickly".
        # And Inspiration 1: "inefficient for the specific...".
        
        # If I strictly follow "Linear", my vertex strategy is perfect.
        # If there's a slight non-linearity or if the "minimizer in [-4,4]" 
        # hint suggests the optimum is interior (implying non-linear, e.g., 
        # quadratic), then CMA-ES is better.
        
        # Given the ambiguity, a hybrid is safest.
        # The vertex strategy gives a very good starting point (likely the 
        # global min if linear).
        # We then run CMA-ES from that point to refine if it's not exactly 
        # optimal (e.g. if it's actually a shifted quadratic).
        
        if self.calls_made < self.budget and time.time() < deadline:
            # Use CMA-ES for refinement
            # Pop size default is 4 + floor(3 ln n) = 4 + 3*ln(5) ~ 9
            # We have ~490 calls left.
            try:
                # Set maxiter to use up the budget
                # CMA-ES stops on convergence.
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='CMA', 
                    options={
                        'maxiter': 1000, # Let it run until budget/time
                        'popsize': 10,
                        'tolx': 1e-8,
                        'tolfun': 1e-8
                    }
                )
            except Exception:
                pass

        return self.best_x, self.best_f
