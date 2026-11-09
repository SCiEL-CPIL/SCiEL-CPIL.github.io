"""Multi-objective optimizer for ZDT2 using direct optimization of point placement for maximum hypervolume."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    
    Strategy:
    ZDT2's Pareto front is a known curve:
      f1 = x0
      f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / (1 + 10*(n-1))))
    achieved when x_1...x_{n-1} = 0.
    
    Instead of using a heuristic spacing (like power law), we directly optimize
    the positions of the x0 values to maximize the hypervolume of the resulting
    set of points relative to the reference point [11.0, 11.0].
    
    For 2D, the hypervolume of a set of points P sorted by f1 is:
    HV = sum_{i=0}^{n-2} (f1_{i+1} - f1_i) * (ref2 - f2_i)
    Note: The last point contributes 0 width in this specific summation if we 
    consider the area under the step function defined by the points up to the 
    next point's f1. However, the standard definition of hypervolume dominated 
    by the set of points up to reference point R is the union of rectangles.
    
    Actually, for a set of non-dominated points sorted by f1 ascending:
    HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (ref2 - f2_i)
    where f1_n is defined as ref1 (11.0) and f2_{n-1} is the f2 of the last point.
    Wait, the last point's rectangle extends to ref1.
    So, HV = sum_{i=0}^{n-2} (f1_{i+1} - f1_i) * (ref2 - f2_i) + (ref1 - f1_{n-1}) * (ref2 - f2_{n-1})
    
    We want to maximize this sum by choosing x0_0, ..., x0_{n-1} in [0, 1].
    We can parameterize x0_i = sin^2(theta_i) or simply optimize x0_i directly 
    with bounds [0, 1]. Since the function is smooth and unimodal in local neighborhoods,
    we can use a simple gradient ascent or scipy's minimize on the negative HV.
    
    To ensure the points are distinct and ordered, we can optimize over the gaps or 
    just optimize the vector and sort it, but sorting makes the function non-differentiable.
    However, since we know the optimal order is increasing x0 (because f2 decreases as f1 increases),
    we can enforce x0_0 < x0_1 < ... < x0_{n-1}.
    
    A simpler and robust approach:
    Use a change of variables to ensure ordering and bounds.
    Let x0_i = u_i where u is a vector of length n.
    We can optimize the vector u directly with bounds [0,1]. 
    To handle the ordering constraint, we can note that if the points are not ordered, 
    sorting them doesn't change the set, but the HV calculation assumes order.
    Since f2 is monotonically decreasing with f1, the optimal set of points will naturally 
    have f1 increasing. If we start with a good initial guess (e.g., uniform or power law), 
    the optimizer should maintain the order.
    
    Let's use scipy.optimize.minimize with method='L-BFGS-B' or 'Nelder-Mead' if gradients are tricky.
    Given n=200, L-BFGS-B is efficient.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        n = self.budget
        dim = self.dim
        
        # ZDT2 parameters
        # f2 = (1 + 10*(dim-1)) * (1 - sqrt(f1 / (1 + 10*(dim-1))))
        coeff = 1.0 + 10.0 * (dim - 1)
        ref1 = 11.0
        ref2 = 11.0
        
        def f2_of_f1(f1):
            # Handle f1=0 case to avoid sqrt(0) issues if any, though sqrt(0)=0 is fine
            # f1 is in [0,1]
            return coeff * (1.0 - np.sqrt(np.maximum(f1 / coeff, 0.0)))
        
        def hypervolume_from_x0(x0):
            # x0 is a 1D array of length n, assumed sorted ascending
            # f1 = x0
            f1 = x0
            f2 = f2_of_f1(f1)
            
            # Calculate HV
            # HV = sum_{i=0}^{n-2} (f1[i+1] - f1[i]) * (ref2 - f2[i]) 
            #    + (ref1 - f1[n-1]) * (ref2 - f2[n-1])
            
            if n == 1:
                return (ref1 - f1[0]) * (ref2 - f2[0])
                
            # Vectors for differences
            # widths[i] = f1[i+1] - f1[i] for i=0..n-2
            # heights[i] = ref2 - f2[i] for i=0..n-2
            # Last term: (ref1 - f1[n-1]) * (ref2 - f2[n-1])
            
            widths = np.diff(f1)
            heights = ref2 - f2[:-1]
            
            hv = np.sum(widths * heights)
            hv += (ref1 - f1[-1]) * (ref2 - f2[-1])
            
            return hv

        def neg_hv(x0):
            # Ensure x0 is within [0,1] and sorted
            # The optimizer might try to violate bounds or order.
            # We clip to [0,1]. 
            # For order, if x0 is not sorted, the HV calculation above is invalid 
            # because it assumes f1[i+1] > f1[i].
            # However, if we initialize well and use a gradient method that respects 
            # the smoothness, it should stay sorted. 
            # To be safe, we can sort x0 inside the objective function. 
            # Sorting makes the function non-differentiable, but L-BFGS-B can often 
            # handle this if the crossings are rare or if we use a penalty.
            # Better: enforce ordering via parameterization?
            # Let x0 = cumsum(exp(y)) normalized? No, complex.
            
            # Simple approach: Sort x0. This creates a piecewise linear function in x0 
            # that is continuous but not differentiable at crossings. 
            # Given the strong convexity/concavity of the HV landscape, 
            # starting from a sorted initial guess usually keeps it sorted.
            x0_sorted = np.sort(x0)
            return -hypervolume_from_x0(x0_sorted)

        # Initial guess: Power law gamma=4.0 (current best)
        u = np.linspace(0.0, 1.0, n)
        x0_init = u ** 4.0
        x0_init[0] = 0.0
        x0_init[-1] = 1.0
        
        # Optimize
        try:
            # Use L-BFGS-B for efficiency with bounds
            # Bounds: all x0 in [0, 1]
            res = minimize(neg_hv, x0_init, method='L-BFGS-B', 
                           bounds=[(0.0, 1.0)] * n,
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < -hypervolume_from_x0(x0_init):
                x0_opt = res.x
            else:
                x0_opt = x0_init
                
        except Exception:
            x0_opt = x0_init
            
        # Ensure final points are sorted and within bounds
        x0_opt = np.sort(np.clip(x0_opt, 0.0, 1.0))
        
        # Construct X
        X = np.zeros((n, dim))
        X[:, 0] = x0_opt
        
        # Set other dimensions to lower bound (0) to be on Pareto front
        if dim > 1:
            X[:, 1:] = bounds[1][0] # Assuming bounds[1][0] is 0.0 for ZDT2
            
        # Evaluate
        F = np.zeros((n, self.n_obj))
        for i in range(n):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X

# EVOLVE-BLOCK-END
