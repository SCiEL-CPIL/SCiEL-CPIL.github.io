"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Multi-Start Adaptive Nelder-Mead Simplex Method.
    
    The Bent Cigar function (BBOB f12) is highly anisotropic, with one easy direction
    and several very narrow directions. Isotropic methods (like CMA-ES or DE) struggle
    to make progress along the narrow axes within a tight budget because they waste
    evaluations exploring the wide axis. Nelder-Mead, being a local method that adapts
    its shape to the local landscape, can align with the narrow directions and exploit
    them efficiently once it finds a good region.
    
    Key features:
    1. Multi-start: We initialize with a small grid to ensure we find the narrow valley.
    2. Adaptive Simplex: The simplex shape adapts to the local curvature, allowing
       rapid convergence along the narrow axes.
    3. Robustness: We use a time guard to ensure we finish before the deadline.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        try:
            # 1. Multi-Start Initialization
            # For Bent Cigar, the narrow valley is often hard to find.
            # We use a coarse grid to find a good starting point for Nelder-Mead.
            # A 3^dim grid is 243 points for dim=5, which is affordable.
            # However, to be safe with budget, we can use a smaller grid or random starts.
            # Let's use a grid of size 5 in each dimension? 5^5 = 3125 is too many.
            # Let's use a grid of size 3 in each dimension: 3^5 = 243.
            # We will evaluate these and pick the best few to start Nelder-Mead.
            
            # Actually, 243 calls is nearly half the budget. 
            # Let's use a random multi-start instead, or a very sparse grid.
            # The Bent Cigar function is convex but anisotropic. 
            # A random start might miss the valley. 
            # Let's try a grid of 2 points per dimension (corners) + center? 2^5 + 1 = 33.
            # Or a random set of points.
            # Given the "hidden seeded shift, rotation and offset", the valley location is random.
            # Random multi-start is more robust than a fixed grid.
            
            n_starts = 10
            starts = []
            # Ensure diversity
            for i in range(n_starts):
                if i == 0:
                    x0 = np.zeros(dim)
                else:
                    x0 = np.random.uniform(-4.5, 4.5, size=dim)
                starts.append(x0)
            
            # Evaluate starts
            f_starts = []
            for x0 in starts:
                try:
                    fx = wrapped_func(x0)
                    f_starts.append(fx)
                except (TimeoutError, ValueError):
                    break
            
            if not f_starts:
                return np.zeros(dim), float('inf')

            # Sort starts by function value
            order = np.argsort(f_starts)
            # Use the best 3-5 starts for Nelder-Mead
            
            # 2. Nelder-Mead Refinement
            # We will run Nelder-Mead from the best starts, using the remaining budget.
            # We need to manage the budget carefully.
            
            # Parameters for Nelder-Mead
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            # Initial step size for simplex
            # For Bent Cigar, a small step size in the narrow direction is needed.
            # We can use an adaptive step size based on the initial spread.
            # Or just a fixed small step size, e.g., 0.1 or 0.5.
            # Since the function is scaled, let's start with 0.5.
            step_size = 0.5
            
            best_overall_x = self.best_x
            best_overall_f = self.best_f
            
            # We will run Nelder-Mead on the top 2 starts to balance exploration/exploitation
            top_k = min(2, len(order))
            
            for k in range(top_k):
                idx = order[k]
                x_current = starts[idx].copy()
                f_current = f_starts[idx]
                
                # Construct initial simplex
                simplex = [x_current]
                f_simplex = [f_current]
                
                for i in range(dim):
                    x_new = x_current.copy()
                    # Use a small perturbation
                    # To handle anisotropy, we might need to adapt the step size.
                    # For now, use a standard simplex construction.
                    x_new[i] += step_size
                    simplex.append(x_new)
                    try:
                        f_new = wrapped_func(x_new)
                        f_simplex.append(f_new)
                    except (TimeoutError, ValueError):
                        break
                
                if len(simplex) < dim + 1:
                    continue
                
                # Nelder-Mead Loop
                # We limit the number of iterations to ensure we don't exceed budget/time
                max_iterations = 500 # Generous, but budget/time will stop us
                
                for _ in range(max_iterations):
                    if time.time() > deadline or self.calls >= budget:
                        break
                    
                    # Sort simplex by function value
                    order_simplex = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in order_simplex]
                    f_simplex = [f_simplex[i] for i in order_simplex]
                    
                    # Check convergence
                    if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                        break
                    
                    # Compute centroid of all but the worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_reflect = centroid + alpha * (centroid - simplex[-1])
                    try:
                        f_reflect = wrapped_func(x_reflect)
                    except (TimeoutError, ValueError):
                        break
                    
                    if f_simplex[0] <= f_reflect < f_simplex[-2]:
                        # Accept reflection
                        simplex[-1] = x_reflect
                        f_simplex[-1] = f_reflect
                    elif f_reflect < f_simplex[0]:
                        # Expansion
                        x_expand = centroid + gamma * (x_reflect - centroid)
                        try:
                            f_expand = wrapped_func(x_expand)
                        except (TimeoutError, ValueError):
                            break
                        
                        if f_expand < f_reflect:
                            simplex[-1] = x_expand
                            f_simplex[-1] = f_expand
                        else:
                            simplex[-1] = x_reflect
                            f_simplex[-1] = f_reflect
                    else:
                        # Contraction
                        if f_reflect < f_simplex[-1]:
                            # Outside contraction
                            x_contract = centroid + rho * (x_reflect - centroid)
                        else:
                            # Inside contraction
                            x_contract = centroid - rho * (simplex[-1] - centroid)
                        
                        try:
                            f_contract = wrapped_func(x_contract)
                        except (TimeoutError, ValueError):
                            break
                        
                        if f_contract <= min(f_reflect, f_simplex[-1]):
                            simplex[-1] = x_contract
                            f_simplex[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                try:
                                    f_shrink = wrapped_func(simplex[i])
                                except (TimeoutError, ValueError):
                                    break
                                f_simplex[i] = f_shrink
                            
                            if time.time() > deadline or self.calls >= budget:
                                break

                    # Update best overall
                    if f_simplex[0] < best_overall_f:
                        best_overall_f = f_simplex[0]
                        best_overall_x = simplex[0].copy()

            return best_overall_x, best_overall_f

        except (TimeoutError, ValueError):
            return self.best_x if self.best_x is not None else np.zeros(dim), self.best_f if self.best_f != float('inf') else 0.0

# EVOLVE-BLOCK-END
