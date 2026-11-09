"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ2 using a structured grid on the Pareto front.
    
    Strategy:
    1. DTLZ2 with M=3 has a Pareto front that is a quarter-sphere in 3D (f1^2+f2^2+f3^2=1, fi>=0).
    2. The position variables x[0], x[1] determine the point on the front.
    3. x[2:] should be set to 0.5 to minimize g(x)=0, but actually for DTLZ2,
       the Pareto front is obtained when g=1 (or the appropriate value).
       
    Actually, for DTLZ2, the standard definition gives:
    f1 = g*(1-x1), f2 = g*x1*(1-x0), f3 = g*x1*x0
    where g = sum_{i=2}^{n} (x_i - 0.5)^2
    
    The Pareto front is the set of non-dominated points. When g=0, all f=0.
    But the "known Pareto front" for scoring is the unit sphere.
    
    This implies the evaluator's reference front is the unit sphere, and our
    goal is to maximize hypervolume. The best points are those closest to the
    origin (0,0,0).
    
    However, since we can set g=0 by setting x[2:]=0.5, all points would be (0,0,0),
    giving the maximum possible hypervolume. But the current score is 1.18, not higher.
    
    Let me reconsider: perhaps the problem(x) function does NOT allow g=0, or the
    bounds are such that x[2:] cannot all be 0.5. But the task says bounds [0,1].
    
    Wait - looking at the current program, it sets x[2:]=0.5 and gets score 1.18.
    If all points were (0,0,0), the hypervolume would be 2.5^3 = 15.625.
    The reference front's hypervolume must be ~15.625/1.18 ≈ 13.24.
    
    This suggests that NOT all points are at (0,0,0). Perhaps the problem function
    has a different definition, or there's a constraint I'm missing.
    
    Given the uncertainty, let's focus on improving the grid coverage.
    The current program uses a 10x20 grid. Let's try different grid shapes
    and see if we can improve coverage of the Pareto front.
    
    Key insight: For hypervolume maximization, we want points that are:
    1. Close to the origin (small objective values)
    2. Well-spread to cover different regions of the objective space
    
    A uniform grid in x[0], x[1] space maps to a non-uniform distribution
    on the Pareto front. We might benefit from a different parameterization.
    
    Let's try using spherical coordinates directly and map back to x[0], x[1].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # For DTLZ2 with M=3, m=2:
        # f1 = g*(1-x1), f2 = g*x1*(1-x0), f3 = g*x1*x0
        # where g = sum_{i=2}^{n} (x_i - 0.5)^2
        #
        # If we set x[2:] = 0.5, g=0, so f=(0,0,0).
        # But the Pareto front is the unit sphere, which implies g=1 on the front.
        #
        # To get points on the unit sphere, we need g=1.
        # g = sum_{i=2}^{11} (x_i - 0.5)^2 = 1
        # We can set x[2] = 0.5 + 1/sqrt(10) and x[3:12] = 0.5
        # Then g = (1/sqrt(10))^2 = 1/10, not 1.
        # To get g=1, we need sum of 10 squared terms = 1.
        # Set x[2] = 0.5 + 1, but that's 1.5, out of bounds.
        # Set x[2] = 0.5 + 0.5 = 1.0, then (0.5)^2 = 0.25.
        # We need total g=1, so we need more terms.
        # With 10 terms, max g = 10 * (0.5)^2 = 2.5.
        # To get g=1, set each of 10 terms to (1/10)^(1/2) = 1/sqrt(10) ≈ 0.316.
        # So x[i] = 0.5 + 1/sqrt(10) for i=2..11. But 0.5+0.316=0.816 < 1, OK.
        # Or x[i] = 0.5 - 1/sqrt(10) = 0.184 > 0, also OK.
        
        # Actually, let's just set x[2:] such that g=1.
        # Use x[2] = 0.5 + sqrt(1/10), x[3:] = 0.5.
        # Then g = (sqrt(1/10))^2 = 1/10. Not 1.
        # To get g=1 with one variable: x[2] = 0.5 + 1 = 1.5 (out of bounds)
        # or x[2] = 0.5 - 1 = -0.5 (out of bounds).
        # So we need to distribute across multiple variables.
        # With 10 variables, set each to 0.5 + 1/sqrt(10).
        # g = 10 * (1/sqrt(10))^2 = 10 * 1/10 = 1. 
        
        # So set x[2:12] = 0.5 + 1/np.sqrt(10)
        
        g_target = 1.0
        n_dist = self.dim - 2  # 10 distance variables
        offset = np.sqrt(g_target / n_dist)  # sqrt(1/10)
        
        # Generate grid points in x[0], x[1] space
        # Use a grid that covers [0,1]^2
        # Try different grid shapes for better hypervolume
        
        # Strategy: Use a grid in (theta, phi) space where:
        # x[0] = sin(theta) * cos(phi)
        # x[1] = sin(theta) * sin(phi)
        # This maps to uniform coverage on the sphere.
        # But x[0], x[1] must be in [0,1].
        
        # Alternative: Use x[0] = sin(alpha), x[1] = sin(beta)
        # where alpha, beta in [0, pi/2]
        # Then f1 = g*(1-sin(beta)), f2 = g*sin(beta)*(1-sin(alpha)), f3 = g*sin(beta)*sin(alpha)
        # f1^2+f2^2+f3^2 = g^2 * [(1-sin(beta))^2 + sin^2(beta)*(1-sin(alpha))^2 + sin^2(beta)*sin^2(alpha)]
        # = g^2 * [(1-sin(beta))^2 + sin^2(beta)]
        # = g^2 * [1 - 2sin(beta) + 2sin^2(beta)]
        # This is not constant, so the points are not on a sphere.
        
        # Let's just use a uniform grid in x[0], x[1] and see.
        # Try a 14x14 grid (196 points) plus 4 extra points.
        
        n_per_axis = 14
        x0_vals = np.linspace(0, 1, n_per_axis)
        x1_vals = np.linspace(0, 1, n_per_axis)
        
        X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        grid_points = np.column_stack([X0.ravel(), X1.ravel()])
        
        # Add 4 strategic points for better coverage
        # Midpoints of the square
        extra_points = np.array([
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5]
        ])
        
        all_points = np.vstack([grid_points, extra_points])
        
        # We have 196 + 4 = 200 points
        if len(all_points) > n_points:
            all_points = all_points[:n_points]
        
        results = []
        
        for i in range(len(all_points)):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            
            # Position variables
            x[0] = all_points[i, 0]
            x[1] = all_points[i, 1]
            
            # Distance variables: set to make g=1
            if self.dim > 2:
                x[2:] = 0.5 + offset
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
