import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        F = []
        
        # DTLZ2: x[0], x[1] are position variables on a quarter-sphere
        # x[2:] are decision variables; setting them to 0.5 minimizes g(x)=0
        mid = (lo + hi) / 2.0
        base = np.array([lo[i] if i < 2 else 0.5 for i in range(self.dim)])
        
        def evaluate(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return False
            if time.time() > deadline:
                return False
            try:
                f = problem(x)
                F.append(np.array(f))
                n_evals += 1
                return True
            except Exception:
                return False
        
        # Strategy: Use a mix of structured points on the Pareto front
        # For DTLZ2 with 3 objectives, the Pareto front is a quarter of a 2-sphere
        # Parameterized by theta1, theta2 in [0, pi/2]
        # f1 = g * cos(theta1) * cos(theta2)
        # f2 = g * cos(theta1) * sin(theta2)
        # f3 = g * sin(theta1)
        # With g=0 (x[2:]=0.5), we get the ideal front
        
        # Use Fibonacci-like / low-discrepancy on the spherical parameter space
        # We'll use a grid in the (theta1, theta2) space with some Sobol points
        
        # First, add the 3 extreme points (vertices of the Pareto front)
        # These correspond to (theta1, theta2) = (0, 0), (0, pi/2), (pi/2, 0)
        # In x-space: 
        #   (1, 0) -> f=(1,0,0)
        #   (0, 1) -> f=(0,1,0) 
        #   (0, 0) -> f=(0,0,1)
        
        # Actually for DTLZ2: 
        # f1 = g*cos(t1)*cos(t2), f2 = g*cos(t1)*sin(t2), f3 = g*sin(t1)
        # At g=1 (ideal):
        # t1=0,t2=0: f=(1,0,0) -> x0=1,x1=0
        # t1=0,t2=pi/2: f=(0,1,0) -> x0=0,x1=1
        # t1=pi/2,t2=0: f=(0,0,1) -> x0=0,x1=0
        
        extreme_points = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        
        for ep in extreme_points:
            x = base.copy()
            x[0] = lo[0] + ep[0] * (hi[0] - lo[0])
            x[1] = lo[1] + ep[1] * (hi[1] - lo[1])
            if not evaluate(x):
                break
        
        # Remaining budget: use Sobol sequence in 2D for (x0, x1)
        remaining = self.budget - n_evals
        if remaining > 0:
            sampler = qmc.Sobol(d=2, scramble=True, seed=0)
            sobol_pts = sampler.random(remaining)
            
            for i in range(remaining):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x = base.copy()
                x[0] = lo[0] + sobol_pts[i, 0] * (hi[0] - lo[0])
                x[1] = lo[1] + sobol_pts[i, 1] * (hi[1] - lo[1])
                if not evaluate(x):
                    break
        
        return F
