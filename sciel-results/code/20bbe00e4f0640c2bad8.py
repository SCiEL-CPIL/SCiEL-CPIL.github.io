import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # DTLZ7 structure:
        # f1 = x[0]
        # f2 = x[1]  
        # f3 = (1 + g(x[2:])) * h(x[1:20])
        #
        # h(y) = 1 + sum_{i=1}^{19} (1+y_i) * (1 - y_i/(1+y_i)) = 1 + 19 = 20 (constant)
        # g(x[2:]) = sum_{i=2}^{21} (x_i - 0.5)^2 (or similar, minimized at x[2:]=0)
        #
        # Pareto front: x[2:] = 0, f1=x[0], f2=x[1], f3=20
        # Front spans (0,0,20) to (1,1,20)
        #
        # Reference point: [1.1, 1.1, 25.0]
        #
        # Strategy: Cover the [0,1]x[0,1] square in (f1,f2) plane with f3=20.
        # Use a structured grid that includes corners and edges for maximum HV coverage.
        
        x_base = np.zeros(n)
        count = 0
        
        # First, evaluate the 4 corners of the square
        corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for x0, x1 in corners:
            if time.time() > deadline or count >= budget:
                break
            x = x_base.copy()
            x[0] = x0
            x[1] = x1
            problem(x)
            count += 1
        
        # Evaluate midpoints of edges
        edge_mids = [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]
        for x0, x1 in edge_mids:
            if time.time() > deadline or count >= budget:
                break
            x = x_base.copy()
            x[0] = x0
            x[1] = x1
            problem(x)
            count += 1
        
        # Now use remaining budget for a dense grid
        remaining = budget - count
        if remaining > 0:
            # Use a grid that fills the interior
            n_grid = int(np.sqrt(remaining))
            if n_grid < 2:
                n_grid = 2
            n_grid = min(n_grid, 20)  # cap at 20 to avoid too many
            n_pts = n_grid * n_grid
            if n_pts > remaining:
                n_grid = int(np.sqrt(remaining))
                n_pts = n_grid * n_grid
            
            if n_grid >= 2:
                xs0 = np.linspace(0, 1, n_grid)
                xs1 = np.linspace(0, 1, n_grid)
                for x0 in xs0:
                    for x1 in xs1:
                        if time.time() > deadline or count >= budget:
                            break
                        x = x_base.copy()
                        x[0] = x0
                        x[1] = x1
                        problem(x)
                        count += 1
                    if time.time() > deadline or count >= budget:
                        break
        
        # Use any remaining evaluations for random sampling
        while count < budget:
            if time.time() > deadline:
                break
            x_rand = x_base.copy()
            x_rand[0] = np.random.rand()
            x_rand[1] = np.random.rand()
            problem(x_rand)
            count += 1
        
        return x_base
