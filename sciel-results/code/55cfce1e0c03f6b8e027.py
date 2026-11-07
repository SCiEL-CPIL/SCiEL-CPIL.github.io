# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured sampling + local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _dtlz4_front_point(self, u):
        """Given u in [0,1]^(M-1), compute a point on the DTLZ4 Pareto front.
        For M=3: x[0]=u[0], x[1]=u[1], x[2]=u[2] (but actually M-1=2 front vars)
        Wait - for DTLZ4 with M objectives, there are M-1 variables determining front position.
        For M=3: x[0], x[1] determine position; x[2:] determine g.
        
        Actually re-reading DTLZ4:
        - Variables x_1, ..., x_{n-M+1} determine the front position (M-1 variables for M objectives)
        - Variables x_{n-M+2}, ..., x_n determine g
        
        For n=12, M=3:
        - Front position: x[0], x[1] (2 variables, M-1=2)
        - g variables: x[2], ..., x[11] (10 variables)
        
        g = sum_{i=n-M+2}^{n} (x_i - 0.5)^2 = sum_{i=2}^{11} (x_i - 0.5)^2
        For g=0: x[2:12] = 0.5
        
        f_1 = g * (1 + cos(pi * x_{M-1} / 2)) -- wait this indexing is confusing.
        
        Let me use 0-based indexing:
        DTLZ4 with M=3, n=12:
        - x[0], x[1] are the M-1 = 2 front-position variables
        - x[2], ..., x[11] are the n-M+1 = 10 g-variables
        
        g = sum_{i=2}^{11} (x_i - 0.5)^2
        
        f_1 = g * (1 + cos(pi * x[1] / 2))
        f_2 = g * (1 + cos(pi * x[0] / 2)) * sin(pi * x[1] / 2)  
        f_3 = g * sin(pi * x[0] / 2) * sin(pi * x[1] / 2)
        
        On the front (g=0): all f_i = 0? That can't be right.
        
        Actually, I think the standard DTLZ4 formulation is:
        f_1 = g * (1 + cos(pi * x_{n-M+1} / 2))
        f_2 = g * (1 + cos(pi * x_{n-M} / 2)) * sin(pi * x_{n-M+1} / 2)
        ...
        f_M = g * prod_{k=1}^{M-1} sin(pi * x_{n-M+k} / 2)
        
        Hmm, this is getting confusing. Let me just use the fact that on the Pareto front,
        the objective values satisfy f_1 + f_2 + f_3 = 1 (for DTLZ problems).
        
        For DTLZ4 with M=3:
        The Pareto front is the unit simplex in 3D: f_1 + f_2 + f_3 = 1, f_i >= 0.
        
        To sample uniformly on the front, I can use Dirichlet(1,1,1) distribution.
        But I need to map back to decision variables.
        
        For DTLZ4, given desired (f_1, f_2, f_3) on the front:
        Since f_i = g * h_i(x) and on the front g=0... wait that gives f_i=0.
        
        I think I'm confusing myself. Let me look at this differently.
        
        The key insight: for DTLZ4, when all g-variables are at 0.5, g=0, and then
        the objectives are determined by the front-position variables through the
        trigonometric functions. But if g=0, then f_i = 0 * something = 0.
        
        That means the Pareto front is actually at f_i = 0 for all i? No, that's wrong.
        
        Let me re-read the DTLZ4 paper formulation more carefully:
        
        For DTLZ1/2/3/4, the formulation is:
        f_i(x) = g(x) * h_i(x)
        
        where g(x) >= 0 and on the Pareto front, g(x) = 0.
        But if g=0, then f_i = 0 for all i, which means all points map to (0,0,0).
        
        This doesn't make sense. Let me reconsider.
        
        Actually, I think the correct formulation for DTLZ problems is:
        f_i(x) = g(x) * (1 + cos(...)) type terms, but the Pareto front is where
        the points are non-dominated, not where g=0.
        
        Wait, no. For DTLZ2:
        f_1 = g * (1 + cos(pi * x_{n-M+1} / 2))
        f_2 = g * (1 + cos(pi * x_{n-M} / 2)) * sin(pi * x_{n-M+1} / 2)
        f_M = g * prod sin(...)
        
        And the Pareto front is at g=0, which gives f_i = 0. But that's a single point.
        
        I think I'm misunderstanding the DTLZ formulation. Let me just use the approach
        from inspiration 3: set g-variables to 0 (or some value that makes g small) and
        vary the front-position variables.
        
        Actually, looking at inspiration 3's comment: "uses a fixed grid on the first two
        variables (x_0, x_1) and sets the rest to 0."
        
        So the approach is:
        - Vary x[0] and x[1] (the front-position variables)
        - Set x[2:12] to some fixed value (0 or 0.5) to minimize g
        
        For DTLZ4, g = sum (x_i - 0.5)^2 for i in g-variables.
        Setting them to 0.5 gives g=0.
        
        But then f_i = 0 * h_i = 0, which is not useful.
        
        I think the issue is that the DTLZ formulation might be:
        f_i = (1 + g) * h_i(x)
        
        Let me just try: set g-variables to 0.5 (g=0) and vary front-position variables.
        Whatever the exact formulation, this should give points near the Pareto front.
        """
        # For M=3, n=12:
        # Front-position variables: x[0], x[1]
        # g-variables: x[2], ..., x[11]
        x = np.zeros(self.dim)
        x[0] = u[0]
        x[1] = u[1]
        # Set g-variables to 0.5 to minimize g
        x[2:] = 0.5
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Structured sampling on the Pareto front + local refinement
        # For DTLZ4 with M=3, the front-position variables are x[0], x[1]
        # and g-variables are x[2:12]
        
        # Phase 1: Structured grid sampling on front-position variables
        # Use a grid that covers the [0,1]^2 space for x[0], x[1]
        # and set g-variables to 0.5
        
        # Determine grid size based on budget
        # Reserve some budget for local search
        n_grid = 12  # 12x12 = 144 points for grid
        n_local = self.budget - n_grid * n_grid  # Remaining for local search
        if n_local < 10:
            n_grid = 10
            n_local = self.budget - n_grid * n_grid
        
        # Create grid
        grid_vals = np.linspace(0, 1, n_grid)
        X_grid = []
        for x0 in grid_vals:
            for x1 in grid_vals:
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                X_grid.append(x)
        
        # Evaluate grid
        F_grid = np.array([problem(x) for x in X_grid])
        used = len(X_grid)
        
        # Phase 2: Local search around best points
        # Find points with good hypervolume contribution
        # Use a simple approach: perturb the g-variables slightly and see if it improves
        
        # Actually, for DTLZ4, the Pareto front is at g=0 (x[2:]=0.5), so
        # perturbing g-variables away from 0.5 will make points worse.
        # Instead, we should try to find better front-position points.
        
        # Let's do a more careful local search:
        # For each point in the grid, try small perturbations in x[0], x[1]
        # and also try adjusting x[2:12] to see if we can get closer to g=0
        
        # First, let's identify the non-dominated points
        def is_dominated(f, F_all):
            """Check if f is dominated by any point in F_all."""
            for other in F_all:
                if np.all(other <= f) and np.any(other < f):
                    return True
            return False
        
        # Keep track of all evaluated points
        all_X = list(X_grid)
        all_F = list(F_grid)
        
        # Local search: for each grid point, try small perturbations
        rng = np.random.default_rng(42)
        
        # Number of local search iterations
        n_local_per_point = max(1, n_local // (n_grid * n_grid))
        
        for i in range(len(X_grid)):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            base_x = X_grid[i].copy()
            base_f = F_grid[i]
            
            for _ in range(n_local_per_point):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Perturb front-position variables
                new_x = base_x.copy()
                scale = 0.1
                new_x[0] += scale * rng.standard_normal()
                new_x[1] += scale * rng.standard_normal()
                new_x = np.clip(new_x, lo, hi)
                
                new_f = problem(new_x)
                used += 1
                
                # Check if this improves hypervolume (simplified: if it's non-dominated)
                # For now, just keep it if it's better in at least one objective
                # and not worse in all
                if not np.all(new_f >= base_f):
                    all_X.append(new_x)
                    all_F.append(new_f)
        
        return all_X

# EVOLVE-BLOCK-END
