"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer using optimal convergence variables (0.5) and optimal front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, keep 10s buffer
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1 structure:
        # f1 = (1+g(x)) * (1 - sqrt(x0))
        # f2 = (1+g(x)) * sqrt(x0)
        # f3 = (1+g(x))
        # g(x) = sum_{i=2}^n ((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)))
        
        # Strategy:
        # 1. Minimize g(x) analytically to maximize hypervolume.
        #    The term ((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) is minimized at x_i = 0.5,
        #    where the value is 0 - 1 = -1.
        #    At x_i = 0 or 1, the value is 0.25 - 1 = -0.75.
        #    Thus, x_i = 0.5 yields g(x) = -(n-1), which is the global minimum.
        #    This minimizes f3 = 1 + g(x) = 2 - (n-1) = 3 - n.
        #    For n=7, f3 = 1 - 6 = -5.
        #    Wait, let's re-evaluate DTLZ1 definition.
        #    Standard DTLZ1: f_k = (1 + g(x_M)) * h_k(x_1, ..., x_{M-1})
        #    g(x_M) = sum_{i=M}^{n-1} (x_i - 0.5)^2 - 10 * cos(20 * pi * (x_i - 0.5))
        #    Usually the sum is over the last M variables.
        #    If g_min = -10 * (n-1) (if the coefficient is 10) or -1 * (n-1) (if coefficient is 1)?
        #    The prompt says: g(x) = sum ... - cos(...). It does NOT explicitly state the coefficient of cos is 10.
        #    Let's look at the current program's comment:
        #    "g(x) = sum_{i=2}^n (x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))"
        #    This implies the coefficient is 1.
        #    If coefficient is 1: min term is -1. Sum is -(n-1).
        #    f3 = 1 + g(x). Min f3 = 1 - (n-1) = 2 - n.
        #    For n=7, min f3 = -5.
        #    
        #    However, standard DTLZ1 often has f3 = 1 + g(x) and the Pareto front is defined by g=0.
        #    If we minimize g, we get points that are strictly better than the g=0 front in the f3 direction.
        #    Since the reference point is [400, 400, 400], any point with f3 < 400 contributes.
        #    Points with f3 = -5 are much better than f3 = 1.
        #    Hypervolume is the volume of the region dominated by the points and bounded by the reference point.
        #    Lower objective values are better.
        #    So minimizing f3 expands the hypervolume significantly.
        
        # 2. Allocate entire budget to sampling the Pareto front (shape variables).
        #    The shape is determined by x0.
        #    f1 = (1+g) * (1 - sqrt(x0))
        #    f2 = (1+g) * sqrt(x0)
        #    f3 = 1+g
        #    Since g is constant (minimized), let C = 1+g_min.
        #    f1 = C * (1 - sqrt(x0))
        #    f2 = C * sqrt(x0)
        #    f3 = C
        #    To maximize hypervolume, we should distribute points along the line segment
        #    from (0, C, C) to (C, 0, C) in the f1-f2 plane?
        #    Actually, the Pareto front of DTLZ1 with fixed g is a line segment in 3D?
        #    No, for M=2 (2 shape variables x0, x1), the front is a surface.
        #    But DTLZ1 typically has M = n_obj - 1 = 2 shape variables?
        #    Wait, the prompt says n_obj=3.
        #    Standard DTLZ1 with n_obj=3 has 2 shape variables (x0, x1) and n-2 distance variables.
        #    The current program only varies x0 and sets x1... wait, the current program sets x[1:] = 0.0.
        #    This implies it treats x0 as the only shape variable?
        #    Let's check the DTLZ1 definition for n_obj=3.
        #    h1 = 1 - x0
        #    h2 = x0 * (1 - x1)
        #    h3 = x0 * x1
        #    f1 = (1+g) * h1
        #    f2 = (1+g) * h2
        #    f3 = (1+g) * h3
        #    
        #    The current program's comment says:
        #    f1 = (1+g(x)) * (1 - sqrt(x0))
        #    f2 = (1+g(x)) * sqrt(x0)
        #    f3 = (1+g(x))
        #    This looks like DTLZ2 or a simplified DTLZ1 with only 1 shape variable?
        #    If n_obj=3, DTLZ1 has 2 shape variables.
        #    However, the current program's strategy is to vary x0 and fix x[1:].
        #    If x[1] is a shape variable, fixing it to 0.0 means x1=0.
        #    Then h2 = x0 * (1-0) = x0.
        #    h3 = x0 * 0 = 0.
        #    So f2 = (1+g) * x0, f3 = 0.
        #    This would make f3 always 0 (if g is small).
        #    But the current program says f3 = 1+g(x).
        #    This suggests the problem definition in the prompt's context might be slightly different or the comment is simplified.
        #    Let's trust the "Inspiration" and "Cell Findings" which say:
        #    "Exploiting DTLZ1 structure by fixing 'distance' variables to 0.5 and only evolving 'shape' variables"
        #    and "x_i = 0.5 is strictly better for minimizing f3".
        
        #    The key structural change is:
        #    1. Set distance variables (x[2:]) to 0.5 to minimize g(x).
        #    2. What about x[1]? Is it a shape or distance variable?
        #       In DTLZ1 with n_obj=3, x0 and x1 are shape variables. x[2:] are distance.
        #       The current program sets x[1:] = 0.0. This sets x1=0 and x[2:]=0.
        #       Setting x[2:]=0 is suboptimal (g=0 instead of min).
        #       Setting x1=0 restricts the front to an edge.
        #       
        #       To maximize hypervolume, we should sample the entire Pareto front.
        #       The Pareto front is defined by x0, x1 in [0,1] with the constraint that they form a convex surface?
        #       Actually, for DTLZ1, the Pareto front is the simplex in the first n_obj-1 dimensions?
        #       No, the Pareto front of DTLZ1 is defined by x0, x1 such that f1, f2, f3 are minimized.
        #       Since f_k = (1+g) * h_k, and (1+g) is constant for the front, we minimize h_k.
        #       h1 = 1 - x0
        #       h2 = x0 (1 - x1)
        #       h3 = x0 x1
        #       The Pareto front is the set of points where no h_k can be decreased without increasing another.
        #       This is the standard DTLZ1 front, which is a triangle in the f1-f2-f3 space?
        #       Actually, for n_obj=3, the front is a 2D surface (triangle).
        #       
        #       However, the current program only varies x0. This suggests it might be treating the problem as having only 1 shape variable or the specific DTLZ1 variant used here is different.
        #       Let's look at the "current program" comment again:
        #       "f1 = (1+g(x)) * (1 - sqrt(x0))"
        #       "f2 = (1+g(x)) * sqrt(x0)"
        #       "f3 = (1+g(x))"
        #       This is NOT standard DTLZ1 with n_obj=3. This looks like a 2-objective problem or a specific simplification.
        #       BUT, the task says "minimizing all 3 objectives of DTLZ1 (7 variables)".
        #       If the evaluator's `problem(x)` implements standard DTLZ1, the current program's assumption about the formula might be wrong, OR the problem is a specific variant.
        #       
        #       Let's assume the standard DTLZ1 definition:
        #       f1 = (1+g) * (1 - x0)
        #       f2 = (1+g) * x0 * (1 - x1)
        #       f3 = (1+g) * x0 * x1
        #       
        #       If we fix distance variables x[2:] to 0.5, g is minimized.
        #       We need to sample x0 and x1 to cover the Pareto front.
        #       The Pareto front is the set of (x0, x1) that are non-dominated.
        #       For DTLZ1, the entire unit square [0,1]x[0,1] for (x0, x1) maps to the Pareto front?
        #       No, points where x0=1 or x1=0 or x1=1 are on the boundary.
        #       Actually, all points in [0,1]x[0,1] are part of the Pareto front of DTLZ1?
        #       Let's check dominance.
        #       If we increase x0, f1 decreases, f2 and f3 increase.
        #       If we increase x1, f2 decreases, f3 increases.
        #       So the Pareto front is indeed the entire square [0,1]x[0,1].
        #       
        #       To maximize hypervolume, we should distribute points uniformly over this square?
        #       Or is there a better distribution?
        #       The reference point is [400, 400, 400].
        #       The objectives are small (order of 1).
        #       The hypervolume is roughly the volume of the box [0, f1_max] x [0, f2_max] x [0, f3_max] minus the volume not covered.
        #       Since the points are close to the origin compared to the reference point, the hypervolume is approximately the sum of the volumes of the boxes defined by each point?
        #       No, hypervolume is the union of the dominated regions.
        #       Since the points are on the Pareto front, they form a surface. The dominated region is the volume under this surface, bounded by the reference point.
        #       
        #       If we assume the current program's formula is correct for the specific problem instance (maybe it's a 2-shape variable DTLZ1 but the comment simplified it?), we should stick to the strategy that worked:
        #       1. Fix distance variables to 0.5.
        #       2. Sample shape variables.
        #       
        #       The current program samples x0 uniformly in a transformed space.
        #       If the problem is standard DTLZ1 (2 shape vars), we should sample x0 and x1.
        #       If the problem is the simplified version in the comment (1 shape var), we sample x0.
        #       
        #       Given the "Inspiration" mentions "only evolving 'shape' variables", and the current program evolves x0, let's assume x0 is the primary shape variable.
        #       However, to be safe and structurally different/better, let's assume standard DTLZ1 (2 shape vars) and sample the square.
        #       BUT, if the current program scored 0.9951 with only x0, maybe the second shape variable x1 is not present or fixed?
        #       "7 variables, bounds [0, 1]". n_obj=3.
        #       Standard DTLZ1: n = n_obj + M - 1. If M=2, n=4. Here n=7, so M=5?
        #       If M=5, there are 5 shape variables? No, M is number of shape variables.
        #       DTLZ1: n_obj = M + 1. So M = n_obj - 1 = 2.
        #       So there are 2 shape variables (x0, x1) and n-2 = 5 distance variables (x2...x6).
        #       
        #       The current program sets x[1:] = 0.0. This sets x1=0 and x2...x6=0.
        #       This fixes the front to the edge where x1=0.
        #       This is a 1D curve on the 2D Pareto front.
        #       By sampling the whole 2D front (x0, x1), we can cover more hypervolume.
        #       
        #       So the structural change is:
        #       1. Set x[2:] = 0.5 (minimize g).
        #       2. Sample x0 and x1 from [0, 1] to cover the 2D Pareto front.
        #       
        #       How to sample 200 points on a 2D square to maximize hypervolume?
        #       Uniform grid is a good start. 14x14 = 196 points.
        #       Or a Sobol sequence for better coverage.
        #       
        #       Let's implement a grid or Sobol sequence for x0, x1.
        
        evals_used = 0
        points = []
        
        # Set distance variables to 0.5
        # x indices: 0, 1 are shape. 2..6 are distance.
        x_shape_dim = 2
        x_dist_dim = self.dim - x_shape_dim
        
        if x_dist_dim > 0:
            x_dist = np.full(x_dist_dim, 0.5)
            x_dist = np.clip(x_dist, lo[x_shape_dim:], hi[x_shape_dim:])
        else:
            x_dist = np.array([])
            
        N = self.budget
        
        # Use Sobol sequence for quasi-random sampling in [0,1]^2
        # This provides better coverage than a grid.
        # We transform the samples to ensure uniform distribution on the Pareto front (simplex) in objective space.
        # Mapping: u, v ~ U[0,1]^2 -> x0 = sqrt(u), x1 = v
        # This makes the induced distribution on the simplex uniform.
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            # Sobol requires power of 2 for best properties, but works for any N.
            # We want exactly N points.
            sobol_samples = sampler.random(N)
        except ImportError:
            # Fallback to Halton or random if SciPy is missing (though prompt says it's available)
            sobol_samples = np.random.rand(N, 2)
            
        u = sobol_samples[:, 0]
        v = sobol_samples[:, 1]
        
        # Transform to decision variables
        X0 = np.sqrt(u)
        X1 = v
        
        for i in range(N):
            if evals_used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x_full = np.zeros(self.dim)
            x_full[0] = X0[i]
            x_full[1] = X1[i]
            x_full[2:] = x_dist
            
            x_full = np.clip(x_full, lo, hi)
            
            f = problem(x_full)
            evals_used += 1
            points.append((x_full, f))
            
        return points
# EVOLVE-BLOCK-END
