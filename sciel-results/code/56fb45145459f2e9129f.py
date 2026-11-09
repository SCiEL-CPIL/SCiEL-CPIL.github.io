"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    DTLZ1 Optimizer using Sobol sampling on the Pareto front manifold.
    
    Strategy:
    1. Assume standard DTLZ1 structure: first M-1 variables are 'position' (simplex coords),
       remaining are 'distance' (g function).
    2. Phase 1: Use 150 evaluations to sample the Pareto front densely using Sobol sequences
       in the position space, setting distance variables to their lower bound (assuming g_min at bound).
    3. Phase 2: Use remaining 50 evaluations to refine. We estimate the current Pareto front 
       and identify "gaps" in the objective space. We place new points by perturbing the 
       position variables of existing points that are likely to fill these gaps, or by 
       doing a local search around the most "central" points if we suspect the distance 
       assumption is slightly off (though for DTLZ1, g=0 is usually exact if bounds allow).
       
    Actually, for DTLZ1, if we set distance vars to 0 (or lo), we are exactly on the front.
    The only thing to optimize is the distribution of points on the front to maximize hypervolume.
    A uniform grid in the simplex is good, but Sobol might be slightly better for low N.
    However, the "adaptive" part is key. If we can identify which parts of the front are 
    under-represented, we can add points there.
    
    Hypervolume contribution of a point depends on the "box" it dominates.
    For a 3-objective problem, the hypervolume is the union of boxes from each point to the ref point.
    Points in the "center" of the front (balanced objectives) often contribute more to the 
    volume near the reference point if the reference point is far away? 
    Actually, with ref [400, 400, 400], the entire front is "small" compared to the ref.
    The hypervolume is roughly the volume of the convex hull of the points and the ref point?
    No, it's the union of hyperrectangles.
    
    For a dense sample, the HV is dominated by the outermost points.
    A uniform distribution on the simplex maximizes the coverage.
    
    Let's stick to a high-quality quasi-random sequence (Sobol) for the position variables.
    We will use all 200 evaluations for the front sampling.
    Why? Because refining distance variables is likely unnecessary if we assume g_min=0.
    If g_min != 0, we would be off the front, but the task says "DTLZ1", implying standard structure.
    The "adaptive" part can be a simple check: if the first few points show large g (distance), 
    we might need to optimize distance. But for DTLZ1 with bounds [0,1], x_d=0 gives g=0.
    
    So, the strategy is:
    1. Generate 200 Sobol points in [0,1]^(M-1).
    2. Convert to simplex coordinates? No, DTLZ1 usually takes x_1..x_{M-1} in [0,1] 
       and maps them to the front via sin/cos or linear?
       Standard DTLZ1:
       f_1 = 0.5 * (1+g) * x_1
       f_2 = 0.5 * (1+g) * (1-x_1) * x_2
       f_3 = 0.5 * (1+g) * (1-x_1) * (1-x_2)
       
       If we set g=0 (by x_d=0), then:
       f_1 = 0.5 * x_1
       f_2 = 0.5 * (1-x_1) * x_2
       f_3 = 0.5 * (1-x_1) * (1-x_2)
       
       The range of objectives is [0, 0.5].
       Reference point is [400, 400, 400].
       The hypervolume is dominated by the volume of the box defined by the "worst" points 
       (largest objectives) and the ref point.
       Actually, since all points are much smaller than 400, the HV is approximately:
       Sum over points of (400-f_1)(400-f_2)(400-f_3) ... but they overlap.
       The union volume is determined by the "envelope" of the points.
       To maximize the union volume of boxes extending to (400,400,400), we want points 
       that are as "large" as possible in each dimension, but since they are on the front 
       (sum is fixed roughly), we want to cover the entire front.
       
       Specifically, the hypervolume of a set of points P with ref R is the volume of 
       union_{p in P} [p_1, R_1] x [p_2, R_2] x [p_3, R_3].
       This is maximized if the points "spread out" to cover the entire simplex.
       
       A uniform grid or Sobol sequence on the simplex is optimal.
       
    Let's implement Sobol sampling for the position variables (x_1, x_2) in [0,1].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        n_pos = n_obj - 1
        n_dist = dim - n_pos

        # Generate Sobol sequence for position variables
        # We want `budget` points.
        # Sobol generator requires power of 2 for efficiency, but we can just take the first N.
        sampler = qmc.Sobol(d=n_pos, scramble=True)
        
        # Generate slightly more than needed to ensure we have enough, 
        # but we can just generate `budget` directly.
        # Note: Sobol in scipy returns points in [0,1].
        sobol_points = sampler.random(self.budget)
        
        # Map to bounds [lo, hi] for position variables
        pos_points = lo[:n_pos] + (hi[:n_pos] - lo[:n_pos]) * sobol_points
        
        # Construct full candidates
        # Distance variables set to lower bound (assuming g_min at lo)
        candidates = np.zeros((self.budget, dim))
        candidates[:, :n_pos] = pos_points
        candidates[:, n_pos:] = lo[n_pos:]
        
        # Evaluate all points
        # We must be careful with time. 200 evaluations should be fast.
        for i in range(self.budget):
            if time.time() > deadline:
                break
            problem(candidates[i])
            
        return candidates

# EVOLVE-BLOCK-END
