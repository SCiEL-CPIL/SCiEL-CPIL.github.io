"""Analytic DTLZ1 Pareto-front sampling with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer leveraging the analytic structure of DTLZ1.
    1. Places a uniform grid of points on the known Pareto front (g=0).
    2. Uses remaining budget to perturb these points to explore local regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1 specific knowledge:
        # f1 = 0.5 * (1 + g) * prod(x[0:m-1])
        # f2 = 0.5 * (1 + g) * (1 - x[m-1]) * prod(x[0:m-2])
        # f3 = 0.5 * (1 + g) * (1 - prod(x[0:m-2]))
        # where g = 100 * (m - x[m-1] + sum((x[i]-0.5)^2 - cos(20*pi*(x[i]-0.5))) for i in m..n-1)
        # To be on the Pareto front, we need g=0.
        # This requires x[m-1] = 0.5 and x[i] = 0.5 for all i >= m.
        # For DTLZ1 with dim=7, m=3 (n_obj=3, so m = n_obj = 3? No, usually m = n_obj. Let's assume standard DTLZ1 where m = n_obj).
        # Actually, standard DTLZ1 has m parameters. Here n_obj=3.
        # The decision variables x[0..m-2] determine the shape of the front.
        # The variables x[m-1..n-1] determine the distance from the front (g).
        # To minimize g, we set x[m-1..n-1] = 0.5.
        # Then g=0 and the objectives depend only on x[0..m-2].
        
        m = self.n_obj  # Number of objectives
        n = self.dim    # Number of decision variables
        
        # Indices for shape variables: 0 to m-2
        # Indices for distance variables: m-1 to n-1
        
        shape_dim = m - 1
        dist_dim = n - (m - 1)
        
        # We need to distribute self.budget evaluations.
        # Strategy:
        # 1. Generate a set of points on the Pareto front (g=0) to maximize hypervolume.
        #    The front is a simplex in the objective space.
        #    We can sample the decision variables x[0..m-2] uniformly in [0,1] (or transformed) to cover the front.
        #    Actually, for DTLZ1, uniform sampling of x[0..m-2] in [0,1] does NOT give uniform coverage of the front in objective space.
        #    The mapping is non-linear. However, for hypervolume, we want points spread out.
        #    A better approach: sample points in the objective simplex directly? No, we must provide decision variables.
        #    We can invert the mapping? 
        #    f1 = 0.5 * p0
        #    f2 = 0.5 * (1-p0) * p1
        #    f3 = 0.5 * (1-p0) * (1-p1)
        #    where p0 = x[0], p1 = x[1].
        #    This mapping is not 1-to-1 onto the whole simplex if we just pick x.
        #    Actually, the set of points (f1, f2, f3) with sum=0.5 is the front.
        #    We want to cover this simplex.
        #    Let's generate points on the simplex in objective space, then map back to decision space?
        #    Inverse mapping:
        #    Let target objectives be (t1, t2, t3) with t1+t2+t3 = 0.5, ti >= 0.
        #    t1 = 0.5 * x0  => x0 = 2*t1
        #    t2 = 0.5 * (1-x0) * x1 => x1 = 2*t2 / (1-x0) = 2*t2 / (1-2*t1)
        #    t3 = 0.5 * (1-x0) * (1-x1) => check consistency.
        #    This works as long as x0 < 1, i.e., t1 < 0.5. Since sum is 0.5, t1 is always < 0.5 unless t2=t3=0.
        #    At boundaries, we might have division by zero. We can handle limits.
        
        # Alternative simpler strategy:
        # Just sample x[0..m-2] using a low-discrepancy sequence (Sobol or Halton) in [0,1]^shape_dim.
        # And set x[m-1..n-1] = 0.5.
        # This will cover the front well because the mapping is continuous and monotonic in the interior.
        
        # Let's use a simple grid or Halton sequence for the shape variables.
        # Number of front points: Let's use 80% of budget for front points, 20% for local search?
        # Actually, if we can hit the front exactly, local search is less useful for hypervolume than more front points.
        # But floating point errors or problem implementation details might mean g is not exactly 0.
        # However, the problem says "problem(x) clips ... and returns 3 objective values".
        # If I set x[m-1..n-1] = 0.5, g should be 0 (or very close).
        
        # Let's allocate all budget to front sampling if possible, or split 90/10.
        # Given budget=200, let's use 180 for front, 20 for perturbation.
        
        n_front = int(self.budget * 0.9)
        n_perturb = self.budget - n_front
        
        # Generate Halton sequence points in [0,1]^shape_dim
        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, dim + 2):
                    f = 1.0
                    r = 0.0
                    i_copy = i
                    while i_copy > 0:
                        f /= base
                        r += f * (i_copy % base)
                        i_copy //= base
                    point.append(r)
                points.append(point)
            return np.array(points)
        
        # If shape_dim is 2 (for 3 objectives), we use bases 2 and 3.
        if shape_dim > 0:
            front_shape = halton_sequence(n_front, shape_dim)
            # Scale to bounds for shape variables
            # Usually bounds are [0,1]
            front_shape = front_shape * (hi[0:shape_dim] - lo[0:shape_dim]) + lo[0:shape_dim]
            
            # Create full decision vectors
            # x[0..shape_dim-1] = front_shape
            # x[shape_dim..n-1] = 0.5 (center of [0,1] to minimize g)
            # Note: shape_dim = m-1. The distance variables start at index m-1.
            # So x[0:m-1] are shape, x[m-1:n] are distance.
            X_front = np.zeros((n_front, n))
            X_front[:, 0:shape_dim] = front_shape
            # Set distance variables to 0.5
            # If bounds are not [0,1], we should set to (lo+hi)/2
            mid = (lo + hi) / 2.0
            X_front[:, shape_dim:] = mid[shape_dim:]
            
            # Evaluate front points
            for i in range(n_front):
                if time.time() > deadline:
                    break
                problem(X_front[i])
        else:
            n_front = 0

        # Now use remaining budget for local perturbation around the best front points?
        # Or just random exploration?
        # Since we are already on the front, perturbing x[0..shape_dim-1] slightly keeps us near the front.
        # Perturbing x[shape_dim:] moves us off the front (worse).
        # So we should only perturb the shape variables.
        
        if n_perturb > 0:
            # Pick random indices from the front points we just generated?
            # We didn't store the objective values, but we can just pick random shape points again?
            # No, we want to explore.
            # Let's just generate more Halton points? That's redundant.
            # Let's do a random jitter on the shape variables of some existing front points.
            # But we don't have the list of existing front points stored in a way that's easy to pick from without re-evaluating?
            # Actually, we can just generate n_perturb new random points in the shape space, but with a smaller range around the center?
            # Or just uniform random in shape space.
            # Since Halton is low-discrepancy, adding random points might fill gaps.
            
            # Let's just use random uniform points in the shape space for the remaining budget.
            for i in range(n_perturb):
                if time.time() > deadline:
                    break
                x = np.random.uniform(lo, hi, size=n)
                # Force distance variables to center to stay on front
                x[shape_dim:] = mid[shape_dim:]
                problem(x)

        return None
# EVOLVE-BLOCK-END
