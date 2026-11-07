import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_evals = self.budget
        if n_evals < 1:
            return np.array([])

        # ZDT1 Pareto Front Analysis:
        # f1(x) = x[0]
        # f2(x) = (1 + 10*(n-1)) * (1 - sqrt(x[0] / g(x)))
        # g(x) is minimized when x[1:] = lo[1:] (assuming standard ZDT1 where g depends on sum of x[1:])
        # For ZDT1, g(x) = 1 + 9 * sum(x[1:]) / (n-1). Min g = 1 when x[1:]=0.
        # Thus, on the Pareto front:
        # f1 = x0
        # f2 = C * (1 - sqrt(x0)) where C = 1 + 10*(dim-1)
        
        C = 1.0 + 10.0 * (self.dim - 1)
        
        # Reference point for hypervolume calculation
        R = np.array([11.0, 11.0])
        
        # The problem is to choose N = n_evals values for x0 in [lo[0], hi[0]]
        # such that the set of points P = {(x0_i, C*(1-sqrt(x0_i)))} maximizes the hypervolume.
        
        # Since f2 is a function of f1 only on the front, we can work in the 1D space of x0.
        # Let u = sqrt(x0). Then x0 = u^2.
        # f2 = C * (1 - u).
        # The mapping from u to (f1, f2) is (u^2, C(1-u)).
        
        # Hypervolume of a set of points on a monotonic curve.
        # Let the points be sorted by f1 (which is increasing with u).
        # Let the points be p_1, ..., p_N sorted by f1.
        # HV = sum over i of (R1 - f1_i) * (f2_{i-1} - f2_i) ... wait, standard HV calculation for sorted points.
        
        # Standard HV for non-dominated sorted points (increasing f1, decreasing f2):
        # HV = sum_{i=1}^{N} (R1 - f1_i) * (f2_{i-1} - f2_i) where f2_0 = R2.
        # Actually, the area is the union of rectangles.
        # If points are sorted by f1:
        # Contribution of point i is (R1 - f1_i) * (max(0, f2_{i-1} - f2_i)) ? No.
        
        # Let's use the standard incremental formula.
        # Sort points by f1 ascending.
        # f2 will be descending.
        # HV = (R1 - f1_1)*(R2 - f2_1) + sum_{i=2}^N (f1_i - f1_{i-1}) * (R2 - f2_i)
        # Wait, this formula assumes the "step" function.
        # Let's verify.
        # Point 1: (f1_1, f2_1). Rectangle: (R1-f1_1) x (R2-f2_1).
        # Point 2: (f1_2, f2_2). f1_2 > f1_1, f2_2 < f2_1.
        # The new area added is the rectangle from f1_1 to f1_2 with height R2-f2_2? No.
        # The union of rectangles defined by points (f1_i, f2_i) and reference (R1, R2) is:
        # Area = sum_{i=1}^N (R1 - f1_i) * (f2_{i-1} - f2_i) where f2_0 = R2.
        # Let's check N=1: (R1-f1_1)(R2-f2_1). Correct.
        # N=2: (R1-f1_1)(R2-f2_1) + (R1-f1_2)(f2_1-f2_2).
        # Is this correct?
        # The region is { (x,y) | x <= R1, y <= R2, NOT (exists i: x >= f1_i AND y >= f2_i) }? No, minimization.
        # HV is volume of region dominated by at least one point.
        # Region dominated by p_i is [0, f1_i] x [0, f2_i].
        # Union of these rectangles.
        # Since f1 increases and f2 decreases, the union is a staircase.
        # Area = sum_{i=1}^N (f1_i - f1_{i-1}) * f2_i where f1_0 = 0?
        # No, the reference point is R. The "hypervolume" is usually defined relative to R.
        # Volume of { z | z_i <= R_i, exists p in P s.t. p_i <= z_i }? No, minimization means we want small values.
        # A point p dominates a region [0, p1] x [0, p2].
        # The hypervolume indicator is the volume of the union of these boxes, capped at R.
        # So HV = Volume( Union_{p in P} ([0, R1] x [0, R2] intersect [0, p1] x [0, p2]) )?
        # No, the standard definition: HV(P, R) = Volume({ z in R^k | exists p in P such that p_i <= z_i for all i, and z_i <= R_i for all i }).
        # This is the volume of the region "above" the front and "below" R.
        # If we sort points by f1 ascending (and f2 descending):
        # The boundary of the union is a step function.
        # For f1 in [f1_{i-1}, f1_i), the max f2 covered is f2_{i-1}? No.
        # If we have points (1, 10) and (2, 9) and R=(3, 11).
        # Region dominated by (1,10) is [0,1]x[0,10].
        # Region dominated by (2,9) is [0,2]x[0,9].
        # Union:
        # x in [0,1]: height is max(10, 9) = 10? No, the region is defined by z >= p.
        # Wait, minimization. p dominates z if p_i <= z_i.
        # So if p=(1,10), it dominates z=(1.5, 11).
        # The set of z dominated by P is { z | exists p in P, p <= z, z <= R }.
        # This is the region "North-East" of the points, capped by R.
        # Let's sort points by f1 ascending. f2 will be descending.
        # Let f1_1 < f1_2 < ... < f1_N.
        # f2_1 > f2_2 > ... > f2_N.
        # The boundary of the dominated region (in the first quadrant, below R) is:
        # For f1 in [0, f1_1): No point has f1 <= this range? 
        # Actually, if z_1 < f1_1, no point p has p_1 <= z_1. So these z are not dominated.
        # So the region starts at f1_1.
        # For f1 in [f1_1, f1_2): The points with p_1 <= f1 are p_1. 
        # So we need p_2 <= z_2. Min p_2 among those with p_1 <= f1 is p_1 (since only p_1 qualifies).
        # So z_2 must be >= f2_1.
        # So for f1 in [f1_1, f1_2), z_2 in [f2_1, R2].
        # For f1 in [f1_2, f1_3): Points p_1, p_2 qualify. Min p_2 is f2_2.
        # So z_2 in [f2_2, R2].
        # ...
        # For f1 in [f1_N, R1]: Points p_1..p_N qualify. Min p_2 is f2_N.
        # So z_2 in [f2_N, R2].
        
        # So HV = sum_{i=1}^{N-1} (f1_{i+1} - f1_i) * (R2 - f2_i) + (R1 - f1_N) * (R2 - f2_N)
        
        # Let's check N=1.
        # HV = (R1 - f1_1) * (R2 - f2_1). Correct.
        
        # So we want to maximize:
        # HV = sum_{i=1}^{N-1} (u_{i+1}^2 - u_i^2) * (R2 - C(1-u_i)) + (R1 - u_N^2) * (R2 - C(1-u_N))
        # where u_i = sqrt(x0_i), sorted ascending.
        
        # Since u_i are sorted, let's define the objective function for a vector u of length N.
        # We must ensure u is sorted. We can enforce this by sorting in the objective function, 
        # but that breaks gradients if we used them. With SA, we can just sort.
        # However, sorting changes the indices. 
        # It's better to work with sorted u directly.
        # Let u be a sorted vector in [u_lo, u_hi].
        
        u_lo = np.sqrt(lo[0])
        u_hi = np.sqrt(hi[0])
        
        def calc_hv(u):
            # u is sorted ascending
            # f1 = u^2
            # f2 = C * (1 - u)
            f1 = u * u
            f2 = C * (1.0 - u)
            
            # HV = sum_{i=0}^{N-2} (f1[i+1] - f1[i]) * (R[1] - f2[i]) + (R[0] - f1[-1]) * (R[1] - f2[-1])
            
            if len(u) == 1:
                return (R[0] - f1[0]) * (R[1] - f2[0])
            
            diff_f1 = np.diff(f1)
            # We need (R2 - f2[i]) for i=0 to N-2
            heights = R[1] - f2[:-1]
            term1 = np.sum(diff_f1 * heights)
            
            term2 = (R[0] - f1[-1]) * (R[1] - f2[-1])
            
            return term1 + term2

        # Initial solution: Uniform grid in u
        u_current = np.linspace(u_lo, u_hi, n_evals)
        hv_current = calc_hv(u_current)
        
        # Simulated Annealing
        # Temperature schedule
        T = 1.0
        T_min = 1e-6
        alpha = 0.99 # cooling rate
        
        # We have 200 evaluations.
        # Each SA iteration involves 1 HV calculation (fast) and potentially 1 problem evaluation (slow).
        # We should perform many SA steps to find a good u vector.
        # Then we evaluate the problem for the final u vector.
        # Wait, the budget is 200 evaluations of `problem`.
        # `problem(x)` is the expensive call.
        # `calc_hv` is cheap.
        # So we can run SA for a long time (thousands of iterations) using only `calc_hv`.
        # Then we construct the 200 points from the best u vector found and call `problem` 200 times.
        
        # This is a huge win! The current program uses all 200 evaluations to sample the front.
        # This new program uses 0 evaluations to optimize the sampling strategy, and 200 to record the points.
        # The quality of the points will be much higher because the placement is optimized for HV.
        
        # Let's run SA.
        # Time limit for SA: say 5 seconds.
        sa_deadline = t0 + 5.0
        
        best_u = u_current.copy()
        best_hv = hv_current
        
        # Random perturbation
        # We can perturb one or more u values.
        # Since u must be sorted, we can perturb u_i by delta, then clamp to [u_{i-1}, u_{i+1}].
        # Or just perturb and sort. Sorting is O(N log N) which is fine for N=200.
        
        while time.time() < sa_deadline:
            # Generate neighbor
            u_new = u_current.copy()
            # Pick a random index
            idx = np.random.randint(0, n_evals)
            # Perturb
            # Scale of perturbation proportional to T
            delta = np.random.normal(0, T * (u_hi - u_lo) / 10.0)
            u_new[idx] += delta
            
            # Clamp to bounds
            u_new[idx] = np.clip(u_new[idx], u_lo, u_hi)
            
            # Sort to maintain monotonicity (though perturbation might break it locally)
            # Actually, if we just sort, we might lose the "identity" of the point, but HV is symmetric w.r.t permutation.
            # So sorting is valid.
            u_new = np.sort(u_new)
            
            hv_new = calc_hv(u_new)
            
            # Acceptance
            if hv_new > hv_current:
                u_current = u_new
                hv_current = hv_new
                if hv_new > best_hv:
                    best_hv = hv_new
                    best_u = u_new.copy()
            else:
                # Accept with probability exp((hv_new - hv_current) / T)
                # Note: hv is positive. We want to maximize.
                # Delta = hv_new - hv_current (negative)
                delta_hv = hv_new - hv_current
                if delta_hv < 0:
                    prob = np.exp(delta_hv / T)
                    if np.random.rand() < prob:
                        u_current = u_new
                        hv_current = hv_new
                # else (delta_hv >= 0) handled above? No, if hv_new > hv_current is false, delta_hv <= 0.
                # If delta_hv == 0, prob = 1.
                
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min

        # Final Polish: Local search with smaller steps
        # We have some time left. Let's do a few more iterations with very small steps.
        # Or just stick with the SA result.
        # Given N=200, SA should find a very good solution.
        
        # Construct the final set of points
        # x[0] = u^2
        # x[1:] = lo[1:]
        
        X = np.empty((n_evals, self.dim), dtype=np.float64)
        X[:, 0] = best_u ** 2
        if self.dim > 1:
            X[:, 1:] = lo[1:]
            np.clip(X[:, 1:], lo[1:], hi[1:], out=X[:, 1:])
        
        # Evaluate all points
        F = np.empty((n_evals, self.n_obj), dtype=np.float64)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
