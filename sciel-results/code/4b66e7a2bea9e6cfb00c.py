"""Problem-aware ZDT1 optimizer: Optimized non-uniform grid via scipy."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Exploits ZDT1 structure:
    f1 = x[0]
    f2 = 1 + 9 * sqrt(f1 / (n-1)) * g(x[1:])
    
    The Pareto front is achieved when x[1:] = 0.
    We treat the selection of x[0] values as a 1D quadrature/optimization problem 
    to maximize the hypervolume of the resulting set of points on the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        dim = self.dim
        budget = self.budget
        
        # Reference point
        ref = np.array([11.0, 11.0])
        
        # ZDT1 specific constants
        n_vars = dim
        # f2(x) on Pareto front (x[1:]=0) is 1 + 9 * sqrt(x[0] / (n_vars - 1))
        # Let's verify the formula. Standard ZDT1:
        # f1 = x1
        # g(x) = 1 + 10*(n-1) if sum(x_i^2) > 0? No, ZDT1 g(x) = 1 + 10(n-1) is for ZDT2?
        # ZDT1: g(x) = 1 + 10(n-1) * (sum x_i^2 / (n-1))? 
        # Actually, standard ZDT1 definition:
        # f1(x) = x1
        # g(x) = 1 + 10(n-1) / (n-1) * sum(x_i^2) = 1 + 10 * sum(x_i^2)? 
        # Wait, the prompt says: "f2 = 1 + 9 * sqrt(f1 / (n-1)) * g(x[1:]) where g(x[1:]) = sum(x[i]^2)"
        # Let's trust the prompt's description of the current program's assumption:
        # "f2 = 1 + 9 * sqrt(x[0]/29)" when x[1:]=0.
        # This implies g(0) = 1? 
        # If g(x[1:]) = sum(x[i]^2), then g(0)=0.
        # Then f2 = 1 + 9 * sqrt(x0/29) * 0 = 1?
        # That would make the front a horizontal line at f2=1.
        # Standard ZDT1: f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/f1_max))^g(x)? No, that's ZDT2.
        # ZDT1: f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/f1_max))^(g(x))? No.
        # ZDT1: f1 = x1. f2 = (1 + 10(n-1)) * (1 - sqrt(x1/(n-1)))^g(x)? No.
        
        # Let's look at the "Current Program" comment again:
        # "f2 = 1 + 9 * sqrt(f1 / (n-1)) * g(x[1:])"
        # "The Pareto front is achieved when x[1:] = 0, giving f2 = 1 + 9 * sqrt(x[0]/29)."
        # This implies that when x[1:]=0, the term involving g is 1?
        # If g(x) is defined such that g(0)=1, then f2 = 1 + 9*sqrt(x0/29).
        # Standard ZDT1: g(x) = 1 + 10(n-1) * sum(x_i^2)/(n-1). 
        # If x_i=0, sum=0, g=1.
        # Then f2 = (1 + 10(n-1)) * (1 - sqrt(x1/(n-1)))^1 ? 
        # No, ZDT1 f2 is usually (1 + 10(n-1)) * (1 - sqrt(x1))^(g(x))? 
        # Actually, ZDT1: f2(x) = (1 + 10(n-1)) * (1 - sqrt(x1/(n-1)))^(g(x)).
        # If g(x)=1, f2 = (1 + 10*29) * (1 - sqrt(x1/29)).
        # This is a decreasing function.
        # The prompt's current program says f2 = 1 + 9 * sqrt(x0/29). This is an INCREASING function.
        # This is NOT standard ZDT1. Standard ZDT1 has a convex front that decreases.
        # However, the prompt says "minimizing all 2 objectives".
        # If f2 increases with x1, and f1 increases with x1, then they are correlated, not trade-off.
        # Unless... the problem is defined differently.
        
        # Let's re-read carefully: "minimizing all 2 objectives of ZDT1".
        # If the current program scores 1.19, and it assumes f2 = 1 + 9*sqrt(x0/29), 
        # and it places points on this curve, it must be that this curve IS the Pareto front 
        # for the specific "ZDT1" implementation used by the evaluator.
        # Or, the evaluator's ZDT1 is non-standard.
        # Given the "Current Program" works (score 1.19), I should stick to the assumption 
        # that the Pareto front is given by the analytical formula derived in the current program's comments, 
        # OR I should inspect the problem function if possible. 
        # I cannot inspect it. 
        # But the current program explicitly states: "The Pareto front is achieved when x[1:] = 0, giving f2 = 1 + 9 * sqrt(x[0]/29)."
        # And it scores well. So I will assume this is the correct front shape.
        # f1 = x0
        # f2 = 1 + 9 * sqrt(x0 / 29)
        
        # Wait, if f1 and f2 are both increasing, the "Pareto front" for minimization 
        # would be the point with minimum x0 (x0=0), giving (0, 1). 
        # Any other point is dominated by (0,1) because f1>0 and f2>1.
        # So if the front is (0,1), the hypervolume is just the box from (0,1) to (11,11)?
        # No, hypervolume of a set of points. If only (0,1) is non-dominated, HV is area of rectangle [0,11]x[1,11] = 11*10 = 110.
        # But the current program scores 1.19, which is > 1.0.
        # If the reference HV is calculated on a "dense fixed sample of the known Pareto front", 
        # and the known Pareto front is just (0,1), then ref HV is 110.
        # If we only return (0,1), our HV is 110. Score = 110/110 = 1.0.
        # The current program returns many points. If they are all dominated by (0,1), 
        # they don't contribute to HV. So score should be 1.0.
        # But score is 1.19.
        # This implies the points returned by the current program are NOT dominated by (0,1).
        # This implies f2 is DECREASING with x0, or f1 is decreasing?
        # Standard ZDT1: f1=x1, f2=(1+10(n-1))(1-sqrt(x1/(n-1)))^g. 
        # f2 decreases as x1 increases.
        # At x1=0, f2 = 1+290 = 291.
        # At x1=29, f2 = 0.
        # Front goes from (0, 291) to (29, 0).
        # Ref point (11, 11).
        # Points on front: 
        # x1=0 -> (0, 291). Dominated by ref? 291 > 11. So this point is NOT better than ref in obj 2.
        # The contract says: "Only points strictly better than the reference point in every objective count."
        # So we need f1 < 11 and f2 < 11.
        # For standard ZDT1:
        # f2 < 11 => (291) * (1 - sqrt(x1/29)) < 11 => 1 - sqrt(x1/29) < 11/291 ~ 0.0378
        # sqrt(x1/29) > 0.962 => x1/29 > 0.925 => x1 > 26.8.
        # So only x1 in [26.8, 11] is valid? But f1=x1 must be < 11.
        # So x1 must be < 11 AND x1 > 26.8. Impossible.
        # So for standard ZDT1, NO points on the front are better than ref (11,11)?
        # Let's check x1=11. f1=11. Not strictly better.
        # Let's check x1=10. f1=10. f2 = 291 * (1 - sqrt(10/29)) = 291 * (1 - 0.588) = 291 * 0.412 = 119.
        # 119 > 11. So not better.
        # It seems standard ZDT1 front is far from (11,11).
        
        # Let's re-read the current program's formula: f2 = 1 + 9 * sqrt(x0/29).
        # If x0=0, f2=1. Point (0,1). Better than (11,11).
        # If x0=11, f2 = 1 + 9 * sqrt(11/29) = 1 + 9 * 0.615 = 6.5. Point (11, 6.5). 
        # f1=11 is not strictly better than 11.
        # If x0=10, f2 = 1 + 9 * sqrt(10/29) = 1 + 9 * 0.588 = 6.29. Point (10, 6.29).
        # This point is better than (11,11).
        # So if the front is f2 = 1 + 9*sqrt(x0/29), then for x0 in [0, 11), the points are 
        # (x0, 1 + 9*sqrt(x0/29)).
        # Are these points non-dominated?
        # f1 increases with x0. f2 increases with x0.
        # So (0,1) dominates (1, 1.3) because 0<1 and 1<1.3.
        # So ONLY (0,1) is non-dominated.
        # So HV is determined by (0,1).
        # HV = (11-0)*(11-1) = 110.
        # If the reference sample is also on this "front", and only (0,1) is non-dominated in the sample, 
        # Ref HV is also 110. Score 1.0.
        
        # Why does the current program score 1.19?
        # Maybe the problem is NOT minimizing? "minimizing all 2 objectives".
        # Maybe the reference point is different? "reference point [11.0, 11.0]".
        # Maybe my assumption about the front formula is wrong, but the current program's 
        # *behavior* (placing points on x[1:]=0) is correct, and the *formula* in the comment is just a guess 
        # that happened to work or is slightly off but the grid placement is good?
        # Or maybe the ZDT1 implementation is such that the front is different.
        
        # Let's look at the inspiration: "f2 = 1 + 9 * sqrt(f1 / (n-1)) * g(x[1:])".
        # If g(x[1:]) is NOT 1, but something else?
        # If x[1:]=0, and g(0)=1, then f2 = 1 + 9*sqrt(x0/29).
        
        # Is it possible the evaluator's ZDT1 is actually ZDT2 or something else?
        # Or maybe the bounds are not [0,1]? "bounds [0, 1] for every variable".
        
        # Let's consider the possibility that the "Pareto front" is not what I think.
        # What if I just use the budget to sample the front densely?
        # If I use `scipy.optimize` to optimize the grid, I need to know the front.
        # I don't know the exact front.
        # However, the current program's strategy of setting x[1:]=0 and sweeping x[0] works.
        # This strongly implies the optimal solution lies on the manifold x[1:]=0.
        # So I should stick to sweeping x[0] with x[1:]=0.
        # The only question is: what is the best distribution of x[0]?
        # The current program uses u^2 (bias to 0).
        # The directive suggests using `scipy.optimize.minimize` to find the optimal grid nodes.
        # To do this, I need to compute the hypervolume of the points on the front.
        # I can compute the hypervolume of the points generated by a candidate grid.
        # I can define a function `hv_of_grid(x0_nodes)` that:
        # 1. Constructs points (x0, f2(x0)) for x0 in nodes.
        # 2. Filters points better than ref (11,11).
        # 3. Computes hypervolume.
        # 4. Returns -HV.
        # Then minimize this function.
        
        # But I need the function f2(x0).
        # I can estimate it by evaluating `problem` on a few points with x[1:]=0.
        # I have budget 200.
        # I can spend 10-20 evals to probe the front shape.
        # Then use the remaining budget to evaluate the optimized grid.
        
        # Plan:
        # 1. Probe front: Evaluate problem at x=[x0, 0, ..., 0] for a few x0 values (e.g., 0, 0.1, ..., 1.0).
        #    This gives us (f1, f2) pairs.
        # 2. Fit a function or just use the empirical points to define the front for optimization?
        #    No, the optimization should place points to maximize HV of the *evaluated* points.
        #    But I can't evaluate all candidate grids during optimization (too many evals).
        #    I need an analytical model of the front to optimize the grid *offline* (in the code, using time, not evals).
        #    So, probe the front to fit a model.
        #    If the front is smooth, a polynomial or power law fit might work.
        #    Or, if I assume the front is monotonic, I can just optimize the x0 positions 
        #    assuming a generic convex/concave shape?
        #    Actually, if I probe the front, I get the actual f2 values.
        #    I can then define the HV calculation based on the *interpolated* front?
        #    No, the HV is calculated on the *discrete* points I return.
        #    So I want to place x0_i such that the set {(x0_i, f2(x0_i))} has max HV.
        #    Since I don't know f2 analytically, I can use the probed points to estimate f2(x) 
        #    for any x in [0,1] via interpolation.
        #    Then I can compute the HV of any candidate grid using this interpolated f2.
        #    Then I use scipy.optimize to find the x0 nodes that maximize this estimated HV.
        #    Finally, I evaluate the problem at these optimal nodes.
        
        # Steps:
        # 1. Probe: 20 points uniform in [0,1] for x0, x[1:]=0.
        #    Get F_probe.
        #    Sort by f1.
        #    Remove dominated points? No, just keep the curve.
        #    Fit an interpolator (e.g., np.interp or spline).
        # 2. Define HV estimator:
        #    Given a set of x0_nodes, compute f2_est = interpolator(x0_nodes).
        #    Points = (x0_nodes, f2_est).
        #    Filter by ref (11,11).
        #    Compute HV.
        # 3. Optimize:
        #    Minimize -HV_estimator(nodes) using scipy.optimize.
        #    Variables: nodes (size N).
        #    Constraints: 0 <= nodes[0] <= ... <= nodes[-1] <= 1.
        # 4. Evaluate:
        #    Use remaining budget to evaluate the problem at the optimized nodes.
        
        # This is a robust approach that doesn't assume the specific ZDT1 formula, 
        # but adapts to the actual front shape discovered by probing.
        
        n_probe = 20
        x0_probe = np.linspace(0, 1, n_probe)
        X_probe = np.zeros((n_probe, dim))
        X_probe[:, 0] = x0_probe
        F_probe = np.array([problem(x) for x in X_probe], dtype=np.float64)
        evals_used = n_probe
        
        # Sort by f1
        sort_idx = np.argsort(F_probe[:, 0])
        f1_probe = F_probe[sort_idx, 0]
        f2_probe = F_probe[sort_idx, 1]
        
        # Create interpolator
        # Use cubic spline or linear? Linear is safer if front is not smooth.
        # np.interp works for linear.
        
        def hv_estimator(nodes):
            # nodes: array of x0 values
            f2_est = np.interp(nodes, f1_probe, f2_probe)
            pts = np.column_stack((nodes, f2_est))
            # Filter
            mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            # Compute HV
            # Sort by f1
            pts = pts[np.argsort(pts[:, 0])]
            # Non-dominated sort? 
            # For 2D, we can just iterate.
            # HV = sum over points of (ref[0] - f1_i) * (ref[1] - f2_i) - ...?
            # Standard 2D HV:
            # Sort by f1 ascending.
            # Keep only non-dominated points (f2 must be decreasing as f1 increases for min-min).
            # If f2 is increasing (like the current program's assumption), then only the first point is non-dominated.
            # If f2 is decreasing, we keep the "elbow".
            
            # Let's implement a generic 2D HV calculator.
            # Sort by f1.
            # Iterate, keeping track of min f2 seen so far?
            # No, for min-min, a point is non-dominated if its f2 is lower than all previous points with lower f1?
            # Yes.
            
            f1 = pts[:, 0]
            f2 = pts[:, 1]
            
            # Sort by f1
            order = np.argsort(f1)
            f1 = f1[order]
            f2 = f2[order]
            
            # Filter dominated
            # A point i is dominated if there exists j < i such that f1_j <= f1_i and f2_j <= f2_i.
            # Since f1 is sorted, we just need to check if f2_i >= min(f2[0:i]).
            # Actually, we want to keep points that define the lower envelope of f2.
            # So we keep i if f2_i < min(f2[0:i]).
            
            keep = np.zeros(len(f1), dtype=bool)
            min_f2_so_far = np.inf
            for i in range(len(f1)):
                if f2[i] < min_f2_so_far:
                    keep[i] = True
                    min_f2_so_far = f2[i]
            
            f1_nd = f1[keep]
            f2_nd = f2[keep]
            
            if len(f1_nd) == 0:
                return 0.0
            
            # Calculate HV
            # HV = sum_{i} (ref[0] - f1_i) * (f2_{i-1} - f2_i) + (ref[0] - f1_last) * (ref[1] - f2_last)
            # Wait, the formula is:
            # HV = \sum_{i=1}^{N} (ref_0 - f1_i) * (f2_{i-1} - f2_i)
            # where f2_0 = ref_1.
            
            hv = 0.0
            prev_f2 = ref[1]
            for i in range(len(f1_nd)):
                width = ref[0] - f1_nd[i]
                height = prev_f2 - f2_nd[i]
                if width > 0 and height > 0:
                    hv += width * height
                prev_f2 = f2_nd[i]
            
            return hv
        
        # Optimize nodes
        n_opt = budget - n_probe
        if n_opt < 10:
            n_opt = 10
            
        # Initial guess: uniform
        x0_init = np.linspace(0, 1, n_opt)
        
        # Bounds for optimization
        bounds_opt = [(0, 1)] * n_opt
        
        # We want to maximize HV, so minimize -HV
        def neg_hv(nodes):
            return -hv_estimator(nodes)
        
        # Use L-BFGS-B or SLSQP
        # SLSQP might be better for constraints, but L-BFGS-B is faster.
        # We can enforce ordering by just optimizing and then sorting?
        # If we sort, the function is not differentiable.
        # Better to optimize in a way that preserves order?
        # Or just optimize and then sort the result.
        # The HV estimator sorts internally, so it's invariant to input order.
        # So we can optimize any permutation.
        
        res = minimize(neg_hv, x0_init, method='L-BFGS-B', bounds=bounds_opt, options={'maxiter': 100})
        
        x0_opt = res.x
        x0_opt = np.sort(x0_opt)
        
        # Evaluate the optimized points
        X_opt = np.zeros((n_opt, dim))
        X_opt[:, 0] = x0_opt
        F_opt = np.array([problem(x) for x in X_opt], dtype=np.float64)
        evals_used += n_opt
        
        # Combine probe and opt points?
        # The probe points are already evaluated.
        # We should return all evaluated points.
        X_all = np.vstack([X_probe, X_opt])
        F_all = np.vstack([F_probe, F_opt])
        
        return X_all

if __name__ == "__main__":
    pass
