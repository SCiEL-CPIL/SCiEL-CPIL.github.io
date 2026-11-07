import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        # ZDT3 Analysis:
        # The Pareto front for ZDT3 (dim=30) is located at x[1:] = 0.
        # On this manifold:
        # g(x) = 1 (minimum possible value for g in ZDT3 definition usually 1+10(n-1) but wait, 
        # standard ZDT3: g(x) = 1 + 10(n-1) + sum(2x_i^2 - 4cos(4pi x_i)). 
        # Min of g is 1 + 10(n-1) when x_i=0.
        # f1 = x0
        # f2 = (1 + g) * (1 - sqrt(f1 / (1 + g)))
        # Let C = 1 + g_min = 1 + (1 + 10*(dim-1)) = 2 + 10*(dim-1).
        # For dim=30, C = 2 + 290 = 292.
        # f2(t) = C * (1 - sqrt(t/C)) = C - sqrt(C*t)
        
        # We want to maximize Hypervolume with ref = [11, 11].
        # The points must be strictly better than [11, 11].
        # f1 is in [0, 1]. f2(0) = C = 292. f2(1) = C - sqrt(C) = 292 - 17.08 = 274.9.
        # WAIT. The reference point is [11, 11].
        # If f2 is around 275, then f2 > 11. 
        # The scoring says: "Only points strictly better than the reference point in every objective count."
        # Better means smaller for minimization.
        # If f2_min is 274.9, and ref is 11, then NO points on the true ZDT3 front (with standard scaling) 
        # are better than [11, 11] in the second objective!
        
        # Let's re-read the task carefully.
        # "minimizing all 2 objectives of ZDT3"
        # "reference point [11.0, 11.0]"
        # "hypervolume of all recorded vectors with reference point [11.0, 11.0]"
        # "Only points strictly better than the reference point in every objective count."
        
        # If the problem is standard ZDT3, f2 is huge. 
        # However, the "CURRENT PROGRAM" (score 1.18) and "INSPIRATIONS" (score 1.26) 
        # imply that the score is not 0. 
        # This suggests either:
        # 1. The problem instance has different scaling (e.g., f2 is scaled down).
        # 2. The reference point is effectively [inf, inf] in the code logic but defined as [11,11] for normalization? 
        #    No, "exact hypervolume ... with reference point [11.0, 11.0]".
        # 3. Maybe the "problem" provided by the evaluator is NOT standard ZDT3, but a scaled version?
        #    Or maybe my assumption of g_min is wrong?
        
        # Let's look at the current program's calibration.
        # It checks `mae < 1e-3` against `one_plus_g * (1.0 - np.sqrt(f1s / one_plus_g))`.
        # If the current program scores 1.18, it means it IS finding points that contribute to HV.
        # This implies that for the specific `problem` function provided, the values of f2 ARE less than 11.
        
        # How can f2 < 11?
        # f2 = (1+g) * (1 - sqrt(f1/(1+g)))
        # If f1 is close to 1, f2 is close to (1+g)(1 - 1/sqrt(1+g)) = (1+g) - sqrt(1+g).
        # For f2 < 11, we need (1+g) - sqrt(1+g) < 11.
        # Let y = sqrt(1+g). y^2 - y - 11 < 0.
        # Roots of y^2 - y - 11 = 0 are (1 + sqrt(1+44))/2 = (1+6.7)/2 = 3.85.
        # So sqrt(1+g) < 3.85 => 1+g < 14.8 => g < 13.8.
        # In standard ZDT3, g_min = 1 + 10*(29) = 291. This is way too high.
        
        # Conclusion: The "problem" passed to the optimizer is NOT standard ZDT3 with dim=30.
        # It is likely a scaled version, or a different ZDT variant, or the "dim" passed to Optimizer 
        # is not the actual dimension used in the g function, or the bounds are different.
        
        # HOWEVER, the current program and inspirations work by assuming x[1:]=0 is the front.
        # And they calculate f2 based on the observed values.
        
        # The key insight from the "failed" child (1.18 vs 1.26) is that the greedy 
        # insertion with a *predicted* f2 might be placing points at x0 values that, 
        # when actually evaluated, yield slightly different f2 values due to numerical 
        # noise or non-linearity not captured by the simple sqrt model, OR the greedy 
        # algorithm itself is inefficient/suboptimal.
        
        # A better strategy: 
        # 1. Use a small number of evaluations to map the front precisely.
        # 2. Since the front is 1D, we can just sample x0 uniformly? 
        #    No, uniform sampling is not optimal for HV if the curve is non-linear.
        # 3. Optimal quantization for HV on a 1D curve:
        #    The hypervolume contribution of a point (x_i, y_i) in a sorted set is related 
        #    to the area of the rectangle it "owns".
        #    For a convex decreasing function, the optimal points are NOT uniform.
        
        # Let's try a different approach: 
        # Since we have a budget of 200, and the front is 1D, we can afford to do a 
        # high-resolution calibration and then use a simple, robust placement strategy.
        
        # Strategy:
        # 1. Spend 10 evaluations to calibrate the front shape (x0 in [0,1], x[1:]=0).
        # 2. Fit a high-order polynomial or use the analytic form if it matches.
        # 3. Use the remaining 190 evaluations to place points at positions that maximize 
        #    the "local" hypervolume gain. 
        #    Actually, for a 1D front, the optimal placement for maximizing HV 
        #    (area under the curve relative to ref) is to equalize the "sag" of the 
        #    chord connecting consecutive points to the curve.
        
        # But wait, the current program's failure suggests that the *prediction* was off.
        # Let's assume the problem IS well-behaved on x[1:]=0.
        
        # Alternative Strategy: 
        # Instead of greedy insertion which is O(N^2), we can use a **Binary Search for Equal Area** approach.
        # Or, simpler: **Uniform Sampling in the "Objective Space"**.
        # If the front is smooth, sampling uniformly in f1 is often close to optimal.
        # But for convex curves, sampling more densely where the curvature is higher (near f1=0) 
        # is better.
        
        # Let's look at the score 1.26 (inspiration 2).
        # "samples x[0] uniformly in [0, 1] with x[1:]=0".
        # This simple approach scored 1.26.
        # The current program (1.18) tried to be smarter with greedy insertion and scored LOWER.
        # Why? 
        # 1. The calibration might have been noisy.
        # 2. The greedy algorithm might have gotten stuck.
        # 3. The "predicted" f2 might have been wrong, leading to bad x0 choices.
        
        # Hypothesis: The "smart" placement is worse than uniform because the prediction 
        # error dominates the gain from optimal placement.
        
        # New Strategy: 
        # Go back to basics but optimize the uniform sampling.
        # 1. Verify the front is at x[1:]=0 with 2 points.
        # 2. If yes, use ALL remaining budget to sample x0 uniformly in [0, 1].
        # 3. BUT, to beat 1.26, we need to do slightly better than uniform.
        
        # How to beat uniform?
        # The HV is maximized when the points are placed such that the "rectangles" 
        # they form have equal area? No, that's for L1 error.
        # For HV, we want to cover the area under the curve.
        
        # Let's try **Adaptive Quadrature** style sampling.
        # 1. Start with a coarse grid.
        # 2. Identify the interval with the largest "error" (difference between chord and curve).
        # 3. Refine that interval.
        # This is essentially what the greedy algorithm does, but done more carefully.
        
        # However, given the time constraint and the fact that the previous "smart" attempt failed,
        # I will implement a **Robust Hybrid**:
        # 1. Calibrate the front with 5 points.
        # 2. If the front matches the analytic ZDT3 form (within tolerance), use the 
        #    analytic derivative to find the optimal quantization points.
        # 3. If not, fall back to uniform sampling (which scored 1.26).
        
        # The analytic optimal quantization for a convex function f(x) to maximize 
        # the area of the union of rectangles (HV) is to place points such that 
        # the integral of (f(x) - chord) is equal for each interval? 
        # Actually, for a convex function, the optimal points for minimizing the 
        # "loss" of HV (which is the area between the curve and the step function) 
        # are the same as the optimal points for L2 approximation? No.
        
        # Let's stick to the most robust improvement: 
        # The previous program's greedy loop was O(N^2) and used a potentially flawed 
        # predictor. 
        # I will use a **Fixed Grid of High Resolution** but optimized for the 
        # known curvature of ZDT3.
        
        # For ZDT3, f2(t) = C - sqrt(Ct).
        # The curvature is highest at t=0.
        # We should sample more densely near t=0.
        # A good distribution is t_i = (i/N)^2 or similar.
        
        # Let's test this hypothesis:
        # If the problem is standard ZDT3 scaled to fit ref [11,11], the shape is still 
        # determined by the sqrt.
        
        # Plan:
        # 1. Calibrate with 5 points to get the actual C (or scale factor).
        # 2. Assume the form f2(t) = A - B*sqrt(t).
        # 3. Fit A and B to the calibration points.
        # 4. Generate 195 points using a distribution that clusters near t=0.
        #    Specifically, t_i = (i / (N-1))^p for some p > 1.
        #    For sqrt curvature, p=2 is a good start.
        # 5. Evaluate all points.
        
        # This avoids the iterative greedy overhead and prediction errors during the loop.
        
        cal_n = 5
        x0_cal = np.linspace(0.0, 1.0, cal_n)
        cal_pts = []
        
        for x0 in x0_cal:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            try:
                f = problem(x)
                if len(f) == 2:
                    cal_pts.append((float(f[0]), float(f[1])))
            except:
                continue
        
        if not cal_pts:
            # Fallback: Random
            for _ in range(10):
                if time.time() > deadline:
                    break
                x = np.random.rand(self.dim)
                try:
                    f = problem(x)
                    if len(f) == 2:
                        cal_pts.append((float(f[0]), float(f[1])))
                except:
                    continue
            if not cal_pts:
                return None

        # Sort calibration points
        cal_pts.sort(key=lambda p: p[0])
        f1s = np.array([p[0] for p in cal_pts])
        f2s = np.array([p[1] for p in cal_pts])
        
        # Fit f2 = A - B * sqrt(f1)
        # This is linear in parameters A, B if we let z = sqrt(f1).
        # f2 = A - B*z
        # We can use least squares.
        
        z = np.sqrt(np.clip(f1s, 0.0, 1.0))
        # Design matrix: [1, -z]
        # y = f2
        # [A, -B] * [1, -z]^T = f2
        # Actually, let's just use np.polyfit on z? 
        # f2 = A + B*z where B is negative.
        
        try:
            coeffs = np.polyfit(z, f2s, 1)
            B, A = coeffs # f2 = B*z + A
            # Check fit quality
            f2_fit = B * z + A
            mae = np.mean(np.abs(f2s - f2_fit))
            
            # If fit is good, use analytic placement.
            # If not, fallback to uniform.
            if mae < 0.1: # Threshold for "good" fit
                # Generate points using t = (i/N)^2
                # This clusters points near 0.
                n_remaining = self.budget - len(cal_pts)
                if n_remaining <= 0:
                    return None
                
                # We want to generate n_remaining new x0 values.
                # Total points will be cal_n + n_remaining.
                # Let's generate a total grid of size self.budget.
                N_total = self.budget
                # Use a power law distribution
                # t_i = (i / (N_total - 1)) ** 2
                i_vals = np.arange(N_total)
                t_vals = (i_vals / (N_total - 1)) ** 2
                t_vals = np.clip(t_vals, 0.0, 1.0)
                
                # We already evaluated cal_pts. We should reuse them if they match the grid?
                # No, cal_pts are at 0, 0.25, 0.5, 0.75, 1.0.
                # The new grid will be different.
                # To maximize efficiency, we should just evaluate the new grid points.
                # But we have already used cal_n evaluations.
                # We have budget - cal_n left.
                # So we should generate (budget - cal_n) points from the optimal distribution,
                # excluding points that are too close to already evaluated ones?
                # Or just generate the full optimal set of size `budget` and skip the ones 
                # we already have?
                
                # Simpler: Generate `budget` points from the optimal distribution.
                # Evaluate the ones we haven't evaluated yet.
                # Since cal_n is small (5), we can just evaluate the rest.
                
                # However, the cal_pts might not be exactly on the optimal grid.
                # Let's just generate `n_remaining` points from the optimal distribution,
                # ensuring they are spread out.
                
                # Actually, the best way is to generate the full set of `budget` points 
                # from the optimal distribution, and then evaluate ONLY the ones that 
                # are not in cal_pts.
                # But cal_pts are specific values.
                
                # Let's just generate `n_remaining` points using the distribution, 
                # but offset to avoid duplicates.
                
                # Better: Generate the full grid of `budget` points.
                # Check which ones are "close" to cal_pts.
                # Evaluate the rest.
                
                full_grid = t_vals
                # cal_pts f1s are f1s.
                # For each point in full_grid, if it's close to any f1 in f1s, skip it.
                # Otherwise, evaluate it.
                
                evaluated_f1s = set(np.round(f1s, 6))
                
                for t in full_grid:
                    if time.time() > deadline:
                        break
                    
                    t_key = round(float(t), 6)
                    if t_key in evaluated_f1s:
                        continue
                    
                    x = np.zeros(self.dim)
                    x[0] = t
                    try:
                        f = problem(x)
                        if len(f) == 2:
                            # We don't need to store, the evaluator records it.
                            pass
                    except:
                        continue
            else:
                # Fallback to Uniform
                n_remaining = self.budget - len(cal_pts)
                for _ in range(n_remaining):
                    if time.time() > deadline:
                        break
                    x0 = np.random.rand()
                    x = np.zeros(self.dim)
                    x[0] = x0
                    try:
                        f = problem(x)
                    except:
                        continue
        except:
            # Fallback to Uniform if fitting fails
            n_remaining = self.budget - len(cal_pts)
            for _ in range(n_remaining):
                if time.time() > deadline:
                    break
                x0 = np.random.rand()
                x = np.zeros(self.dim)
                x[0] = x0
                try:
                    f = problem(x)
                except:
                    continue

        return None
