"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Smooth surrogate objective:
    # We want to minimize (max_d / min_d)^2.
    # This is equivalent to minimizing max_d * max(1/d_i).
    # A smooth approximation for max(x) is log-sum-exp(x).
    # Let L = log-sum-exp(log(d_i)) approx max(log(d_i)) = log(max(d_i)).
    # Let M = log-sum-exp(-log(d_i)) approx max(-log(d_i)) = -log(min(d_i)).
    # Objective approx: exp(L + M) = exp(log(max_d) - log(min_d)) = max_d / min_d.
    # We want to minimize this ratio.
    # To improve stability, we can minimize log(ratio) = L + M.
    # However, log-sum-exp can suffer from numerical issues if distances are very small or large.
    # We will work with squared distances to avoid sqrt, but log-sum-exp works on logs.
    # Let's use a soft-max/soft-min on the squared distances directly?
    # Ratio of sq_dists: R_sq = max(d_sq) / min(d_sq).
    # log(R_sq) = log(max(d_sq)) - log(min(d_sq)).
    # Approx: LSE(log(d_sq)) - LSE(-log(d_sq)).
    # We minimize this value.
    
    def smooth_objective(x):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        # Clip to avoid log(0)
        d_sq = np.clip(d_sq, 1e-12, 1e12)
        logs = np.log(d_sq)
        # Soft max of logs = log(sum(exp(logs))) = log(sum(d_sq))
        # Wait, LSE(logs) is NOT log(max(d_sq)).
        # LSE(y) approx max(y). So LSE(log(d_sq)) approx log(max(d_sq)).
        # LSE(-log(d_sq)) approx max(-log(d_sq)) = -log(min(d_sq)).
        # Sum approx log(max(d_sq)) - log(min(d_sq)) = log(max(d_sq)/min(d_sq)).
        # This is log(Ratio_sq). Minimizing this minimizes Ratio_sq.
        
        # Numerical stability for LSE: subtract max
        max_log = np.max(logs)
        lse_max = max_log + np.log(np.sum(np.exp(logs - max_log)))
        
        min_logs = -logs
        max_neg_log = np.max(min_logs)
        lse_min = max_neg_log + np.log(np.sum(np.exp(min_logs - max_neg_log)))
        
        # The objective is lse_max - lse_min
        # We add a small penalty for degeneracy (points too close or far)
        # If points are identical, d_sq=0, log=-inf. Clipping helps.
        # If points are very far, d_sq=inf. Clipping helps.
        
        # Add a small term to encourage isotropy? 
        # The ratio objective itself encourages compactness relative to spread.
        # We might need to prevent the points from collapsing to a line or point if the ratio allows it?
        # No, if points collapse, min_d -> 0, ratio -> inf. So it's naturally penalized.
        # If points expand to infinity, min_d -> inf? No, if we scale, min_d is fixed?
        # The ratio is scale invariant. So the optimizer might drift in scale.
        # We should fix the scale to prevent drift. 
        # Let's normalize inside the objective? No, that breaks gradients.
        # Instead, we can add a penalty on the mean squared distance from centroid to be 1.
        # Or simply rely on the fact that L-BFGS-B is local and we start from normalized points.
        # Actually, scale invariance means the landscape is flat along the radial direction.
        # This can slow down convergence.
        # Let's add a small penalty: (mean(d_sq) - 1)^2 * 1e-4
        # This anchors the scale.
        
        center = np.mean(pts, axis=0)
        dists_from_center = np.sum((pts - center)**2, axis=1)
        scale_penalty = (np.mean(dists_from_center) - 1.0)**2
        
        return lse_max - lse_min + 1e-4 * scale_penalty

    # Initializations
    
    # 1. 4x4 Grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    grid_pts = normalize(grid_pts)
    # Scale so that mean sq dist from center is 1?
    # Or just let the optimizer handle it.
    inits = [grid_pts.ravel()]
    
    # 2. Hexagonal Lattice Subset
    # Generate a hex grid and pick 16 points.
    # Center at (0,0).
    # Ring 0: 1 point.
    # Ring 1: 6 points.
    # Ring 2: 12 points. Total 19. Too many.
    # Let's take Ring 0 (1) + Ring 1 (6) + 9 from Ring 2?
    # Or just use a distorted grid.
    # Let's try a "hexagonal" arrangement of 16 points.
    # Rows: 4, 4, 4, 4? No, that's square.
    # Rows: 3, 4, 4, 3, 2? 
    # Let's use a standard hex packing for 16.
    # It turns out a 4x4 grid is often a good local minimum.
    # Let's add a random perturbed grid.
    
    for _ in range(5):
        pts = grid_pts + 0.1 * np.random.normal(0, 1, (N, 2))
        pts = normalize(pts)
        inits.append(pts.ravel())
        
    # 3. Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    circle_pts = normalize(circle_pts)
    inits.append(circle_pts.ravel())

    best_x = None
    best_score = 1e10

    # Stage 1: Differential Evolution for global search
    # DE is slow but robust. We use it to find a good basin.
    # Bounds: [-2, 2] for each coordinate.
    bounds = [(-2.0, 2.0)] * (N * 2)
    
    # We only have time for a few DE runs or a short DE.
    # Let's try one DE run with limited iterations.
    # If time is tight, we skip DE and go straight to local polish from inits.
    
    if time.time() < deadline - 60:
        try:
            # DE objective wrapper
            def de_obj(x):
                if time.time() > deadline - 10:
                    return 1e10
                return smooth_objective(x)
            
            # Use a subset of inits as starting points? DE is population based.
            # We can run DE for a short time.
            # popsize=10, maxiter=50 is ~500 evaluations.
            # Each eval is fast.
            de_res = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                polish=False # We will polish manually
            )
            if de_res.success or de_res.fun < best_score:
                x_de = de_res.x
                s_de = score_raw(x_de.reshape(N, 2))
                if s_de < best_score:
                    best_score = s_de
                    best_x = x_de
        except Exception:
            pass

    # Stage 2: Local Polish with L-BFGS-B
    # Try polishing from the best DE result (if any) and all initializations.
    
    candidates = []
    if best_x is not None:
        candidates.append(best_x)
    candidates.extend(inits)
    
    # Remove duplicates in candidates roughly
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for u in unique_candidates:
            if np.linalg.norm(c - u) < 1e-3:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
            
    for x0 in unique_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                smooth_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            final_pts = res.x.reshape(N, 2)
            raw_score = score_raw(final_pts)
            
            if raw_score < best_score:
                best_score = raw_score
                best_x = res.x
        except Exception:
            continue

    # Stage 3: Final high-precision polish
    if best_x is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                smooth_objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            final_pts = res_final.x.reshape(N, 2)
            raw_score = score_raw(final_pts)
            if raw_score < best_score:
                best_score = raw_score
                best_x = res_final.x
        except Exception:
            pass

    if best_x is None:
        best_x = inits[0]
        
    return best_x.reshape(N, 2)

# EVOLVE-BLOCK-END
