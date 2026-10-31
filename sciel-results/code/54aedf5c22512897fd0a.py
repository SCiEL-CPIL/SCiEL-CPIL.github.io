import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

    def score_points(pts):
        d = pdist(pts, 'euclidean')
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        max_d = np.max(d)
        return (max_d / min_d) ** 2

    def soft_score(pts, beta):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Soft max and min of squared distances
        max_d_sq = np.max(d_sq)
        soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        min_d_sq = np.min(d_sq)
        soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min < 1e-8:
            return 1e10
        return soft_max / soft_min

    def generate_hex_seed():
        # Generate a hexagonal lattice and pick 16 closest to origin
        pts = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                # Convert axial to cartesian
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                pts.append([x, y])
        pts = np.array(pts)
        dists = np.sum(pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        return pts[idx]

    def generate_ring_seed():
        # Two concentric rings
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r1 = 1.0
        r2 = 2.0
        pts1 = np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1)])
        pts2 = np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def generate_polygon_seed():
        # Regular 16-gon
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        r = 1.0
        return np.column_stack([r*np.cos(angles), r*np.sin(angles)])

    def generate_random_seed(rng):
        return rng.uniform(-2, 2, size=(16, 2))

    def refine(pts):
        """
        Multi-stage refinement for a single seed.
        Uses soft objectives with increasing beta, then Powell, then BasinHopping.
        """
        current_pts = pts.copy()
        
        # Check initial score
        curr_score = score_points(current_pts)
        
        # Optimization schedule: Start with smooth objectives, then sharpen
        betas = [10, 50, 200, 1000]
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    lambda p: soft_score(p.reshape(-1, 2), beta),
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                new_pts = res.x.reshape(-1, 2)
                new_score = score_points(new_pts)
                
                if new_score < curr_score:
                    curr_score = new_score
                    current_pts = new_pts
            except Exception:
                pass
        
        # Final polish with Powell on exact objective
        if time.time() < deadline - 5:
            try:
                res_powell = minimize(
                    lambda p: score_points(p.reshape(-1, 2)),
                    current_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': 2000}
                )
                new_pts = res_powell.x.reshape(-1, 2)
                new_score = score_points(new_pts)
                
                if new_score < curr_score:
                    curr_score = new_score
                    current_pts = new_pts
            except Exception:
                pass
        
        # Basin Hopping for global escape
        if time.time() < deadline - 10:
            try:
                def bh_objective(p):
                    return score_points(p.reshape(-1, 2))
                
                # Use a simpler local minimizer for basin hopping to ensure speed
                def bh_minimizer(p):
                    try:
                        res = minimize(
                            bh_objective, p,
                            method='L-BFGS-B',
                            options={'maxiter': 50, 'ftol': 1e-8}
                        )
                        return res.x, res.fun, {'nit': 1}
                    except:
                        return p, bh_objective(p), {'nit': 1}
                
                # Use a custom step function that allows larger jumps
                def bh_step(x, f, p):
                    # p is the step size parameter
                    return x + p * np.random.normal(size=x.shape)
                
                res_bh = basinhopping(
                    bh_objective, 
                    current_pts.flatten(), 
                    minimizer_kwargs={'args': ()},
                    minimizer=bh_minimizer,
                    stepsize=0.5,
                    niter=10,
                    T=1.0,
                    restarts=1,
                    accept_test=lambda f_new, f_old, T: f_new < f_old or np.random.rand() < np.exp((f_old - f_new) / T)
                )
                
                new_pts = res_bh.x.reshape(-1, 2)
                new_score = score_points(new_pts)
                
                if new_score < curr_score:
                    curr_score = new_score
                    current_pts = new_pts
            except Exception:
                pass
                
        return current_pts, curr_score

    seeds = []
    hex_seed = generate_hex_seed()
    seeds.append(hex_seed)
    
    ring_seed = generate_ring_seed()
    seeds.append(ring_seed)
    
    poly_seed = generate_polygon_seed()
    seeds.append(poly_seed)
    
    rng = np.random.default_rng(42)
    # Add perturbed versions
    for s in [hex_seed, ring_seed, poly_seed]:
        for _ in range(3):
            seeds.append(s + rng.normal(0, 0.2, size=(16, 2)))
            
    # Add some random seeds
    for _ in range(5):
        seeds.append(generate_random_seed(rng))

    best_score = 1e10
    best_pts = None

    # Iterate through seeds
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
            
        # Determine time budget for this seed
        # If we have many seeds, give less time to each.
        # With ~14 seeds and 125s available, ~8s per seed is reasonable.
        # But BasinHopping can be slow. Let's cap the time per seed.
        
        try:
            refined_pts, refined_score = refine(seed)
            
            if refined_score < best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        return hex_seed
        
    # Verify distinctness
    d_check = pdist(best_pts, 'euclidean')
    if np.min(d_check) < 1e-10:
        return hex_seed
        
    return best_pts
