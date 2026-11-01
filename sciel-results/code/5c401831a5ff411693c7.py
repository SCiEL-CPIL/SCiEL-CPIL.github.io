import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def objective_smooth(x, tau=50.0):
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max
    # Use stable log-sum-exp
    # log(sum(exp(tau * (sq_dists - m_max)))) + m_max
    # = m_max + log(sum(exp(tau * (sq_dists - m_max)))) / tau
    
    # To avoid overflow/underflow in exp, we subtract max/min
    # For max: exp(tau * (sq_dists - m_max)) is <= 1
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # For min: exp(-tau * (sq_dists - m_min)) is <= 1
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # Return log ratio
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        # Ensure distinct points
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-8:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Icosahedron based
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * 2.0
    
    # Add 2 points. Try placing them at the origin (bad) or slightly off.
    # Better: place them at positions that break symmetry but are close to existing structure.
    # Or just random points on the sphere.
    for i in range(20):
        p1 = rng.normal(size=D) * 0.1
        p2 = rng.normal(size=D) * 0.1
        pts = np.vstack([ico, p1, p2])
        # Normalize to sphere? No, just use as is, optimizer will adjust.
        # Actually, scaling invariance means we can fix scale.
        # Let's normalize the whole set to have radius 1 or something?
        # The ratio is scale invariant.
        # Let's just use the raw points.
        candidates.append(pts.flatten())
        
        # Also try placing p1, p2 at "gaps"
        # Gaps in icosahedron are centers of faces.
        # There are 20 faces.
        # Let's pick 2 random face centers.
        # This is complex to compute on the fly. Random is fine for now.

    # 2. Random points on sphere
    for i in range(20):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
        
    # 3. Lattice points
    # Simple grid
    for i in range(5):
        pts = np.array([
            [0,0,0], [1,0,0], [0,1,0], [0,0,1],
            [1,1,0], [1,0,1], [0,1,1], [1,1,1],
            [-1,0,0], [0,-1,0], [0,0,-1],
            [-1,-1,0], [0,-1,-1], [1,1,2] # Random fill
        ])
        # Center
        pts = pts - np.mean(pts, axis=0)
        # Scale
        pts = pts / np.max(np.linalg.norm(pts, axis=1)) * 2.0
        # Add random noise
        pts += rng.normal(size=(N, D)) * 0.1
        candidates.append(pts.flatten())

    # Optimization Loop
    # We have ~45 candidates.
    # Each L-BFGS-B run should be fast (< 1s).
    # 45 * 1s = 45s. Plenty of time for polish.
    
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 10:
            break
            
        try:
            # Use L-BFGS-B
            res = minimize(
                objective_smooth, 
                init_x, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None:
                update_best(res.x.reshape(N, D))
        except Exception:
            pass

    # If time remains, do a final polish on the best point found
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            # Use a tighter tolerance and maybe a different method or just more iterations
            res_final = minimize(
                objective_smooth, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_final.x is not None:
                update_best(res_final.x.reshape(N, D))
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
