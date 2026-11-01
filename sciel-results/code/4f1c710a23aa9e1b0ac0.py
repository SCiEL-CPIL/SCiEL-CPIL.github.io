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

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def generate_d3d_points(x, y, z, h):
    # Generate 12 points from orbit of (x, y, z) under D3d
    # and 2 poles (0, 0, h), (0, 0, -h)
    
    pts = []
    
    # C3 rotation angles
    c3 = np.cos(2 * np.pi / 3)
    s3 = np.sin(2 * np.pi / 3)
    
    # Base point
    p0 = np.array([x, y, z])
    
    # C3 rotated
    p1 = np.array([c3 * x - s3 * y, s3 * x + c3 * y, z])
    
    # C3^2 rotated
    p2 = np.array([c3 * c3 - s3 * s3 * 0 - (1-c3*c3)*0, 0, 0]) # Just use matrix or re-calc
    # Easier: apply C3 to p1
    p2 = np.array([c3 * p1[0] - s3 * p1[1], s3 * p1[0] + c3 * p1[1], p1[2]])
    
    # Inversions
    ip0 = -p0
    ip1 = -p1
    ip2 = -p2
    
    # C2' rotations (180 deg around axes in xy plane)
    # Axis 1: x-axis. (x, y, z) -> (x, -y, -z)
    q0 = np.array([x, -y, -z])
    
    # Axis 2: rotated 120 deg. Apply C3 to q0
    q1 = np.array([c3 * q0[0] - s3 * q0[1], s3 * q0[0] + c3 * q0[1], q0[2]])
    
    # Axis 3: rotated 240 deg. Apply C3 to q1
    q2 = np.array([c3 * q1[0] - s3 * q1[1], s3 * q1[0] + c3 * q1[1], q1[2]])
    
    # Inversions of C2' rotated points
    iq0 = -q0
    iq1 = -q1
    iq2 = -q2
    
    pts.extend([p0, p1, p2, ip0, ip1, ip2, q0, q1, q2, iq0, iq1, iq2])
    
    # Poles
    pts.append(np.array([0, 0, h]))
    pts.append(np.array([0, 0, -h]))
    
    return np.array(pts)

def objective(params):
    x, y, z, h = params
    pts = generate_d3d_points(x, y, z, h)
    pts = normalize_points(pts)
    
    # Check for coincident points
    sq_dists = get_pairwise_sq_dists(pts)
    if np.min(sq_dists) < 1e-8:
        return 1e10
    
    return raw_score(pts)

def construct():
    start_time = time.time()
    timeout = 145.0
    
    best_score = np.inf
    best_params = None
    
    # Initial guesses
    # Try a few random starting points
    rng = np.random.default_rng(42)
    
    # Heuristic initial guess:
    # Icosahedron vertices are roughly at radius 1.
    # z-coords of icosahedron: 0, +/- 1/phi, +/- phi
    # Let's try x=1, y=0, z=1/phi, h=1
    initial_guesses = [
        [1.0, 0.0, 0.618, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.5, 0.5, 0.8],
        [0.8, 0.4, 0.2, 1.2],
        [1.0, 0.0, 1.0, 0.5]
    ]
    
    # Add random guesses
    for _ in range(20):
        initial_guesses.append(rng.uniform(-1, 1, size=4))
        
    for guess in initial_guesses:
        if time.time() - start_time > timeout - 10:
            break
            
        try:
            res = minimize(
                objective,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception as e:
            pass

    # If Nelder-Mead didn't work well, try L-BFGS-B on a smooth approximation?
    # The raw_score is non-smooth (min/max).
    # We can use a smooth approximation for L-BFGS-B if needed, but Nelder-Mead is derivative-free and often works well for low-dim non-smooth problems.
    
    # Let's do a second pass with L-BFGS-B using a smooth objective if we have time
    # Smooth objective: log-sum-exp based
    
    def smooth_objective(params):
        x, y, z, h = params
        pts = generate_d3d_points(x, y, z, h)
        pts = normalize_points(pts)
        sq_dists = get_pairwise_sq_dists(pts)
        
        alpha = 50.0
        max_sq = np.max(sq_dists)
        min_sq = np.min(sq_dists)
        
        term_max = alpha * max_sq + np.log(np.sum(np.exp(alpha * (sq_dists - max_sq))))
        term_min = -alpha * min_sq + np.log(np.sum(np.exp(-alpha * (sq_dists - min_sq))))
        
        if term_min == 0:
            return 1e10
        return term_max / (-term_min)

    if best_params is not None and time.time() - start_time < timeout - 20:
        try:
            res2 = minimize(
                smooth_objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            # Evaluate raw score of the result
            pts2 = generate_d3d_points(*res2.x)
            pts2 = normalize_points(pts2)
            score2 = raw_score(pts2)
            if score2 < best_score:
                best_score = score2
                best_params = res2.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_params = initial_guesses[0]
        
    best_points = generate_d3d_points(*best_params)
    best_points = normalize_points(best_points)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
