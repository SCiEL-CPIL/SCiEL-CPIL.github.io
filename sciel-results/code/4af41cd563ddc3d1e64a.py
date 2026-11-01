import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_exact(x):
    """
    Exact objective: minimize (max_sq_dist / min_sq_dist).
    """
    points = x.reshape(N, D)
    # Center to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return 1e10
    return np.max(sq_dists) / min_sq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Icosahedron-based: 12 vertices + 2 additional points
    phi = (1 + np.sqrt(5)) / 2
    ico_base = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_base = ico_base / np.linalg.norm(ico_base, axis=1, keepdims=True)
    
    # Add 2 points: try poles
    pts_poles = np.vstack([ico_base, np.array([[0,0,1],[0,0,-1]])])
    candidates.append(pts_poles.flatten())
    
    # Add 2 random points on sphere
    for _ in range(5):
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        pts = np.vstack([ico_base, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
    
    # 2. Two rings of 7 points each
    for z_off in [0.3, 0.5, 0.7]:
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        ring2_angles = ring1_angles + np.pi/7
        r = np.sqrt(1 - z_off**2)
        ring1 = np.column_stack([r*np.cos(ring1_angles), r*np.sin(ring1_angles), np.full(7, z_off)])
        ring2 = np.column_stack([r*np.cos(ring2_angles), r*np.sin(ring2_angles), np.full(7, -z_off)])
        pts = np.vstack([ring1, ring2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
    
    # 3. Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * i / (n - 1)
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    fib_pts = fibonacci_sphere(N)
    rot = Rotation.from_quat(rng.normal(size=4))
    fib_pts = rot.apply(fib_pts)
    candidates.append(fib_pts.flatten())
    
    # 4. Random spherical points
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())
    
    # Optimization: Use Nelder-Mead with the exact objective
    for i, x0 in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                objective_exact,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass
    
    # Second pass: refine top candidates
    scored = []
    for x0 in candidates:
        pts = x0.reshape(N, D)
        score = raw_score(pts)
        scored.append((score, x0))
    scored.sort()
    
    for score, x0 in scored[:5]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                objective_exact,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass
    
    # Third pass: local perturbation search on best solution
    if best_points is not None and time.time() < deadline - 2:
        current_best = best_points.copy()
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            perturbation = rng.normal(size=(N, D), scale=0.01)
            perturbed = current_best + perturbation
            perturbed = perturbed - np.mean(perturbed, axis=0, keepdims=True)
            update_best(perturbed)
            if best_score < raw_score(perturbed):
                current_best = best_points.copy()
    
    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
