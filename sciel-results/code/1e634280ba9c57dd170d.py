"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 130
DEADLINE = 125.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use pairwise distances
        d = 1.0 - pts @ pts.T
        np.fill_diagonal(d, 0.0)
        d = np.maximum(d, 0.0)
        return np.degrees(np.arccos(np.clip(1.0 - np.max(d), -1.0, 1.0)))
    
    max_angle = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        for sign in [1.0, -1.0]:
            x = sign * n
            da = np.dot(x, a)
            db = np.dot(x, b)
            dc = np.dot(x, c)
            if abs(da - db) < 1e-8 and abs(da - dc) < 1e-8:
                angle = np.arccos(np.clip(da, -1.0, 1.0))
                if angle > max_angle:
                    max_angle = angle
                break
    
    return np.degrees(max_angle)


def _fibonacci_sphere(n, rng=None):
    """Generate n points on sphere using Fibonacci spiral."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random perturbation to break symmetry
    if rng is not None:
        pts += rng.randn(n, 3) * 1e-3
        pts = _normalize(pts)
    
    return pts


def _repulsion_energy(pts, k=5):
    """Compute repulsion energy: sum of exp(-k * min_angle_to_nearest) for each point.
    Minimizing this spreads points uniformly."""
    pts = _normalize(pts)
    d = pts @ pts.T
    np.fill_diagonal(d, 1.0)
    min_dots = np.min(d, axis=1)
    min_dots = np.clip(min_dots, -1.0, 1.0)
    min_angles = np.arccos(min_dots)
    return np.sum(np.exp(-k * min_angles))


def _combined_loss(pts, target_cr=11.5):
    """Combined loss: repulsion energy + covering radius penalty.
    The penalty strongly penalizes configurations where covering radius exceeds target."""
    pts = _normalize(pts)
    
    # Repulsion term
    rep = _repulsion_energy(pts, k=8.0)
    
    # Covering radius penalty
    cr = _covering_radius_deg(pts)
    penalty = 0.0
    if cr > target_cr:
        penalty = 100.0 * (cr - target_cr) ** 2
    
    return rep + penalty


def _local_search_greedy(pts, time_limit=10.0, step_size=0.1):
    """Greedy local search: try moving each point in random directions, accept if loss decreases."""
    start_time = time.time()
    pts = _normalize(pts).copy()
    best_loss = _combined_loss(pts)
    rng = np.random.RandomState(int(start_time * 1000) % 2**31)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        improved = False
        for i in range(N):
            if time.time() - start_time >= time_limit:
                break
            
            # Try several random perturbations
            for _ in range(5):
                # Random direction on sphere
                dir_vec = rng.randn(3)
                dir_vec /= np.linalg.norm(dir_vec)
                
                # Move point i in this direction (geodesic step)
                theta = step_size * rng.uniform(0.5, 1.5)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                new_pt = cos_t * pts[i] + sin_t * dir_vec
                new_pt /= np.linalg.norm(new_pt)
                
                # Evaluate
                trial = pts.copy()
                trial[i] = new_pt
                trial_loss = _combined_loss(trial)
                
                if trial_loss < best_loss - 1e-10:
                    pts = trial
                    best_loss = trial_loss
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            # Reduce step size
            step_size *= 0.8
    
    return pts, best_loss


def _simulated_annealing(pts, time_limit=30.0):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_loss = _combined_loss(pts)
    rng = np.random.RandomState(int(start_time * 1000) % 2**31)
    
    T = 0.5  # Initial temperature
    T_min = 0.001
    cooling = 0.995
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate candidate
        i = rng.randint(N)
        dir_vec = rng.randn(3)
        dir_vec /= np.linalg.norm(dir_vec)
        theta = 0.1 * rng.uniform(0.5, 1.5)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        new_pt = cos_t * pts[i] + sin_t * dir_vec
        new_pt /= np.linalg.norm(new_pt)
        
        trial = pts.copy()
        trial[i] = new_pt
        trial_loss = _combined_loss(trial)
        
        delta = trial_loss - _combined_loss(pts)
        
        # Metropolis criterion
        if delta < 0 or rng.random() < math.exp(-delta / T):
            pts = trial
            if trial_loss < best_loss:
                best_loss = trial_loss
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_loss


def construct():
    global DEADLINE
    t_start = time.time()
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting configurations
    starts = []
    
    # Fibonacci spiral with different phases
    for phase in [0.0, 1.0, 2.0, 3.0]:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + phase
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts += rng.randn(N, 3) * 1e-2
        starts.append(_normalize(pts))
    
    # Random starts
    for _ in range(3):
        pts = rng.randn(N, 3)
        starts.append(_normalize(pts))
    
    best_pts = None
    best_score = np.inf
    
    for s_idx, start_pts in enumerate(starts):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 20:
            break
        
        # Phase 1: Greedy local search
        try:
            opt_pts, loss1 = _local_search_greedy(start_pts, time_limit=15.0)
        except Exception:
            continue
        
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 15:
            break
        
        # Phase 2: Simulated annealing to escape local minima
        try:
            sa_pts, loss2 = _simulated_annealing(opt_pts, time_limit=20.0)
        except Exception:
            sa_pts = opt_pts
        
        # Phase 3: Final greedy polish
        try:
            final_pts, loss3 = _local_search_greedy(sa_pts, time_limit=10.0, step_size=0.05)
        except Exception:
            final_pts = sa_pts
        
        # Evaluate
        try:
            score = _covering_radius_deg(final_pts)
        except Exception:
            score = np.inf
        
        if score < best_score:
            best_score = score
            best_pts = final_pts.copy()
        
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 10:
            break
    
    # If we didn't find anything, use the first Fibonacci spiral
    if best_pts is None:
        best_pts = starts[0]
    
    # Final normalization and deduplication
    best_pts = _normalize(best_pts)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] += rng.randn(3) * 1e-4
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
