"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
N_TRIS = len(_idx_i)


def _bary_to_cart(bary):
    """Convert barycentric coords (N,3) summing to 1 to cartesian (N,2)."""
    # bary[:,0]*A + bary[:,1]*B + bary[:,2]*C
    # A=(0,0), B=(1,0), C=(0.5, H)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert cartesian (N,2) to barycentric (N,3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    b2 = y / H  # weight of C
    b1 = x - 0.5 * b2  # weight of B
    b0 = 1.0 - b1 - b2  # weight of A
    return np.column_stack([b0, b1, b2])


def _min_area_from_bary(bary):
    """Compute minimum triangle area given barycentric coordinates (N,3)."""
    pts = _bary_to_cart(bary)
    p1 = pts[_idx_i]
    p2 = pts[_idx_j]
    p3 = pts[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _min_area_from_cart(pts):
    """Compute minimum triangle area given cartesian points (N,2)."""
    p1 = pts[_idx_i]
    p2 = pts[_idx_j]
    p3 = pts[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _objective_bary(params):
    """Objective for optimization: negative min area (we minimize)."""
    bary = params.reshape(N, 3)
    # Ensure barycentric coords are valid (non-negative, sum to 1)
    # We'll use a penalty approach
    s = bary.sum(axis=1)
    bary = bary / s[:, np.newaxis]
    # Penalty for negative coordinates
    penalty = np.sum(np.clip(-bary, 0, None) ** 2) * 10.0
    if penalty > 0.01:
        return 1.0 + penalty
    area = _min_area_from_bary(bary)
    return -area


def _project_to_bary(params):
    """Project parameters to valid barycentric coordinates."""
    bary = params.reshape(N, 3)
    bary = np.clip(bary, 0, None)
    s = bary.sum(axis=1)
    s[s == 0] = 1
    bary = bary / s[:, np.newaxis]
    return bary


def _local_refine(pts, max_iter=500):
    """Local refinement: iteratively move the worst triangle's vertices to improve min area."""
    best = _min_area_from_cart(pts)
    pts = pts.copy()
    for it in range(max_iter):
        p1 = pts[_idx_i]
        p2 = pts[_idx_j]
        p3 = pts[_idx_k]
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        worst_idx = np.argmin(areas)
        wi, wj, wk = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
        
        # Try to move one of the three vertices slightly to increase this triangle's area
        # while not decreasing other triangles too much
        best_move = None
        best_improvement = 0
        for vi in [wi, wj, wk]:
            for dx in np.linspace(-0.02, 0.02, 9):
                for dy in np.linspace(-0.02, 0.02, 9):
                    new_pts = pts.copy()
                    new_pts[vi, 0] += dx
                    new_pts[vi, 1] += dy
                    # Check if still inside triangle
                    bary = _cart_to_bary(new_pts)
                    if np.any(bary < -1e-9):
                        continue
                    new_area = _min_area_from_cart(new_pts)
                    if new_area > best + 1e-12:
                        if new_area - best > best_improvement:
                            best_improvement = new_area - best
                            best_move = new_pts
        if best_move is not None:
            pts = best_move
            best = _min_area_from_cart(pts)
        else:
            break
    return pts, best


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Phase 1: Lattice-based initialization in barycentric space
    # Use a grid of barycentric coordinates
    best_pts = None
    best_score = 0
    
    # Try multiple lattice resolutions
    for n_grid in [3, 4, 5]:
        if time.time() > deadline:
            break
        # Generate all lattice points in barycentric space
        lattice_pts = []
        for a in range(n_grid + 1):
            for b in range(n_grid + 1 - a):
                c = n_grid - a - b
                if a + b + c == n_grid:
                    lattice_pts.append([a / n_grid, b / n_grid, c / n_grid])
        lattice_pts = np.array(lattice_pts)
        
        # Select N points from lattice to maximize min area
        # Use greedy: start with corners, then add points that maximize min distance
        # Actually, let's use a simple approach: random subsets with replacement
        rng = np.random.RandomState(42)
        for trial in range(200):
            if time.time() > deadline:
                break
            # Random selection with some structure
            idx = rng.choice(len(lattice_pts), size=N, replace=False)
            bary = lattice_pts[idx]
            # Add small random perturbation
            bary = bary + rng.normal(0, 0.05, size=bary.shape)
            bary = np.clip(bary, 0, None)
            s = bary.sum(axis=1)
            bary = bary / s[:, np.newaxis]
            pts = _bary_to_cart(bary)
            score = _min_area_from_cart(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    
    # Phase 2: Differential evolution on barycentric coordinates
    if time.time() < deadline:
        # Parameters: N*3 barycentric coords, but we'll normalize
        # Use a simpler parameterization: N*2 cartesian, with penalty for being outside
        def obj_cart(params):
            pts = params.reshape(N, 2)
            # Check if inside triangle
            bary = _cart_to_bary(pts)
            if np.any(bary < -0.01):
                return 1.0 + np.sum(np.clip(-bary, 0, None) ** 2) * 10
            return -_min_area_from_cart(pts)
        
        # Bounds for cartesian coordinates
        bounds = [(0, 1), (0, H)] * N
        
        # Use differential evolution
        de_result = differential_evolution(
            obj_cart, bounds, maxiter=100, popsize=15,
            seed=42, tol=1e-8, polish=False,
            updating='deferred'
        )
        if de_result.fun < 0:
            de_pts = de_result.x.reshape(N, 2)
            de_score = -de_result.fun
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
    
    # Phase 3: Local refinement
    if time.time() < deadline and best_pts is not None:
        refined_pts, refined_score = _local_refine(best_pts, max_iter=1000)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    # Phase 4: More local refinement with different starting points
    if time.time() < deadline:
        # Try refining from several perturbed versions
        rng = np.random.RandomState(123)
        for trial in range(20):
            if time.time() > deadline:
                break
            perturbed = best_pts + rng.normal(0, 0.01, size=best_pts.shape)
            # Project back inside
            bary = _cart_to_bary(perturbed)
            bary = np.clip(bary, 0, None)
            s = bary.sum(axis=1)
            bary = bary / s[:, np.newaxis]
            perturbed = _bary_to_cart(bary)
            refined_pts, refined_score = _local_refine(perturbed, max_iter=500)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts
    
    return best_pts


# Quick test
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min area:", _min_area_from_cart(pts))
    print("Area of equilateral triangle:", 0.5 * np.sqrt(3))
    print("Normalized min area:", _min_area_from_cart(pts) / (0.5 * np.sqrt(3)))
