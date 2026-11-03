"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree

N = 26
DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _compute_radii(centers):
    """Given centers (n,2), compute max radii such that circles are disjoint and inside [0,1]^2.
    For each circle, radius = min over:
      - distance to each neighbor (from Delaunay) / 2
      - distance to each boundary
    Then we take the minimum of all these constraints per circle.
    Actually, for non-overlapping: r_i + r_j <= dist(i,j) for all pairs.
    And r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for boundary.
    The tightest packing sets r_i = min over neighbors j of (d_ij/2) and boundary dist.
    But this is only valid if the packing is "tight" - i.e., each circle touches at least one neighbor or boundary.
    For optimization, we use: r_i = min( min_j d_ij/2, boundary_dist_i )
    This gives a lower bound on feasible radii. The sum of these is our objective.
    """
    n = len(centers)
    if n < 2:
        bd = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1 - centers[:, 1]))
        return bd

    # Boundary distances
    bd = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1 - centers[:, 1]))

    # Delaunay triangulation to get neighbor pairs
    try:
        tri = Delaunay(centers)
        # Extract unique edges from triangles
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
    except Exception:
        # Fallback: use k-d tree nearest neighbors
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=4)  # 3 neighbors + self
        edges = set()
        for i in range(n):
            for j in idxs[i, 1:]:
                if j > i:
                    edges.add((i, j))
                else:
                    edges.add((j, i))

    # Compute pairwise distances for all edges
    edge_list = list(edges)
    if not edge_list:
        return bd

    edge_arr = np.array(edge_list, dtype=int)
    diffs = centers[edge_arr[:, 0]] - centers[edge_arr[:, 1]]
    dists = np.linalg.norm(diffs, axis=1)

    # For each circle, the radius constraint from neighbors is d_ij / 2 for each neighbor j
    # r_i <= d_ij / 2 for all neighbors j of i
    # So r_i = min( min_{j neighbor of i} d_ij/2, bd_i )
    neighbor_min = np.full(n, np.inf)
    for k, (a, b) in enumerate(edge_list):
        half_d = dists[k] / 2.0
        if half_d < neighbor_min[a]:
            neighbor_min[a] = half_d
        if half_d < neighbor_min[b]:
            neighbor_min[b] = half_d

    # Where no neighbors found (shouldn't happen), use boundary only
    neighbor_min[~np.isfinite(neighbor_min)] = np.inf

    radii = np.minimum(neighbor_min, bd)
    # Ensure non-negative
    radii = np.maximum(radii, 0.0)
    return radii


def _objective(centers):
    """Negative sum of radii (we minimize negative = maximize sum)."""
    radii = _compute_radii(centers)
    return -np.sum(radii), radii


def _gradient_ascent(centers, step_size=0.01, n_iter=2000, lr_decay=0.999):
    """Simple gradient ascent on center positions to maximize sum of radii.
    Uses finite differences for gradient of the (nonsmooth) objective."""
    n = len(centers)
    centers = centers.copy()
    best_centers = centers.copy()
    best_val, _ = _objective(centers)
    best_val = -best_val  # actual sum of radii

    for it in range(n_iter):
        # Compute gradient via central differences
        grad = np.zeros_like(centers)
        eps = 1e-5
        for i in range(n):
            for d in range(2):
                cp = centers.copy()
                cm = centers.copy()
                cp[i, d] += eps
                cm[i, d] -= eps
                # Keep within [0,1]
                cp = np.clip(cp, 0, 1)
                cm = np.clip(cm, 0, 1)
                f_plus, _ = _objective(cp)
                f_minus, _ = _objective(cm)
                grad[i, d] = (f_plus - f_minus) / (2 * eps)  # gradient of -sum_r

        # Update: gradient ascent on sum_r = gradient descent on -sum_r
        centers = centers - step_size * grad
        centers = np.clip(centers, 1e-6, 1 - 1e-6)

        val, _ = _objective(centers)
        val = -val
        if val > best_val:
            best_val = val
            best_centers = centers.copy()

        step_size *= lr_decay

    return best_centers, best_val


def _repulsion_force(centers, radii, n_steps=500, dt=0.005):
    """Apply repulsion forces to push overlapping circles apart, and attraction to keep them tight.
    Force model:
    - If two circles overlap (r_i + r_j > d_ij), repel them apart.
    - If a circle is outside, push it inward.
    - Additionally, a small attractive force to keep circles close (for tight packing).
    """
    centers = centers.copy()
    n = len(centers)

    for step in range(n_steps):
        forces = np.zeros_like(centers)

        # Boundary forces
        for i in range(n):
            for d in range(2):
                if centers[i, d] < radii[i]:
                    forces[i, d] += (radii[i] - centers[i, d]) * 10.0
                elif centers[i, d] > 1 - radii[i]:
                    forces[i, d] -= (centers[i, d] - (1 - radii[i])) * 10.0

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([1e-6, 1e-6])
                    dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Overlapping: repel
                    force_mag = (min_dist - dist) * 5.0
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist

        centers = centers + dt * forces
        centers = np.clip(centers, 1e-6, 1 - 1e-6)

        # Recompute radii
        radii = _compute_radii(centers)

    return centers, radii


def _polish(centers, n_iter=5000, tol=1e-10):
    """High-precision polish using projected gradient with line search."""
    centers = centers.copy()
    best_val, _ = _objective(centers)
    best_val = -best_val

    lr = 0.005
    for it in range(n_iter):
        # Finite difference gradient
        grad = np.zeros_like(centers)
        eps = 1e-7
        for i in range(centers.shape[0]):
            for d in range(2):
                cp = centers.copy()
                cm = centers.copy()
                cp[i, d] += eps
                cm[i, d] -= eps
                cp = np.clip(cp, 0, 1)
                cm = np.clip(cm, 0, 1)
                fp, _ = _objective(cp)
                fm, _ = _objective(cm)
                grad[i, d] = (fp - fm) / (2 * eps)

        # Gradient ascent step
        new_centers = centers - lr * grad
        new_centers = np.clip(new_centers, 1e-8, 1 - 1e-8)

        new_val, _ = _objective(new_centers)
        new_val = -new_val

        if new_val > best_val:
            best_val = new_val
            centers = new_centers
            lr = min(lr * 1.01, 0.01)
        else:
            lr *= 0.95

        if lr < 1e-8:
            break

    return centers, best_val


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_centers = None
    best_value = -np.inf

    # Multiple random restarts
    rng = np.random.default_rng(42)

    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break

        # Initialize centers
        if restart == 0:
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))
            centers = np.array([
                [(i + 0.5) / side, (j + 0.5) / side]
                for i in range(side) for j in range(side)
            ][:N])
        elif restart == 1:
            # Random initialization
            centers = rng.random((N, 2)) * 0.8 + 0.1
        else:
            # Perturbed best or random
            if best_centers is not None and rng.random() < 0.5:
                centers = best_centers + rng.normal(0, 0.05, (N, 2))
                centers = np.clip(centers, 0.05, 0.95)
            else:
                centers = rng.random((N, 2)) * 0.8 + 0.1

        # Step 1: Repulsion-based relaxation
        radii = _compute_radii(centers)
        centers, radii = _repulsion_force(centers, radii, n_steps=300, dt=0.003)

        if time.time() > deadline - 25:
            break

        # Step 2: Gradient ascent
        centers, val = _gradient_ascent(centers, step_size=0.01, n_iter=1000, lr_decay=0.9995)

        if time.time() > deadline - 15:
            break

        # Step 3: Polish
        centers, val = _polish(centers, n_iter=2000, tol=1e-10)

        if val > best_value:
            best_value = val
            best_centers = centers.copy()

    if best_centers is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        centers = np.array([
            [(i + 0.5) / side, (j + 0.5) / side]
            for i in range(side) for j in range(side)
        ][:N])
        radii = _compute_radii(centers)
        return np.column_stack([centers, radii])

    # Final: compute radii and return
    radii = _compute_radii(best_centers)

    # Validate: check all circles are inside and disjoint
    # (defensive check)
    for i in range(N):
        x, y, r = best_centers[i, 0], best_centers[i, 1], radii[i]
        if r < 0:
            radii[i] = 0
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            radii[i] = min(r, x, 1 - x, y, 1 - y)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            if radii[i] + radii[j] > d + 1e-9:
                # Shrink to fix
                scale = d / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return np.column_stack([best_centers, radii])
