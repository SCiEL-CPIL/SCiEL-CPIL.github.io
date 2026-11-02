"""Force-directed circle packing with Gauss-Newton polish and Delaunay contact graph."""
import numpy as np
from scipy.spatial import Delaunay
import time

N = 26

# EVOLVE-BLOCK-START
DEADLINE = 150.0
TIME_MARGIN = 2.0  # seconds safety margin


def _delaunay_edges(points):
    """Return set of edge indices (i,j) from Delaunay triangulation."""
    if len(points) < 3:
        return set()
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    if i > j:
                        i, j = j, i
                    edges.add((i, j))
        return edges
    except Exception:
        return set()


def _pairwise_edges(points):
    """Fallback: all pairs (expensive but always works)."""
    n = len(points)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            edges.add((i, j))
    return edges


def _force_directed_step(pos, radii, edges, dt, repulse_strength, attract_strength):
    """One step of force-directed relaxation.
    pos: (N,2) centers, radii: (N,)
    Returns updated pos, radii.
    """
    n = len(pos)
    forces = np.zeros_like(pos)
    dr = np.zeros(n)

    # Repulsion between overlapping (or close) circles
    for (i, j) in edges:
        diff = pos[i] - pos[j]
        d = np.linalg.norm(diff)
        if d < 1e-12:
            diff = np.array([1e-8, 1e-8])
            d = np.linalg.norm(diff)
        min_dist = radii[i] + radii[j]
        if d < min_dist:
            # Overlapping: push apart
            overlap = min_dist - d
            f = repulse_strength * overlap / d
            forces[i] += f * diff
            forces[j] -= f * diff
        else:
            # Attract: pull together to grow
            gap = d - min_dist
            f = -attract_strength * gap / d
            forces[i] += f * diff
            forces[j] -= f * diff

    # Wall constraints: push away from walls
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        # Left wall
        if x < r:
            forces[i, 0] += repulse_strength * (r - x)
            dr[i] -= attract_strength * (r - x)
        # Right wall
        if x > 1.0 - r:
            forces[i, 0] -= repulse_strength * (x - (1.0 - r))
            dr[i] -= attract_strength * (x - (1.0 - r))
        # Bottom wall
        if y < r:
            forces[i, 1] += repulse_strength * (r - y)
            dr[i] -= attract_strength * (r - y)
        # Top wall
        if y > 1.0 - r:
            forces[i, 1] -= repulse_strength * (y - (1.0 - r))
            dr[i] -= attract_strength * (y - (1.0 - r))

    # Grow radii if not too constrained
    for i in range(n):
        # Limit radius growth
        max_r = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
        if radii[i] < max_r:
            dr[i] += attract_strength * 0.1
        else:
            dr[i] = min(dr[i], 0)

    # Update
    pos = pos + dt * forces
    radii = radii + dt * dr

    # Clamp radii to be non-negative and within walls
    for i in range(n):
        radii[i] = max(radii[i], 1e-6)
        max_r = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
        radii[i] = min(radii[i], max(0.01, max_r))

    # Clamp positions to be inside square
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
        pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])

    return pos, radii


def _gauss_newton_polish(pos, radii, edges, max_iter=50, tol=1e-8):
    """Gauss-Newton polish on distance constraints.
    We solve for small adjustments to pos and radii to satisfy:
    - For each edge (i,j): ||pos_i - pos_j|| >= r_i + r_j (equality at contact)
    - For each i: r_i <= pos_i[x], r_i <= pos_i[y], 1-r_i <= pos_i[x], 1-r_i <= pos_i[y]
    We use a least-squares formulation to push toward feasibility while maximizing radii.
    """
    n = len(pos)
    # Variables: dx (2n), dr (n) -> total 3n
    # For each contact edge, we want distance == r_i + r_j
    # For walls, we want r_i == min distance to walls (active constraints)

    for it in range(max_iter):
        # Build Jacobian and residual for contact constraints
        m = len(edges)
        if m == 0:
            break
        J = np.zeros((m, 3 * n))
        res = np.zeros(m)
        edge_list = list(edges)
        for k, (i, j) in enumerate(edge_list):
            diff = pos[i] - pos[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                diff = np.array([1e-8, 1e-8])
                d = np.linalg.norm(diff)
            res[k] = d - (radii[i] + radii[j])
            # d/dx_i = diff/d, d/dx_j = -diff/d
            J[k, 2 * i] = diff[0] / d
            J[k, 2 * i + 1] = diff[1] / d
            J[k, 2 * j] = -diff[0] / d
            J[k, 2 * j + 1] = -diff[1] / d
            J[k, 2 * n + i] = 1.0
            J[k, 2 * n + j] = 1.0

        # Solve least squares: J @ delta = -res
        try:
            delta, residuals, rank, sv = np.linalg.lstsq(J, -res, rcond=None)
        except Exception:
            break

        # Split delta
        dpos = delta[:2 * n].reshape(n, 2)
        dr = delta[2 * n:]

        # Damping
        step = 0.5
        # Try to take step, reduce if it increases violation
        best_step = step
        best_viol = np.sum(np.maximum(res, 0) ** 2)
        for trial in range(10):
            trial_pos = pos + step * dpos
            trial_radii = radii + step * dr
            # Check feasibility
            feasible = True
            for i in range(n):
                if trial_radii[i] < 0 or trial_pos[i, 0] < trial_radii[i] or \
                   trial_pos[i, 0] > 1 - trial_radii[i] or \
                   trial_pos[i, 1] < trial_radii[i] or \
                   trial_pos[i, 1] > 1 - trial_radii[i]:
                    feasible = False
                    break
            if feasible:
                # Check overlap violations
                for (i, j) in edge_list:
                    d = np.linalg.norm(trial_pos[i] - trial_pos[j])
                    if d < trial_radii[i] + trial_radii[j] - 1e-9:
                        feasible = False
                        break
            if feasible:
                best_step = step
                break
            step *= 0.5

        pos = pos + best_step * dpos
        radii = radii + best_step * dr

        # Clamp
        for i in range(n):
            radii[i] = max(radii[i], 1e-6)
            max_r = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            radii[i] = min(radii[i], max(0.01, max_r))
            pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
            pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])

        if np.linalg.norm(delta) < tol:
            break

    return pos, radii


def _initial_grid(n):
    """Initial grid placement."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells[:n])


def _random_init(n, seed=42):
    """Random initial placement."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.05)
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - TIME_MARGIN

    # Start with grid
    pos, radii = _initial_grid(N)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()

    # Force-directed relaxation
    edges = _delaunay_edges(pos)
    dt = 0.01
    repulse = 10.0
    attract = 5.0

    for step in range(2000):
        if time.time() > deadline:
            break
        pos, radii = _force_directed_step(pos, radii, edges, dt, repulse, attract)
        # Rebuild edges periodically
        if step % 50 == 0:
            edges = _delaunay_edges(pos)
        # Decay step size
        dt *= 0.999
        repulse *= 0.998
        attract *= 0.998

        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()

    # Gauss-Newton polish
    if time.time() < deadline - 10:
        edges = _delaunay_edges(best_pos)
        polished_pos, polished_radii = _gauss_newton_polish(best_pos, best_radii, edges, max_iter=100)
        score = np.sum(polished_radii)
        if score > best_score:
            best_score = score
            best_pos = polished_pos.copy()
            best_radii = polished_radii.copy()

    # Final check: ensure validity
    for i in range(N):
        best_radii[i] = max(best_radii[i], 1e-6)
        max_r = min(best_pos[i, 0], best_pos[i, 1], 1.0 - best_pos[i, 0], 1.0 - best_pos[i, 1])
        best_radii[i] = min(best_radii[i], max(0.01, max_r))
        best_pos[i, 0] = np.clip(best_pos[i, 0], best_radii[i], 1.0 - best_radii[i])
        best_pos[i, 1] = np.clip(best_pos[i, 1], best_radii[i], 1.0 - best_radii[i])

    result = np.column_stack([best_pos, best_radii])
    return result
# EVOLVE-BLOCK-END
