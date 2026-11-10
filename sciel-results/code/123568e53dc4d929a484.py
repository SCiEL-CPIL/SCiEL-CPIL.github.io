"""Combined program: hex lattice init + smooth gradient L-BFGS + gradient-based local refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Compute the radius score: min of pairwise distances/2 and edge distances."""
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_edge = np.min(edge_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_edge, min_pair)

def _neg_score(centers_flat):
    centers = centers_flat.reshape(-1, 2)
    if not np.all(np.isfinite(centers)):
        return 1e10
    if not (np.all(centers >= -0.5) and np.all(centers <= 0.5)):
        return 1e10
    return -_score(centers)

def _smooth_gradient(centers_flat, beta=50.0):
    """Compute smooth gradient using log-sum-exp soft-min for near-active constraints."""
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    grad = np.zeros_like(centers_flat)

    # Edge constraints: for each circle, distance to each of 4 edges
    # Edge distances for circle i: [0.5-x_i, 0.5+y_i, 0.5-y_i, 0.5+x_i]
    edge_dists = np.column_stack([
        0.5 - centers[:, 0],   # left
        0.5 + centers[:, 0],   # right
        0.5 - centers[:, 1],   # bottom
        0.5 + centers[:, 1]    # top
    ])
    # Gradients w.r.t. x,y for each edge constraint
    # d(0.5-x)/dx = -1, d(0.5+x)/dx = 1, d(0.5-y)/dy = -1, d(0.5+y)/dy = 1
    edge_grads = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # shape (4, 2)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)

    # Soft-min over all constraints (edges and pairs)
    # For each constraint, we have a value and a gradient contribution
    # Edge constraints: n*4 values, each with gradient
    # Pair constraints: n*(n-1)/2 values, each with gradient

    # Collect all constraint values
    edge_vals = edge_dists.flatten()  # shape (4n,)
    pair_vals = dists[np.triu_indices(n, k=1)] / 2.0  # shape (n*(n-1)/2,)

    all_vals = np.concatenate([edge_vals, pair_vals])

    # Soft-min: -logsumexp(-beta * vals) / beta approximates min
    # Gradient of soft-min w.r.t. each val: -exp(-beta*val) / sum(exp(-beta*val))
    # But we want gradient of -soft_min, which is the negative of that

    # To avoid overflow, use log-sum-exp trick
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # weights sum to 1, weight_k = exp(-beta*val_k)/sum

    # The gradient of the soft-min objective w.r.t. centers:
    # d/dx [-softmin] = sum_k weights_k * grad_k
    # where grad_k is the gradient of constraint k w.r.t. centers

    # Edge constraint gradients
    # For edge constraint k (circle i, edge e), grad = edge_grads[e] for that circle
    edge_weights = weights[:4*n].reshape(n, 4)
    for i in range(n):
        for e in range(4):
            grad[2*i:2*i+2] += edge_weights[i, e] * edge_grads[e]

    # Pair constraint gradients
    # For pair (i,j) with i<j, value = dist(i,j)/2
    # d/dx_i (dist/2) = (x_i - x_j) / dist
    # d/dx_j (dist/2) = (x_j - x_i) / dist
    pair_weights = weights[4*n:]
    pair_idx = np.triu_indices(n, k=1)
    for idx in range(len(pair_weights)):
        i, j = pair_idx[0][idx], pair_idx[1][idx]
        w = pair_weights[idx]
        if dists[i, j] < 1e-12:
            continue
        vec = centers[i] - centers[j]
        dvec = vec / dists[i, j]
        grad[2*i:2*i+2] += w * dvec
        grad[2*j:2*j+2] -= w * dvec

    return -grad  # gradient of -softmin is what we want for minimization

def _hex_init(rng, jitter=0.0):
    """Generate a hexagonal lattice, pick N points nearest origin, scale to fit."""
    s = 1.0
    pts = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            pts.append([x, y])
    pts = np.array(pts)
    d2 = np.sum(pts ** 2, axis=1)
    idx = np.argsort(d2)[:N]
    pos = pts[idx].copy()
    if jitter > 0:
        pos += rng.uniform(-jitter, jitter, size=pos.shape)
    max_abs = np.max(np.abs(pos))
    if max_abs > 0:
        pos *= (0.5 / max_abs)
    pos = np.clip(pos, -0.4999, 0.4999)
    return pos

def _grid_init(rng, jitter=0.0):
    """Generate a grid lattice, pick N points, scale to fit."""
    # Try different grid sizes
    best_pos = None
    best_score = -1
    for cols in range(7, 9):
        rows = int(np.ceil(N / cols))
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) < N:
                    pts.append([c, r])
        pts = np.array(pts, dtype=float)
        if jitter > 0:
            pts += rng.uniform(-jitter, jitter, size=pts.shape)
        max_abs = np.max(np.abs(pts))
        if max_abs > 0:
            pts *= (0.5 / max_abs)
        pts = np.clip(pts, -0.4999, 0.4999)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pos = pts.copy()
    return best_pos

def _local_refine(positions, deadline, rng, beta=50.0):
    """Gradient-based local refinement using smooth gradient."""
    pos = positions.copy().flatten()
    current_r = _score(pos.reshape(-1, 2))
    best_r = current_r
    best_pos = pos.copy()
    step = 0.01
    it = 0
    no_improve = 0

    while time.time() < deadline - 3.0 and it < 15000:
        it += 1
        # Compute smooth gradient
        try:
            grad = _smooth_gradient(pos, beta=beta)
        except Exception:
            break

        # Take a step in the gradient direction (maximize radius = minimize -radius)
        # We want to move in the direction that increases radius
        # _smooth_gradient returns gradient of -softmin, so we move opposite
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            no_improve += 1
            if no_improve > 50:
                break
            # Random perturbation
            k = rng.randint(N)
            pos[2*k:2*k+2] += rng.uniform(-0.003, 0.003, size=2)
            pos = np.clip(pos, -0.4999, 0.4999)
            continue

        # Adaptive step size
        trial_step = step
        for _ in range(5):
            new_pos = pos - trial_step * grad
            new_pos = np.clip(new_pos, -0.4999, 0.4999)
            new_r = _score(new_pos.reshape(-1, 2))
            if new_r > current_r:
                pos = new_pos
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = pos.copy()
                    no_improve = 0
                    break
            else:
                trial_step *= 0.5

        else:
            # No improvement, try smaller step or random kick
            no_improve += 1
            if no_improve > 30:
                # Random kick from best
                pos = best_pos.copy()
                current_r = best_r
                k = rng.randint(N)
                pos[2*k:2*k+2] += rng.uniform(-0.005, 0.005, size=2)
                pos = np.clip(pos, -0.4999, 0.4999)
                no_improve = 0
                continue
            if no_improve > 100:
                break

        # Anneal step size occasionally
        if it % 200 == 0 and step > 0.001:
            step *= 0.9

    return best_pos.reshape(-1, 2), best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    rng = np.random.RandomState(42)
    best_centers = None
    best_score = -1.0

    def _try_optimize(init_pos, maxiter=3000):
        """Run L-BFGS with smooth gradient from initial position."""
        nonlocal best_centers, best_score
        try:
            result = minimize(
                _neg_score,
                init_pos.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50},
                jac=lambda x: _smooth_gradient(x, beta=50.0)
            )
            c = result.x.reshape(-1, 2)
            s = _score(c)
            if s > best_score:
                best_centers = c.copy()
                best_score = s
                # Local refinement
                if time.time() < deadline - 10.0:
                    c2, s2 = _local_refine(best_centers, deadline, rng, beta=50.0)
                    if s2 > best_score:
                        best_centers = c2
                        best_score = s2
        except Exception:
            pass

    # Phase 1: Hex lattice init
    hex_pos = _hex_init(rng, jitter=0.0)
    _try_optimize(hex_pos, maxiter=3000)

    # Phase 2: Grid init
    if time.time() < deadline - 15.0:
        grid_pos = _grid_init(rng, jitter=0.0)
        if grid_pos is not None:
            _try_optimize(grid_pos, maxiter=2500)

    # Phase 3: Multiple restarts with jitter
    for restart in range(5):
        if time.time() > deadline - 15.0:
            break
        jitter = 0.01 * (restart + 1)
        hex_j = _hex_init(rng, jitter=jitter)
        _try_optimize(hex_j, maxiter=2000)

    # Phase 4: Final polish with very tight tolerance
    if time.time() < deadline - 5.0 and best_centers is not None:
        try:
            result = minimize(
                _neg_score,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100},
                jac=lambda x: _smooth_gradient(x, beta=100.0)
            )
            c = result.x.reshape(-1, 2)
            s = _score(c)
            if s > best_score:
                best_centers = c.copy()
                best_score = s
        except Exception:
            pass

    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers
# EVOLVE-BLOCK-END
