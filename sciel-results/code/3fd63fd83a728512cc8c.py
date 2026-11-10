"""Combined program: hex lattice init + gradient L-BFGS + targeted local refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the radius score: min of pairwise distances/2 and edge distances."""
    n = len(centers)
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

def _gradient(centers_flat):
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    edge_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_edge = np.min(edge_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_full = np.min(dists)
    min_pair = min_pair_full / 2.0
    grad = np.zeros_like(centers_flat)

    if min_edge < min_pair:
        for i in range(n):
            if edge_dists[i] <= min_edge + 1e-12:
                d_left = 0.5 - centers[i, 0]
                d_right = centers[i, 0] + 0.5
                d_bottom = 0.5 - centers[i, 1]
                d_top = centers[i, 1] + 0.5
                min_d = min(d_left, d_right, d_bottom, d_top)
                if d_left <= min_d + 1e-12:
                    grad[2*i] = -1.0
                if d_right <= min_d + 1e-12:
                    grad[2*i] = 1.0
                if d_bottom <= min_d + 1e-12:
                    grad[2*i+1] = -1.0
                if d_top <= min_d + 1e-12:
                    grad[2*i+1] = 1.0
    else:
        min_dist_threshold = min_pair_full + 1e-10
        for i in range(n):
            for j in range(i+1, n):
                if dists[i, j] <= min_dist_threshold:
                    vec = centers[i] - centers[j]
                    d_vec = vec / (2.0 * dists[i, j])
                    grad[2*i:2*i+2] += d_vec
                    grad[2*j:2*j+2] -= d_vec
    return -grad

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

def _local_refine(positions, deadline, rng):
    """Targeted local search: move the bottleneck circle(s) to increase radius."""
    pos = positions.copy()
    current_r = _score(pos)
    best_r = current_r
    best_pos = pos.copy()
    step = 0.005
    it = 0
    while time.time() < deadline - 2.0 and it < 20000:
        it += 1
        # Recompute bottleneck
        abs_pos = np.abs(pos)
        min_bound = 0.5 - np.max(abs_pos)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_bound = min_bound
        r_dist = min_dist / 2.0

        if r_bound < r_dist:
            # Boundary limiting: move the offending circle toward center
            flat_idx = np.argmax(abs_pos)
            ci = flat_idx // 2
            dir_vec = -pos[ci]
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-8:
                dir_vec = np.array([1.0, 0.0])
            else:
                dir_vec /= norm
            new_pos = pos[ci].copy() + dir_vec * step
            if np.any(np.abs(new_pos) > 0.5):
                continue
            pos[ci] = new_pos
        else:
            # Distance limiting: separate the closest pair
            flat_idx = np.argmin(dists)
            i, j = np.unravel_index(flat_idx, dists.shape)
            vec = pos[j] - pos[i]
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                ang = rng.uniform(0, 2 * np.pi)
                dir_i = np.array([np.cos(ang), np.sin(ang)])
            else:
                dir_i = -vec / norm
            dir_j = -dir_i
            new_i = pos[i].copy() + dir_i * step
            new_j = pos[j].copy() + dir_j * step
            if np.any(np.abs(new_i) > 0.5) or np.any(np.abs(new_j) > 0.5):
                # Try moving only the one that fits
                if not np.any(np.abs(new_i) > 0.5):
                    pos[i] = new_i
                elif not np.any(np.abs(new_j) > 0.5):
                    pos[j] = new_j
                else:
                    continue
            else:
                pos[i] = new_i
                pos[j] = new_j

        new_r = _score(pos)
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pos = pos.copy()
                if it % 500 == 0 and step > 0.0005:
                    step *= 0.9  # anneal
        else:
            # Revert: recompute from best and try a random kick
            pos = best_pos.copy()
            current_r = best_r
            # Random small perturbation to escape
            k = rng.randint(N)
            pos[k] += rng.uniform(-0.002, 0.002, size=2)
            pos = np.clip(pos, -0.4999, 0.4999)
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s

    rng = np.random.RandomState(42)
    best_centers = None
    best_score = -1.0

    # Phase 1: Hex lattice init + L-BFGS
    hex_pos = _hex_init(rng, jitter=0.0)
    from scipy.optimize import minimize

    result = minimize(
        _neg_score,
        hex_pos.flatten(),
        method='L-BFGS-B',
        bounds=[(-0.4999, 0.4999)] * (2 * N),
        options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50},
        jac=lambda x: _gradient(x)
    )
    c1 = result.x.reshape(-1, 2)
    s1 = _score(c1)
    best_centers = c1.copy()
    best_score = s1

    # Phase 2: Local refinement on the L-BFGS result
    if time.time() < deadline - 5.0:
        c2, s2 = _local_refine(best_centers, deadline, rng)
        if s2 > best_score:
            best_centers = c2
            best_score = s2

    # Phase 3: Try a few more hex inits with different jitter
    for restart in range(3):
        if time.time() > deadline - 10.0:
            break
        hex_j = _hex_init(rng, jitter=0.02)
        try:
            res = minimize(
                _neg_score,
                hex_j.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50},
                jac=lambda x: _gradient(x)
            )
            cr = res.x.reshape(-1, 2)
            sr = _score(cr)
            if sr > best_score:
                best_centers = cr
                best_score = sr
                if time.time() < deadline - 5.0:
                    c3, s3 = _local_refine(best_centers, deadline, rng)
                    if s3 > best_score:
                        best_centers = c3
                        best_score = s3
        except Exception:
            pass

    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers
# EVOLVE-BLOCK-END
