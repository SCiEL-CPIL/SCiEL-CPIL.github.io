"""Rewritten program: BCC lattice init + soft-min L-BFGS-B + exact polish."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def compute_radius(x):
        """Compute the maximum radius for given positions."""
        positions = x.reshape(-1, 3)
        boundary_dists = 0.5 - np.abs(positions)
        min_boundary = np.min(boundary_dists)
        if N > 1:
            min_pair = np.min(pdist(positions)) / 2.0
        else:
            min_pair = min_boundary
        return min(min_boundary, min_pair)

    def soft_min(values, beta=50.0):
        """Smooth approximation of min using log-sum-exp."""
        vals = np.clip(values, -1e10, 1e10)
        return -np.logaddexp.reduce(-vals / beta) * beta

    def negative_soft_radius(x, beta=50.0):
        """Smooth objective: negative of soft-min radius."""
        positions = x.reshape(-1, 3)
        boundary_dists = 0.5 - np.abs(positions)
        if N > 1:
            pair_dists = pdist(positions) / 2.0
        else:
            pair_dists = np.array([])
        all_dists = np.concatenate([boundary_dists, pair_dists])
        return -soft_min(all_dists, beta)

    def negative_soft_radius_grad(x, beta=50.0):
        """Gradient of the smooth objective."""
        positions = x.reshape(-1, 3)
        boundary_dists = 0.5 - np.abs(positions)
        if N > 1:
            pair_dists = pdist(positions) / 2.0
        else:
            pair_dists = np.array([])
        all_dists = np.concatenate([boundary_dists, pair_dists])

        neg_vals = -all_dists / beta
        max_neg = np.max(neg_vals)
        exp_vals = np.exp(neg_vals - max_neg)
        weights = exp_vals / np.sum(exp_vals)

        grad_dists = -weights  # d(-soft_min)/d(dist_i) = -w_i

        grad = np.zeros_like(x)

        # Boundary part
        sign_x = np.sign(positions)
        grad_boundary = -sign_x * grad_dists[:N*3].reshape(-1, 3)
        grad[:N*3] += grad_boundary.flatten()

        # Pairwise part
        if N > 1:
            idx_i, idx_j = np.triu_indices(N, 1)
            diff = positions[idx_i] - positions[idx_j]
            dists = np.linalg.norm(diff, axis=1)
            dists = np.maximum(dists, 1e-12)
            unit_vecs = diff / dists[:, None]
            pair_weights = grad_dists[N*3:]

            for k in range(len(idx_i)):
                i, j = idx_i[k], idx_j[k]
                w = pair_weights[k]
                grad[3*i:3*i+3] += w * unit_vecs[k] / 2.0
                grad[3*j:3*j+3] -= w * unit_vecs[k] / 2.0

        return grad

    # BCC lattice initialization: more isotropic and denser than simple cubic
    def bcc_init():
        """Create BCC lattice points scaled to fit the unit cube."""
        # BCC lattice: corners + body center of each cube cell
        # We need 50 points. Use a 3x3x3 grid of BCC cells = 27 cells * 2 = 54 points
        # Take first 50
        m = 3
        pts = []
        spacing = 1.0 / m
        for ix in range(m):
            for iy in range(m):
                for iz in range(m):
                    # Corner
                    cx = -0.5 + (ix + 0.5) * spacing
                    cy = -0.5 + (iy + 0.5) * spacing
                    cz = -0.5 + (iz + 0.5) * spacing
                    pts.append([cx, cy, cz])
                    # Body center
                    bx = -0.5 + (ix + 1.0) * spacing
                    by = -0.5 + (iy + 1.0) * spacing
                    bz = -0.5 + (iz + 1.0) * spacing
                    pts.append([bx, by, bz])
        pts = np.array(pts[:N])
        return pts.flatten()

    def simple_cubic_init():
        """Simple cubic grid initialization."""
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].flatten()

    def random_init(rng, scale=0.3):
        """Random initialization within the cube."""
        return (rng.rand(N, 3) * 2.0 - 1.0) * scale + (rng.rand(3) * 2.0 - 1.0) * 0.1

    # Generate multiple initial configurations
    rng = np.random.RandomState(42)
    inits = [
        bcc_init(),
        simple_cubic_init(),
        random_init(rng, 0.35),
        random_init(rng, 0.30),
        random_init(rng, 0.40),
    ]

    best_x = None
    best_r = -np.inf
    bounds = [(-0.5, 0.5)] * (N * 3)

    # Phase 1: Optimize each initial configuration with soft-min L-BFGS-B
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        # Clip to bounds
        x0 = np.clip(x0, -0.49, 0.49)
        try:
            result = minimize(negative_soft_radius, x0, method='L-BFGS-B',
                             bounds=bounds,
                             jac=lambda x: negative_soft_radius_grad(x, beta=30.0),
                             options={'maxiter': 400, 'gtol': 1e-7, 'ftol': 1e-9})
            r_new = compute_radius(result.x)
            if r_new > best_r:
                best_r = r_new
                best_x = result.x.copy()
        except Exception:
            pass

    if best_x is None:
        best_x = bcc_init()
        best_r = compute_radius(best_x)

    # Phase 2: High-precision polish with tighter beta
    if time.time() < deadline - 20.0:
        try:
            result = minimize(negative_soft_radius, best_x, method='L-BFGS-B',
                             bounds=bounds,
                             jac=lambda x: negative_soft_radius_grad(x, beta=100.0),
                             options={'maxiter': 600, 'gtol': 1e-9, 'ftol': 1e-11})
            r_new = compute_radius(result.x)
            if r_new > best_r:
                best_r = r_new
                best_x = result.x.copy()
        except Exception:
            pass

    # Phase 3: Exact local solve with COBYLA on the nonsmooth objective
    if time.time() < deadline - 10.0:
        def negative_radius_exact(x):
            return -compute_radius(x)
        try:
            result = minimize(negative_radius_exact, best_x, method='COBYLA',
                             options={'maxiter': 2000, 'rhobeg': 1e-6, 'catol': 1e-12})
            r_new = compute_radius(result.x)
            if r_new > best_r:
                best_r = r_new
                best_x = result.x.copy()
        except Exception:
            pass

    # Phase 4: Final tiny perturbation search to escape shallow local minima
    if time.time() < deadline - 5.0:
        rng2 = np.random.RandomState(123)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            x_pert = best_x + rng2.randn(N * 3) * 1e-4
            x_pert = np.clip(x_pert, -0.499, 0.499)
            try:
                result = minimize(negative_soft_radius, x_pert, method='L-BFGS-B',
                                 bounds=bounds,
                                 jac=lambda x: negative_soft_radius_grad(x, beta=200.0),
                                 options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
                r_new = compute_radius(result.x)
                if r_new > best_r:
                    best_r = r_new
                    best_x = result.x.copy()
            except Exception:
                pass

    # Ensure final result is within bounds
    best_x = np.clip(best_x, -0.5, 0.5)

    return best_x.reshape(-1, 3)
# EVOLVE-BLOCK-END
