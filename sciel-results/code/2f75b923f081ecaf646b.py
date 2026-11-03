"""
Combined approach: Spectral-inspired initialization + vectorized smooth surrogate
optimization with beta-annealing for 50 equal spheres in a unit cube.

The structural idea from the inspiration: use a soft-min (log-sum-exp) surrogate
with vectorized gradients for fast L-BFGS-B optimization, plus beta-annealing
for global-to-local convergence. The refinement machinery from the current
program: multi-stage optimization with perturbation refinement and deadline
guarding.

Additionally, we add spectral-inspired initializations (eigenvector-based)
to provide diverse starting points that may escape local minima inherent in
grid-based initializations.
"""
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_radius(positions):
        """Compute the exact minimum radius for given positions."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (N, 3) or not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0

        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)

        # Wall distances
        dist_to_walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall_dist = np.min(dist_to_walls)

        return float(min(min_pair_dist / 2.0, min_wall_dist))

    def smooth_objective_and_grad(c_flat, beta):
        """
        Computes the smooth surrogate objective (-soft_min) and its gradient.
        c_flat: (3N,)
        beta: temperature parameter
        Returns: (value, grad_flat)
        """
        c = c_flat.reshape(N, 3)

        # 1. Calculate all "radius candidates" g_k
        # Pairwise: d_ij / 2
        tri_i, tri_j = np.triu_indices(N, k=1)
        diff = c[tri_i] - c[tri_j]
        d_pairs = np.linalg.norm(diff, axis=1) / 2.0

        # Walls: 6 distances per point
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])

        # Combine all g_k
        g_all = np.concatenate([d_pairs, wall_dists])

        # 2. Soft Min Calculation
        scaled = -beta * g_all
        m = np.max(scaled)
        log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
        val = -log_sum_exp / beta

        # 3. Gradient Calculation
        weights = np.exp(-beta * g_all - log_sum_exp)

        grad_c = np.zeros_like(c)

        # Pairwise gradient - fully vectorized using np.add.at
        n_pairs = len(d_pairs)
        safe_d = np.maximum(d_pairs, 1e-12)
        factors = weights[:n_pairs] / (4.0 * safe_d)
        factors = np.where(d_pairs > 1e-12, factors, 0.0)

        weighted_diffs = factors[:, np.newaxis] * diff

        # Scatter: add to index i, subtract from index j
        np.add.at(grad_c, tri_i, weighted_diffs)
        np.add.at(grad_c, tri_j, -weighted_diffs)

        # Wall part
        w_start = n_pairs
        grad_c[:, 0] -= weights[w_start: w_start + N]
        grad_c[:, 0] += weights[w_start + N: w_start + 2 * N]
        grad_c[:, 1] -= weights[w_start + 2 * N: w_start + 3 * N]
        grad_c[:, 1] += weights[w_start + 3 * N: w_start + 4 * N]
        grad_c[:, 2] -= weights[w_start + 4 * N: w_start + 5 * N]
        grad_c[:, 2] += weights[w_start + 5 * N: w_start + 6 * N]

        # We want to MINIMIZE -soft_min.
        return -val, -grad_c.flatten()

    def local_optimize_smooth(c_init, beta, maxiter, time_budget):
        """Run L-BFGS-B with smooth objective."""
        if remaining() < time_budget:
            return c_init

        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_flat(x):
            val, _ = smooth_objective_and_grad(x, beta)
            return val

        def grad_flat(x):
            _, grad = smooth_objective_and_grad(x, beta)
            return grad

        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), jac=grad_flat, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def local_optimize_direct(c_init, maxiter, time_budget):
        """Run L-BFGS-B on the exact non-smooth objective."""
        if remaining() < time_budget:
            return c_init

        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_flat(x):
            return -compute_radius(x.reshape(N, 3))

        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def make_initial_grid():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        """Body-centered cubic inspired."""
        positions = []
        a = 0.5
        spacing = (2 * a) / 3.0
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * spacing
                    y = -a + (iy + 0.5) * spacing
                    z = -a + (iz + 0.5) * spacing
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + spacing / 2, y + spacing / 2, z + spacing / 2])
        return np.array(positions[:N])

    def make_spectral_init(seed):
        """
        Spectral-inspired initialization: generate a random symmetric positive-definite
        matrix, take its top N eigenvectors, and map them to 3D space.
        This leverages orthogonality of eigenvectors to encourage uniform spacing.
        """
        rng = np.random.RandomState(seed)
        # Generate a random symmetric matrix
        A = rng.randn(3 * N, 3 * N)
        A = (A + A.T) / 2.0
        # Make it positive definite
        A = A + 3 * N * np.eye(3 * N)

        # Get eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
        except np.linalg.LinAlgError:
            return make_random_init(seed)

        # Take top N eigenvectors (largest eigenvalues)
        idx = np.argsort(eigvals)[::-1][:N]
        V = eigvecs[:, idx]  # shape (3N, N)

        # Map to 3D: each column is a 3N-vector, reshape to N points in 3D
        # Actually, V is (3N, N), so each row group of 3 corresponds to one coordinate
        # Let's reshape: V.T is (N, 3N), reshape to (N, 3, N) -> take first 3*? 
        # Better approach: use the eigenvectors as directions and scale them
        # Simpler: use PCA-like approach on random points
        # Alternative: generate N random points, compute covariance, use eigenvectors
        pts = rng.randn(N, 3)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Project onto top 3 eigenvectors (which is just the original space)
        # Instead, let's use a different spectral approach:
        # Generate a random graph Laplacian and use its eigenvectors
        
        # Simpler spectral init: use eigenvectors of a random SPD matrix to create
        # well-spread points in 3D
        B = rng.randn(3, 3)
        B = (B + B.T) / 2.0 + 3 * np.eye(3)
        eigvals_b, eigvecs_b = np.linalg.eigh(B)
        
        # Create points as linear combinations of eigenvectors with different scales
        positions = []
        for i in range(N):
            t = i / N
            # Interpolate between eigenvectors with varying weights
            w = np.array([t, 1 - t, 0.5])
            w = w / np.linalg.norm(w)
            p = np.dot(w, eigvecs_b)
            # Scale to fit in cube
            p = p * 0.4
            positions.append(p)
        
        positions = np.array(positions)
        
        # Ensure they're spread out: add small random perturbations
        positions += rng.randn(N, 3) * 0.05
        positions = np.clip(positions, -0.499999, 0.499999)
        
        return positions

    def make_random_init(seed):
        """Random initialization."""
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.45, 0.45, size=(N, 3))

    def make_fcc_like():
        """FCC-inspired initialization."""
        a = 0.32
        positions = []
        for iz in range(4):
            for iy in range(4):
                for ix in range(4):
                    x = (ix - 1.5) * a
                    y = (iy - 1.5) * a
                    z = (iz - 1.5) * a
                    positions.append([x, y, z])
                    if ix < 3:
                        positions.append([x + a / 2, y, z])
                    if iy < 3:
                        positions.append([x, y + a / 2, z])
                    if iz < 3:
                        positions.append([x, y, z + a / 2])
        
        positions = np.array(positions, dtype=np.float64)
        mask = np.all(np.abs(positions) <= 0.48, axis=1)
        positions = positions[mask]
        
        if len(positions) > N:
            # Select N points with best local spacing
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(-min_dists)[:N]
            positions = positions[idx]
        elif len(positions) < N:
            # Pad with random points
            rng = np.random.RandomState(999)
            extra = rng.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])
        
        return positions[:N].copy()

    best_positions = None
    best_radius = -1.0

    def update_best(pos):
        nonlocal best_positions, best_radius
        r = compute_radius(pos)
        if r > best_radius:
            best_radius = r
            best_positions = pos.copy()
        return r

    # Define initializations
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("fcc", make_fcc_like),
        ("spectral0", lambda: make_spectral_init(0)),
        ("spectral1", lambda: make_spectral_init(42)),
        ("spectral2", lambda: make_spectral_init(123)),
        ("rand0", lambda: make_random_init(0)),
        ("rand1", lambda: make_random_init(42)),
        ("rand2", lambda: make_random_init(777)),
        ("rand3", lambda: make_random_init(2024)),
    ]

    # Stage 1: Try each initialization with smooth optimization
    for name, init_func in initializations:
        if remaining() < 15:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)

            # Beta-annealing: start with low beta (smooth), increase to high beta (sharp)
            betas = [5.0, 10.0, 20.0, 40.0, 80.0]
            pos_opt = positions

            for beta in betas:
                if remaining() < 10:
                    break
                pos_opt = local_optimize_smooth(pos_opt, beta, maxiter=100, time_budget=5)
                update_best(pos_opt)

            # Final direct polish
            if remaining() > 10:
                pos_opt = local_optimize_direct(pos_opt, maxiter=200, time_budget=5)
                update_best(pos_opt)

        except Exception:
            continue

    # Phase 2: Perturbation refinement of the best solution
    if best_positions is not None and remaining() > 30:
        n_perturb = 6
        for k in range(n_perturb):
            if remaining() < 10:
                break
            rng = np.random.RandomState(1000 + k)
            noise_scale = 0.015 * (1.0 - k / (n_perturb + 1))
            perturbed = best_positions + rng.normal(0, noise_scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)

            # Smooth optimization with annealing
            pos_opt = perturbed
            for beta in [10.0, 30.0, 60.0]:
                if remaining() < 8:
                    break
                pos_opt = local_optimize_smooth(pos_opt, beta, maxiter=80, time_budget=4)

            # Direct polish
            if remaining() > 8:
                pos_opt = local_optimize_direct(pos_opt, maxiter=150, time_budget=4)

            update_best(pos_opt)

    # Final validity clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    else:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    # Final high-precision polish if time remains
    if remaining() > 10:
        try:
            # Use very high beta for near-exact optimization
            polished = local_optimize_smooth(best_positions, beta=100.0, maxiter=300, time_budget=5)
            polished = local_optimize_direct(polished, maxiter=200, time_budget=5)
            if compute_radius(polished) > best_radius:
                best_positions = np.clip(polished, -0.499999, 0.499999)
        except Exception:
            pass

    return best_positions
