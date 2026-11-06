"""
Fifty equal circles in a circle - contact-graph based optimization with Newton-Raphson refinement.
Uses a hybrid approach: initial ring/hex configurations, then optimizes via
1) Soft-min objective with L-BFGS-B for coarse optimization
2) Newton-Raphson on contact constraints for fine-tuning
3) Iterative contact graph updates
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    best_score = 0.0
    best_centers = None

    # Pre-compute pair indices
    iu = np.triu_indices(N, 1)
    num_pairs = len(iu[0])

    def _score_raw(centers):
        """Compute the actual score (minimum radius) for a set of centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0

        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms >= 1.0):
            return 0.0

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)

        min_pair_dist = np.min(dists)
        min_edge_dist = np.min(1.0 - norms)

        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(r, 0.0)

    def _objective(centers, beta=300.0):
        """Smooth approximation of negative score using log-sum-exp."""
        centers = centers.reshape(-1, 2)

        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_constraints = 1.0 - norms

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        pair_constraints = dists[iu] / 2.0

        all_constraints = np.concatenate([edge_constraints, pair_constraints])

        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta

        return -soft_min_val

    def _gradient(centers, beta=300.0):
        """Compute gradient of the objective - vectorized."""
        centers = centers.reshape(-1, 2)
        grad = np.zeros_like(centers)

        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-15
        edge_constraints = 1.0 - norms

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-15
        pair_constraints = dists[iu] / 2.0

        all_constraints = np.concatenate([edge_constraints, pair_constraints])

        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp

        # Edge gradient (vectorized)
        edge_grad = -centers / norms[:, np.newaxis]
        grad += -weights[:N, np.newaxis] * beta * edge_grad

        # Pair gradients (vectorized using scatter)
        i_idx = iu[0]
        j_idx = iu[1]
        w_pair = weights[N:N+num_pairs]

        diff_pairs = centers[i_idx] - centers[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]

        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * beta * dir_vecs)
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * beta * dir_vecs)

        return grad.reshape(-1)

    def _newton_refine(centers, r_target, max_iter=100, tol=1e-12):
        """
        Newton-Raphson refinement assuming contact graph.
        For each contact pair, enforce |c_i - c_j| = 2*r
        For boundary contacts, enforce |c_i| = 1 - r
        """
        centers = centers.copy()
        r = r_target

        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            # Compute all distances
            norms = np.sqrt(np.sum(centers**2, axis=1))
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)

            # Identify active constraints (near contact)
            # Pair contacts: dist ≈ 2r
            contact_threshold = 2 * r * 1.01
            pair_contact_mask = dists < contact_threshold
            np.fill_diagonal(pair_contact_mask, False)

            # Boundary contacts: norm ≈ 1-r
            boundary_contact_mask = norms > (1 - r) * 0.99

            # Build constraint system
            # For each contact pair (i,j): (c_i - c_j) · (c_i - c_j) = (2r)^2
            # For each boundary contact (i): c_i · c_i = (1-r)^2

            # Get contact pairs
            contact_i, contact_j = np.where(np.triu(pair_contact_mask, 1))
            n_pair_contacts = len(contact_i)
            n_boundary_contacts = np.sum(boundary_contact_mask)
            n_constraints = n_pair_contacts + n_boundary_contacts

            if n_constraints == 0:
                break

            # Build Jacobian and residual
            # Variables: 2*N (x_0, y_0, x_1, y_1, ...)
            # Residuals: for pair (i,j): |c_i - c_j|^2 - (2r)^2
            #            for boundary i: |c_i|^2 - (1-r)^2

            # Compute residual
            residual = np.zeros(n_constraints)

            # Pair residuals
            if n_pair_contacts > 0:
                ci = centers[contact_i]
                cj = centers[contact_j]
                pair_diff = ci - cj
                pair_dists_sq = np.sum(pair_diff**2, axis=1)
                residual[:n_pair_contacts] = pair_dists_sq - (2 * r)**2

            # Boundary residuals
            if n_boundary_contacts > 0:
                boundary_indices = np.where(boundary_contact_mask)[0]
                boundary_norms_sq = norms[boundary_indices]**2
                residual[n_pair_contacts:] = boundary_norms_sq - (1 - r)**2

            # Check convergence
            if np.max(np.abs(residual)) < tol:
                break

            # Build Jacobian (sparse-like, but we'll use dense for simplicity with N=50)
            # J has shape (n_constraints, 2*N)
            J = np.zeros((n_constraints, 2 * N))

            # Pair constraint (i,j): f = |c_i - c_j|^2 - (2r)^2
            # df/d(c_i) = 2*(c_i - c_j)
            # df/d(c_j) = -2*(c_i - c_j)
            if n_pair_contacts > 0:
                for k in range(n_pair_contacts):
                    i = contact_i[k]
                    j = contact_j[k]
                    d = centers[i] - centers[j]
                    J[k, 2*i] = 2 * d[0]
                    J[k, 2*i+1] = 2 * d[1]
                    J[k, 2*j] = -2 * d[0]
                    J[k, 2*j+1] = -2 * d[1]

            # Boundary constraint (i): f = |c_i|^2 - (1-r)^2
            # df/d(c_i) = 2*c_i
            if n_boundary_contacts > 0:
                for k in range(n_boundary_contacts):
                    i = boundary_indices[k]
                    J[n_pair_contacts + k, 2*i] = 2 * centers[i][0]
                    J[n_pair_contacts + k, 2*i+1] = 2 * centers[i][1]

            # Solve J * delta = -residual using least squares
            try:
                delta, residuals_ls, rank, sv = np.linalg.lstsq(J, -residual, rcond=None)
            except np.linalg.LinAlgError:
                break

            # Damping to ensure stability
            step_norm = np.linalg.norm(delta)
            if step_norm > 0.1:
                delta *= 0.1 / step_norm

            # Update
            new_centers = centers.copy()
            new_centers[:, 0] += delta[0::2]
            new_centers[:, 1] += delta[1::2]

            # Check if this improves or maintains validity
            new_norms = np.sqrt(np.sum(new_centers**2, axis=1))
            if np.any(new_norms >= 1.0):
                # Reject if any circle goes outside
                break

            # Check pair distances
            new_diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
            new_dists = np.sqrt(np.sum(new_diff**2, axis=2))
            np.fill_diagonal(new_dists, np.inf)
            if np.min(new_dists) < 2 * r * 0.99:
                # Reject if circles overlap too much
                break

            centers = new_centers

        return centers

    def _make_ring_config(ring_radii, ring_counts, angles_offsets=None):
        """Create a ring-based configuration."""
        centers = []

        # Center
        if ring_counts[0] > 0:
            centers.append([0.0, 0.0])

        for ring_idx in range(1, len(ring_radii)):
            r = ring_radii[ring_idx]
            count = ring_counts[ring_idx]
            offset = angles_offsets[ring_idx] if angles_offsets else 0.0
            for k in range(count):
                theta = 2 * np.pi * k / count + offset
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                centers.append([x, y])

        centers = np.array(centers[:N])
        return centers

    def _hexagonal_config(spacing=0.25):
        """Create a hexagonal lattice configuration."""
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x*x + y*y)
                if r <= 1.0 - 0.13 - 1e-9:
                    points.append([x, y])

        points = np.array(points)
        if len(points) < N:
            return None

        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]

    # Generate diverse initial configurations
    configs = []

    # Configuration 1: Standard rings
    radii1 = [0.0, 0.25, 0.50, 0.75, 0.95]
    counts1 = [1, 6, 12, 18, 13]
    configs.append(_make_ring_config(radii1, counts1))

    # Configuration 2: Slightly compressed
    radii2 = [0.0, 0.23, 0.48, 0.73, 0.94]
    counts2 = [1, 6, 12, 18, 13]
    configs.append(_make_ring_config(radii2, counts2))

    # Configuration 3: Expanded
    radii3 = [0.0, 0.27, 0.52, 0.74, 0.93]
    counts3 = [1, 6, 12, 18, 13]
    configs.append(_make_ring_config(radii3, counts3))

    # Configuration 4: Angle offset on ring 2
    offsets4 = [0.0, 0.0, np.pi/12, 0.0, 0.0]
    configs.append(_make_ring_config(radii1, counts1, offsets4))

    # Configuration 5: Angle offset on ring 3
    offsets5 = [0.0, 0.0, 0.0, np.pi/18, 0.0]
    configs.append(_make_ring_config(radii1, counts1, offsets5))

    # Configuration 6: Hexagonal lattice
    hex_cfg = _hexagonal_config(0.25)
    if hex_cfg is not None:
        configs.append(hex_cfg)

    hex_cfg2 = _hexagonal_config(0.26)
    if hex_cfg2 is not None:
        configs.append(hex_cfg2)

    # Random configurations
    for seed in range(8):
        np.random.seed(42 + seed * 7)
        cfg = np.random.uniform(-0.85, 0.85, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        if np.any(mask):
            cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        configs.append(cfg)

    # Optimize each configuration
    for idx, x0 in enumerate(configs):
        if time.time() > deadline - 15.0:
            break

        current_centers = x0.copy()

        # Progressive beta refinement with L-BFGS-B
        for beta in [100.0, 300.0, 600.0, 1000.0]:
            if time.time() > deadline - 20.0:
                break

            try:
                result = minimize(
                    _objective,
                    current_centers.flatten(),
                    args=(beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
                )

                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)

                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()

                current_centers = centers
            except Exception:
                continue

        # Newton-Raphson refinement on the best found so far
        if best_centers is not None:
            r_target = best_score
            newton_result = _newton_refine(best_centers, r_target, max_iter=50)
            newton_score = _score_raw(newton_result)

            if newton_score > best_score:
                best_score = newton_score
                best_centers = newton_result.copy()

    # Additional refinement passes on the best solution
    if best_centers is not None:
        # Multiple Newton-Raphson passes with slightly adjusted r_target
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break

            r_target = best_score
            newton_result = _newton_refine(best_centers, r_target, max_iter=100)
            newton_score = _score_raw(newton_result)

            if newton_score > best_score:
                best_score = newton_score
                best_centers = newton_result.copy()
            else:
                # Try with a slightly smaller target to allow movement
                r_target = best_score * 0.999
                newton_result = _newton_refine(best_centers, r_target, max_iter=50)
                newton_score = _score_raw(newton_result)
                if newton_score > best_score:
                    best_score = newton_score
                    best_centers = newton_result.copy()

        # Perturbation-based local search
        np.random.seed(12345)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break

            perturbed = best_centers + np.random.normal(0, 0.001, size=(N, 2))
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.98
            if np.any(mask):
                perturbed[mask] *= 0.98 / norms[mask, np.newaxis]

            try:
                result = minimize(
                    _objective,
                    perturbed.flatten(),
                    args=(800.0,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
                )
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
            except Exception:
                continue

    # Final validation
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers

    # Fallback
    return _make_ring_config([0.0, 0.25, 0.50, 0.75, 0.95], [1, 6, 12, 18, 13])
