"""QAP tai256c solver: continuous phase-field embedding with gradient-based relaxation and permutation extraction."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=20):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2
        C = M1 + M2
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def phase_field_embedding(a, b, num_iters=300, lr=0.01):
        """
        Continuous phase-field embedding:
        1. Initialize facility positions in 2D
        2. Minimize energy E = sum_{i,j} a[i,j] * exp(-||x_i - x_j||^2 / sigma^2)
           where x_i are 2D positions
        3. Extract permutation by matching facility positions to location grid
        """
        # Initialize positions on a circle with some noise
        angles = 2 * np.pi * np.arange(n) / n + np.random.randn(n) * 0.1
        radius = 10.0 + np.random.randn(n) * 0.5
        positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

        # Precompute flow matrix
        flow = a.astype(np.float64)

        # Multi-scale Gaussian kernel
        sigma_scales = [1.0, 2.0, 5.0]

        # Gradient-based optimization
        for it in range(num_iters):
            if time.time() > deadline - 30:
                break

            # Compute energy and gradient
            # E = sum_{i,j} a[i,j] * exp(-||x_i - x_j||^2 / sigma^2)
            # dE/dx_i = sum_j a[i,j] * exp(-||x_i - x_j||^2 / sigma^2) * (-2/sigma^2) * (x_i - x_j)

            grad = np.zeros_like(positions)

            for sigma in sigma_scales:
                # Compute pairwise distances
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                dist_sq = np.sum(diff * diff, axis=2)

                # Gaussian kernel
                kernel = np.exp(-dist_sq / (sigma * sigma))

                # Weight by flow
                weighted_kernel = flow * kernel

                # Gradient: sum_j a[i,j] * kernel * (-2/sigma^2) * (x_i - x_j)
                # = (-2/sigma^2) * sum_j a[i,j] * kernel * (x_i - x_j)
                # = (-2/sigma^2) * (x_i * sum_j a[i,j]*kernel[j] - sum_j a[i,j]*kernel[j]*x_j)

                flow_sum = np.sum(weighted_kernel, axis=1)
                weighted_pos_sum = np.sum(weighted_kernel[:, :, np.newaxis] * positions[np.newaxis, :, :], axis=1)

                grad += (-2.0 / (sigma * sigma)) * (positions * flow_sum[:, np.newaxis] - weighted_pos_sum)

            # Update positions with adaptive learning rate
            lr_current = lr * (1.0 - it / num_iters) * 0.1
            positions -= lr_current * grad

            # Keep positions bounded
            positions = np.clip(positions, -50, 50)

        return positions

    def extract_permutation(positions, b):
        """
        Extract permutation from 2D embedding:
        1. Sort locations by their "centrality" (distance from center of mass)
        2. Sort facilities by their radial distance in embedding
        3. Match them, then refine with local search
        """
        # Compute center of mass of facility positions
        center = np.mean(positions, axis=0)

        # Radial distances of facilities
        rad_fac = np.linalg.norm(positions - center, axis=1)

        # For locations, use their "centrality" in the distance matrix
        # Central locations have smaller average distance to all other locations
        b_avg = b.mean(axis=1)
        loc_order = np.argsort(b_avg)  # Most central first

        # Sort facilities by radial distance (most central first)
        fac_order = np.argsort(rad_fac)

        # Initial assignment: most central facility -> most central location
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order

        return p

    def angular_sort_permutation(positions, b):
        """
        Alternative extraction: sort by angle in 2D embedding
        """
        center = np.mean(positions, axis=0)
        angles_fac = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
        fac_order = np.argsort(angles_fac)

        # For locations, use row sums of b as a proxy for "position"
        b_sums = b.sum(axis=1)
        loc_order = np.argsort(b_sums)

        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order

        return p

    def hungarian_init(a, b):
        """Hungarian-based initialization using linearized cost"""
        M1 = a @ b.T
        M2 = a.T @ b
        M = M1 + M2
        row_ind, col_ind = linear_sum_assignment(M)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def rank_init(a, b):
        """Rank-matching initialization"""
        a_row_sums = a.sum(axis=1)
        b_col_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_col_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # === Main solver: combine phase-field with combinatorial refinement ===

    best_p = None
    best_cost = np.inf

    # Strategy 1: Phase-field embedding -> extract permutation -> local search
    if time.time() < deadline - 60:
        positions = phase_field_embedding(a, b, num_iters=200, lr=0.05)
        p1 = extract_permutation(positions, b)
        p1, c1 = local_search(p1.copy(), compute_cost(p1))
        if c1 < best_cost:
            best_cost = c1
            best_p = p1.copy()

        p2 = angular_sort_permutation(positions, b)
        p2, c2 = local_search(p2.copy(), compute_cost(p2))
        if c2 < best_cost:
            best_cost = c2
            best_p = p2.copy()

    # Strategy 2: Hungarian initialization (proven effective)
    if time.time() < deadline - 40:
        p_h = hungarian_init(a, b)
        p_h, c_h = local_search(p_h.copy(), compute_cost(p_h))
        if c_h < best_cost:
            best_cost = c_h
            best_p = p_h.copy()

    # Strategy 3: Rank-matching initialization
    if time.time() < deadline - 30:
        p_r = rank_init(a, b)
        p_r, c_r = local_search(p_r.copy(), compute_cost(p_r))
        if c_r < best_cost:
            best_cost = c_r
            best_p = p_r.copy()

    # Iterated local search with diversification
    if best_p is not None:
        current_p = best_p.copy()
        current_cost = best_cost

        iteration = 0
        while time.time() < deadline - 15:
            iteration += 1

            # Perturb: random block reversal
            p = current_p.copy()
            block_len = np.random.randint(3, 11)
            start = np.random.randint(0, n - block_len)
            p[start:start+block_len] = p[start:start+block_len][::-1]

            p, cost = local_search(p, compute_cost(p))

            # Periodic Hungarian re-optimization
            if iteration % 5 == 0 and time.time() < deadline - 25:
                h_p = hungarian_reopt(p)
                h_cost = compute_cost(h_p)
                h_p, h_cost = local_search(h_p, h_cost)
                if h_cost < cost:
                    p, cost = h_p, h_cost

            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                current_p = p.copy()
                current_cost = cost

    # Final polish
    if best_p is not None and time.time() < deadline - 3:
        final_p = best_p.copy()
        final_cost = best_cost
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            final_p, final_cost = local_search(final_p, final_cost, max_passes=30)
            if final_cost < best_cost:
                best_cost = final_cost
                best_p = final_p.copy()

    return best_p.tolist()
