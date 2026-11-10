import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    max_time = 115.0

    def compute_dists(centers):
        """Compute full pairwise distance matrix."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        return dists

    def compute_score(centers, dists):
        """Compute score from centers and precomputed distance matrix."""
        min_pair = np.min(dists) / 2.0
        norms = np.linalg.norm(centers, axis=1)
        min_boundary = np.min(1.0 - norms)
        return min(min_pair, min_boundary)

    def update_dists_row(dists, centers, idx):
        """Update row and column idx of distance matrix after moving center[idx]."""
        diff = centers[idx] - centers
        new_dists = np.sqrt(np.sum(diff**2, axis=1))
        new_dists[idx] = np.inf
        dists[idx, :] = new_dists
        dists[:, idx] = new_dists

    def local_search(centers, time_budget, step_sizes):
        """
        Structural change: Use a multi-scale local search that directly
        addresses the binding constraints. Instead of random SA moves, we
        iteratively identify the most violated constraint and apply a
        targeted perturbation, with adaptive step sizes.
        """
        centers = centers.copy()
        dists = compute_dists(centers)
        current_score = compute_score(centers, dists)
        best_score = current_score
        best_centers = centers.copy()
        best_dists = dists.copy()

        end_time = start_time + time_budget
        no_improve = 0
        max_no_improve = 200

        # Multi-scale step sizes: start coarse, refine
        base_steps = step_sizes
        current_step_idx = 0
        scale_no_improve = 0

        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break

            step = base_steps[current_step_idx]

            # Find the most violated constraint
            min_pair_dist = np.min(dists)
            i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)

            norms = np.linalg.norm(centers, axis=1)
            min_boundary_val = np.min(1.0 - norms)
            min_boundary_idx = np.argmin(1.0 - norms)

            # Determine which constraint is more binding
            # Score = min(min_pair/2, min_boundary)
            pair_contribution = min_pair_dist / 2.0
            boundary_contribution = min_boundary_val

            if pair_contribution <= boundary_contribution:
                # Pair constraint is binding: push i and j apart
                vec = centers[j_min] - centers[i_min]
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 1e-10:
                    direction = vec / norm_vec
                    # Try several perturbation directions around the direct separation
                    angles = np.linspace(-np.pi/3, np.pi/3, 7)
                    best_new_score = current_score
                    best_new_centers = None
                    best_new_dists = None

                    for angle in angles:
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        # Rotate the separation direction slightly
                        rot_dir = np.array([
                            direction[0] * cos_a - direction[1] * sin_a,
                            direction[0] * sin_a + direction[1] * cos_a
                        ])

                        new_i = centers[i_min] - rot_dir * step * 0.5
                        new_j = centers[j_min] + rot_dir * step * 0.5

                        # Also try perpendicular component
                        perp_dir = np.array([-rot_dir[1], rot_dir[0]])
                        for perp_frac in [0.0, 0.3, -0.3]:
                            trial_i = new_i + perp_dir * step * 0.3 * perp_frac
                            trial_j = new_j - perp_dir * step * 0.3 * perp_frac

                            # Check boundary
                            if np.linalg.norm(trial_i) > 1.0 or np.linalg.norm(trial_j) > 1.0:
                                continue

                            # Trial move
                            old_i = centers[i_min].copy()
                            old_j = centers[j_min].copy()
                            centers[i_min] = trial_i
                            centers[j_min] = trial_j

                            # Update distances incrementally
                            update_dists_row(dists, centers, i_min)
                            update_dists_row(dists, centers, j_min)

                            new_score = compute_score(centers, dists)

                            if new_score > best_new_score:
                                best_new_score = new_score
                                best_new_centers = (centers.copy(), trial_i.copy(), trial_j.copy())

                            # Revert for next trial
                            centers[i_min] = old_i
                            centers[j_min] = old_j
                            update_dists_row(dists, centers, i_min)
                            update_dists_row(dists, centers, j_min)

                    if best_new_centers is not None and best_new_score > current_score:
                        centers[i_min] = best_new_centers[1]
                        centers[j_min] = best_new_centers[2]
                        update_dists_row(dists, centers, i_min)
                        update_dists_row(dists, centers, j_min)
                        current_score = best_new_score
                        no_improve = 0
                        scale_no_improve = 0

                        if current_score > best_score:
                            best_score = current_score
                            best_centers = centers.copy()
                            best_dists = dists.copy()
                    else:
                        no_improve += 1
                        scale_no_improve += 1
                else:
                    no_improve += 1
                    scale_no_improve += 1
            else:
                # Boundary constraint is binding: move that point inward
                idx = min_boundary_idx
                vec = centers[idx]
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 1e-10:
                    direction = vec / norm_vec
                    # Try moving inward with slight angular variations
                    angles = np.linspace(-np.pi/4, np.pi/4, 5)
                    best_new_score = current_score
                    best_new_center = None

                    for angle in angles:
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        rot_dir = np.array([
                            direction[0] * cos_a - direction[1] * sin_a,
                            direction[0] * sin_a + direction[1] * cos_a
                        ])

                        trial = centers[idx] - rot_dir * step
                        if np.linalg.norm(trial) > 1.0:
                            continue

                        old = centers[idx].copy()
                        centers[idx] = trial
                        update_dists_row(dists, centers, idx)

                        new_score = compute_score(centers, dists)

                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_center = trial.copy()

                        centers[idx] = old
                        update_dists_row(dists, centers, idx)

                    if best_new_center is not None and best_new_score > current_score:
                        centers[idx] = best_new_center
                        update_dists_row(dists, centers, idx)
                        current_score = best_new_score
                        no_improve = 0
                        scale_no_improve = 0

                        if current_score > best_score:
                            best_score = current_score
                            best_centers = centers.copy()
                            best_dists = dists.copy()
                    else:
                        no_improve += 1
                        scale_no_improve += 1
                else:
                    no_improve += 1
                    scale_no_improve += 1

            # Adapt step size scale
            if scale_no_improve > max_no_improve:
                # Move to finer scale
                if current_step_idx < len(base_steps) - 1:
                    current_step_idx += 1
                    scale_no_improve = 0
                    no_improve = 0
                else:
                    # Try random perturbation to escape local minimum
                    if no_improve > 500:
                        # Random restart-like perturbation
                        idx = np.random.randint(N)
                        old = centers[idx].copy()
                        noise = np.random.normal(0, 0.05, 2)
                        new_pos = centers[idx] + noise
                        if np.linalg.norm(new_pos) < 1.0:
                            centers[idx] = new_pos
                            update_dists_row(dists, centers, idx)
                            new_score = compute_score(centers, dists)
                            if new_score > current_score:
                                current_score = new_score
                                no_improve = 0
                                scale_no_improve = 0
                            else:
                                centers[idx] = old
                                update_dists_row(dists, centers, idx)
                                no_improve += 1
                        else:
                            no_improve += 1

            # Early termination if stuck at finest scale
            if current_step_idx == len(base_steps) - 1 and no_improve > 3000:
                break

        return best_centers, best_score

    def sa_phase(centers, time_budget):
        """
        Simulated annealing with efficient distance updates.
        Uses the cached distance matrix and only updates affected rows.
        """
        centers = centers.copy()
        dists = compute_dists(centers)
        current_score = compute_score(centers, dists)
        best_score = current_score
        best_centers = centers.copy()
        best_dists = dists.copy()

        T = 0.005
        cooling_rate = 0.9995
        step_size = 0.015
        no_improve = 0
        end_time = start_time + time_budget

        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break

            # Choose a random point to move
            idx = np.random.randint(N)

            # Generate perturbation: mix of random and gradient-like
            angle = np.random.uniform(0, 2 * np.pi)
            magnitude = np.random.uniform(0.5, 1.5) * step_size
            delta = np.array([np.cos(angle), np.sin(angle)]) * magnitude

            new_pos = centers[idx] + delta

            # Keep inside container
            norm = np.linalg.norm(new_pos)
            if norm > 0.995:
                new_pos = new_pos / norm * 0.995

            old_pos = centers[idx].copy()
            centers[idx] = new_pos

            # Update distances incrementally
            update_dists_row(dists, centers, idx)

            new_score = compute_score(centers, dists)
            delta_E = new_score - current_score

            if delta_E > 0 or (T > 1e-9 and np.random.rand() < np.exp(delta_E / T)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
                    best_dists = dists.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                centers[idx] = old_pos
                update_dists_row(dists, centers, idx)
                no_improve += 1

            # Cooling
            T *= cooling_rate

            # Reheat if stuck
            if no_improve > 200:
                T = 0.003
                step_size = min(0.05, step_size * 1.2)
                no_improve = 0
            elif no_improve > 50:
                step_size = max(0.005, step_size * 0.95)

            if T < 1e-9:
                T = 0.0005

        return best_centers, best_score

    # Initialization: Multiple strategies
    def hex_init(spacing):
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(dists)
            return points[sorted_indices[:N]]
        return np.random.uniform(-0.8, 0.8, (N, 2))

    def random_init():
        # Random initialization with rejection sampling to avoid overlaps
        centers = np.zeros((N, 2))
        centers[0] = np.array([0.0, 0.0])
        min_r = 0.12
        for i in range(1, N):
            for attempt in range(1000):
                r = np.random.uniform(0, 0.9)
                theta = np.random.uniform(0, 2*np.pi)
                pos = np.array([r * np.cos(theta), r * np.sin(theta)])
                if np.all(np.linalg.norm(centers[:i] - pos, axis=1) > min_r * 2):
                    centers[i] = pos
                    break
            else:
                centers[i] = pos
        return centers

    # Try multiple initializations and keep the best
    inits = []
    inits.append(hex_init(0.22))
    inits.append(hex_init(0.20))
    inits.append(hex_init(0.25))
    inits.append(random_init())

    # Score each initialization
    best_init = None
    best_init_score = -1
    for init in inits:
        init_dists = compute_dists(init)
        init_score = compute_score(init, init_dists)
        if init_score > best_init_score:
            best_init_score = init_score
            best_init = init

    # Phase 1: SA for exploration
    sa_time = 60.0
    sa_end = start_time + sa_time
    remaining_for_sa = min(sa_time, max_time - (time.time() - start_time))
    if remaining_for_sa > 5.0:
        best_centers, best_score = sa_phase(best_init, remaining_for_sa)
    else:
        best_centers = best_init
        best_score = best_init_score

    # Phase 2: Multi-scale local search for refinement
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 5.0:
        step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        polished_centers, polished_score = local_search(best_centers, remaining_time - 2.0, step_sizes)
        if polished_score >= best_score:
            best_centers = polished_centers
            best_score = polished_score

    # Final safety check
    if best_centers is None or len(best_centers) != N:
        best_centers = np.random.uniform(-0.9, 0.9, (N, 2))

    # Ensure validity
    norms = np.linalg.norm(best_centers, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_centers[i] = best_centers[i] / norms[i] * 0.9999

    return best_centers
