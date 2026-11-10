import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    max_time = 115.0

    def calculate_score(centers):
        # Calculate min distance between pairs and to boundary
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
        return min(min_pair, min_boundary)

    def run_repulsion_search(initial_centers, time_budget):
        centers = initial_centers.copy()
        current_score = calculate_score(centers)
        best_score = current_score
        best_centers = centers.copy()

        T = 0.005
        cooling_rate = 0.9995
        step_size = 0.01

        no_improve_count = 0
        end_time = start_time + time_budget

        # Pre-allocate for distance matrix
        diff = np.zeros((N, N, 2))
        dists = np.zeros((N, N))

        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break

            # Compute distance matrix
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)

            # Find nearest neighbor for each point
            nn_idx = np.argmin(dists, axis=1)

            # Compute repulsion forces
            forces = np.zeros((N, 2))
            for i in range(N):
                j = nn_idx[i]
                vec = centers[i] - centers[j]  # Vector from j to i (away from neighbor)
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 1e-9:
                    # Repulsion strength inversely proportional to distance
                    strength = 1.0 / (norm_vec + 0.01)
                    forces[i] += vec / norm_vec * strength

                # Boundary repulsion
                dist_to_center = np.linalg.norm(centers[i])
                if dist_to_center > 0.85:
                    if dist_to_center > 1e-9:
                        boundary_force = (dist_to_center - 0.85) * 2.0
                        forces[i] -= centers[i] / dist_to_center * boundary_force

            # Select point to move
            idx = np.random.randint(N)

            # Move in direction of force + noise
            force_norm = np.linalg.norm(forces[idx])
            if force_norm > 1e-9:
                direction = forces[idx] / force_norm
            else:
                direction = np.random.normal(0, 1, 2)
                direction /= np.linalg.norm(direction)

            noise = np.random.normal(0, step_size * 0.3, 2)
            displacement = direction * step_size + noise

            new_point = centers[idx] + displacement

            # Keep inside circle
            norm = np.linalg.norm(new_point)
            if norm > 0.99:
                new_point = new_point / norm * 0.99

            old_point = centers[idx]
            centers[idx] = new_point

            new_score = calculate_score(centers)
            delta_E = new_score - current_score

            if delta_E > 0:
                accept = True
            else:
                if T > 1e-8:
                    p = np.exp(delta_E / T)
                    accept = (np.random.rand() < p)
                else:
                    accept = False

            if accept:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                centers[idx] = old_point
                no_improve_count += 1

            # Cooling
            T *= cooling_rate

            # Reheating
            if no_improve_count > 500:
                T = 0.02
                step_size = 0.03
                no_improve_count = 0

            # Adapt step size
            if no_improve_count < 50:
                step_size = max(0.002, step_size * 0.98)
            else:
                step_size = min(0.05, step_size * 1.02)

            if T < 1e-8:
                T = 0.001

        return best_centers, best_score

    def run_multistart(num_starts, time_per_start):
        best_centers = None
        best_score = -1.0

        for s in range(num_starts):
            if time.time() > start_time + max_time - 5.0:
                break

            # Different initialization strategies
            if s % 3 == 0:
                # Hexagonal grid
                spacing = 0.18
                k = int(1.0 / spacing) + 3
                points = []
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * spacing
                        y = j * spacing * np.sqrt(3.0) / 2.0
                        if x * x + y * y <= 0.95:
                            points.append([x, y])
                points = np.array(points)
                if len(points) >= N:
                    dists = np.linalg.norm(points, axis=1)
                    sorted_indices = np.argsort(dists)
                    init_centers = points[sorted_indices[:N]]
                else:
                    init_centers = np.random.uniform(-0.8, 0.8, (N, 2))
            elif s % 3 == 1:
                # Random uniform
                init_centers = np.random.uniform(-0.9, 0.9, (N, 2))
                norms = np.linalg.norm(init_centers, axis=1)
                mask = norms > 0.99
                init_centers[mask] = init_centers[mask] / norms[mask, np.newaxis] * 0.99
            else:
                # Concentric rings
                init_centers = np.zeros((N, 2))
                # Center point
                init_centers[0] = [0, 0]
                remaining = N - 1
                ring_radius = 0.15
                points_per_ring = 6
                idx = 1
                while idx < N:
                    for angle in np.linspace(0, 2 * np.pi, points_per_ring, endpoint=False):
                        if idx >= N:
                            break
                        init_centers[idx] = [ring_radius * np.cos(angle), ring_radius * np.sin(angle)]
                        idx += 1
                    ring_radius += 0.15
                    points_per_ring += 6

            # Run search
            centers, score = run_repulsion_search(init_centers, time_per_start)
            if score > best_score:
                best_score = score
                best_centers = centers

        return best_centers, best_score

    # Run multistart
    num_starts = 8
    time_per_start = (max_time - 10.0) / num_starts

    best_centers, best_score = run_multistart(num_starts, time_per_start)

    # Ensure validity
    if best_centers is None or len(best_centers) != N:
        best_centers = np.random.uniform(-0.9, 0.9, (N, 2))

    norms = np.linalg.norm(best_centers, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_centers[i] = best_centers[i] / norms[i] * 0.99

    return best_centers
