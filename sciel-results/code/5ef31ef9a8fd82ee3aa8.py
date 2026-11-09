"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Grid initialization
    m = int(math.ceil(math.sqrt(N)))  # 14 for N=200
    # Use a slightly denser grid to have more initial points, then pick N
    # Actually, just use the standard grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()

    # Calculate initial radius
    def compute_radius(centers):
        # Min distance between centers / 2
        # Min distance to boundary
        # Boundary: max(0.5 - |x|, 0.5 - |y|)
        tree = cKDTree(centers)
        min_dist = np.inf
        # Efficient way to find min distance
        # Use query_pairs for efficiency
        pairs = tree.query_pairs(1.0)  # We need to find the actual min
        # Actually, let's just compute it properly
        # For 200 points, O(N^2) is fine
        min_pair_dist = np.inf
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_pair_dist:
                    min_pair_dist = d
        min_boundary = np.min(np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])))
        return min(min_pair_dist / 2.0, min_boundary)

    current_radius = compute_radius(centers)

    # Optimization using gradient ascent on radius
    # We want to maximize r such that all constraints are satisfied.
    # Alternatively, we can use a repulsive force model.
    # Let's try a simple iterative improvement:
    # 1. Find the closest pair or boundary constraint.
    # 2. Move points to increase the minimum distance.

    # A better approach: Use scipy.optimize.minimize to minimize the negative of the radius.
    # But the radius function is non-differentiable.
    # Let's use a soft-min approximation or just do a local search.

    # Local search approach:
    # Randomly perturb points and keep if radius improves.
    # Or, use a force-directed method.

    # Let's implement a simple force-directed relaxation.
    # Forces:
    # 1. Repulsion between all pairs (inverse square or similar).
    # 2. Attraction to center (to keep them in bounds, or just hard constraints).
    # Actually, we want to maximize the minimum distance.
    # This is equivalent to packing.

    # Let's try a simple hill-climbing with random restarts or just a single run.
    # Given the time limit, let's do a deterministic local optimization.

    # Algorithm:
    # Repeat until time limit:
    #   Find the pair of points (or point-boundary) with the minimum distance.
    #   Move these points apart slightly.
    #   Check if any constraint is violated (points leaving square).
    #   If a point leaves, project it back.

    # This is essentially a simple version of the algorithm used in packing problems.

    def optimize(centers, max_time):
        # Precompute some things?
        # Let's just loop.
        while time.time() < max_time:
            # Find min distance pair
            min_dist = np.inf
            min_i = 0
            min_j = -1
            min_boundary_dist = np.inf
            min_boundary_i = 0
            min_boundary_axis = 0 # 0 for x, 1 for y, -1 for none

            # Pairwise distances
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_dist:
                        min_dist = d
                        min_i = i
                        min_j = j

            # Boundary distances
            for i in range(N):
                # Distance to left/right
                d_left = centers[i, 0] - (-0.5)
                d_right = 0.5 - centers[i, 0]
                # Distance to top/bottom
                d_bottom = centers[i, 1] - (-0.5)
                d_top = 0.5 - centers[i, 1]

                # We want the min distance to boundary
                # Note: distance to boundary is min(d_left, d_right, d_bottom, d_top)
                # But we also need to know which boundary to push against.
                # Actually, the constraint is 0.5 - |x| >= r and 0.5 - |y| >= r.
                # So the "distance" is 0.5 - |x| and 0.5 - |y|.
                # Let's compute these.
                dist_x = 0.5 - np.abs(centers[i, 0])
                dist_y = 0.5 - np.abs(centers[i, 1])

                if dist_x < min_boundary_dist:
                    min_boundary_dist = dist_x
                    min_boundary_i = i
                    min_boundary_axis = 0
                if dist_y < min_boundary_dist:
                    min_boundary_dist = dist_y
                    min_boundary_i = i
                    min_boundary_axis = 1

            current_min = min(min_dist / 2.0, min_boundary_dist)

            # If the limiting constraint is a pair
            if min_dist / 2.0 <= min_boundary_dist:
                i, j = min_i, min_j
                # Move i and j apart
                # Direction from i to j
                diff = centers[j] - centers[i]
                norm = np.linalg.norm(diff)
                if norm == 0:
                    # Random direction
                    diff = np.random.randn(2)
                    norm = np.linalg.norm(diff)
                dir = diff / norm

                # How much to move?
                # We want to increase the distance.
                # Let's move each by a small epsilon in opposite directions.
                # But we must not violate other constraints.
                # A safe step is to move them apart by half the difference to the next smallest distance?
                # Simpler: move by a small fixed step, or adaptive.
                # Let's try moving them apart by a small amount, say 0.01 * min_dist?
                # Or just try to increase the distance to be slightly larger than current.
                # Actually, we can't just increase it arbitrarily if other points are close.
                # Let's use a small step size.
                step = 0.01
                # Move i by -step * dir, j by +step * dir
                centers[i] -= step * dir
                centers[j] += step * dir

                # Check if they went out of bounds
                # If so, project back
                for k in [i, j]:
                    centers[k, 0] = np.clip(centers[k, 0], -0.5, 0.5)
                    centers[k, 1] = np.clip(centers[k, 1], -0.5, 0.5)

            else:
                # Limiting constraint is boundary
                i = min_boundary_i
                axis = min_boundary_axis
                # Move point i towards center
                # Direction is towards 0
                # If x-axis: move x towards 0
                # If y-axis: move y towards 0
                step = 0.01
                if axis == 0:
                    # Move x towards 0
                    if centers[i, 0] > 0:
                        centers[i, 0] -= step
                    else:
                        centers[i, 0] += step
                else:
                    if centers[i, 1] > 0:
                        centers[i, 1] -= step
                    else:
                        centers[i, 1] += step

                # Clip
                centers[i, 0] = np.clip(centers[i, 0], -0.5, 0.5)
                centers[i, 1] = np.clip(centers[i, 1], -0.5, 0.5)

            # Optional: Add some random jitter to escape local minima?
            # Maybe every 1000 iterations?
            # For now, let's keep it deterministic.

        return centers

    # Run optimization
    centers = optimize(centers, deadline - 5.0)

    # Final check and cleanup
    # Ensure all points are strictly inside? No, "on or outside" is invalid.
    # So we must ensure |x| <= 0.5 and |y| <= 0.5.
    # The clipping ensures this.
    # But what if they are exactly on the boundary? That's valid ("on or outside" -> invalid? "A centre on or outside the square ... make the candidate invalid").
    # Wait, the text says: "A centre on or outside the square ... make the candidate invalid."
    # So we must be STRICTLY inside.
    # Let's shrink the square slightly for the final result to be safe.
    # Or just ensure they are well inside.
    # The grid starts at 0.5/m - 0.5, which is inside.
    # The optimization might push them to the boundary.
    # Let's re-clip with a small margin.
    margin = 1e-9
    centers[:, 0] = np.clip(centers[:, 0], -0.5 + margin, 0.5 - margin)
    centers[:, 1] = np.clip(centers[:, 1], -0.5 + margin, 0.5 - margin)

    return centers

# EVOLVE-BLOCK-END
