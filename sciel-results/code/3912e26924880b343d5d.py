"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # We will optimize 2N center coordinates.
    # Radius of circle i is min over all constraints:
    #   - distance to boundary: x_i, 1-x_i, y_i, 1-y_i
    #   - distance to other circle j: ||p_i - p_j|| / 2
    # We want to maximize sum of these radii.
    #
    # Use a soft-min via log-sum-exp to make the objective smooth.
    # r_i = softmin of all constraint distances.
    #
    # We use gradient ascent with adaptive step size.

    def softmin(vals, beta):
        """Soft minimum: approximates min but is smooth.
        softmin(v) = -logsumexp(-beta * v) / beta
        """
        # For numerical stability
        m = np.min(vals)
        return m - np.log(np.sum(np.exp(beta * (vals - m)))) / beta

    def compute_radii(positions, beta):
        """Compute soft radii for all circles given positions.
        positions: (N, 2) array of (x, y)
        Returns: (N,) array of radii
        """
        n = positions.shape[0]
        radii = np.zeros(n)
        for i in range(n):
            # Boundary distances
            boundary = np.array([positions[i, 0], 1.0 - positions[i, 0],
                                  positions[i, 1], 1.0 - positions[i, 1]])
            # Distances to other circles / 2
            if n > 1:
                diffs = positions - positions[i]
                dists = np.sqrt(np.sum(diffs**2, axis=1)) / 2.0
                # Exclude self
                dists = np.concatenate([dists[:i], dists[i+1:]])
            else:
                dists = np.array([])
            
            all_dists = np.concatenate([boundary, dists])
            radii[i] = softmin(all_dists, beta)
        return radii

    def objective_and_grad(positions, beta):
        """Compute objective (sum of radii) and gradient via finite differences."""
        n = positions.shape[0]
        obj = np.sum(compute_radii(positions, beta))
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(positions)
        for d in range(positions.size):
            pos_plus = positions.copy()
            idx = d // 2
            comp = d % 2
            pos_plus[idx, comp] += eps
            obj_plus = np.sum(compute_radii(pos_plus, beta))
            
            pos_minus = positions.copy()
            pos_minus[idx, comp] -= eps
            obj_minus = np.sum(compute_radii(pos_minus, beta))
            
            grad[idx, comp] = (obj_plus - obj_minus) / (2 * eps)
        
        return obj, grad

    # Initialize with a grid
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            positions[count, 0] = (i + 0.5) / side
            positions[count, 1] = (j + 0.5) / side
            count += 1

    best_positions = positions.copy()
    best_obj = np.sum(compute_radii(best_positions, 10.0))

    # Start with moderate beta, increase over time for sharper min
    beta = 5.0
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(positions)

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        obj, grad = objective_and_grad(positions, beta)
        
        # Update with momentum
        velocity = momentum * velocity + lr * grad
        positions = positions + velocity
        
        # Clamp to [0, 1]
        positions = np.clip(positions, 0.0, 1.0)
        
        # Check improvement
        if obj > best_obj:
            best_obj = obj
            best_positions = positions.copy()
        
        # Adapt learning rate: reduce if no improvement for a while
        if iteration % 50 == 0:
            lr *= 0.95
            beta *= 1.1  # sharpen the soft-min
        
        # Occasionally add small random perturbations to escape local minima
        if iteration % 100 == 0:
            positions = positions + 0.005 * np.random.randn(*positions.shape)
            positions = np.clip(positions, 0.0, 1.0)
        
        # If stuck, try a restart with perturbation
        if iteration % 200 == 0:
            # Try several random restarts
            for _ in range(3):
                perturbed = best_positions + 0.05 * np.random.randn(*best_positions.shape)
                perturbed = np.clip(perturbed, 0.0, 1.0)
                test_obj = np.sum(compute_radii(perturbed, beta))
                if test_obj > best_obj:
                    best_obj = test_obj
                    best_positions = perturbed.copy()
                    positions = perturbed.copy()
                    break

    # Final polish: use high beta for precise radii
    final_beta = 50.0
    final_radii = compute_radii(best_positions, final_beta)
    
    # Build result
    result = np.zeros((N, 3))
    result[:, :2] = best_positions
    result[:, 2] = final_radii
    
    # Verify validity: ensure circles are inside and non-overlapping
    # Adjust radii down slightly if needed
    for i in range(N):
        x, y, r = result[i]
        # Boundary constraint
        r = min(r, x, 1.0 - x, y, 1.0 - y)
        result[i, 2] = r
    
    # Check pairwise overlaps and shrink if necessary
    for i in range(N):
        for j in range(i + 1, N):
            dx = result[i, 0] - result[j, 0]
            dy = result[i, 1] - result[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            max_r = dist / 2.0
            if result[i, 2] + result[j, 2] > max_r + 1e-9:
                # Shrink both
                scale = max_r / (result[i, 2] + result[j, 2])
                result[i, 2] *= scale
                result[j, 2] *= scale
    
    return result
