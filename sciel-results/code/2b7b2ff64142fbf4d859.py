"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _objective_and_grad(pos, radii):
    """
    Compute objective (sum of radii) and gradients for pos (n,2) and radii (n,).
    Uses smooth min via soft-min to handle constraints.
    Returns: objective_value, grad_pos, grad_radii, is_valid
    """
    n = len(radii)
    # Constraints:
    # 1. r_i >= 0 (we'll enforce via projection or penalty)
    # 2. x_i - r_i >= 0, x_i + r_i <= 1 => r_i <= x_i, r_i <= 1 - x_i
    # 3. y_i - r_i >= 0, y_i + r_i <= 1 => r_i <= y_i, r_i <= 1 - y_i
    # 4. ||p_i - p_j|| >= r_i + r_j for i != j
    
    # We want to maximize sum(r_i) subject to these.
    # Equivalent: maximize sum(r_i) - penalty * sum(violations)
    # Or: minimize -sum(r_i) + penalty * sum(soft_violations)
    
    # Let's use a barrier/penalty approach.
    # Actually, let's just compute the "effective" radius for each circle given positions,
    # which is the min of (dist to walls, dist to other circles).
    # r_i_max = min(x_i, 1-x_i, y_i, 1-y_i, min_{j!=i} ||p_i - p_j|| - r_j) -- this is circular.
    
    # Better: Define the problem as optimizing both positions and radii.
    # Constraints:
    # g_i^wall = r_i - x_i <= 0
    # g_i^wall = r_i - (1-x_i) <= 0
    # g_i^wall = r_i - y_i <= 0
    # g_i^wall = r_i - (1-y_i) <= 0
    # g_ij = r_i + r_j - ||p_i - p_j|| <= 0
    
    # We want to maximize sum(r_i).
    # Let's use a penalty method with a smooth approximation.
    
    # Compute wall distances
    wall_dist = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]),
                           np.minimum(pos[:, 1], 1 - pos[:, 1]))
    
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # For each circle i, the max radius is min(wall_dist[i], min_j (dist[i,j] - r_j))
    # This is still coupled. Let's use a different approach.
    
    # Direct penalty:
    # Objective = sum(r) - lambda * (sum of violations)
    # Violations:
    # v_i_wall = max(0, r_i - wall_dist[i])
    # v_ij = max(0, r_i + r_j - dist[i,j])
    
    # To make it smooth, use softplus or similar.
    # But for now, let's just use a simple gradient-based approach with projection.
    
    # Actually, let's reformulate:
    # For fixed positions, the optimal radius for circle i is:
    # r_i = min(wall_dist[i], min_{j!=i} (dist[i,j] - r_j))
    # This is a system of equations. We can solve it iteratively.
    
    # Let's just do direct optimization with a penalty.
    lam = 10.0  # penalty weight
    
    # Wall violations
    v_wall_x_pos = np.maximum(0, radii - pos[:, 0])
    v_wall_x_neg = np.maximum(0, radii - (1 - pos[:, 0]))
    v_wall_y_pos = np.maximum(0, radii - pos[:, 1])
    v_wall_y_neg = np.maximum(0, radii - (1 - pos[:, 1]))
    
    # Pairwise violations
    # v_ij = max(0, r_i + r_j - dist[i,j])
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    v_pair = np.maximum(0, r_sum - dist)
    np.fill_diagonal(v_pair, 0)
    v_pair_sum = np.sum(v_pair) / 2  # each pair counted twice
    
    total_violation = np.sum(v_wall_x_pos) + np.sum(v_wall_x_neg) + \
                      np.sum(v_wall_y_pos) + np.sum(v_wall_y_neg) + v_pair_sum
    
    objective = np.sum(radii) - lam * total_violation
    
    # Gradients
    grad_pos = np.zeros_like(pos)
    grad_radii = np.zeros_like(radii)
    
    # Wall gradient contributions
    # d/dx_i of max(0, r_i - x_i) = -1 if r_i > x_i else 0
    mask = radii > pos[:, 0]
    grad_pos[mask, 0] += -lam * mask
    mask = radii > (1 - pos[:, 0])
    grad_pos[mask, 0] += lam * mask  # d/dx_i of max(0, r_i - 1 + x_i) = 1
    mask = radii > pos[:, 1]
    grad_pos[mask, 1] += -lam * mask
    mask = radii > (1 - pos[:, 1])
    grad_pos[mask, 1] += lam * mask
    
    # Pairwise gradients
    # d/dp_i of max(0, r_i + r_j - ||p_i - p_j||)
    # If violation > 0, gradient = - (p_i - p_j) / ||p_i - p_j||
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = radii[i] + radii[j] - dist[i, j]
            if v > 0:
                d = pos[i] - pos[j]
                norm = dist[i, j]
                if norm > 1e-10:
                    grad = -d / norm
                    grad_pos[i] += -lam * grad  # d/dp_i of -v = d/dp_i of (dist - r_i - r_j) = (p_i-p_j)/dist
                    # Wait, objective has -lam * v, so d/dp_i = -lam * d(v)/dp_i
                    # d(v)/dp_i = - (p_i - p_j) / dist
                    # So d(-lam*v)/dp_i = -lam * (-(p_i-p_j)/dist) = lam * (p_i-p_j)/dist
                    grad_pos[i] = 0  # reset, let me redo
                    grad_pos[i] += lam * d / norm
                    grad_pos[j] += -lam * d / norm
    
    # Actually, let me be more careful.
    # objective = sum(r) - lam * (sum_wall_violations + sum_pair_violations)
    # d(objective)/d(p_i) = -lam * d(sum_violations)/d(p_i)
    
    # For wall: d(max(0, r_i - x_i))/dx_i = -1 if active
    # d(max(0, r_i - 1 + x_i))/dx_i = 1 if active
    # So d(objective)/dx_i = -lam * (-1 * I[r_i>x_i] + 1 * I[r_i>1-x_i])
    # = lam * (I[r_i>x_i] - I[r_i>1-x_i])
    
    # For pair: v_ij = max(0, r_i + r_j - dist_ij)
    # d(v_ij)/dp_i = - (p_i - p_j)/dist_ij if active
    # d(objective)/dp_i includes -lam * sum_j d(v_ij)/dp_i = -lam * sum_j (- (p_i-p_j)/dist) = lam * sum_j (p_i-p_j)/dist
    
    # So my gradient calculation above was roughly correct but I reset it. Let me redo cleanly.
    
    grad_pos = np.zeros_like(pos)
    grad_radii = np.ones_like(radii)  # d(sum(r))/dr_i = 1
    
    # Wall contributions to grad_pos
    # d(objective)/dx_i = lam * (I[r_i > x_i] - I[r_i > 1 - x_i])
    grad_pos[:, 0] += lam * ((radii > pos[:, 0]).astype(float) - (radii > (1 - pos[:, 0])).astype(float))
    grad_pos[:, 1] += lam * ((radii > pos[:, 1]).astype(float) - (radii > (1 - pos[:, 1])).astype(float))
    
    # Wall contributions to grad_radii
    # d(objective)/dr_i includes -lam * (I[r_i > x_i] + I[r_i > 1-x_i] + I[r_i > y_i] + I[r_i > 1-y_i])
    grad_radii -= lam * ((radii > pos[:, 0]).astype(float) + (radii > (1 - pos[:, 0])).astype(float) +
                         (radii > pos[:, 1]).astype(float) + (radii > (1 - pos[:, 1])).astype(float))
    
    # Pair contributions
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = radii[i] + radii[j] - dist[i, j]
            if v > 0:
                d = pos[i] - pos[j]
                norm = dist[i, j]
                if norm > 1e-10:
                    # d(objective)/dp_i += lam * (p_i - p_j) / dist
                    grad_pos[i] += lam * d / norm
                    grad_pos[j] += -lam * d / norm
                    # d(objective)/dr_i += -lam
                    grad_radii[i] -= lam
                    grad_radii[j] -= lam
    
    # Check validity
    is_valid = (np.all(radii >= 0) and
                np.all(pos[:, 0] >= radii) and np.all(pos[:, 0] <= 1 - radii) and
                np.all(pos[:, 1] >= radii) and np.all(pos[:, 1] <= 1 - radii) and
                np.all(dist >= radii[:, np.newaxis] + radii[np.newaxis, :] - 1e-9))
    
    return objective, grad_pos, grad_radii, is_valid


def _project(pos, radii):
    """Project to feasible region approximately."""
    # Clamp positions
    pos = np.clip(pos, 0, 1)
    # Clamp radii to be non-negative and not exceed wall distance
    wall_dist = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]),
                           np.minimum(pos[:, 1], 1 - pos[:, 1]))
    radii = np.clip(radii, 0, wall_dist)
    return pos, radii


def _physics_relax(pos, radii, n_steps=100):
    """Simple physics-based relaxation to resolve overlaps."""
    for _ in range(n_steps):
        forces = np.zeros_like(pos)
        # Wall forces
        for i in range(len(radii)):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 10
            if pos[i, 0] > 1 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1 - radii[i])) * 10
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 10
            if pos[i, 1] > 1 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1 - radii[i])) * 10
        
        # Pairwise forces
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    force = overlap * 5 * d / dist
                    forces[i] += force
                    forces[j] -= force
        
        pos += forces * 0.01
        pos, radii = _project(pos, radii)
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    np.random.seed(42)
    
    # Initial positions: random
    pos = np.random.rand(N, 2) * 0.8 + 0.1
    radii = np.full(N, 0.02)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_obj = -np.inf
    
    # Initial physics relaxation
    pos, radii = _physics_relax(pos, radii, 50)
    obj, _, _, valid = _objective_and_grad(pos, radii)
    if valid:
        best_pos = pos.copy()
        best_radii = radii.copy()
        best_obj = obj
    
    # Gradient-based optimization
    lr = 0.01
    for step in range(5000):
        if time.time() > deadline:
            break
        
        obj, grad_pos, grad_radii, valid = _objective_and_grad(pos, radii)
        
        if valid:
            if obj > best_obj:
                best_obj = obj
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        # Update
        pos += lr * grad_pos
        radii += lr * grad_radii
        
        # Project
        pos, radii = _project(pos, radii)
        
        # Occasionally do physics relaxation
        if step % 100 == 0:
            pos, radii = _physics_relax(pos, radii, 20)
            obj, _, valid = _objective_and_grad(pos, radii)
            if valid and obj > best_obj:
                best_obj = obj
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        # Decay learning rate
        if step > 3000:
            lr *= 0.99
    
    # Final polish: try to increase radii where possible
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Try to increase radius of circle i
            old_r = best_radii[i]
            new_r = old_r + 0.001
            if new_r < 0.1:  # reasonable upper bound
                # Check if feasible
                test_pos = best_pos.copy()
                test_radii = best_radii.copy()
                test_radii[i] = new_r
                _, _, _, valid = _objective_and_grad(test_pos, test_radii)
                if valid:
                    best_radii[i] = new_r
                    improved = True
        if not improved:
            break
    
    # Final validation
    _, _, _, valid = _objective_and_grad(best_pos, best_radii)
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
