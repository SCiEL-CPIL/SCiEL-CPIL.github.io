import numpy as np
import time
import scipy.optimize as opt

N = 50

def true_radius(positions):
    """Compute the true minimum radius for given positions."""
    n = positions.shape[0]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = 0.5 * np.sqrt(dist_sq[mask])
    min_pair = np.min(dists)
    
    wall_dists = np.concatenate([
        0.5 - positions[:, 0],
        0.5 + positions[:, 0],
        0.5 - positions[:, 1],
        0.5 + positions[:, 1]
    ])
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _hex_init(n):
    """Create hexagonal close-packed initial positions."""
    positions = []
    dx = 1.0 / np.sqrt(n)
    dy = dx * np.sqrt(3) / 2
    
    row = 0
    while len(positions) < n:
        for col in range(int(1.0 / dx) + 1):
            if len(positions) >= n:
                break
            x = (col + 0.5 * (row % 2)) * dx - 0.5
            y = (row + 0.5) * dy - 0.5
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                positions.append([x, y])
        row += 1
    
    return np.array(positions[:n])

def _square_grid_init(n):
    """Create square grid initial positions."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=42):
    """Create random initial positions."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (n, 2))

def _slsqp_optimize(positions, deadline):
    """
    Optimize positions using SLSQP with hard constraints.
    Maximizes r subject to:
      - ||pos_i - pos_j|| >= 2r for all i < j
      - |pos_i_k| <= 0.5 - r for all i, k
    """
    n = positions.shape[0]
    
    def objective(x):
        """Minimize -r (maximize r)."""
        pos = x[:-1].reshape(n, 2)
        r = x[-1]
        return -r
    
    def objective_grad(x):
        """Gradient of objective."""
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad
    
    def constraint_pairs(x):
        """Constraints: ||pos_i - pos_j|| >= 2r for all i < j."""
        pos = x[:-1].reshape(n, 2)
        r = x[-1]
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                constraints.append(d - 2 * r)
        return np.array(constraints)
    
    def constraint_pairs_grad(x):
        """Gradient of pair constraints."""
        pos = x[:-1].reshape(n, 2)
        r = x[-1]
        n_pairs = n * (n - 1) // 2
        grad = np.zeros((n_pairs, 2 * n + 1))
        
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    d = 1e-10
                # d/dpos_i of (d - 2r) = diff / d
                grad[pair_idx, 2*i:2*i+2] = diff / d
                # d/dpos_j of (d - 2r) = -diff / d
                grad[pair_idx, 2*j:2*j+2] = -diff / d
                # d/dr of (d - 2r) = -2
                grad[pair_idx, -1] = -2.0
                pair_idx += 1
        return grad
    
    def constraint_walls(x):
        """Constraints: |pos_i_k| <= 0.5 - r for all i, k."""
        pos = x[:-1].reshape(n, 2)
        r = x[-1]
        constraints = []
        for i in range(n):
            for k in range(2):
                constraints.append(0.5 - r - abs(pos[i, k]))
        return np.array(constraints)
    
    def constraint_walls_grad(x):
        """Gradient of wall constraints."""
        pos = x[:-1].reshape(n, 2)
        r = x[-1]
        grad = np.zeros((2 * n, 2 * n + 1))
        
        for i in range(n):
            for k in range(2):
                idx = 2 * i + k
                # d/dpos_ik of (0.5 - r - |pos_ik|) = -sign(pos_ik)
                if pos[i, k] >= 0:
                    grad[idx, 2*i + k] = -1.0
                else:
                    grad[idx, 2*i + k] = 1.0
                # d/dr of (0.5 - r - |pos_ik|) = -1
                grad[idx, -1] = -1.0
        return grad
    
    # Combine constraints
    def all_constraints(x):
        return np.concatenate([constraint_pairs(x), constraint_walls(x)])
    
    def all_constraints_grad(x):
        return np.concatenate([constraint_pairs_grad(x), constraint_walls_grad(x)], axis=0)
    
    # Initial radius
    r0 = true_radius(positions)
    x0 = np.concatenate([positions.flatten(), [r0]])
    
    # Bounds: positions in [-0.5, 0.5], r in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.0, 0.5)]
    
    try:
        res = opt.minimize(
            objective, x0, jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': all_constraints, 'jac': all_constraints_grad}
            ],
            options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.success or res.fun < -r0:
            new_pos = res.x[:-1].reshape(n, 2)
            new_r = true_radius(new_pos)
            if new_r > r0:
                return new_pos, new_r
    except Exception:
        pass
    
    return positions, r0

def _force_relax(positions, n_steps=1000, dt=0.01, deadline=None):
    """
    Simple force-directed relaxation to spread circles apart.
    """
    n = positions.shape[0]
    pos = positions.copy()
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    # Random small perturbation if coincident
                    forces[i] += np.random.uniform(-0.01, 0.01, 2)
                    forces[j] -= np.random.uniform(-0.01, 0.01, 2)
                else:
                    # Repulsive force proportional to 1/d
                    f = diff / d / d
                    forces[i] += f
                    forces[j] -= f
        
        # Wall repulsion
        for i in range(n):
            for k in range(2):
                if pos[i, k] > 0:
                    forces[i, k] -= 0.1
                else:
                    forces[i, k] += 0.1
        
        # Update positions
        pos += dt * forces
        pos = np.clip(pos, -0.49, 0.49)
    
    return pos

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_r = 0.0
    best_pos = None
    
    # Multiple initializations
    inits = [
        _square_grid_init(N),
        _hex_init(N),
        _random_init(N, 42),
        _random_init(N, 123),
        _random_init(N, 999),
    ]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        current_pos = init.copy()
        current_r = true_radius(current_pos)
        
        if current_r > best_r:
            best_r = current_r
            best_pos = current_pos.copy()
        
        # Force relaxation to improve initial packing
        if time.time() < deadline - 5:
            relaxed_pos = _force_relax(current_pos, n_steps=500, dt=0.02, deadline=deadline - 5)
            relaxed_r = true_radius(relaxed_pos)
            if relaxed_r > current_r:
                current_pos = relaxed_pos
                current_r = relaxed_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = current_pos.copy()
        
        # SLSQP optimization with hard constraints
        if time.time() < deadline - 3:
            optimized_pos, optimized_r = _slsqp_optimize(current_pos, deadline - 3)
            if optimized_r > current_r:
                current_pos = optimized_pos
                current_r = optimized_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = current_pos.copy()
    
    # Final polish with SLSQP on best result
    if time.time() < deadline - 2 and best_pos is not None:
        polished_pos, polished_r = _slsqp_optimize(best_pos, deadline - 2)
        if polished_r > best_r:
            best_r = polished_r
            best_pos = polished_pos
    
    if best_pos is None:
        return _square_grid_init(N)
    
    # Ensure validity
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    return best_pos
