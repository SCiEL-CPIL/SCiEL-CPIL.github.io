"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import KDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import time


def _score(pos):
    """Compute the packing score (minimum radius)."""
    if not np.all(np.isfinite(pos)):
        return 0.0
    if pos.shape != (N, 2):
        return 0.0
    # Check inside square
    if np.any(np.abs(pos) > 0.5 + 1e-12):
        return 0.0
    # Wall distances
    wall_dists = 0.5 - np.abs(pos)
    min_wall = np.min(wall_dists)
    # Pair distances
    tree = KDTree(pos)
    pairs, dists = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        min_pair = 1.0
    else:
        d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
        min_pair = np.min(d)
    return min(min_wall, min_pair / 2.0)


def _force_relax(pos, max_time=10.0, dt=0.0001, damping=0.98, k_rep=100.0, k_wall=500.0):
    """Velocity-Verlet style force-directed relaxation with adaptive stepping."""
    pos = pos.copy().astype(float)
    vel = np.zeros_like(pos)
    start = time.time()
    
    for iteration in range(500000):
        if time.time() - start > max_time:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        tree = KDTree(pos)
        # Use a reasonable cutoff for neighbor search
        pairs, dists = tree.query_pairs(0.15, output_type='ndarray')
        
        for i, j in pairs:
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Random nudge for coincident points
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            if dist > 0:
                # Repulsive force: strong when close
                f = k_rep * d / (dist ** 3)
                forces[i] += f
                forces[j] -= f
        
        # Wall repulsion
        for axis in range(2):
            # Right wall
            over = np.maximum(0.0, pos[:, axis] - (0.5 - 0.01))
            forces[:, axis] -= k_wall * over
            # Left wall
            under = np.maximum(0.0, -0.5 - 0.01 - pos[:, axis])
            forces[:, axis] -= k_wall * under
            # Top wall
            over = np.maximum(0.0, pos[:, axis] - (0.5 - 0.01))
            forces[:, axis] -= k_wall * over
            # Bottom wall
            under = np.maximum(0.0, -0.5 - 0.01 - pos[:, axis])
            forces[:, axis] -= k_wall * under
        
        # Actually, let's do wall forces properly per axis
        forces = np.zeros_like(pos)
        for i, j in pairs:
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            if dist > 0:
                f = k_rep * d / (dist ** 3)
                forces[i] += f
                forces[j] -= f
        
        # Wall forces
        for axis in range(2):
            # Push away from right wall (x=0.5)
            over_r = np.maximum(0.0, pos[:, axis] - (0.5 - 0.005))
            forces[:, axis] -= k_wall * over_r
            # Push away from left wall (x=-0.5)
            over_l = np.maximum(0.0, (-0.5 + 0.005) - pos[:, axis])
            forces[:, axis] += k_wall * over_l
        
        # Velocity Verlet integration
        vel = vel * damping + forces * dt
        pos = pos + vel * dt
        
        # Clamp to square
        pos = np.clip(pos, -0.5, 0.5)
        
        # Adaptive time step: reduce if any pair is too close
        if len(pairs) > 0:
            d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
            min_d = np.min(d)
            if min_d < 0.005:
                dt *= 0.5
            elif min_d > 0.03:
                dt = min(dt * 1.05, 0.001)
    
    return pos


def _lbfgs_polish(pos, max_time=15.0):
    """Polish with L-BFGS-B maximizing the minimum distance."""
    from scipy.optimize import minimize
    
    start = time.time()
    
    def neg_score(x):
        pos = x.reshape(-1, 2)
        return -_score(pos)
    
    def neg_score_grad(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            g[i] = (neg_score(xp) - neg_score(xm)) / (2 * eps)
        return g
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            neg_score, x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_pos = None
    best_score = 0.0
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Square grid (baseline)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 2. Hexagonal packing
    hex_pos = []
    row = 0
    while len(hex_pos) < N:
        x_offset = (row % 2) * 0.25
        for col in range(m + 1):
            x = col * 0.5 - 0.25 + x_offset
            y = row * 0.433 - 0.25
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                hex_pos.append([x, y])
        row += 1
    if len(hex_pos) >= N:
        inits.append(np.array(hex_pos[:N]))
    
    # 3. Random initialization
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Clustered random
    centers = np.random.uniform(-0.3, 0.3, (8, 2))
    cluster_pos = []
    per_cluster = N // 8
    for c in range(8):
        cluster = centers[c] + np.random.randn(per_cluster, 2) * 0.05
        cluster = np.clip(cluster, -0.45, 0.45)
        cluster_pos.append(cluster)
    inits.append(np.vstack(cluster_pos)[:N])
    
    # 5. Another random with different seed
    np.random.seed(123)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # Process each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # Force relaxation
        relaxed = _force_relax(init.copy(), max_time=10.0)
        score = _score(relaxed)
        
        if score > best_score:
            best_score = score
            best_pos = relaxed.copy()
        
        # L-BFGS polish
        if time.time() > deadline - 15.0:
            break
        polished = _lbfgs_polish(relaxed, max_time=15.0)
        score = _score(polished)
        
        if score > best_score:
            best_score = score
            best_pos = polished.copy()
    
    # Final polish on best
    if best_pos is not None and time.time() < deadline - 5.0:
        final = _lbfgs_polish(best_pos, max_time=10.0)
        score = _score(final)
        if score > best_score:
            best_pos = final
    
    if best_pos is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
