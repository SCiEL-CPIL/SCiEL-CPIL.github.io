"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np


def _check_validity(pos, tol=1e-8):
    """Check validity: circles inside [0,1]^2 and disjoint interiors."""
    n = len(pos)
    x, y, r = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # Check bounds
    if np.any(x < -tol) or np.any(x > 1 + tol):
        return False
    if np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    
    # Check pairwise using vectorized computation
    # Compute all pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx * dx + dy * dy
    min_dist = r[:, None] + r[None, :]
    
    # Check upper triangle only
    iu = np.triu_indices(n, k=1)
    if np.any(dist_sq[iu] < min_dist[iu] ** 2 - tol * 1e-2):
        return False
    
    return True


def _vectorized_check(pos, tol=1e-8):
    """Vectorized validity check."""
    return _check_validity(pos, tol)


def _build_grid(n):
    """Create a grid initial placement."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                positions.append([x, y, r])
    return np.array(positions[:n])


def _build_hex(n):
    """Create a hexagonal packing initial placement."""
    positions = []
    row_height = 1.0 / (n + 2)
    col_width = row_height * 2 / np.sqrt(3)
    y = row_height * 0.5
    row = 0
    while len(positions) < n:
        x_offset = 0.0
        if row % 2 == 1:
            x_offset = col_width * 0.5
        x = x_offset + col_width * 0.5
        col = 0
        while len(positions) < n and x <= 1.0:
            r = row_height * 0.5 - 1e-4
            positions.append([x, y, r])
            x += col_width
            col += 1
        y += row_height
        row += 1
    return np.array(positions[:n])


def _relax_and_grow(positions, max_iters=1000, tol=1e-10, time_budget=None):
    """
    Iteratively relax overlaps and grow radii using vectorized operations.
    """
    n = len(positions)
    pos = positions.copy()
    start_time = time.time()
    
    for it in range(max_iters):
        if time_budget is not None and time.time() > time_budget:
            break
        
        # Compute pairwise forces (vectorized)
        x = pos[:, 0]
        y = pos[:, 1]
        r = pos[:, 2]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, None] + r[None, :]
        
        # Compute overlap forces
        overlap = np.maximum(min_dist - dist, 0)
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-12, dist, 1e-12)
        fx = dx / safe_dist * overlap
        fy = dy / safe_dist * overlap
        
        # Sum forces for each circle
        fx_sum = np.sum(fx, axis=1)
        fy_sum = np.sum(fy, axis=1)
        
        # Apply forces
        pos[:, 0] += fx_sum * 0.5
        pos[:, 1] += fy_sum * 0.5
        
        # Clamp to unit square
        pos[:, 0] = np.clip(pos[:, 0], pos[:, 2], 1 - pos[:, 2])
        pos[:, 1] = np.clip(pos[:, 1], pos[:, 2], 1 - pos[:, 2])
        
        # Try to grow radii
        max_growth = 0
        for i in range(n):
            max_r = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = (dist - pos[j][2])
                max_r = min(max_r, available)
            max_r = max(max_r, 0)
            if max_r > pos[i][2] + 1e-10:
                growth = (max_r - pos[i][2]) * 0.15
                pos[i][2] += growth
                max_growth = max(max_growth, growth)
        
        # Clamp radii to be valid
        for i in range(n):
            pos[i][2] = min(pos[i][2], pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        
        if max_growth < tol and np.max(overlap) < 1e-10:
            break
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_build_grid(N))
    inits.append(_build_hex(N))
    
    # Add random inits
    rng = np.random.RandomState(42)
    for seed in range(15):
        if time.time() > deadline - 5:
            break
        positions = []
        for _ in range(N):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            r = 0.01 + rng.random() * 0.05
            positions.append([x, y, r])
        inits.append(np.array(positions[:N]))
    
    best_result = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() > deadline - 5:
            break
        
        # Relax and grow
        relaxed = _relax_and_grow(init, max_iters=500, time_budget=deadline - 5)
        
        score = float(np.sum(relaxed[:, 2]))
        
        if _check_validity(relaxed):
            if score > best_score:
                best_score = score
                best_result = relaxed
    
    # Final polish: try to grow radii more aggressively
    if best_result is not None and time.time() < deadline - 2:
        polished = _relax_and_grow(best_result, max_iters=300, tol=1e-12, time_budget=deadline - 2)
        polished_score = float(np.sum(polished[:, 2]))
        if _check_validity(polished) and polished_score > best_score:
            best_result = polished
            best_score = polished_score
    
    if best_result is None:
        # Fallback to grid
        best_result = _build_grid(N)
    
    return best_result
# EVOLVE-BLOCK-END
