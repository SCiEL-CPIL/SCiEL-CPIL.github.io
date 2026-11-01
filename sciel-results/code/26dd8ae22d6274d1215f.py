"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the score (minimum radius) for a configuration."""
    c = np.asarray(c)
    # Check boundaries
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    # Pairwise distances
    d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to walls
    wall_dists = np.minimum(np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1]))
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _hex_init(n, jitter=0.0):
    """Generate a hexagonal lattice initialization."""
    if jitter == 0:
        rng = np.random.default_rng(42)
    else:
        rng = np.random.default_rng()
    
    # Hexagonal packing in a unit square
    # Try different orientations and spacings
    best_c = None
    best_score = -1
    
    for spacing in np.linspace(0.035, 0.045, 20):
        # Hexagonal lattice
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        
        pts = []
        row = 0
        y = -0.5 + dy / 2
        while y <= 0.5 - dy / 2 + 1e-9:
            x_offset = (dx / 2) if (row % 2 == 1) else 0
            x = -0.5 + dx / 2 + x_offset
            while x <= 0.5 - dx / 2 + 1e-9:
                if x + dx/2 <= 0.5 + 1e-9 and x - dx/2 >= -0.5 - 1e-9:
                    pts.append([x, y])
            row += 1
            y += dy
        
        pts = np.array(pts)
        if len(pts) >= n:
            # Add jitter
            if jitter > 0:
                pts = pts + rng.normal(0, jitter, size=pts.shape)
                # Clip to valid region
                pts[:, 0] = np.clip(pts[:, 0], -0.5 + 0.001, 0.5 - 0.001)
                pts[:, 1] = np.clip(pts[:, 1], -0.5 + 0.001, 0.5 - 0.001)
            
            # Take first n points (or random subset if too many)
            if len(pts) > n:
                idx = rng.choice(len(pts), n, replace=False)
                pts = pts[idx]
            
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_c = pts.copy()
    
    if best_c is None:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:n]
    
    return best_c

def _gradient_ascent(c, max_iter=5000, lr=0.001, tol=1e-12):
    """Simple gradient ascent to maximize the minimum distance."""
    c = np.array(c, dtype=float)
    n = len(c)
    
    for it in range(max_iter):
        # Compute all pairwise distances
        d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum distance
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        min_dist = dists[i, j]
        
        if min_dist < 1e-12:
            break
        
        # Gradient of min distance with respect to positions
        grad = np.zeros_like(c)
        
        # Contribution from the closest pair
        diff = c[i] - c[j]
        norm = np.linalg.norm(diff)
        if norm > 1e-12:
            unit = diff / norm
            grad[i] += 0.5 * unit
            grad[j] -= 0.5 * unit
        
        # Wall constraints
        for k in range(n):
            # x boundaries
            if c[k, 0] > 0:
                grad[k, 0] -= 0.5
            else:
                grad[k, 0] += 0.5
            if c[k, 1] > 0:
                grad[k, 1] -= 0.5
            else:
                grad[k, 1] += 0.5
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        
        # Update
        c = c + lr * grad / gnorm
        
        # Project back to square
        c[:, 0] = np.clip(c[:, 0], -0.5 + 0.0001, 0.5 - 0.0001)
        c[:, 1] = np.clip(c[:, 1], -0.5 + 0.0001, 0.5 - 0.0001)
        
        # Adaptive learning rate
        new_score = _score(c)
        old_score = min_dist / 2
        if new_score > old_score:
            lr = min(lr * 1.05, 0.01)
        else:
            lr = max(lr * 0.95, 0.0001)
    
    return c

def _polish(c, max_iter=10000):
    """High-precision polish using alternating optimization."""
    c = np.array(c, dtype=float)
    n = len(c)
    
    for it in range(max_iter):
        # Find all close pairs (within 2x current min distance)
        d = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        
        # Pairs that are close
        close_mask = dists < min_dist * 1.5
        np.fill_diagonal(close_mask, False)
        
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if close_mask[i, j]:
                    diff = c[i] - c[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-12:
                        continue
                    # Push apart
                    push = 0.001 * diff / norm
                    c[i] += push
                    c[j] -= push
                    improved = True
        
        # Push away from walls
        for k in range(n):
            for axis in range(2):
                if c[k, axis] > 0.45:
                    c[k, axis] -= 0.001
                    improved = True
                elif c[k, axis] < -0.45:
                    c[k, axis] += 0.001
                    improved = True
        
        # Project to square
        c[:, 0] = np.clip(c[:, 0], -0.5 + 1e-6, 0.5 - 1e-6)
        c[:, 1] = np.clip(c[:, 1], -0.5 + 1e-6, 0.5 - 1e-6)
        
        if not improved:
            break
    
    return c

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 second safety margin
    
    best_c = None
    best_score = -1
    
    # Try multiple initializations
    inits = []
    
    # Hexagonal with different jitter levels
    for jitter in [0.0, 0.001, 0.002, 0.005]:
        c = _hex_init(N, jitter)
        inits.append(c)
    
    # Grid initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # Random initialization
    rng = np.random.default_rng(123)
    c_rand = rng.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # Process each initialization
    for c0 in inits:
        if time.time() > deadline:
            break
        
        # Gradient ascent
        c = _gradient_ascent(c0, max_iter=2000, lr=0.001)
        s = _score(c)
        
        if s > best_score:
            best_score = s
            best_c = c.copy()
        
        # Polish
        if time.time() > deadline:
            break
        c_polished = _polish(c, max_iter=2000)
        s_polished = _score(c_polished)
        
        if s_polished > best_score:
            best_score = s_polished
            best_c = c_polished.copy()
    
    # Final polish with more iterations if time allows
    if best_c is not None and time.time() < deadline - 5:
        c_final = _polish(best_c, max_iter=5000)
        s_final = _score(c_final)
        if s_final > best_score:
            best_c = c_final
            best_score = s_final
    
    if best_c is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
