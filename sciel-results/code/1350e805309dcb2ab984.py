"""Improved packing: hexagonal init + local push optimization with deadline guard."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def _hex_init(n):
    """Generate a hexagonal lattice packing of n circles in [-0.5, 0.5]^2."""
    # We want to fit n circles in a hexagonal arrangement.
    # Try different row counts to find a good fit.
    best_pts = None
    best_score = -1
    
    # For hexagonal packing, rows alternate offset
    # Try various numbers of rows
    for nrows in range(1, n + 1):
        ncols = int(np.ceil(n / nrows))
        if ncols < 1:
            continue
        
        # Spacing for hexagonal: horizontal = s, vertical = s * sqrt(3)/2
        # We need to fit ncols in width 1 and nrows in height 1
        # With hexagonal offset, the effective width for odd rows is (ncols-1)*s + s/2 (offset)
        # Let's just compute positions and check if they fit
        
        # Horizontal spacing
        s_h = 1.0 / (ncols - 1) if ncols > 1 else 1.0
        # Vertical spacing for hexagonal
        s_v = s_h * np.sqrt(3) / 2
        
        # Check if it fits vertically
        total_h = (nrows - 1) * s_v
        if total_h > 1.0:
            continue
        
        # Scale to fit
        # We want the circles to be as large as possible, so we want to maximize min distance
        # and min distance to boundary
        
        # Let's just place them and compute the score
        pts = []
        for i in range(nrows):
            n_in_row = min(ncols, n - len(pts))
            if n_in_row <= 0:
                break
            offset = s_h / 2 if i % 2 == 1 else 0.0
            for j in range(n_in_row):
                x = -0.5 + s_h / 2 + j * s_h + offset
                y = -0.5 + total_h / 2 + i * s_v - total_h / 2 + s_v / 2
                # Actually let me redo the y positioning
                pts.append((x, y))
        
        if len(pts) < n:
            continue
            
        pts = np.array(pts[:n])
        
        # Check if all points are in [-0.5, 0.5]^2
        if np.any(pts < -0.5 + 1e-9) or np.any(pts > 0.5 - 1e-9):
            continue
            
        # Compute score
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:n]
    
    return best_pts


def _compute_score(pts):
    """Compute the packing score: min of all pair distances/2 and boundary distances."""
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:2, 1]
    # Actually let's do it properly with broadcasting
    dx = pts[:, 0:1] - pts[0:1, 0]  # This is wrong, let me redo
    
    # Proper pairwise distance computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    
    # Boundary distances
    min_boundary = np.min(np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    ))
    
    return min(min_pair, min_boundary)


def _local_push(pts, max_iters=500, step_size=0.01, deadline=None):
    """Local optimization by pushing overlapping circles apart and pulling boundary circles inward."""
    n = len(pts)
    pts = pts.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n):
            # Compute forces on point i
            fx = 0.0
            fy = 0.0
            
            # Pairwise repulsion
            for j in range(n):
                if i == j:
                    continue
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    # Random small push
                    fx += 0.001 * np.random.randn()
                    fy += 0.001 * np.random.randn()
                    continue
                if dist < 1.0:  # Overlapping (in terms of radius=0.5 scale, but we use actual coords)
                    # Push apart proportional to overlap
                    overlap = 1.0 - dist
                    fx += (dx / dist) * overlap
                    fy += (dy / dist) * overlap
            
            # Boundary repulsion
            # Left boundary
            if pts[i, 0] < 0:
                fx += (0.5 - pts[i, 0]) if pts[i, 0] < -0.4 else 0
            # Actually, boundary force: push away from boundaries
            # Left: if x < -0.5 + r, push right
            # Right: if x > 0.5 - r, push left
            # Top: if y > 0.5 - r, push down
            # Bottom: if y < -0.5 + r, push up
            
            # We don't know r yet, so use a heuristic: push if within 0.1 of boundary
            margin = 0.1
            if pts[i, 0] < -0.5 + margin:
                fx += (-0.5 + margin - pts[i, 0]) * 0.5
            if pts[i, 0] > 0.5 - margin:
                fx -= (pts[i, 0] - (0.5 - margin)) * 0.5
            if pts[i, 1] < -0.5 + margin:
                fy += (-0.5 + margin - pts[i, 1]) * 0.5
            if pts[i, 1] > 0.5 - margin:
                fy -= (pts[i, 1] - (0.5 - margin)) * 0.5
            
            # Update position
            new_x = pts[i, 0] + step_size * fx
            new_y = pts[i, 1] + step_size * fy
            
            # Clamp to square
            new_x = np.clip(new_x, -0.5, 0.5)
            new_y = np.clip(new_y, -0.5, 0.5)
            
            if abs(new_x - pts[i, 0]) > 1e-12 or abs(new_y - pts[i, 1]) > 1e-12:
                pts[i, 0] = new_x
                pts[i, 1] = new_y
                improved = True
        
        if not improved:
            break
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Step 1: Hexagonal initialization
    pts = _hex_init(N)
    
    # Step 2: Local push optimization
    if time.time() < deadline - 5.0:
        pts = _local_push(pts, max_iters=200, step_size=0.005, deadline=deadline - 5.0)
    
    # Step 3: Fine-tune with smaller steps
    if time.time() < deadline - 2.0:
        pts = _local_push(pts, max_iters=100, step_size=0.001, deadline=deadline - 2.0)
    
    # Final validation: ensure all points are strictly inside
    pts = np.clip(pts, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return pts
# EVOLVE-BLOCK-END
