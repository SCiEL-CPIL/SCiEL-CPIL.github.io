"""Circle packing in perimeter-4 rectangle: contact-graph + greedy + radius growth."""
import numpy as np
import time

N = 21

def _valid_pack(cx, cy, r, W, H, tol=1e-7):
    """Check circles fit in [0,W]x[0,H] with disjoint interiors."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _greedy_place(W, H, seed=0):
    """Place N circles greedily, maximizing clearance at each step."""
    rng = np.random.default_rng(seed)
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    # Start with a reasonable base radius
    base_r = min(W, H) / 2 / np.sqrt(N) * 0.7
    for i in range(N):
        # Sample candidate positions
        best_pos = None
        best_clearance = -1
        for _ in range(200):
            x = rng.uniform(0, W)
            y = rng.uniform(0, H)
            # Compute clearance: min distance to walls minus r, and to other circles
            wall_clear = min(x, W - x, y, H - y)
            if wall_clear < base_r * 0.5:
                continue
            min_circ_clear = wall_clear
            for j in range(i):
                dx = x - cx[j]
                dy = y - cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                clear = dist - (base_r + r[j])
                min_circ_clear = min(min_circ_clear, clear)
            clearance = min(wall_clear - base_r, min_circ_clear)
            if clearance > best_clearance:
                best_clearance = clearance
                best_pos = (x, y)
        if best_pos is None:
            # Fallback: grid position
            side = int(np.ceil(np.sqrt(N)))
            idx = i
            gx = (idx % side + 0.5) / side
            gy = (idx // side + 0.5) / side
            best_pos = (gx * W, gy * H)
            best_clearance = 0
        cx[i] = best_pos[0]
        cy[i] = best_pos[1]
        r[i] = base_r
    return cx, cy, r

def _grow_radii(cx, cy, r, W, H, max_iter=500, dt=0.002):
    """Grow radii while maintaining validity via force-directed relaxation."""
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    for it in range(max_iter):
        # Try to grow all radii slightly
        dr = 0.001
        r_new = r + dr
        # Check if growth is valid
        if _valid_pack(cx, cy, r_new, W, H, tol=1e-8):
            r = r_new
            continue
        # Otherwise, do force-directed relaxation to make room
        fx = np.zeros(N)
        fy = np.zeros(N)
        # Wall repulsion
        for i in range(N):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 50
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 50
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 50
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 50
        # Pairwise repulsion with target distance r[i]+r[j]+dr
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                target = r[i] + r[j] + dr
                if d < target and d > 1e-12:
                    overlap = target - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10
        cx += dt * fx
        cy += dt * fy
        # Clamp to rectangle
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        # Check if we can now grow
        if _valid_pack(cx, cy, r_new, W, H, tol=1e-8):
            r = r_new
    return cx, cy, r

def _contact_graph_optimize(cx, cy, r, W, H, max_iter=2000):
    """Optimize positions using contact-graph (bar-joint) model."""
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for it in range(max_iter):
        # Build contact graph: edges where dist < r[i]+r[j]+margin
        margin = 0.05 * min(W, H)
        contacts = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] + margin:
                    contacts.append((i, j, dist))
        
        # Force-directed: springs for contacts, repulsion for others
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall forces
        for i in range(N):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 100
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 100
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 100
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 100
        
        # Contact spring forces: want dist = r[i]+r[j]
        for i, j, dist in contacts:
            if dist < 1e-12:
                continue
            target = r[i] + r[j]
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            force = (dist - target) * 50  # spring constant
            fx[i] += (dx / dist) * force
            fy[i] += (dy / dist) * force
            fx[j] -= (dx / dist) * force
            fy[j] -= (dy / dist) * force
        
        # Repulsion for non-contacts (soft)
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    fx[i] += (dx / dist) * overlap * 20
                    fy[i] += (dy / dist) * overlap * 20
                    fx[j] -= (dx / dist) * overlap * 20
                    fy[j] -= (dy / dist) * overlap * 20
        
        # Update positions
        cx += 0.01 * fx
        cy += 0.01 * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    
    return cx, cy, r

def _try_aspect(W, H, deadline, seed=0):
    """Try multiple strategies for a given aspect ratio."""
    best_val = -1
    best_result = None
    
    # Strategy 1: Greedy placement + radius growth
    cx, cy, r = _greedy_place(W, H, seed=seed)
    cx, cy, r = _grow_radii(cx, cy, r, W, H, max_iter=300)
    if _valid_pack(cx, cy, r, W, H):
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_result = (cx.copy(), cy.copy(), r.copy())
    
    if time.time() > deadline:
        return best_result, best_val
    
    # Strategy 2: Grid init + contact graph optimization
    side = int(np.ceil(np.sqrt(N)))
    cx = np.array([(i % side + 0.5) / side * W for i in range(N)])
    cy = np.array([(i // side + 0.5) / side * H for i in range(N)])
    r = np.full(N, min(W / side, H / side) * 0.45)
    cx, cy, r = _contact_graph_optimize(cx, cy, r, W, H, max_iter=1000)
    # Grow radii
    cx, cy, r = _grow_radii(cx, cy, r, W, H, max_iter=200)
    if _valid_pack(cx, cy, r, W, H):
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_result = (cx.copy(), cy.copy(), r.copy())
    
    if time.time() > deadline:
        return best_result, best_val
    
    # Strategy 3: Random init + contact graph
    rng = np.random.default_rng(seed + 100)
    cx = rng.uniform(0, W, N)
    cy = rng.uniform(0, H, N)
    r = np.full(N, min(W, H) / 2 / np.sqrt(N) * 0.5)
    cx, cy, r = _contact_graph_optimize(cx, cy, r, W, H, max_iter=1000)
    cx, cy, r = _grow_radii(cx, cy, r, W, H, max_iter=200)
    if _valid_pack(cx, cy, r, W, H):
        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_result = (cx.copy(), cy.copy(), r.copy())
    
    return best_result, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_val = 0.0
    best_result = None
    
    # Try various aspect ratios (W + H = 2)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 0.9, 1.1]
    
    for idx, W in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        result, val = _try_aspect(W, H, deadline, seed=idx * 7)
        if val > best_val:
            best_val = val
            best_result = result
    
    # If no valid result, fall back to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    cx, cy, r = best_result
    return np.column_stack([cx, cy, r])
