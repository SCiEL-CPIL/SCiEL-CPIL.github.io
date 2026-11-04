"""Circle packing in unit square - contact-graph force-directed + SA strategy."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Contact graph: for each circle, track its nearest neighbors (circle-circle) and wall constraints
# We use a sparse representation: for each circle i, store list of (j, type) where type=0 is circle, 1 is wall

def _build_contact_graph(xy, r, k=4):
    """Build a contact graph: for each circle, find its k nearest other circles and wall constraints."""
    n = len(xy)
    # For each circle, find nearest neighbors
    contacts = [[] for _ in range(n)]
    # Wall contacts: which walls are close
    wall_contacts = [set() for _ in range(n)]
    
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        # Wall distances
        d_left = x - ri
        d_right = 1.0 - x - ri
        d_bottom = y - ri
        d_top = 1.0 - y - ri
        # Track walls that are "close" (gap < 2*r_i)
        if d_left < 2.0 * ri:
            wall_contacts[i].add('L')
        if d_right < 2.0 * ri:
            wall_contacts[i].add('R')
        if d_bottom < 2.0 * ri:
            wall_contacts[i].add('B')
        if d_top < 2.0 * ri:
            wall_contacts[i].add('T')
        
        # Circle-circle distances
        dists = []
        for j in range(n):
            if i == j:
                continue
            dx = x - xy[j, 0]
            dy = y - xy[j, 1]
            d = math.sqrt(dx*dx + dy*dy)
            dists.append((d, j))
        dists.sort()
        for d, j in dists[:k]:
            contacts[i].append(j)
    
    return contacts, wall_contacts

def _contact_energy(xy, r, contacts, wall_contacts, stiffness=1.0):
    """
    Energy based on contact graph: penalize gaps (under-constraint) and overlaps (over-constraint).
    For circle-circle: energy = stiffness * max(0, r_i + r_j - dist)^2 + soft_term * max(0, gap_target - gap)^2
    For walls: energy = stiffness * max(0, r_i - dist_to_wall)^2 + soft_term * max(0, gap_target - gap)^2
    """
    n = len(xy)
    energy = 0.0
    gap_target = 0.001  # Small target gap
    
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        # Wall constraints
        for wall in wall_contacts[i]:
            if wall == 'L':
                gap = x - ri
            elif wall == 'R':
                gap = 1.0 - x - ri
            elif wall == 'B':
                gap = y - ri
            elif wall == 'T':
                gap = 1.0 - y - ri
            else:
                continue
            if gap < 0:
                energy += stiffness * gap * gap
            elif gap < gap_target:
                energy += 0.1 * (gap_target - gap) ** 2
        
        # Circle-circle constraints
        for j in contacts[i]:
            if j <= i:  # Avoid double counting
                continue
            dx = x - xy[j, 0]
            dy = y - xy[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            gap = dist - ri - r[j]
            if gap < 0:
                energy += stiffness * gap * gap
            elif gap < gap_target:
                energy += 0.1 * (gap_target - gap) ** 2
    
    return energy

def _grow_radii(xy, r, contacts, wall_contacts, max_iter=100):
    """
    Grow radii iteratively: for each circle, increase radius until it hits the nearest constraint.
    """
    n = len(xy)
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            # Find maximum radius that keeps circle inside square and non-overlapping
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in contacts[i]:
                dx = x - xy[j, 0]
                dy = y - xy[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - r[j])
            # Also check all circles, not just contacts (for safety)
            for j in range(n):
                if i == j:
                    continue
                dx = x - xy[j, 0]
                dy = y - xy[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - r[j])
            max_r = max(max_r, 0.001)
            if max_r > ri + 1e-9:
                r[i] = max_r
                improved = True
        if not improved:
            break
    return r

def _sa_packing(xy, r, deadline, max_iter=50000):
    """
    Simulated annealing on the contact graph.
    Perturb positions and radii, accept if energy decreases or with probability exp(-delta/T).
    """
    n = len(xy)
    start_time = time.time()
    
    # Build initial contact graph
    contacts, wall_contacts = _build_contact_graph(xy, r)
    
    # Initial energy
    current_energy = _contact_energy(xy, r, contacts, wall_contacts)
    
    # Track best valid packing
    best_xy = xy.copy()
    best_r = r.copy()
    best_total = 0.0
    
    # Check initial validity
    total, valid = _eval_packing(xy, r)
    if valid:
        best_total = total
    
    # SA parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9999
    step_size = 0.02  # Initial perturbation size
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Perturb a random circle's position and radius
        i = np.random.randint(n)
        old_x, old_y, old_r = xy[i, 0], xy[i, 1], r[i]
        
        # New position: small random shift
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        new_x = old_x + dx
        new_y = old_y + dy
        
        # New radius: small random change
        dr = np.random.uniform(-step_size * 0.5, step_size * 0.5)
        new_r = max(0.001, old_r + dr)
        
        # Keep inside square (clip)
        new_x = np.clip(new_x, new_r, 1.0 - new_r)
        new_y = np.clip(new_y, new_r, 1.0 - new_r)
        
        # Apply perturbation
        xy[i, 0] = new_x
        xy[i, 1] = new_y
        r[i] = new_r
        
        # Recompute energy (only for affected circle and its neighbors)
        # For simplicity, recompute full energy (n=26 is small)
        new_energy = _contact_energy(xy, r, contacts, wall_contacts)
        
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            current_energy = new_energy
            # Update best if valid
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_xy = xy.copy()
                best_r = r.copy()
        else:
            # Revert
            xy[i, 0] = old_x
            xy[i, 1] = old_y
            r[i] = old_r
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        step_size *= 0.99995  # Slowly reduce step size
    
    # Final: grow radii
    contacts, wall_contacts = _build_contact_graph(best_xy, best_r)
    best_r = _grow_radii(best_xy, best_r, contacts, wall_contacts)
    
    # Validate final
    total, valid = _eval_packing(best_xy, best_r)
    if valid:
        best_total = total
    else:
        # Fallback: shrink radii until valid
        for i in range(n):
            best_r[i] *= 0.95
        total, valid = _eval_packing(best_xy, best_r)
        if valid:
            best_total = total
    
    return best_xy, best_r, best_total, valid

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Initial radius: half the cell size minus a small margin
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _spiral_init():
    """Create a spiral initialization for better coverage."""
    positions = []
    center_x, center_y = 0.5, 0.5
    for i in range(N):
        angle = i * 2.399963229728653  # Golden angle
        radius = 0.05 + 0.15 * math.sqrt(i / N)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        r = 0.03
        positions.append([x, y, r])
    return np.array(positions[:N])

def _diagonal_init():
    """Create a diagonal initialization."""
    positions = []
    for i in range(N):
        t = i / (N - 1)
        x = 0.1 + 0.8 * t
        y = 0.1 + 0.8 * (1 - t)
        r = 0.03
        positions.append([x, y, r])
    return np.array(positions[:N])

def _random_init():
    """Create a random initialization."""
    np.random.seed(123)
    positions = []
    for i in range(N):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = 0.03
        positions.append([x, y, r])
    return np.array(positions[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (20s buffer)

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Strategy 1: Grid initialization + SA
    for side in [5, 6]:
        if time.time() > deadline - 40.0:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        # Run SA
        out_xy, out_r, total, valid = _sa_packing(xy, r, deadline - 30.0)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Strategy 2: Spiral initialization + SA
    if time.time() < deadline - 30.0:
        xy = _spiral_init()
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _sa_packing(xy, r, deadline - 20.0)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Strategy 3: Diagonal initialization + SA
    if time.time() < deadline - 20.0:
        xy = _diagonal_init()
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _sa_packing(xy, r, deadline - 10.0)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Strategy 4: Random initialization + SA
    if time.time() < deadline - 10.0:
        xy = _random_init()
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _sa_packing(xy, r, deadline - 5.0)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # If no valid result found, fall back to a simple grid
    if best_xy is None or not best_valid:
        side = int(math.ceil(math.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        # Quick validation
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            # Last resort: tiny circles in a grid
            side = int(math.ceil(math.sqrt(N)))
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        # Ensure circle is inside unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to simple grid
        side = int(math.ceil(math.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
