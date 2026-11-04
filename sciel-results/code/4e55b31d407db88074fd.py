"""Physics-based bubble relaxation for circle packing."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


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
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _bubble_relaxation(x, y, r, dt, n_steps, deadline):
    """
    Physics-based relaxation: circles are pressurized bubbles that expand.
    Internal pressure drives radius growth; repulsive forces prevent overlap;
    wall constraints keep circles inside [0,1]^2.
    
    x, y: (n,) arrays of center coordinates
    r: (n,) array of radii
    dt: initial time step
    n_steps: number of simulation steps
    deadline: absolute time limit
    """
    n = len(x)
    x = x.copy().astype(np.float64)
    y = y.copy().astype(np.float64)
    r = r.copy().astype(np.float64)
    
    # Initial forces/velocities
    vx = np.zeros(n)
    vy = np.zeros(n)
    vr = np.zeros(n)
    
    # Parameters
    pressure_coeff = 0.5       # Internal pressure: F_r = pressure_coeff * r
    repulsion_k = 50.0         # Repulsive force stiffness
    wall_k = 100.0             # Wall constraint stiffness
    damping = 0.95             # Velocity damping
    max_r = 0.5                # Cap on individual radius
    
    best_total = 0.0
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Internal pressure (drives expansion)
        fr += pressure_coeff * r
        
        # Pairwise repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    dx, dy = 1e-5, 1e-5
                    dist = np.sqrt(dx * dx + dy * dy)
                
                # Overlap: want dist >= r[i] + r[j]
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    force = repulsion_k * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                    
                    # Radii should shrink if overlapping (but pressure pushes back)
                    # Net effect: equilibrium where pressure balances repulsion
                    fr[i] -= repulsion_k * overlap * 0.5
                    fr[j] -= repulsion_k * overlap * 0.5
        
        # Wall constraints
        for i in range(n):
            # Left wall
            if x[i] - r[i] < 0:
                pen = r[i] - x[i]
                fx[i] += wall_k * pen
                fr[i] -= wall_k * pen * 0.5
            # Right wall
            if 1.0 - x[i] - r[i] < 0:
                pen = x[i] + r[i] - 1.0
                fx[i] -= wall_k * pen
                fr[i] -= wall_k * pen * 0.5
            # Bottom wall
            if y[i] - r[i] < 0:
                pen = r[i] - y[i]
                fy[i] += wall_k * pen
                fr[i] -= wall_k * pen * 0.5
            # Top wall
            if 1.0 - y[i] - r[i] < 0:
                pen = y[i] + r[i] - 1.0
                fy[i] -= wall_k * pen
                fr[i] -= wall_k * pen * 0.5
        
        # Update velocities with damping
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        vr = vr * damping + fr * dt
        
        # Update positions and radii
        x += vx * dt
        y += vy * dt
        r += vr * dt
        
        # Enforce hard constraints: radii must be positive and bounded
        r = np.clip(r, 0.001, max_r)
        
        # Project positions to be inside square (hard constraint)
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - r[i])
        
        # Evaluate current state
        total, valid = _eval_packing(np.column_stack([x, y]), r)
        if valid and total > best_total:
            best_total = total
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Adaptive time step: reduce if oscillating
        if step % 100 == 0 and step > 0:
            # Check if we're making progress
            pass  # Keep dt constant for stability
    
    return best_x, best_y, best_r, best_total


def _hex_init():
    """Initialize circles in a hexagonal pattern."""
    positions = []
    # Hexagonal packing: rows offset by half spacing
    side = int(np.ceil(np.sqrt(N)))
    spacing = 1.0 / (side + 0.5)
    r_init = spacing * 0.45
    
    row = 0
    while len(positions) < N:
        for col in range(side + 1):
            if len(positions) >= N:
                break
            x = (col + 0.25) * spacing + (0.25 if row % 2 == 1 else 0)
            y = (row + 0.5) * spacing
            if 0 < x < 1 and 0 < y < 1:
                positions.append([x, y, r_init])
        row += 1
    
    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    
    return np.array(positions[:N])


def _random_init(seed=42):
    """Initialize circles at random positions."""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(N):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = 0.03
        positions.append([x, y, r])
    return np.array(positions)


def _grid_init(side=5):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.4 / side
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_total = 0.0
    best_result = None

    # Try multiple initializations
    initializations = [
        ("hex", _hex_init()),
        ("random42", _random_init(42)),
        ("random123", _random_init(123)),
        ("grid5", _grid_init(5)),
        ("grid6", _grid_init(6)),
    ]

    for name, init_pos in initializations:
        if time.time() > deadline:
            break
        
        x = init_pos[:, 0].copy()
        y = init_pos[:, 1].copy()
        r = init_pos[:, 2].copy()
        
        # Run bubble relaxation with adaptive parameters
        dt = 0.001
        n_steps = 5000
        
        out_x, out_y, out_r, total = _bubble_relaxation(x, y, r, dt, n_steps, deadline)
        
        if total > best_total:
            best_total = total
            best_result = np.column_stack([out_x, out_y, out_r])
    
    # If no valid result, fall back to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and cleanup
    result = best_result.copy()
    n = len(result)
    for i in range(n):
        x, y, ri = result[i, 0], result[i, 1], result[i, 2]
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
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
