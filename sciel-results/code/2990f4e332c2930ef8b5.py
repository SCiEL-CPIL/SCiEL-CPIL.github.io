"""Rewritten program: local-search circle packing in the unit square."""
import time
import math
import numpy as np

N = 26
DEADLINE = 115.0  # safety margin before the 120 s kill


def _max_radius(x, y, other_idx, xs, ys, rs):
    """Largest radius at (x, y) given other circles and square walls."""
    r = min(x, 1.0 - x, y, 1.0 - y)
    for j in other_idx:
        d = math.hypot(x - xs[j], y - ys[j])
        r = min(r, d - rs[j])
    return max(r, 1e-9)


def _score(xs, ys, rs):
    return float(np.sum(rs))


def _validate(xs, ys, rs, tol=1e-7):
    if len(xs) != N:
        return False
    for i in range(N):
        if xs[i] < -tol or xs[i] > 1.0 + tol or ys[i] < -tol or ys[i] > 1.0 + tol:
            return False
        if rs[i] < 0:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _clamp(x, y, r):
    x = min(max(x, r), 1.0 - r)
    y = min(max(y, r), 1.0 - r)
    return x, y


def _soft_min(values, temp=0.01):
    """Smooth approximation of min using log-sum-exp."""
    if len(values) == 0:
        return 0.0
    v = np.array(values, dtype=np.float64)
    vmax = np.max(v)
    if vmax <= 0:
        return vmax
    log_sum = np.log(np.sum(np.exp((v - vmax) / temp)))
    return vmax - temp * log_sum


def _soft_min_grad(values, temp=0.01):
    """Gradient of soft_min with respect to input values."""
    if len(values) == 0:
        return np.array([])
    v = np.array(values, dtype=np.float64)
    vmax = np.max(v)
    if vmax <= 0:
        return np.zeros_like(v)
    exps = np.exp((v - vmax) / temp)
    probs = exps / np.sum(exps)
    return -probs


def _polish_smooth(xs, ys, rs, deadline, temp=0.01):
    """Polish using smooth surrogate for the max radius objective."""
    n = len(xs)
    best_score = _score(xs, ys, rs)
    
    for iteration in range(200):
        if time.time() >= deadline:
            break
        
        # Compute soft-min radius and gradient for each circle
        improved_any = False
        for i in range(n):
            # Constraints: distance to walls and other circles
            constraints = [xs[i], 1.0 - xs[i], ys[i], 1.0 - ys[i]]
            constraint_grads = [1.0, -1.0, 1.0, -1.0]  # d/dx, d/dy for each
            
            for j in range(n):
                if j == i:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = math.hypot(dx, dy)
                if d < 1e-12:
                    d = 1e-12
                constraints.append(d - rs[j])
                # Gradient of (d - r_j) w.r.t. x_i, y_i
                constraint_grads.append(dx / d)
                constraint_grads.append(dy / d)
            
            # Soft min of constraints
            soft_r = _soft_min(constraints, temp)
            if soft_r < rs[i] * 0.5:
                # If soft radius is much smaller, skip this circle
                continue
            
            # Gradient of soft min w.r.t. position
            probs = _soft_min_grad(constraints, temp)
            
            # dx, dy gradient
            gx = 0.0
            gy = 0.0
            for k in range(len(constraints)):
                if k < 4:
                    # Wall constraints
                    if k == 0:  # x
                        gx += probs[k] * 1.0
                    elif k == 1:  # 1-x
                        gx += probs[k] * (-1.0)
                    elif k == 2:  # y
                        gy += probs[k] * 1.0
                    elif k == 3:  # 1-y
                        gy += probs[k] * (-1.0)
                else:
                    # Circle constraints
                    idx = k - 4
                    gx += probs[k] * constraint_grads[idx * 2]
                    gy += probs[k] * constraint_grads[idx * 2 + 1]
            
            # Step size
            step = 0.01
            new_x = xs[i] + step * gx
            new_y = ys[i] + step * gy
            
            # Clamp
            new_x, new_y = _clamp(new_x, new_y, max(rs[i] * 0.5, 0.01))
            
            # Evaluate actual max radius at new position
            other = [j for j in range(n) if j != i]
            new_r = _max_radius(new_x, new_y, other, xs, ys, rs)
            
            if new_r > rs[i] + 1e-12:
                xs[i], ys[i], rs[i] = new_x, new_y, new_r
                improved_any = True
                new_score = _score(xs, ys, rs)
                if new_score > best_score:
                    best_score = new_score
        
        if not improved_any:
            break
    
    return xs, ys, rs, best_score


def _local_search(xs, ys, rs, deadline):
    """Iterative improvement: move each circle to maximize its own radius."""
    best_score = _score(xs, ys, rs)
    n = len(xs)
    step_size = 0.05
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            old_r = rs[i]
            best_r = old_r
            best_x, best_y = xs[i], ys[i]
            
            # Adaptive step sizes: try multiple scales
            for scale in [step_size, step_size * 0.3, step_size * 3]:
                # Random perturbations
                for _ in range(15):
                    dx = np.random.uniform(-scale, scale)
                    dy = np.random.uniform(-scale, scale)
                    nx, ny = xs[i] + dx, ys[i] + dy
                    nx, ny = _clamp(nx, ny, old_r)
                    other = [j for j in range(n) if j != i]
                    new_r = _max_radius(nx, ny, other, xs, ys, rs)
                    if new_r > best_r + 1e-12:
                        best_r = new_r
                        best_x, best_y = nx, ny
                
                # Grid search around current position
                for dx in [-scale, -scale/2, 0, scale/2, scale]:
                    for dy in [-scale, -scale/2, 0, scale/2, scale]:
                        nx, ny = xs[i] + dx, ys[i] + dy
                        nx, ny = _clamp(nx, ny, old_r)
                        other = [j for j in range(n) if j != i]
                        new_r = _max_radius(nx, ny, other, xs, ys, rs)
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_x, best_y = nx, ny
            
            if best_r > old_r + 1e-12:
                xs[i], ys[i], rs[i] = best_x, best_y, best_r
                improved = True
                new_score = _score(xs, ys, rs)
                if new_score > best_score:
                    best_score = new_score
        
        if not improved:
            # Try larger random jumps to escape local optima
            for _ in range(80):
                if time.time() >= deadline:
                    break
                i = np.random.randint(n)
                old_r = rs[i]
                # Jump to random location
                nx = np.random.uniform(0.05, 0.95)
                ny = np.random.uniform(0.05, 0.95)
                nx, ny = _clamp(nx, ny, old_r)
                other = [j for j in range(n) if j != i]
                new_r = _max_radius(nx, ny, other, xs, ys, rs)
                if new_r > old_r + 1e-12:
                    xs[i], ys[i], rs[i] = nx, ny, new_r
                    improved = True
                    new_score = _score(xs, ys, rs)
                    if new_score > best_score:
                        best_score = new_score
            
            # Also try swapping two circles
            for _ in range(30):
                if time.time() >= deadline:
                    break
                i, j = np.random.choice(n, 2, replace=False)
                xs[i], xs[j] = xs[j], xs[i]
                ys[i], ys[j] = ys[j], ys[i]
                rs[i], rs[j] = rs[j], rs[i]
                new_score = _score(xs, ys, rs)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                else:
                    # Swap back
                    xs[i], xs[j] = xs[j], xs[i]
                    ys[i], ys[j] = ys[j], ys[i]
                    rs[i], rs[j] = rs[j], rs[i]
            
            if not improved:
                # Reduce step size
                step_size *= 0.8
                if step_size < 0.001:
                    break
    return xs, ys, rs, best_score


def _initial_grid():
    """Create an initial grid-based packing."""
    side = int(math.ceil(math.sqrt(N)))
    xs, ys, rs = [], [], []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    return np.array(xs), np.array(ys), np.array(rs)


def _initial_random():
    """Create a random initial packing with small circles."""
    xs, ys, rs = [], [], []
    for _ in range(N):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = 0.02
        xs.append(x)
        ys.append(y)
        rs.append(r)
    return np.array(xs), np.array(ys), np.array(rs)


def _initial_hexagonal():
    """Create a hexagonal initial packing."""
    xs, ys, rs = [], [], []
    # Hexagonal packing with 26 circles
    rows = 6
    cols = 5
    dx = 0.17
    dy = 0.15
    offset = dx * 0.5
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= N:
                break
            x = 0.1 + col * dx + (row % 2) * offset
            y = 0.1 + row * dy
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                xs.append(x)
                ys.append(y)
                rs.append(0.06)
                count += 1
    
    # If we don't have enough, add more
    while len(xs) < N:
        xs.append(np.random.uniform(0.1, 0.9))
        ys.append(np.random.uniform(0.1, 0.9))
        rs.append(0.02)
    
    return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Try multiple restarts with different initializations
    best_xs, best_ys, best_rs = None, None, None
    best_score = -1.0

    # Start with grid initialization
    xs, ys, rs = _initial_grid()
    xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
    if score > best_score:
        best_score = score
        best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Try hexagonal initialization
    if time.time() < deadline - 10.0:
        xs, ys, rs = _initial_hexagonal()
        xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Try several random restarts
    while time.time() < deadline - 10.0:
        xs, ys, rs = _initial_random()
        xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Final smooth polish on the best solution
    if best_xs is not None and time.time() < deadline - 3.0:
        best_xs, best_ys, best_rs, score = _polish_smooth(
            best_xs, best_ys, best_rs, deadline, temp=0.005
        )
        if score > best_score:
            best_score = score

    # Final validation
    if best_xs is not None:
        if not _validate(best_xs, best_ys, best_rs):
            # Fallback to grid if validation fails
            xs, ys, rs = _initial_grid()
            best_xs, best_ys, best_rs = xs, ys, rs

    result = np.column_stack([best_xs, best_ys, best_rs])
    return result
