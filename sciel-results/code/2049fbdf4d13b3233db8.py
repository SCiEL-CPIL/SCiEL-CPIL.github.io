"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _eval_packing(xy, r):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _force_directed(xy, r, deadline, n_steps=3000, dt=0.002, damping=0.95):
    """
    Physics-based relaxation: circles repel each other (inverse-square)
    and are attracted to walls proportional to how much room they have.
    Momentum is carried between steps for faster convergence.
    """
    n = len(xy)
    xy = xy.astype(np.float64).copy()
    r = r.astype(np.float64).copy()
    vel = np.zeros_like(xy)
    
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(xy, r)
    if not best_valid:
        best_total = 0.0
    
    # Precompute pair indices for speed
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    step = 0
    while step < n_steps:
        if time.time() > deadline:
            break
        
        # Adaptive dt: start large, shrink
        progress = step / n_steps
        cur_dt = dt * (1.0 - 0.5 * progress)
        
        # Compute forces
        forces = np.zeros_like(xy)
        
        # Wall forces: push circle toward center if it's too close to a wall
        # Strength proportional to how far it can grow (i.e., clearance)
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            # If circle is near left wall, push right
            if x < ri:
                forces[i, 0] += 50.0 * (ri - x)
            if x > 1.0 - ri:
                forces[i, 0] -= 50.0 * (x - (1.0 - ri))
            if y < ri:
                forces[i, 1] += 50.0 * (ri - y)
            if y > 1.0 - ri:
                forces[i, 1] -= 50.0 * (y - (1.0 - ri))
        
        # Pair repulsion: if overlapping, push apart; if close, mild repulsion
        for i, j in pairs:
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
            min_dist = r[i] + r[j]
            if dist < min_dist:
                # Strong repulsion when overlapping
                overlap = min_dist - dist
                f = 100.0 * overlap / (dist + 1e-6)
                fx = f * dx / (dist + 1e-6)
                fy = f * dy / (dist + 1e-6)
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
            elif dist < min_dist * 1.5:
                # Mild repulsion in the "near" zone
                gap = dist - min_dist
                f = 5.0 * (min_dist - gap) / (dist + 1e-6) * (1.0 - gap / (0.5 * min_dist))
                fx = f * dx / (dist + 1e-6)
                fy = f * dy / (dist + 1e-6)
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
        
        # Update velocities with damping
        vel = vel * damping + forces * cur_dt
        # Clamp velocity
        vmax = 0.05
        vel = np.clip(vel, -vmax, vmax)
        
        # Update positions
        xy += vel
        
        # Enforce hard wall constraints
        for i in range(n):
            ri = r[i]
            xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
            xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)
        
        step += 1
        
        # Periodic evaluation
        if step % 200 == 0:
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
    
    # Final eval
    total, valid = _eval_packing(xy, r)
    if valid and total > best_total:
        best_total = total
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _best_fit_radius(xy, r, deadline, max_sweeps=20):
    """
    For each circle, find the maximum radius it can have at its current position
    given all other circles. This is a 1D problem per circle (position fixed).
    Then re-optimize position to maximize radius via local perturbation.
    """
    n = len(xy)
    xy = xy.astype(np.float64).copy()
    r = r.astype(np.float64).copy()
    
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(xy, r)
    if not best_valid:
        best_total = 0.0
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Find max radius for circle i at position xy[i]
            max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            
            for j in range(n):
                if j == i:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - r[j])
            
            max_r = max(max_r, 0.001)
            
            if max_r > r[i] + 1e-10:
                r[i] = max_r
                improved = True
            
            # Try to improve position: move circle toward the "gap" center
            # Find the most constraining neighbor
            min_gap = np.inf
            constraining_j = -1
            for j in range(n):
                if j == i:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                gap = dist - r[i] - r[j]
                if gap < min_gap:
                    min_gap = gap
                    constraining_j = j
            
            # Also check walls
            wall_gaps = [xy[i, 0] - r[i], 1.0 - xy[i, 0] - r[i],
                         xy[i, 1] - r[i], 1.0 - xy[i, 1] - r[i]]
            min_wall_gap = min(wall_gaps)
            
            if min_wall_gap < min_gap and min_wall_gap < 0.01:
                # Move away from the tightest wall
                if wall_gaps[0] == min_wall_gap:
                    xy[i, 0] += 0.005
                elif wall_gaps[1] == min_wall_gap:
                    xy[i, 0] -= 0.005
                elif wall_gaps[2] == min_wall_gap:
                    xy[i, 1] += 0.005
                else:
                    xy[i, 1] -= 0.005
                improved = True
            elif constraining_j >= 0 and min_gap < 0.01:
                # Move away from constraining circle
                dx = xy[i, 0] - xy[constraining_j, 0]
                dy = xy[i, 1] - xy[constraining_j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 1e-9:
                    xy[i, 0] += 0.005 * dx / dist
                    xy[i, 1] += 0.005 * dy / dist
                    improved = True
        
        # Enforce walls
        for i in range(n):
            ri = r[i]
            xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
            xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r, best_total, best_valid


def _simulated_annealing(xy, r, deadline, n_iter=50000):
    """
    SA on positions only (radii are recomputed as max feasible at each position).
    Accept moves that increase total radius; accept worse moves with probability
    exp(-delta/T).
    """
    n = len(xy)
    xy = xy.astype(np.float64).copy()
    r = r.astype(np.float64).copy()
    
    # Compute initial radii
    for i in range(n):
        max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        for j in range(n):
            if j == i:
                continue
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - r[j])
        r[i] = max(max_r, 0.001)
    
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(xy, r)
    if not best_valid:
        best_total = 0.0
    
    T_start = 0.05
    T_end = 0.001
    T = T_start
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Pick a random circle to move
        i = random.randint(0, n - 1)
        old_x, old_y = xy[i, 0], xy[i, 1]
        
        # Random perturbation
        scale = 0.02 * (1.0 + T / T_start)
        new_x = old_x + random.gauss(0, scale)
        new_y = old_y + random.gauss(0, scale)
        
        # Clamp to valid wall range (using current radius)
        ri = r[i]
        new_x = np.clip(new_x, ri, 1.0 - ri)
        new_y = np.clip(new_y, ri, 1.0 - ri)
        
        # Try the move
        xy[i, 0] = new_x
        xy[i, 1] = new_y
        
        # Recompute radius for moved circle
        max_r = min(new_x, 1.0 - new_x, new_y, 1.0 - new_y)
        for j in range(n):
            if j == i:
                continue
            dx = new_x - xy[j, 0]
            dy = new_y - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - r[j])
        new_r = max(max_r, 0.001)
        
        delta = new_r - r[i]
        
        if delta > 0 or random.random() < np.exp(delta / T):
            r[i] = new_r
        else:
            xy[i, 0] = old_x
            xy[i, 1] = old_y
        
        # Track best
        if it % 500 == 0:
            total = np.sum(r)
            # Quick validity check
            valid = True
            for k in range(n):
                xk, yk, rk = xy[k, 0], xy[k, 1], r[k]
                if xk - rk < -1e-9 or 1.0 - xk - rk < -1e-9 or yk - rk < -1e-9 or 1.0 - yk - rk < -1e-9:
                    valid = False
                    break
            if valid:
                for k in range(n):
                    for m in range(k + 1, n):
                        dx = xy[k, 0] - xy[m, 0]
                        dy = xy[k, 1] - xy[m, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < r[k] + r[m] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
    
    total, valid = _eval_packing(xy, r)
    if valid and total > best_total:
        best_total = total
        best_valid = True
        best_xy = xy.copy()
        best_r = r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _grid_init(side):
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init():
    positions = []
    r = 0.02
    for i in range(N):
        x = random.uniform(r, 1 - r)
        y = random.uniform(r, 1 - r)
        positions.append([x, y, r])
    return np.array(positions[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Phase 1: Multiple initializations with force-directed relaxation
    inits = []
    for side in [5, 6, 7]:
        xy = _grid_init(side)
        inits.append((xy[:, :2].copy(), xy[:, 2].copy()))
    inits.append((_random_init()[:, :2].copy(), _random_init()[:, 2].copy()))
    
    for idx, (xy_init, r_init) in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # Force-directed relaxation
        phase_deadline = deadline - 20.0
        fd_xy, fd_r, fd_total, fd_valid = _force_directed(xy_init, r_init, phase_deadline, n_steps=2000)
        
        if fd_valid and fd_total > best_total:
            best_total = fd_total
            best_valid = True
            best_xy = fd_xy
            best_r = fd_r
    
    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result
    
    # Phase 2: Best-fit radius optimization
    if time.time() < deadline - 15.0:
        bf_xy, bf_r, bf_total, bf_valid = _best_fit_radius(best_xy, best_r, deadline - 15.0)
        if bf_valid and bf_total > best_total:
            best_total = bf_total
            best_valid = True
            best_xy = bf_xy
            best_r = bf_r
    
    # Phase 3: Simulated annealing for global exploration
    if time.time() < deadline - 10.0:
        sa_xy, sa_r, sa_total, sa_valid = _simulated_annealing(best_xy, best_r, deadline - 10.0, n_iter=30000)
        if sa_valid and sa_total > best_total:
            best_total = sa_total
            best_valid = True
            best_xy = sa_xy
            best_r = sa_r
    
    # Phase 4: Final force-directed polish
    if time.time() < deadline - 3.0:
        fd_xy2, fd_r2, fd_total2, fd_valid2 = _force_directed(best_xy, best_r, deadline - 3.0, n_steps=1500, dt=0.001, damping=0.9)
        if fd_valid2 and fd_total2 > best_total:
            best_total = fd_total2
            best_valid = True
            best_xy = fd_xy2
            best_r = fd_r2
    
    # Final best-fit radius pass
    if time.time() < deadline - 1.0:
        bf_xy2, bf_r2, bf_total2, bf_valid2 = _best_fit_radius(best_xy, best_r, deadline - 1.0, max_sweeps=5)
        if bf_valid2 and bf_total2 > best_total:
            best_total = bf_total2
            best_valid = True
            best_xy = bf_xy2
            best_r = bf_r2
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
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
