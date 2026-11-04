import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-6):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _constraint_violation(circles, w, h):
    """Compute total constraint violation (boundary + overlap)."""
    n = len(circles)
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Boundary violations
    viol = 0.0
    viol += np.sum(np.maximum(0, r - cx))
    viol += np.sum(np.maximum(0, cx + r - w))
    viol += np.sum(np.maximum(0, r - cy))
    viol += np.sum(np.maximum(0, cy + r - h))
    
    # Overlap violations
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[i] + r[i+1:]
        viol += np.sum(np.maximum(0, min_dist - dist))
    
    return float(viol)

def _make_hexagonal_initial(w, h, rows=None):
    """Create a hexagonal-like arrangement for 21 circles."""
    if rows is None:
        rows = [5, 4, 5, 4, 3]
    n_rows = len(rows)
    circles = []
    r_est = np.sqrt(w * h / (21 * np.pi * 0.9))
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / n_rows
        for col_j in range(count):
            x = (col_j + 0.5) * w / count
            r = r_est
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _make_grid_initial(w, h, rows=None, cols=None):
    """Create a grid arrangement for 21 circles."""
    if rows is None or cols is None:
        best_r, best_c = 3, 7
        best_diff = abs(w/h - 7/3)
        for r in range(1, 22):
            for c in range(1, 22):
                if r * c >= 21:
                    diff = abs(w/h - c/r)
                    if diff < best_diff:
                        best_diff = diff
                        best_r, best_c = r, c
        rows, cols = best_r, best_c
        
    circles = []
    r_est = min(w / cols, h / rows) / 2
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= N:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            circles.append([x, y, r_est])
            
    return np.array(circles[:N])

def _make_random_initial(w, h, seed=None):
    """Create a random arrangement for 21 circles."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    circles = []
    # Start with small circles and grow
    r_base = min(w, h) / 8.0
    for _ in range(N):
        r = r_base * rng.uniform(0.5, 1.5)
        x = rng.uniform(r, w - r)
        y = rng.uniform(r, h - r)
        circles.append([x, y, r])
    
    return np.array(circles[:N])

def _relax_circles_safe(circles, w, h, iterations=500, tol=1e-9):
    """Relaxation to ensure validity."""
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for it in range(iterations):
        moved = False
        
        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] < rad[i]:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] > w - rad[i]:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] < rad[i]:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] > h - rad[i]:
                pos[i, 1] = h - rad[i]
                moved = True
        
        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True
        
        if not moved:
            break
    
    new_circles = np.column_stack((pos, rad))
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    return circles

def _local_optimize(circles, w, h, deadline, max_iter=100):
    """Local optimization using L-BFGS-B with penalty method."""
    if time.time() > deadline - 5.0:
        return circles
    
    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))
        
    def objective(x):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:]
        
        obj = -np.sum(r)
        
        def smooth_max(v):
            eps = 1e-4
            return 0.5 * (v + np.sqrt(v*v + eps*eps))
        
        v_left = r - cx
        v_right = cx + r - w
        v_bottom = r - cy
        v_top = cy + r - h
        
        obj += 1000.0 * (np.sum(smooth_max(v_left)**2) + 
                        np.sum(smooth_max(v_right)**2) + 
                        np.sum(smooth_max(v_bottom)**2) + 
                        np.sum(smooth_max(v_top)**2))
        
        dx = cx[:, np.newaxis] - cx[np.newaxis, :]
        dy = cy[:, np.newaxis] - cy[np.newaxis, :]
        dist = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dist, 1e12)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        v_overlap = r_sum - dist
        
        iu, ju = np.triu_indices(n, k=1)
        v_vec = v_overlap[iu, ju]
        obj += 1000.0 * np.sum(smooth_max(v_vec)**2)
        
        return obj
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
        new_circles = _relax_circles_safe(new_circles, w, h, iterations=100)
        
        if _validate(new_circles, w, h, tol=1e-5):
            return new_circles
    except Exception:
        pass
    
    return circles

def _nsga2_select(pop, fitness1, fitness2, pop_size):
    """
    NSGA-II selection: non-dominated sorting + crowding distance.
    fitness1: sum of radii (maximize)
    fitness2: constraint violation (minimize)
    """
    n = len(pop)
    
    # Non-dominated sorting
    fronts = []
    remaining = list(range(n))
    
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # j dominates i if j is better or equal in both, and strictly better in at least one
                if (fitness1[j] >= fitness1[i] and fitness2[j] <= fitness2[i] and 
                    (fitness1[j] > fitness1[i] or fitness2[j] < fitness2[i])):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        
        fronts.append(front)
        for i in front:
            remaining.remove(i)
    
    # Crowding distance for each front
    crowding = np.zeros(n)
    for front in fronts:
        if len(front) <= 2:
            for i in front:
                crowding[i] = np.inf
            continue
        
        for obj_idx in range(2):
            if obj_idx == 0:
                f = fitness1[front]
            else:
                f = fitness2[front]
            
            order = np.argsort(f)
            sorted_front = [front[i] for i in order]
            f_min = f[order[0]]
            f_max = f[order[-1]]
            
            if f_max == f_min:
                for i in sorted_front:
                    crowding[i] += 1.0
                continue
            
            crowding[sorted_front[0]] = np.inf
            crowding[sorted_front[-1]] = np.inf
            
            for k in range(1, len(sorted_front) - 1):
                i = sorted_front[k]
                prev_i = sorted_front[k-1]
                next_i = sorted_front[k+1]
                if obj_idx == 0:
                    denom = fitness1[next_i] - fitness1[prev_i]
                else:
                    denom = fitness2[next_i] - fitness2[prev_i]
                if abs(denom) > 1e-15:
                    crowding[i] += 1.0 / abs(denom)
    
    # Select next generation
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            # Sort by crowding distance (descending) and take remaining
            front_sorted = sorted(front, key=lambda i: crowding[i], reverse=True)
            needed = pop_size - len(selected)
            selected.extend(front_sorted[:needed])
            break
    
    return selected

def _evolve_population(pop, fitness1, fitness2, w_vals, h_vals, pop_size, deadline, n_gen=5):
    """
    Evolve population using NSGA-II style genetic operations.
    """
    best_circles = None
    best_val = -np.inf
    best_w = 1.0
    best_h = 1.0
    
    for gen in range(n_gen):
        if time.time() > deadline - 10.0:
            break
        
        # Update best
        for i in range(len(pop)):
            if fitness1[i] > best_val:
                best_val = fitness1[i]
                best_circles = pop[i]
                best_w = w_vals[i]
                best_h = h_vals[i]
        
        # Selection
        selected_idx = _nsga2_select(pop, fitness1, fitness2, pop_size)
        
        # Create new population
        new_pop = []
        new_w = []
        new_h = []
        
        # Elitism: keep best individuals
        for i in selected_idx[:pop_size//2]:
            new_pop.append(pop[i].copy())
            new_w.append(w_vals[i])
            new_h.append(h_vals[i])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size and time.time() < deadline - 5.0:
            # Crossover
            p1_idx = selected_idx[np.random.randint(0, len(selected_idx))]
            p2_idx = selected_idx[np.random.randint(0, len(selected_idx))]
            
            if np.random.rand() < 0.7:  # Crossover probability
                # Blend crossover
                alpha = np.random.rand()
                child_pos = alpha * pop[p1_idx][:, :2] + (1-alpha) * pop[p2_idx][:, :2]
                child_rad = np.maximum(pop[p1_idx][:, 2], pop[p2_idx][:, 2]) * np.random.uniform(0.95, 1.05, N)
                child = np.column_stack((child_pos, child_rad))
                
                # Aspect ratio crossover
                w = alpha * w_vals[p1_idx] + (1-alpha) * w_vals[p2_idx]
                w = np.clip(w, 0.3, 1.7)
                h = 2.0 - w
            else:
                # Just copy parent
                child = pop[p1_idx].copy()
                w = w_vals[p1_idx]
                h = h_vals[p1_idx]
            
            # Mutation
            if np.random.rand() < 0.5:  # Mutation probability
                # Position mutation
                child[:, 0] += np.random.normal(0, 0.02 * w, N)
                child[:, 1] += np.random.normal(0, 0.02 * h, N)
                # Radius mutation
                child[:, 2] *= np.random.uniform(0.9, 1.1, N)
                # Aspect ratio mutation
                if np.random.rand() < 0.3:
                    w += np.random.normal(0, 0.05)
                    w = np.clip(w, 0.3, 1.7)
                    h = 2.0 - w
            
            # Relax
            child = _relax_circles_safe(child, w, h, iterations=100)
            
            if _validate(child, w, h, tol=1e-5):
                val = _total_radius(child)
                viol = _constraint_violation(child, w, h)
            else:
                val = 0.0
                viol = 100.0
            
            new_pop.append(child)
            new_w.append(w)
            new_h.append(h)
        
        # Local optimization on a few best individuals
        if time.time() < deadline - 15.0:
            # Sort by fitness and optimize top few
            sorted_idx = np.argsort(-fitness1)
            for idx in sorted_idx[:3]:
                if time.time() > deadline - 10.0:
                    break
                w = w_vals[idx]
                h = h_vals[idx]
                opt_circles = _local_optimize(pop[idx], w, h, deadline, max_iter=50)
                if _validate(opt_circles, w, h, tol=1e-5):
                    val = _total_radius(opt_circles)
                    viol = _constraint_violation(opt_circles, w, h)
                    new_pop.append(opt_circles)
                    new_w.append(w)
                    new_h.append(h)
        
        # Update population
        pop = new_pop[:pop_size]
        w_vals = new_w[:pop_size]
        h_vals = new_h[:pop_size]
        
        # Evaluate fitness
        fitness1 = np.zeros(len(pop))
        fitness2 = np.zeros(len(pop))
        for i in range(len(pop)):
            if _validate(pop[i], w_vals[i], h_vals[i], tol=1e-5):
                fitness1[i] = _total_radius(pop[i])
                fitness2[i] = _constraint_violation(pop[i], w_vals[i], h_vals[i])
            else:
                fitness1[i] = -100.0
                fitness2[i] = 100.0
        
        # Update best
        for i in range(len(pop)):
            if fitness1[i] > best_val:
                best_val = fitness1[i]
                best_circles = pop[i]
                best_w = w_vals[i]
                best_h = h_vals[i]
    
    return best_circles, best_val, best_w, best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Population-based evolutionary algorithm
    pop_size = 30
    
    # Initialize population with diverse configurations
    pop = []
    w_vals = []
    h_vals = []
    
    # Different aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for w in aspect_ratios:
        h = 2.0 - w
        if h <= 0.1:
            continue
        
        # Hexagonal init
        circles = _make_hexagonal_initial(w, h)
        circles = _relax_circles_safe(circles, w, h, iterations=300)
        if _validate(circles, w, h, tol=1e-5):
            pop.append(circles)
            w_vals.append(w)
            h_vals.append(h)
        
        # Grid init
        circles = _make_grid_initial(w, h)
        circles = _relax_circles_safe(circles, w, h, iterations=300)
        if _validate(circles, w, h, tol=1e-5):
            pop.append(circles)
            w_vals.append(w)
            h_vals.append(h)
        
        # Random init
        circles = _make_random_initial(w, h, seed=np.random.randint(10000))
        circles = _relax_circles_safe(circles, w, h, iterations=300)
        if _validate(circles, w, h, tol=1e-5):
            pop.append(circles)
            w_vals.append(w)
            h_vals.append(h)
    
    # Ensure minimum population size
    while len(pop) < pop_size and time.time() < deadline - 20.0:
        w = np.random.uniform(0.4, 1.6)
        h = 2.0 - w
        circles = _make_random_initial(w, h, seed=np.random.randint(10000))
        circles = _relax_circles_safe(circles, w, h, iterations=200)
        if _validate(circles, w, h, tol=1e-5):
            pop.append(circles)
            w_vals.append(w)
            h_vals.append(h)
    
    if len(pop) == 0:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    # Initial fitness evaluation
    fitness1 = np.zeros(len(pop))
    fitness2 = np.zeros(len(pop))
    for i in range(len(pop)):
        if _validate(pop[i], w_vals[i], h_vals[i], tol=1e-5):
            fitness1[i] = _total_radius(pop[i])
            fitness2[i] = _constraint_violation(pop[i], w_vals[i], h_vals[i])
        else:
            fitness1[i] = -100.0
            fitness2[i] = 100.0
    
    # Run evolution
    best_circles, best_val, best_w, best_h = _evolve_population(
        pop, fitness1, fitness2, w_vals, h_vals, pop_size, deadline, n_gen=8
    )
    
    # Final local optimization on best
    if best_circles is not None and time.time() < deadline - 10.0:
        opt_circles = _local_optimize(best_circles, best_w, best_h, deadline, max_iter=100)
        if _validate(opt_circles, best_w, best_h, tol=1e-5):
            val = _total_radius(opt_circles)
            if val > best_val:
                best_circles = opt_circles
                best_val = val
    
    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_circles

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Best Value: {_total_radius(res)}")
