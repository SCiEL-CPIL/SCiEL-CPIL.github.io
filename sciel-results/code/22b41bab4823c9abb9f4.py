"""Circle packing in perimeter-4 rectangle.
Approach: Simulated Annealing.
- Formulates the problem as minimizing an energy function that penalizes
  overlaps and wall violations while rewarding the sum of radii.
- Uses stochastic search with a cooling schedule to escape local minima.
- Runs multiple restarts with different aspect ratios and initial temperatures.
"""
import numpy as np
import time

N = 21
DEADLINE_PAD = 10.0


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_PAD

    best = None
    best_val = 0.0

    # Aspect ratios: w + h = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45,
                     1.1, 0.9, 1.3, 0.75, 1.5, 0.65, 1.2, 0.85]

    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Run SA for this aspect ratio
        x, y, r = _simulated_annealing(w0, h0, deadline)

        val = _sum_if_valid(x, y, r, w0, h0)
        if val > best_val:
            best_val = val
            best = (x.copy(), y.copy(), r.copy(), w0, h0)

    if best is None:
        # Fallback: trivial valid grid in a unit square.
        w0, h0 = 1.0, 1.0
        x, y = _grid_init(w0, h0)
        r = np.full(N, 0.05)
        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)
        return np.column_stack((x, y, r))

    x, y, r, w0, h0 = best
    # Final safety shrink to guarantee strict validity.
    for _ in range(20):
        if _is_valid(x, y, r * (1 - 1e-7), w0, h0):
            r = r * (1 - 1e-7)
        else:
            break

    return np.column_stack((x, y, r))


def _grid_init(w, h):
    side = int(np.ceil(np.sqrt(N)))
    x = np.empty(N)
    y = np.empty(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w
        y[i] = (row + 0.5) / side * h
    return x, y


def _simulated_annealing(w, h, deadline):
    """Run Simulated Annealing to find a good packing."""
    # Initial state: Grid-like start helps convergence
    x, y = _grid_init(w, h)
    # Small initial radii to avoid huge initial overlaps
    r = np.full(N, 0.05)
    
    # Ensure initial state is valid (shrink if necessary)
    for _ in range(10):
        if _is_valid(x, y, r, w, h):
            break
        r *= 0.9

    current_x = x.copy()
    current_y = y.copy()
    current_r = r.copy()
    current_energy = _energy(current_x, current_y, current_r, w, h)

    best_x = current_x.copy()
    best_y = current_y.copy()
    best_r = current_r.copy()
    best_energy = current_energy

    T = 0.5  # Initial temperature
    T_min = 1e-6
    total_iters = 50000
    
    # Estimate time per iteration to adjust total_iters
    start = time.time()
    for i in range(1000):
        _energy(np.random.rand(N)*w, np.random.rand(N)*h, np.random.rand(N)*0.1, w, h)
    time_per_iter = (time.time() - start) / 1000.0
    
    # Adjust total iterations to fit within deadline
    remaining_time = deadline - time.time()
    if time_per_iter > 0:
        total_iters = int(min(total_iters, remaining_time / time_per_iter * 0.8))
    else:
        total_iters = 50000
        
    if total_iters < 1000:
        return best_x, best_y, best_r

    for i in range(total_iters):
        if time.time() > deadline - 2:
            break

        # Cool down
        T = T_min + (T - T_min) * (T_min / T) ** (1.0 / total_iters)

        # Generate move
        idx = np.random.randint(N)
        new_x = current_x.copy()
        new_y = current_y.copy()
        new_r = current_r.copy()

        # Step size scales with temperature
        step_size = 0.05 * (1.0 + T)
        
        # Move position
        new_x[idx] += np.random.normal(0, step_size)
        new_y[idx] += np.random.normal(0, step_size)
        
        # Move radius (only grow or shrink slightly)
        if np.random.random() < 0.5:
            new_r[idx] += np.random.normal(0, step_size * 0.5)
            new_r[idx] = max(1e-6, new_r[idx])

        # Clip to walls roughly to keep energy lower
        new_x[idx] = np.clip(new_x[idx], 0.0, w)
        new_y[idx] = np.clip(new_y[idx], 0.0, h)

        new_energy = _energy(new_x, new_y, new_r, w, h)
        
        # Metropolis acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            current_x = new_x
            current_y = new_y
            current_r = new_r
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
                best_y = new_y.copy()
                best_r = new_r.copy()

    # Final local polish: try to grow radii
    for _ in range(50):
        if time.time() > deadline - 1:
            break
        improved = False
        for i in range(N):
            # Try to increase radius i
            orig_r = best_r[i]
            best_r[i] = orig_r * 1.05
            if _is_valid(best_x, best_y, best_r, w, h):
                improved = True
            else:
                best_r[i] = orig_r
        
        # Try to move circles slightly to allow growth
        if not improved:
            for i in range(N):
                dx = np.random.normal(0, 0.01)
                dy = np.random.normal(0, 0.01)
                ox, oy = best_x[i], best_y[i]
                best_x[i] += dx
                best_y[i] += dy
                best_x[i] = np.clip(best_x[i], 0, w)
                best_y[i] = np.clip(best_y[i], 0, h)
                if not _is_valid(best_x, best_y, best_r, w, h):
                    best_x[i], best_y[i] = ox, oy
                    break

    return best_x, best_y, best_r


def _energy(x, y, r, w, h):
    """Calculate energy: penalty for overlaps/walls - reward for radii."""
    e_wall = 0.0
    for i in range(N):
        e_wall += max(0.0, r[i] - x[i])
        e_wall += max(0.0, r[i] - (w - x[i]))
        e_wall += max(0.0, r[i] - y[i])
        e_wall += max(0.0, r[i] - (h - y[i]))
    
    e_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            overlap = r[i] + r[j] - d
            if overlap > 0:
                e_overlap += overlap * overlap
    
    reward = np.sum(r)
    
    # High weight on violations to prioritize valid configurations
    return 100.0 * (e_wall + e_overlap) - reward


def _is_valid(x, y, r, w, h, tol=1e-8):
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > w + tol) or np.any(y + r > h + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_if_valid(x, y, r, w, h):
    if _is_valid(x, y, r, w, h):
        return float(np.sum(r))
    return 0.0
