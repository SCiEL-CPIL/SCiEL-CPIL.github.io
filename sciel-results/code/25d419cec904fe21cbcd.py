"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _validate_and_score(x, y, r, w, h):
    """Check validity and compute score (sum of radii)."""
    eps = 1e-9
    if w <= 0 or h <= 0:
        return -1
    if 2 * (w + h) > 4.0 + eps:
        return -1
    for i in range(N):
        if x[i] < -eps or x[i] > w + eps:
            return -1
        if y[i] < -eps or y[i] > h + eps:
            return -1
        if r[i] < -eps:
            return -1
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - eps:
                return -1
    # Check boundary
    for i in range(N):
        if x[i] < r[i] - eps or x[i] > w - r[i] + eps:
            return -1
        if y[i] < r[i] - eps or y[i] > h - r[i] + eps:
            return -1
    return np.sum(r)


def _make_valid(x, y, r, w, h):
    """Adjust positions/radii to make configuration valid."""
    eps = 1e-9
    # Ensure perimeter constraint
    if 2 * (w + h) > 4.0:
        scale = 2.0 / (w + h)
        x *= scale
        y *= scale
        r *= scale
        w *= scale
        h *= scale
    # Clamp positions
    for i in range(N):
        x[i] = max(r[i], min(x[i], w - r[i]))
        y[i] = max(r[i], min(y[i], h - r[i]))
    # Fix overlaps by reducing radii
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_r = d / 2.0
                if r[i] + r[j] > d + eps:
                    # Reduce both
                    excess = r[i] + r[j] - d
                    r[i] -= excess / 2.0
                    r[j] -= excess / 2.0
                    changed = True
        # Ensure boundary
        for i in range(N):
            max_r = min(x[i], w - x[i], y[i], h - y[i])
            if r[i] > max_r:
                r[i] = max_r
                changed = True
        if not changed:
            break
    # Final clamp
    for i in range(N):
        x[i] = max(r[i], min(x[i], w - r[i]))
        y[i] = max(r[i], min(y[i], h - y[i]))
        r[i] = max(0, min(r[i], x[i], w - x[i], y[i], h - y[i]))
    return x, y, r, w, h


def _local_optimize(x, y, r, w, h):
    """Try to improve the configuration via gradient-free local search."""
    best_score = np.sum(r)
    for _ in range(50):
        # Try small perturbations
        improved = False
        for i in range(N):
            for dx, dy in [(-0.01, 0), (0.01, 0), (0, -0.01), (0, 0.01),
                           (-0.005, 0.005), (0.005, 0.005), (-0.005, -0.005), (0.005, -0.005)]:
                x_new = x.copy()
                y_new = y.copy()
                x_new[i] += dx
                y_new[i] += dy
                x_new[i] = max(r[i], min(x_new[i], w - r[i]))
                y_new[i] = max(r[i], min(y_new[i], h - r[i]))
                # Check if valid
                valid = True
                for j in range(N):
                    if j == i:
                        continue
                    d = np.sqrt((x_new[i] - x_new[j])**2 + (y_new[i] - y_new[j])**2)
                    if d < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if valid:
                    # Check if we can increase radius
                    new_r = min(x_new[i], w - x_new[i], y_new[i], h - y_new[i])
                    for j in range(N):
                        if j == i:
                            continue
                        d = np.sqrt((x_new[i] - x_new[j])**2 + (y_new[i] - y_new[j])**2)
                        new_r = min(new_r, d - r[j])
                    if new_r > r[i] + 1e-10:
                        x[i] = x_new[i]
                        y[i] = y_new[i]
                        r[i] = new_r
                        best_score = np.sum(r)
                        improved = True
                        break
        if not improved:
            break
    return x, y, r, w, h, best_score


def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # Safety margin before 150s kill

    best_x = np.zeros(N)
    best_y = np.zeros(N)
    best_r = np.zeros(N)
    best_w = 1.0
    best_h = 1.0
    best_score = 0.0

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 0.5, 1.25, 0.8, 1.8, 0.6]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        # For perimeter 4: 2(w+h) = 4 => w+h = 2
        # w/h = aspect => h = 2/(1+aspect), w = 2*aspect/(1+aspect)
        h = 2.0 / (1.0 + aspect)
        w = 2.0 * aspect / (1.0 + aspect)
        
        # Grid initialization
        cols = max(1, int(np.round(np.sqrt(N * aspect))))
        rows = max(1, int(np.ceil(N / cols)))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for idx in range(N):
            ci = idx % cols
            ri = idx // cols
            if ri >= rows:
                ri = rows - 1
            x[idx] = (ci + 0.5) * w / cols
            y[idx] = (ri + 0.5) * h / rows
        
        # Initial radii: min distance to boundary or neighbor
        for i in range(N):
            max_r = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if j != i:
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    max_r = min(max_r, d / 2.0)
            r[i] = max(0, max_r)
        
        # Local optimization
        x, y, r, w, h, score = _local_optimize(x, y, r, w, h)
        
        if score > best_score:
            best_score = score
            best_x = x.copy()
            best_y = y.copy()
            best_r = r.copy()
            best_w = w
            best_h = h

    # Also try a few random starts
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        
        aspect = np.random.uniform(0.5, 2.0)
        h = 2.0 / (1.0 + aspect)
        w = 2.0 * aspect / (1.0 + aspect)
        
        x = np.random.uniform(0, w, N)
        y = np.random.uniform(0, h, N)
        
        # Set small initial radii
        for i in range(N):
            max_r = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if j != i:
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    max_r = min(max_r, d / 2.0)
            r[i] = max(0, max_r)
        
        x, y, r, w, h, score = _local_optimize(x, y, r, w, h)
        
        if score > best_score:
            best_score = score
            best_x = x.copy()
            best_y = y.copy()
            best_r = r.copy()
            best_w = w
            best_h = h

    # Final validation and return
    best_x, best_y, best_r, best_w, best_h = _make_valid(best_x, best_y, best_r, best_w, best_h)
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_x[i], best_y[i], best_r[i]]
    
    return result
# EVOLVE-BLOCK-END
