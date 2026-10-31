"""Circle packing in perimeter-4 rectangle: combined topological + aspect-ratio approach."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # --- Phase 1: Try multiple aspect ratios with grid-based SLSQP optimization ---
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]

    for W in aspect_ratios:
        if time.time() > deadline - 5:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue

        result = _optimize_fixed_aspect(W, H, deadline - (time.time() - start_time))
        if result is not None:
            value = np.sum(result[:, 2])
            if value > best_value:
                best_value = value
                best_result = result

    # --- Phase 2: Topological contact-graph optimization (variable rectangle) ---
    for attempt in range(15):
        if time.time() > deadline - 5:
            break

        rng = np.random.default_rng(attempt * 7919 + 31337)
        contact_graph = _generate_contact_graph(N, rng)
        centers, radii = _initialize_packing(N, contact_graph, rng)

        result = _topological_optimize(centers, radii, contact_graph,
                                       deadline - (time.time() - start_time))
        if result is not None:
            value = np.sum(result[:, 2])
            if value > best_value:
                best_value = value
                best_result = result

    # --- Phase 3: Final local polish with force-directed + radius growth ---
    if best_result is not None and time.time() < deadline - 10:
        polished = _force_polish(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            val = np.sum(polished[:, 2])
            if val > best_value:
                best_value = val
                best_result = polished

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result


def _optimize_fixed_aspect(W, H, time_budget):
    """Optimize circles in a fixed rectangle W x H (W+H=2) using SLSQP."""
    # Try a few grid layouts
    for cols in [4, 5, 6, 7]:
        if time.time() > time_budget + 0:  # check against absolute deadline via budget
            break
        rows = int(np.ceil(N / cols))
        if rows * cols < N:
            continue

        W_cell = W / cols
        H_cell = H / rows

        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= N:
                    break
                cx[idx] = (col + 0.5) * W_cell
                cy[idx] = (row + 0.5) * H_cell
                r[idx] = min(W_cell, H_cell) * 0.45
                idx += 1

        # Force-directed relaxation to fix initial overlaps
        cx, cy = _force_relax(cx, cy, r, W, H, max_iter=300)

        # SLSQP optimization
        x0 = np.concatenate([cx, cy, r])

        def objective(x):
            cx_l = x[:N]
            cy_l = x[N:2*N]
            r_l = x[2*N:3*N]

            pen = 0.0
            # Wall constraints
            for i in range(N):
                if cx_l[i] - r_l[i] < 0:
                    pen += (cx_l[i] - r_l[i]) ** 2 * 50
                if cx_l[i] + r_l[i] > W:
                    pen += (cx_l[i] + r_l[i] - W) ** 2 * 50
                if cy_l[i] - r_l[i] < 0:
                    pen += (cy_l[i] - r_l[i]) ** 2 * 50
                if cy_l[i] + r_l[i] > H:
                    pen += (cy_l[i] + r_l[i] - H) ** 2 * 50
            # Pairwise non-overlap
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx_l[i] - cx_l[j]
                    dy = cy_l[i] - cy_l[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r_l[i] + r_l[j]
                    if d < min_d:
                        pen += (min_d - d) ** 2 * 20
            return -np.sum(r_l) + pen

        bounds = []
        for i in range(N):
            bounds.append((r[i], W - r[i]))
        for i in range(N):
            bounds.append((r[i], H - r[i]))
        for i in range(N):
            bounds.append((1e-4, min(W, H) / 2))

        maxiter = int(min(400, time_budget * 40))
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-9, 'maxls': 50})
            if res.success or res.fun < objective(x0):
                cx_opt = res.x[:N]
                cy_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                if _is_valid(cx_opt, cy_opt, r_opt, W, H):
                    return np.column_stack([cx_opt, cy_opt, r_opt])
        except Exception:
            pass

    return None


def _force_relax(cx, cy, r, W, H, max_iter=300):
    """Simple force-directed relaxation to resolve overlaps."""
    n = len(cx)
    cx = cx.copy().astype(float)
    cy = cy.copy().astype(float)
    dt = 0.005

    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 20
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 20
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 20
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 20
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10
        cx += dt * fx
        cy += dt * fy
        # Clamp
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)

    return cx, cy


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _generate_contact_graph(n, rng):
    """Generate a random contact graph for dense packing."""
    side = int(np.ceil(np.sqrt(n)))
    graph = np.zeros((n, n), dtype=bool)

    for i in range(n):
        row, col = divmod(i, side)
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < side and 0 <= nc < side:
                j = nr * side + nc
                if j < n:
                    graph[i, j] = True
                    graph[j, i] = True

    # Add random edges
    for _ in range(n // 2):
        i, j = rng.choice(n, 2, replace=False)
        graph[i, j] = True
        graph[j, i] = True

    return graph


def _initialize_packing(n, graph, rng):
    """Initialize circle positions and radii based on contact graph."""
    side = int(np.ceil(np.sqrt(n)))
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.08)

    for i in range(n):
        row, col = divmod(i, side)
        centers[i, 0] = (col + 0.5) / side + rng.uniform(-0.03, 0.03)
        centers[i, 1] = (row + 0.5) / side + rng.uniform(-0.03, 0.03)

    return centers, radii


def _topological_optimize(centers, radii, graph, time_budget):
    """Optimize with variable rectangle dimensions using L-BFGS-B."""
    n = len(centers)
    w_init = 1.0
    h_init = 1.0
    params_init = np.concatenate(([w_init, h_init], centers.flatten(), radii))

    def objective(params):
        w = params[0]
        h = params[1]
        if w <= 0.05 or h <= 0.05:
            return 1e6
        if w + h > 2.0 + 1e-9:
            return 1e6 + (w + h - 2.0) * 500

        centers_opt = params[2:2 + 2 * n].reshape(n, 2)
        radii_opt = params[2 + 2 * n:2 + 3 * n]

        penalty = 0.0
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            if x - r < 0:
                penalty += (x - r) ** 2 * 100
            if x + r > w:
                penalty += (x + r - w) ** 2 * 100
            if y - r < 0:
                penalty += (y - r) ** 2 * 100
            if y + r > h:
                penalty += (y + r - h) ** 2 * 100

        for i in range(n):
            for j in range(i + 1, n):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii_opt[i] + radii_opt[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 30

        # Contact graph encouragement
        for i in range(n):
            for j in range(i + 1, n):
                if graph[i, j]:
                    dx = centers_opt[i, 0] - centers_opt[j, 0]
                    dy = centers_opt[i, 1] - centers_opt[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    target_dist = radii_opt[i] + radii_opt[j]
                    penalty += (dist - target_dist) ** 2 * 0.05

        return -np.sum(radii_opt) + penalty

    bounds = [(0.1, 1.9), (0.1, 1.9)]
    for i in range(n):
        bounds.append((0, 2))
        bounds.append((0, 2))
        bounds.append((0.005, 0.5))

    maxiter = int(min(400, time_budget * 40))
    try:
        res = minimize(objective, params_init, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-9})

        if res.fun < 1e4:
            w = res.x[0]
            h = res.x[1]
            if w + h <= 2.0 + 1e-6 and w > 0.05 and h > 0.05:
                centers_opt = res.x[2:2 + 2 * n].reshape(n, 2)
                radii_opt = res.x[2 + 2 * n:2 + 3 * n]

                if _is_valid(centers_opt[:, 0], centers_opt[:, 1], radii_opt, w, h):
                    scale = min(1.0, 2.0 / (w + h))
                    centers_final = centers_opt * scale
                    radii_final = radii_opt * scale
                    return np.column_stack([centers_final, radii_final])
    except Exception:
        pass

    return None


def _force_polish(result, time_budget):
    """Polish a valid packing with force-directed radius growth."""
    cx = result[:, 0].copy()
    cy = result[:, 1].copy()
    r = result[:, 2].copy()

    # Determine bounding rectangle
    W = np.max(cx + r)
    H = np.max(cy + r)
    if W + H > 2.0:
        scale = 2.0 / (W + H)
        cx *= scale
        cy *= scale
        r *= scale
        W *= scale
        H *= scale

    n = len(cx)
    dt = 0.002
    max_iter = int(min(1000, time_budget * 80))

    for it in range(max_iter):
        if time.time() > time_budget + 0:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 30
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 30
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 30
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 30

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 15
                    fy[i] += (dy / d) * overlap * 15
                    fx[j] -= (dx / d) * overlap * 15
                    fy[j] -= (dy / d) * overlap * 15

        cx += dt * fx
        cy += dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)

    # Try to grow radii uniformly
    grow_factor = 1.001
    for _ in range(50):
        if time.time() > time_budget + 0:
            break
        r_trial = r * grow_factor
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break

    # Final validation
    if _is_valid(cx, cy, r, W, H):
        return np.column_stack([cx, cy, r])
    return None
