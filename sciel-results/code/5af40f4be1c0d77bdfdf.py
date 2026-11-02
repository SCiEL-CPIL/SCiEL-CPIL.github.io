"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Contact-graph gradient ascent on radius slack with adaptive line search.
We define the objective as the sum of "maximal feasible radii" given current positions.
The gradient of this objective w.r.t. positions is computed from the contact graph
(Delaunay edges). We perform gradient ascent with backtracking line search to
maximize the sum of radii. The contact graph is rebuilt every iteration to ensure
we capture all active constraints. This avoids the overhead of solving a large
constrained NLP and allows the circles to naturally settle into a dense packing.
"""
import time
import numpy as np
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 145.0  # Safety margin for 150s limit


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                dmin = min(dmin, d - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _contact_graph(pos):
    """Return set of (i, j) pairs that are close enough to be in contact or near-contact.
    We use Delaunay triangulation to find candidate pairs, then filter by distance."""
    tri = Delaunay(pos)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                edges.add((min(a, b), max(a, b)))
    # Also add pairs that are very close but not in Delaunay (just in case)
    # But usually Delaunay is sufficient for "contact" in packing.
    # Let's also check all pairs to be safe for the gradient calculation?
    # No, that's O(N^2). Delaunay is O(N log N).
    # However, for gradient of max_radius, we only need pairs where the constraint is active or near-active.
    # Let's stick to Delaunay edges + boundary contacts.
    return list(edges)


def _radius_slack_gradient(pos, r):
    """Compute gradient of sum(max_radius_i) w.r.t. positions.
    max_radius_i = min( boundary_dist_i, min_j (dist_ij - r_j) )
    Let m_i = max_radius_i.
    If m_i is determined by boundary b:
        d(m_i)/dx_i = -sign(x_i - 0.5) for x-boundary, etc.
        Specifically:
        if x_i < r_i (left): d/dx_i = -1
        if 1 - x_i < r_i (right): d/dx_i = 1
        if y_i < r_i (bottom): d/dy_i = -1
        if 1 - y_i < r_i (top): d/dy_i = 1
    If m_i is determined by pair (i, j):
        m_i = dist_ij - r_j
        d(m_i)/dx_i = (x_i - x_j) / dist_ij
        d(m_i)/dx_j = -(x_i - x_j) / dist_ij
    """
    grad = np.zeros((N, 2))
    edges = _contact_graph(pos)
    
    # We need to know which constraint is active for each i.
    # Let's compute the active constraint for each i.
    for i in range(N):
        # Check boundaries
        active_type = None
        active_j = -1
        min_val = 1e9
        
        # Boundaries
        d_left = pos[i, 0]
        d_right = 1.0 - pos[i, 0]
        d_bottom = pos[i, 1]
        d_top = 1.0 - pos[i, 1]
        
        if d_left < min_val:
            min_val = d_left
            active_type = 'left'
        if d_right < min_val:
            min_val = d_right
            active_type = 'right'
        if d_bottom < min_val:
            min_val = d_bottom
            active_type = 'bottom'
        if d_top < min_val:
            min_val = d_top
            active_type = 'top'
            
        # Check neighbors (from contact graph)
        for (a, b) in edges:
            if a == i:
                j = b
            elif b == i:
                j = a
            else:
                continue
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            val = d - r[j]
            if val < min_val:
                min_val = val
                active_type = 'pair'
                active_j = j
        
        # Compute gradient contribution for circle i
        if active_type == 'left':
            grad[i, 0] += -1.0
        elif active_type == 'right':
            grad[i, 0] += 1.0
        elif active_type == 'bottom':
            grad[i, 1] += -1.0
        elif active_type == 'top':
            grad[i, 1] += 1.0
        elif active_type == 'pair' and active_j != -1:
            j = active_j
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.hypot(dx, dy)
            if d < EPS:
                continue
            ux = dx / d
            uy = dy / d
            # d(dist_ij)/dx_i = ux, d(dist_ij)/dx_j = -ux
            # m_i = dist_ij - r_j -> grad_i += u
            # m_j = dist_ij - r_i -> grad_j += -u (but we handle j in its own loop)
            grad[i, 0] += ux
            grad[i, 1] += uy
            # We also need to add the contribution to j's gradient?
            # The objective is sum_k m_k.
            # m_i depends on pos_j. d(m_i)/dx_j = -ux.
            # So grad[j] should also be updated.
            # Let's do it here to be efficient.
            grad[j, 0] += -ux
            grad[j, 1] += -uy

    return grad


def _polish(cands):
    """Final polish: fix positions, maximize radii exactly."""
    if not cands or len(cands) < N:
        return cands
    arr = np.array(cands[:N])
    pos = arr[:, :2].copy()
    rad = arr[:, 2].copy()
    
    # Iterate: fix pos -> max rad
    for it in range(50):
        if time.time() > DEADLINE - 5:
            break
        rad_new = _max_radius(pos, rad)
        if np.allclose(rad_new, rad, atol=1e-12):
            rad = rad_new
            break
        rad = rad_new
    out = np.column_stack([pos, rad])
    return out.tolist()


def _gradient_ascent(pos, r, max_iters=500):
    """Perform gradient ascent on sum(max_radius) w.r.t. positions."""
    step_size = 0.01
    best_score = -1e9
    best_pos = pos.copy()
    best_r = r.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
            
        r = _max_radius(pos, r)
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()
            
        grad = _radius_slack_gradient(pos, r)
        
        # Normalize gradient to avoid huge steps if gradient is large
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-10:
            break
        grad_norm = grad / g_norm
        
        # Backtracking line search
        alpha = step_size
        improved = False
        for k in range(20):
            new_pos = pos + alpha * grad_norm
            # Clip to valid range (roughly, to avoid going too far out)
            new_pos = np.clip(new_pos, 0.01, 0.99)
            new_r = _max_radius(new_pos, r)
            new_score = np.sum(new_r)
            
            if new_score > score + 1e-10:
                pos = new_pos
                r = new_r
                improved = True
                break
            alpha *= 0.5
            
        if not improved:
            # Reduce step size if no improvement
            step_size *= 0.5
            if step_size < 1e-6:
                break

    return best_pos, best_r


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(cands):
        nonlocal best, best_score
        # keep only valid (non-overlapping, in-square) circles
        valid = []
        for x, y, r in cands:
            if r <= 0:
                continue
            if x < r + EPS or x > 1 - r - EPS or y < r + EPS or y > 1 - r - EPS:
                continue
            ok = True
            for x2, y2, r2 in valid:
                if (x - x2) ** 2 + (y - y2) ** 2 < (r + r2) ** 2 - 1e-9:
                    ok = False
                    break
            if ok:
                valid.append((x, y, r))
        if len(valid) < N:
            return
        arr = np.array(valid[:N])
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr

    # ---- Phase A: Generate diverse initializations ----
    all_starts = []

    # 1. Grid initializations
    for side in range(4, 8):
        if time.time() > DEADLINE:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        r = np.full(N, 0.5 / side - 1e-3)
        for _ in range(10):
            r = _max_radius(pos, r)
        all_starts.append((pos, r))

    # 2. Hexagonal packing initialization
    try:
        pos_hex = []
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(pos_hex) >= N:
                    break
                x = (j + 0.5 * (i % 2)) / side
                y = (i + 0.5) / side
                pos_hex.append((x, y))
        pos_hex = np.array(pos_hex[:N])
        pos_hex = np.clip(pos_hex, 0.1, 0.9)
        r_hex = np.full(N, 0.05)
        for _ in range(20):
            r_hex = _max_radius(pos_hex, r_hex)
        all_starts.append((pos_hex, r_hex))
    except Exception:
        pass

    # 3. Random perturbations of grid
    for side in [5, 6, 7]:
        if time.time() > DEADLINE:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        r = np.full(N, 0.5 / side - 1e-3)
        for _ in range(10):
            r = _max_radius(pos, r)
        
        for k in range(5):
            if time.time() > DEADLINE:
                break
            p0 = pos + 0.05 * (rng.random((N, 2)) * 2 - 1)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = r.copy()
            r0 = _max_radius(p0, r0)
            all_starts.append((p0, r0))

    # 4. Pure random
    for k in range(5):
        if time.time() > DEADLINE:
            break
        p0 = rng.random((N, 2)) * 0.8 + 0.1
        r0 = np.full(N, 0.03)
        r0 = _max_radius(p0, r0)
        all_starts.append((p0, r0))

    # ---- Phase B: Gradient ascent on each start ----
    for pos0, r0 in all_starts:
        if time.time() > DEADLINE - 20:
            break
        
        try:
            pos_final, rad_final = _gradient_ascent(pos0, r0, max_iters=300)
            cands = list(zip(pos_final[:, 0], pos_final[:, 1], rad_final))
            polished = _polish(cands)
            consider(polished)
        except Exception:
            pass

    # Final polish on the best solution found
    if best is not None:
        cands = best.tolist()
        polished = _polish(cands)
        consider(polished)

    if best is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
