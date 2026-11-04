"""Circle packing in perimeter-4 rectangle: maximize sum of radii of 21 disjoint circles."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START

def _pack(w, h, x0, y0, r0, max_iter, tol, deadline):
    """Optimize circle positions/radii inside rectangle [0,w]x[0,h]."""
    n = len(x0)
    x = x0.copy()
    y = y0.copy()
    r = r0.copy()

    def neg_sum_radii(p):
        px, py, pr = p[:n], p[n:2*n], p[2*n:3*n]
        # boundary violations
        bpen = np.sum(np.maximum(0, px - (w - pr)))**2 + np.sum(np.maximum(0, py - (h - pr)))**2
        bpen += np.sum(np.maximum(0, pr - px))**2 + np.sum(np.maximum(0, pr - py))**2
        # overlap penalty (soft, smooth)
        dx = px[:, None] - px[None, :]
        dy = py[:, None] - py[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        req = pr[:, None] + pr[None, :]
        overlap = np.maximum(0, req - dist)
        # zero out diagonal
        np.fill_diagonal(overlap, 0)
        open = np.sum(overlap**2)
        # small penalty for negative radii
        rpen = np.sum(np.maximum(0, -pr))**2
        return -(np.sum(pr)) + 10.0 * open + 50.0 * (bpen + rpen)

    def neg_sum_radii_grad(p):
        px, py, pr = p[:n], p[n:2*n], p[2*n:3*n]
        gx = np.zeros(n); gy = np.zeros(n); gr = np.zeros(n)
        # objective gradient: d/dx = -1 for r
        gr[:] = -1.0
        # boundary gradient
        # x > w - r: d = 2*(px - (w-pr)) * d/dpx = 2*(px - w + pr)
        m = px > (w - pr)
        gx[m] += 50.0 * 2.0 * (px[m] - (w - pr[m]))
        gr[m] += 50.0 * 2.0 * (px[m] - (w - pr[m])) * (-1.0)
        m = py > (h - pr)
        gy[m] += 50.0 * 2.0 * (py[m] - (h - pr[m]))
        gr[m] += 50.0 * 2.0 * (py[m] - (h - pr[m])) * (-1.0)
        m = pr > px
        gx[m] += 50.0 * 2.0 * (pr[m] - px[m]) * (-1.0)
        gr[m] += 50.0 * 2.0 * (pr[m] - px[m])
        m = pr > py
        gy[m] += 50.0 * 2.0 * (pr[m] - py[m]) * (-1.0)
        gr[m] += 50.0 * 2.0 * (pr[m] - py[m])
        # negative radius penalty
        m = pr < 0
        gr[m] += 2.0 * (-pr[m]) * (-1.0)
        # overlap gradient
        dx = px[:, None] - px[None, :]
        dy = py[:, None] - py[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        req = pr[:, None] + pr[None, :]
        overlap = np.maximum(0, req - dist)
        np.fill_diagonal(overlap, 0)
        # d(overlap^2)/d(dist) = 2*overlap * (-1) ; d(dist)/d(px_i) = (px_i - px_j)/dist
        # contribution to gx_i from pair (i,j): 10 * 2*overlap_ij * (-(px_i-px_j)/dist_ij)
        coef = 10.0 * 2.0 * overlap / (dist + 1e-12)
        gx += np.sum(coef * (-dx), axis=1)
        gy += np.sum(coef * (-dy), axis=1)
        # d(overlap)/d(pr_i) = 1 for each pair i,j where overlap>0
        gr += 10.0 * 2.0 * np.sum(overlap, axis=1)
        return np.concatenate([gx, gy, gr])

    p = np.concatenate([x, y, r])
    # bounds
    lb = np.concatenate([np.zeros(n), np.zeros(n), np.zeros(n)])
    ub = np.concatenate([np.full(n, w), np.full(n, h), np.full(n, max(w, h)/2.0)])

    from scipy.optimize import minimize
    best_p = p.copy()
    best_val = neg_sum_radii(p)
    t0 = time.time()
    try:
        res = minimize(neg_sum_radii, p, jac=neg_sum_radii_grad, method='L-BFGS-B',
                       bounds=list(zip(lb, ub)),
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxfun': max_iter*3})
        if res.fun < best_val:
            best_val = res.fun
            best_p = res.x
    except Exception:
        pass
    if time.time() > deadline:
        return best_p
    return best_p


def _is_valid(p, w, h, eps=1e-6):
    n = len(p) // 3
    x, y, r = p[:n], p[n:2*n], p[2*n:3*n]
    if np.any(r < -eps):
        return False
    if np.any(x < -eps) or np.any(x > w + eps) or np.any(y < -eps) or np.any(y > h + eps):
        return False
    if np.any(r - x < -eps) or np.any(r - y < -eps) or np.any((w - x) - r < -eps) or np.any((h - y) - r < -eps):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    req = r[:, None] + r[None, :]
    np.fill_diagonal(req, 0)
    if np.any(req - dist > eps):
        return False
    return True


def _separate(p, w, h, iters=200):
    """Push overlapping circles apart and clamp to bounds."""
    n = len(p) // 3
    x, y, r = p[:n].copy(), p[n:2*n].copy(), p[2*n:3*n].copy()
    r = np.maximum(r, 0)
    for _ in range(iters):
        moved = False
        # clamp to bounds
        for i in range(n):
            if x[i] < r[i]: x[i] = r[i]
            if x[i] > w - r[i]: x[i] = w - r[i]
            if y[i] < r[i]: y[i] = r[i]
            if y[i] > h - r[i]: y[i] = h - r[i]
        for i in range(n):
            for j in range(i+1, n):
                dx = x[j] - x[i]; dy = y[j] - y[i]
                d = np.sqrt(dx*dx + dy*dy)
                req = r[i] + r[j]
                if d < req - 1e-9:
                    if d < 1e-9:
                        ang = np.random.uniform(0, 2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx/d, dy/d
                    push = (req - d) / 2.0
                    x[i] -= ux * push; y[i] -= uy * push
                    x[j] += ux * push; y[j] += uy * push
                    moved = True
        if not moved:
            break
    return np.concatenate([x, y, r])


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    t_start = time.time()

    # aspect ratios to try (w/h). perimeter 4 => w+h=2 => h=2/(1+ar), w=2*ar/(1+ar)
    aspect_ratios = [1.0, 1.2, 0.83, 1.4, 0.71, 1.6, 0.625, 2.0, 0.5, 1.8, 2.2, 0.75]

    def grid_init(w, h, n):
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = 0.5 * min(w, h) / side * 0.8
                cells.append((cx, cy, r))
        return np.array(cells)

    best_p = None
    best_score = -np.inf
    best_wh = (1.0, 1.0)

    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        # multiple initializations
        inits = []
        # grid
        inits.append(grid_init(w, h, N))
        # random
        for _ in range(2):
            rr = np.random.uniform(0.05, 0.2, N)
            xx = np.random.uniform(rr, w - rr, N)
            yy = np.random.uniform(rr, h - rr, N)
            inits.append(np.column_stack([xx, yy, rr]))

        for init in inits:
            if time.time() > deadline:
                break
            p = init.copy()
            # separate first
            p = _separate(p, w, h, iters=100)
            # optimize
            p = _pack(w, h, p[0::3], p[1::3], p[2::3], max_iter=300, tol=1e-8, deadline=deadline)
            # separate again to ensure validity
            p = _separate(p, w, h, iters=300)
            # polish: one more optimization pass
            if time.time() < deadline - 5.0:
                p = _pack(w, h, p[0::3], p[1::3], p[2::3], max_iter=200, tol=1e-10, deadline=deadline)
                p = _separate(p, w, h, iters=300)

            if _is_valid(p, w, h, eps=1e-5):
                score = np.sum(p[2::3])
                if score > best_score:
                    best_score = score
                    best_p = p.copy()
                    best_wh = (w, h)

    if best_p is None:
        # fallback: simple grid
        w = h = 1.0
        p = grid_init(w, h, N)
        p = _separate(p, w, h, iters=500)
        return np.column_stack([p[0::3], p[1::3], p[2::3]])

    return np.column_stack([best_p[0::3], best_p[1::3], best_p[2::3]])

# EVOLVE-BLOCK-END
