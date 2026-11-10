import numpy as np
import time
from scipy.optimize import minimize


def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580

    sqrt3 = np.sqrt(3)
    base_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals0 = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)

    def is_separated(c1, c2):
        v = c1 - c2
        return np.any(np.abs(normals0 @ v) > sqrt3 + 1e-9)

    def bounding_hex_side(points, center):
        diffs = points - center
        proj = np.abs(diffs @ normals0.T)
        return 2.0 * proj.max() / sqrt3

    def optimal_center(points):
        c = np.mean(points, axis=0)
        for _ in range(300):
            diffs = points - c
            proj = np.abs(diffs @ normals0.T)
            idx = np.unravel_index(proj.argmax(), proj.shape)
            p, n = points[idx[0]], normals0[idx[1]]
            s = np.sign((p - c) @ n)
            if s == 0:
                s = 1
            c += 0.002 * s * n
        return c

    # Lattice-based greedy initialization
    step = 0.05
    ext = 4.5
    xs = np.arange(-ext, ext + step / 2, step)
    ys = np.arange(-ext, ext + step / 2, step)
    candidates = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    n_cand = len(candidates)

    cs = 3.5
    cells = {}
    for i, c in enumerate(candidates):
        k = (int(c[0] / cs), int(c[1] / cs))
        cells.setdefault(k, []).append(i)

    # Greedy: start from origin, add nearest non-overlapping
    oi = np.argmin(np.sum(candidates ** 2, axis=1))
    sel = [oi]
    sset = {oi}
    for _ in range(10):
        best, bd = None, 1e18
        for i in range(n_cand):
            if i in sset:
                continue
            ok = True
            for s in sel:
                if not is_separated(candidates[i], candidates[s]):
                    ok = False
                    break
            if not ok:
                continue
            cent = np.mean(candidates[sel], axis=0)
            d = np.linalg.norm(candidates[i] - cent)
            if d < bd:
                bd, best = d, i
        if best is None:
            break
        sel.append(best)
        sset.add(best)

    lattice_pts = candidates[sel] if len(sel) >= 11 else np.array([
        [0, 0], [2.5, 0], [0, 4.33], [-2.5, 4.33], [-5, 0],
        [-2.5, -4.33], [0, -4.33], [2.5, -4.33], [5, 0],
        [7.5, 0], [0, 8.66]
    ])

    # Continuous optimization
    def get_normals(a_deg):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad([30, 90, 150, 210, 270, 330])
        return np.stack([np.cos(th), np.sin(th)], axis=1)

    def pair_gap(c1, c2, nm):
        v = c1 - c2
        return np.abs(nm @ v).min() - sqrt3

    def all_inner_in_outer(points, oc, S, nm):
        diffs = points - oc
        proj = np.abs(diffs @ nm.T)
        return (S * sqrt3 / 2) - proj.max(axis=1).min()

    def min_pair_gap(points, nm):
        mg = 1e18
        for i in range(11):
            for j in range(i + 1, 11):
                g = pair_gap(points[i], points[j], nm)
                if g < mg:
                    mg = g
        return mg

    def objective(x):
        points = x[:22].reshape(11, 2)
        oc = x[22:24]
        S = x[24]
        a = x[25]
        nm = get_normals(a)
        mg = min_pair_gap(points, nm)
        gp = max(0.0, -mg) ** 2
        im = all_inner_in_outer(points, oc, S, nm)
        mp = max(0.0, -im) ** 2
        return S + 80.0 * (gp + mp)

    x0 = np.concatenate([lattice_pts[:11].ravel(), [0.0, 0.0, 7.0, 0.0]])

    best_x = x0.copy()
    best_obj = objective(x0)

    for trial in range(8):
        if time.time() - start_time > deadline - 60:
            break
        x_init = x0.copy()
        if trial > 0:
            x_init[:22] += np.random.randn(22) * 0.03
            x_init[25] += np.random.randn() * 3.0

        try:
            r = minimize(objective, x_init, method='Nelder-Mead',
                         options={'maxiter': 6000, 'xatol': 1e-9, 'fatol': 1e-9})
            if r.fun < best_obj:
                best_obj = r.fun
                best_x = r.x.copy()
                print(f"T{trial}: obj={r.fun:.4f}, S={r.x[24]:.4f}", flush=True)
        except Exception:
            pass

    points = best_x[:22].reshape(11, 2)
    oc = best_x[22:24]
    S = best_x[24]
    a = best_x[25]
    nm = get_normals(a)

    mg = min_pair_gap(points, nm)
    if mg < -1e-6:
        inner = np.array([[0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
                          [-1.25, 2.17, 0], [1.25, 2.17, 0],
                          [-1.25, -2.17, 0], [1.25, -2.17, 0],
                          [-3.75, 2.17, 0], [3.75, 2.17, 0],
                          [-3.75, -2.17, 0], [3.75, -2.17, 0]])
        return inner, np.array([0.0, 0.0, 0.0]), 8.0

    diffs = points - oc
    proj = np.abs(diffs @ nm.T)
    S_valid = 2.0 * proj.max() / sqrt3
    S = max(S, S_valid)

    inner = np.column_stack([points[:, 0], points[:, 1], np.zeros(11)])
    outer = np.array([oc[0], oc[1], a])
    print(f"Final: S={S:.4f}, 1/S={1.0 / S:.4f}", flush=True)
    return inner, outer, float(S)


if __name__ == "__main__":
    inner, outer, side = hexagon_packing_11()
    print(f"Side: {side}")
    print(f"1/Side: {1.0 / side}")
