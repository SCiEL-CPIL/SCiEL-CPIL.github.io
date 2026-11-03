"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before the 150s kill

    SQRT3 = np.sqrt(3.0)

    def hex_vertices(x, y, angle_deg):
        a = np.deg2rad(angle_deg)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        base = np.array([
            [1.0, 0.0],
            [0.5, SQRT3 / 2],
            [-0.5, SQRT3 / 2],
            [-1.0, 0.0],
            [-0.5, -SQRT3 / 2],
            [0.5, -SQRT3 / 2],
        ])
        return base @ R.T + np.array([x, y])

    def vertex_violation(v, S):
        # outer hexagon centered at origin, flat-top, side S.
        # apothem = S*sqrt(3)/2. constraint: |dot(v,n_i)| <= apothem for 6 normals
        apothem = S * SQRT3 / 2.0
        mx = abs(v[0])
        my = abs(v[1])
        # normals at 0,60,120,180,240,300 deg -> use |x|, |y|, |x/2 + sqrt3/2*y|, |x/2 - sqrt3/2*y|
        m1 = abs(0.5 * v[0] + (SQRT3 / 2.0) * v[1])
        m2 = abs(0.5 * v[0] - (SQRT3 / 2.0) * v[1])
        return max(mx, my, m1, m2) - apothem

    def container_penalty(xs, ys, thetas, S):
        pen = 0.0
        for k in range(N):
            verts = hex_vertices(xs[k], ys[k], thetas[k])
            for v in verts:
                viol = vertex_violation(v, S)
                if viol > 0:
                    pen += viol
        return pen

    def overlap_penalty(xs, ys, thetas):
        # conservative: min center distance for disjoint unit hexes is sqrt(3)
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < SQRT3:
                    pen += (SQRT3 - d)
        return pen

    def total_penalty(xs, ys, thetas, S):
        return container_penalty(xs, ys, thetas, S) + 10.0 * overlap_penalty(xs, ys, thetas)

    def hex_grid_init():
        d = SQRT3  # neighbor center distance in hex grid
        # 7-hex flower
        pts = [
            (0.0, 0.0),
            (d, 0.0), (d / 2, 1.5), (-d / 2, 1.5),
            (-d, 0.0), (-d / 2, -1.5), (d / 2, -1.5),
        ]
        # 4 satellites: extend the flower in 4 directions
        # place them just beyond the outer ring
        pts.append((2.0 * d, 0.0))
        pts.append((-2.0 * d, 0.0))
        pts.append((d, 3.0))
        pts.append((-d, -3.0))
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), np.zeros(N)

    def local_search(S, iters):
        xs, ys, thetas = hex_grid_init()
        best_pen = total_penalty(xs, ys, thetas, S)
        best_state = (xs.copy(), ys.copy(), thetas.copy())
        rng = np.random.default_rng(42)

        step = 0.3
        for it in range(iters):
            if time.time() > deadline:
                break
            pen = total_penalty(xs, ys, thetas, S)
            if pen < best_pen:
                best_pen = pen
                best_state = (xs.copy(), ys.copy(), thetas.copy())
            if pen < 1e-6:
                break
            # random gradient-free perturbation, annealing step size
            step = max(0.05, 0.3 * (1.0 - it / float(iters)))
            for k in range(N):
                if rng.random() < 0.7:
                    xs[k] += rng.normal(0, step)
                    ys[k] += rng.normal(0, step)
                if rng.random() < 0.3:
                    thetas[k] += rng.normal(0, step * 0.5)
        return best_pen, best_state

    # Lower bound: need to contain a cluster whose extent is at least ~ 2*2 + 1 in some direction.
    # For flat-top hex, width in x for side S is 2S. Our cluster spans ~ 4d + 2 in x ~ 8.9, so S >= ~4.45.
    low = 4.3
    high = 6.5

    best_side = 40.0
    best_inner = None

    # First, find a feasible S by searching upward from a reasonable start.
    # Use binary search: if feasible at mid, go lower; else go higher.
    lo, hi = low, high
    feasible_side = None
    for _ in range(14):
        if time.time() > deadline - 5.0:
            break
        mid = (lo + hi) / 2.0
        pen, state = local_search(mid, iters=400)
        if pen < 1e-4:
            feasible_side = mid
            hi = mid
        else:
            lo = mid

    if feasible_side is not None:
        # refine: we know hi is feasible; tighten a bit more
        for _ in range(8):
            if time.time() > deadline - 2.0:
                break
            mid = (lo + hi) / 2.0
            pen, state = local_search(mid, iters=300)
            if pen < 1e-4:
                feasible_side = mid
                hi = mid
            else:
                lo = mid
        # final state at the best feasible side
        pen, state = local_search(feasible_side, iters=600)
        if pen < 1e-3:
            best_side = feasible_side
            best_inner = [[float(state[0][i]), float(state[1][i]), float(state[2][i])] for i in range(N)]
    else:
        # fallback: use the largest S we tried that gave the lowest penalty
        pen, state = local_search(hi, iters=600)
        best_side = hi
        best_inner = [[float(state[0][i]), float(state[1][i]), float(state[2][i])] for i in range(N)]

    if best_inner is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return best_inner, (0.0, 0.0), float(best_side), 0.0
# EVOLVE-BLOCK-END
