import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 120.0

    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 2))
    rad = np.zeros(N)
    for i in range(N):
        row, col = divmod(i, side)
        pos[i] = [(col + 0.5) / side, (row + 0.5) / side]
        rad[i] = 0.5 / side - 0.005

    def grads(p, r):
        n = len(r)
        gp = np.zeros((n, 2))
        gr = np.zeros(n)
        dx = p[:, 0:1] - p[0:1, :]
        dy = p[:, 1:2] - p[0:1, :]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        np.fill_diagonal(d, 1.0)
        md = r[:, None] + r[None, :]
        v = np.maximum(md - d, 0)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        inv = np.where(d > 1e-10, 1.0 / d, 0.0)
        w = v * inv
        w[~mask] = 0
        gp[:, 0] = np.sum(w * dx, axis=1)
        gp[:, 1] = np.sum(w * dy, axis=1)
        gr += np.sum(v, axis=1)
        vl = np.maximum(r - p[:, 0], 0)
        gp[:, 0] += vl
        gr += vl
        vr = np.maximum(r - (1 - p[:, 0]), 0)
        gp[:, 0] -= vr
        gr += vr
        vb = np.maximum(r - p[:, 1], 0)
        gp[:, 1] += vb
        gr += vb
        vt = np.maximum(r - (1 - p[:, 1]), 0)
        gp[:, 1] -= vt
        gr += vt
        return gp, gr

    def viol(p, r):
        n = len(r)
        dx = p[:, 0:1] - p[0:1, :]
        dy = p[:, 1:2] - p[0:1, :]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        md = r[:, None] + r[None, :]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        cv = np.sum(np.maximum(md - d, 0)[mask])
        bv = (np.sum(np.maximum(r - p[:, 0], 0)) +
              np.sum(np.maximum(r - (1 - p[:, 0]), 0)) +
              np.sum(np.maximum(r - p[:, 1], 0)) +
              np.sum(np.maximum(r - (1 - p[:, 1]), 0)))
        return cv + bv

    # Phase 1: Resolve initial overlaps
    for _ in range(1000):
        if time.time() - start_time > deadline:
            break
        gp, _ = grads(pos, rad)
        if np.max(np.abs(gp)) < 1e-12:
            break
        pos -= 0.002 * gp
        pos = np.clip(pos, -0.05, 1.05)

    # Phase 2: Grow radii with position optimization
    for _ in range(40):
        if time.time() - start_time > deadline:
            break
        rt = rad * 1.015
        pt = pos.copy()
        for _ in range(500):
            if time.time() - start_time > deadline:
                break
            gp, _ = grads(pt, rt)
            if np.max(np.abs(gp)) < 1e-12:
                break
            pt -= 0.001 * gp
            pt = np.clip(pt, -0.05, 1.05)
        if viol(pt, rt) < 0.005:
            pos, rad = pt, rt
        else:
            lo, hi = 1.0, 1.015
            for _ in range(8):
                mid = (lo + hi) / 2
                rt = rad * mid
                pt = pos.copy()
                for _ in range(300):
                    if time.time() - start_time > deadline:
                        break
                    gp, _ = grads(pt, rt)
                    if np.max(np.abs(gp)) < 1e-12:
                        break
                    pt -= 0.001 * gp
                    pt = np.clip(pt, -0.05, 1.05)
                if viol(pt, rt) < 0.005:
                    lo = mid
                    pos, rad = pt, rt
                else:
                    hi = mid

    # Phase 3: Joint fine-tuning
    for _ in range(300):
        if time.time() - start_time > deadline:
            break
        gp, _ = grads(pos, rad)
        if np.max(np.abs(gp)) < 1e-13:
            break
        pos -= 0.0005 * gp
        pos = np.clip(pos, -0.05, 1.05)
        rad = np.maximum(rad + 0.00005, 0.001)

    # Phase 4: Polish
    for _ in range(200):
        if time.time() - start_time > deadline:
            break
        gp, _ = grads(pos, rad)
        if np.max(np.abs(gp)) < 1e-15:
            break
        pos -= 0.0001 * gp
        pos = np.clip(pos, -0.05, 1.05)

    # Final validation
    for i in range(N):
        mr = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        rad[i] = min(rad[i], mr)
        rad[i] = max(rad[i], 0.0)
        pos[i, 0] = np.clip(pos[i, 0], rad[i], 1 - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], 1 - rad[i])

    return np.column_stack([pos, rad])
