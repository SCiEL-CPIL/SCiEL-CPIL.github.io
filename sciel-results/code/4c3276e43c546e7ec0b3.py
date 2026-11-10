# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _incidence_matrix(points, lines):
    # M[i, j] = 1 iff line j passes through point i
    # points: (n,2) int, lines: (m,3) int (A,B,C)
    A = lines[:, 0:1]      # (m,1)
    B = lines[:, 1:2]      # (m,1)
    C = lines[:, 2:3]      # (m,1)
    x = points[:, 0:1]     # (n,1)
    y = points[:, 1:2]     # (n,1)
    val = A * x + B * y + C  # (n,m)
    return (val == 0).astype(np.int64)


def _canonical_line(A, B, C):
    g = int(abs(A))
    if abs(B) > g:
        g = abs(B)
    if abs(C) > g:
        g = abs(C)
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)


def _make_grid(x0, y0, w, h):
    xs = np.arange(x0, x0 + w)
    ys = np.arange(y0, y0 + h)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _steep_lines(points):
    # all lines with |slope| >= 1 that pass through >= 2 points (canonical, no dupes)
    pts = points.tolist()
    seen = set()
    lines = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 or dy == 0:
                continue
            if abs(dx) > abs(dy):
                continue  # shallow (|slope|<1)
            c = _canonical_line(dy, -dx, -dy * x1 + dx * y1)
            if c in seen:
                continue
            seen.add(c)
            lines.append(c)
    return lines


def _all_lines(points):
    pts = points.tolist()
    seen = set()
    lines = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            c = _canonical_line(y2 - y1, x1 - x2, x2 * y1 - x1 * y2)
            if c in seen:
                continue
            seen.add(c)
            lines.append(c)
    return lines


def _greedy_select(M, k, budget):
    n, m = M.shape
    selected = np.zeros(m, dtype=bool)
    sel_count = 0
    best = np.zeros(n, dtype=np.int64)  # best coverage per point
    start = time.time()
    while sel_count < k:
        remaining = budget - (time.time() - start)
        if remaining <= 0.5:
            break
        gains = M.sum(axis=0) - (M & best[:, None]).sum(axis=0)
        cand = np.where((~selected) & (gains > 0))[0]
        if cand.size == 0:
            break
        # tie-break: prefer fewer points (smaller support) for diversity
        sup = M[cand].sum(axis=1)
        order = np.lexsort((sup, -gains[cand]))
        pick = cand[order[0]]
        selected[pick] = True
        sel_count += 1
        best = np.maximum(best, M[:, pick])
    return selected, best.sum()


def _local_search(M, selected, k, budget):
    n, m = M.shape
    cur = selected.copy()
    cur_cov = (M & cur[:, None]).any(axis=1)
    cur_val = int(cur_cov.sum())
    start = time.time()
    improved = True
    while improved:
        improved = False
        if time.time() - start > budget:
            break
        for i in np.where(cur)[0]:
            if time.time() - start > budget:
                break
            for j in np.where(~cur)[0]:
                # quick upper bound: removing i loses its unique points, adding j gains
                # its unique points
                col_i = M[:, i]
                col_j = M[:, j]
                # points only covered by i among selected
                only_i = ((col_i == 1) & (cur_cov == 1))
                # actually need: points covered by i and by no other selected
                others = cur.copy()
                others[i] = False
                covered_by_others = (M & others[:, None]).any(axis=1)
                lost = int(((col_i == 1) & (~covered_by_others)).sum())
                # points not covered at all
                gain = int(((col_j == 1) & (~cur_cov)).sum())
                if gain - lost <= 0:
                    continue
                # do the swap
                cur[i] = False
                cur[j] = True
                cur_cov = (M & cur[:, None]).any(axis=1)
                nv = int(cur_cov.sum())
                if nv > cur_val:
                    cur_val = nv
                    improved = True
                else:
                    # revert
                    cur[i] = True
                    cur[j] = False
                    cur_cov = (M & cur[:, None]).any(axis=1)
    return cur, cur_val


def _try_config(x0, y0, w, h, budget):
    points = _make_grid(x0, y0, w, h)
    n = points.shape[0]
    steep = _steep_lines(points)
    if len(steep) < 27:
        # not enough steep lines; fall back to all lines
        cand = _all_lines(points)
    else:
        # candidate pool: steep lines plus a few horizontals/verticals
        hlines = []
        for yy in range(y0, y0 + h):
            hlines.append(_canonical_line(0, 1, -yy))
        vlines = []
        for xx in range(x0, x0 + w):
            vlines.append(_canonical_line(1, 0, -xx))
        pool = list(steep)
        for L in hlines + vlines:
            if L not in set(pool):
                pool.append(L)
        cand = pool
    cand = np.array(cand, dtype=np.int64)
    M = _incidence_matrix(points, cand)
    sel, val = _greedy_select(M, 27, budget)
    sel, val = _local_search(M, sel, 27, budget)
    lines = cand[sel]
    # verify exactly 27 distinct valid lines
    assert lines.shape == (27, 3)
    return points, lines, int(val)


def szemeredi_trotter_incidences():
    deadline = time.time() + 600.0
    margin = 10.0
    best = None
    best_val = -1
    configs = [
        (1, 1, 6, 9),
        (1, 1, 9, 6),
        (1, 1, 6, 6),
        (1, 1, 5, 11),
        (1, 1, 11, 5),
        (1, 1, 4, 14),
        (1, 1, 14, 4),
    ]
    for (x0, y0, w, h) in configs:
        if time.time() + margin >= deadline:
            break
        try:
            pts, lns, val = _try_config(x0, y0, w, h, max(1.0, deadline - margin - time.time()))
        except Exception:
            continue
        if val > best_val:
            best_val = val
            best = (pts, lns)
    if best is None:
        # fallback to baseline grid
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        return points, lines
    return best


# EVOLVE-BLOCK-END
