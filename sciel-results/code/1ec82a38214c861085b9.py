import numpy as np
import time
import heapq


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3
    n_points = q * q  # 2401
    n_dirs = 50

    # Field operations as lookup tables
    f_mul = np.zeros((q, q), dtype=np.int32)
    f_add = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ia, ib = i // p, i % p
        for j in range(q):
            ja, jb = j // p, j % p
            f_mul[i, j] = ((ia * ja + ib * jb * k) % p) * p + ((ia * jb + ib * ja) % p)
            f_add[i, j] = ((ia + ja) % p) * p + ((ib + jb) % p)

    # Directions
    dir_vx = np.zeros(n_dirs, dtype=np.int32)
    dir_vy = np.zeros(n_dirs, dtype=np.int32)
    for d in range(49):
        dir_vx[d] = p  # field element 1 = (1,0) -> index 7
        dir_vy[d] = d
    dir_vx[49] = 0
    dir_vy[49] = p  # field element (0,1) -> index 1

    # Point coordinates
    pt_xfe = np.zeros(n_points, dtype=np.int32)
    pt_yfe = np.zeros(n_points, dtype=np.int32)
    for i in range(n_points):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        pt_xfe[i] = xa * p + xb
        pt_yfe[i] = ya * p + yb

    # Precompute line_pts[point, dir, t_idx] for t_idx in 0..47 (t = 1..48)
    line_pts = np.zeros((n_points, n_dirs, 48), dtype=np.int32)
    for i in range(n_points):
        xfe = pt_xfe[i]
        yfe = pt_yfe[i]
        for d in range(n_dirs):
            vx = dir_vx[d]
            vy = dir_vy[d]
            for t_idx in range(48):
                t = t_idx + 1
                tx = f_mul[t, vx]
                ty = f_mul[t, vy]
                nx = f_add[xfe, tx]
                ny = f_add[yfe, ty]
                nxa, nxb = nx // p, nx % p
                nya, nyb = ny // p, ny % p
                line_pts[i, d, t_idx] = nxa * 343 + nxb * 49 + nya * 7 + nyb

    # Build reverse_line: for each point p, list of (x, d) where p is on line(x,d)\{x}
    reverse_line = [[] for _ in range(n_points)]
    for x in range(n_points):
        for d in range(n_dirs):
            for t_idx in range(48):
                pt = line_pts[x, d, t_idx]
                reverse_line[pt].append((x, d))

    # Initial set: unital
    in_N = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        xa = i // 343
        ya = (i // 7) % 7
        yb = i % 7
        if (2 * xa + ya * ya - 3 * yb * yb) % p == 0:
            in_N[i] = True

    # Track validity
    valid = np.ones((n_points, n_dirs), dtype=bool)
    covered = np.zeros(n_points, dtype=np.int32)

    for x in range(n_points):
        for d in range(n_dirs):
            if np.all(in_N[line_pts[x, d, :]]):
                covered[x] += 1
            else:
                valid[x, d] = False

    start_time = time.time()
    deadline = start_time + 585

    def compute_cost(p):
        if not in_N[p]:
            return 0
        cost = 0
        for (x, d) in reverse_line[p]:
            if x == p:
                continue
            if valid[x, d] and covered[x] == 1:
                cost += 1
        return cost

    heap = []
    for p in range(n_points):
        if in_N[p]:
            c = compute_cost(p)
            if c == 0:
                heapq.heappush(heap, (c, p))

    while heap:
        if time.time() > deadline:
            break
        c, p = heapq.heappop(heap)
        if not in_N[p]:
            continue
        actual_cost = compute_cost(p)
        if actual_cost != c:
            if actual_cost == 0:
                heapq.heappush(heap, (actual_cost, p))
            continue
        if actual_cost > 0:
            continue
        in_N[p] = False
        for (x, d) in reverse_line[p]:
            if x == p:
                continue
            if valid[x, d]:
                valid[x, d] = False
                covered[x] -= 1
                if in_N[x] and covered[x] == 0:
                    # This shouldn't happen if cost was 0, but guard anyway
                    pass
        # Push newly removable points
        for (x, d) in reverse_line[p]:
            if x == p:
                continue
            if in_N[x]:
                nc = compute_cost(x)
                if nc == 0:
                    heapq.heappush(heap, (nc, x))

    N_indices = [i for i in range(n_points) if in_N[i]]
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        result[idx, 0, 0] = i // 343
        result[idx, 0, 1] = (i // 49) % 7
        result[idx, 1, 0] = (i // 7) % 7
        result[idx, 1, 1] = i % 7

    return result
