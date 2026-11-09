"""3D bin packing: best-fit decreasing construction + move/swap local search."""
import time

_DEADLINE = 140.0  # wall-clock safety margin before the 150 s kill


def pack(boxes, container):
    start = time.time()
    W, H, D = (int(container[0]), int(container[1]), int(container[2]))
    Wf, Hf, Df = float(W), float(H), float(D)
    cont_vol = Wf * Hf * Df
    n = len(boxes)
    if n == 0:
        return []

    dims = [tuple(int(v) for v in boxes[i]) for i in range(n)]

    def overlap(x, y, z, w, h, d, o):
        ox, oy, oz = o[0]
        ow, oh, od = o[1]
        return not (x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y
                    or z + d <= oz or oz + od <= z)

    def fits(b, i, pos, skip):
        x, y, z = pos
        w, h, d = dims[i]
        for j, o in bins[b].items():
            if j == i or j == skip:
                continue
            if overlap(x, y, z, w, h, d, (o, dims[j])):
                return False
        return True

    def try_place(b, i, skip):
        w, h, d = dims[i]
        cands = {(0, 0, 0)}
        for o in bins[b].values():
            cands.add((o[0] + w, o[1], o[2]))
            cands.add((o[0], o[1] + h, o[2]))
            cands.add((o[0], o[1], o[2] + d))
        best = None
        for pos in cands:
            x, y, z = pos
            if x + w > W or y + h > H or z + d > D:
                continue
            if fits(b, i, pos, skip):
                s = x * Wf * Hf + y * Wf * Df + z * Hf * Df + (Wf - x - w) * (Hf - y - h) * (Df - z - d)
                if best is None or s < best[0]:
                    best = (s, pos)
        return best

    # construction: best-fit decreasing
    order = sorted(range(n), key=lambda i: dims[i][0] * dims[i][1] * dims[i][2], reverse=True)
    bins = [dict() for _ in range(n)]
    for i in order:
        placed = False
        for b in range(len(bins)):
            res = try_place(b, i, None)
            if res is not None:
                bins[b][i] = res[1]
                placed = True
                break
        if not placed:
            b = len(bins)
            bins.append(dict())
            bins[b][i] = (0, 0, 0)

    def occupied(b):
        return sum(dims[i][0] * dims[i][1] * dims[i][2] for i in bins[b])

    def maxb():
        return max(bins) if bins else -1

    def last_frac():
        mb = maxb()
        return (occupied(mb) / cont_vol) if cont_vol else 1.0

    def score():
        mb = maxb()
        return mb + 1 + 0.1 * (1.0 - last_frac())

    # local search: move + swap
    improved = True
    while improved:
        if time.time() > _DEADLINE - 1.0:
            break
        improved = False
        for b in range(len(bins)):
            if time.time() > _DEADLINE - 1.0:
                break
            for i in list(bins[b]):
                cur = bins[b].pop(i)
                placed = False
                for t in range(len(bins)):
                    if t == b:
                        continue
                    res = try_place(t, i, None)
                    if res is not None:
                        bins[t][i] = res[1]
                        placed = True
                        improved = True
                        break
                if not placed:
                    bins[b][i] = cur
        for b in range(len(bins)):
            if time.time() > _DEADLINE - 1.0:
                break
            for i in list(bins[b]):
                for t in range(len(bins)):
                    if t == b:
                        continue
                    for j in list(bins[t]):
                        pos_i = try_place(t, i, j)
                        pos_j = try_place(b, j, i)
                        if pos_i is not None and pos_j is not None:
                            del bins[b][i]
                            del bins[t][j]
                            bins[t][i] = pos_i[1]
                            bins[b][j] = pos_j[1]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # renumber bins to be consecutive
    new_bins = [dict() for _ in range(len(bins))]
    mapping = {}
    new_id = 0
    for b in range(len(bins)):
        if bins[b]:
            mapping[b] = new_id
            new_id += 1
    for b in range(len(bins)):
        if bins[b]:
            for i, pos in bins[b].items():
                new_bins[mapping[b]][i] = pos
    bins = new_bins[:new_id]

    placements = []
    for i in range(n):
        for b in range(len(bins)):
            if i in bins[b]:
                x, y, z = bins[b][i]
                placements.append((b, x, y, z, dims[i][0], dims[i][1], dims[i][2]))
                break
    return placements
