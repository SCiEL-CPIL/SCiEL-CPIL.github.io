"""MWIS on 3-AP conflict graph with branch-and-bound + coloring upper bound."""
import time

N = 256
# vertices 0..255 represent numbers 1..256
WEIGHT = [1.0 / (i + 1) for i in range(N)]

# Build conflict adjacency bitmask
def build_conflicts():
    conf = [0] * N
    for p in range(N):
        x = p + 1
        for q in range(p + 1, N):
            y = q + 1
            d = y - x
            # third value possibilities
            # (i) consecutive: third = x - d (if >=1) or y + d (if <=256)
            hit = False
            if x - d >= 1:
                hit = True
            elif y + d <= 256:
                hit = True
            else:
                # (ii) x,y are 1st and 3rd: middle = (x+y)/2 integer and distinct
                if (x + y) % 2 == 0:
                    m = (x + y) // 2
                    if 1 <= m <= 256 and m != x and m != y:
                        hit = True
            if hit:
                conf[p] |= (1 << q)
                conf[q] |= (1 << p)
    return conf

CONF = build_conflicts()

_start_time = None
_best_set = None
_best_val = 0.0
_time_limit = 50.0

def _time_left():
    return _start_time is None or (time.time() - _start_time) < _time_limit

def _color_ub(cand_mask):
    # Greedy color the induced subgraph; return sum over colors of max weight.
    # Order by descending weight for tighter bound.
    verts = [i for i in range(N) if (cand_mask >> i) & 1]
    verts.sort(key=lambda i: -WEIGHT[i])
    color_of = {}
    # max weight per color
    col_max = {}
    for v in verts:
        # colors used by colored neighbors
        used = set()
        cm = CONF[v] & cand_mask
        while cm:
            nb = (cm & -cm).bit_length() - 1
            cm &= cm - 1
            if nb in color_of:
                used.add(color_of[nb])
        c = 0
        while c in used:
            c += 1
        color_of[v] = c
        if c not in col_max or WEIGHT[v] > col_max[c]:
            col_max[c] = WEIGHT[v]
    return sum(col_max.values())

def _greedy_solution(cand_mask):
    # Greedy maximal independent set by weight desc
    verts = [i for i in range(N) if (cand_mask >> i) & 1]
    verts.sort(key=lambda i: -WEIGHT[i])
    chosen = []
    avail = cand_mask
    for v in verts:
        if (avail >> v) & 1:
            chosen.append(v)
            avail &= ~CONF[v]
            avail &= ~(1 << v)
    return chosen

def _search(cand_mask, cur_val, chosen):
    global _best_set, _best_val
    if not _time_left():
        return
    if cand_mask == 0:
        if cur_val > _best_val:
            _best_val = cur_val
            _best_set = list(chosen)
        return
    # upper bound via coloring
    ub = _color_ub(cand_mask)
    if cur_val + ub <= _best_val + 1e-15:
        return
    # pick branching vertex: max weight in cand
    verts = [i for i in range(N) if (cand_mask >> i) & 1]
    v = max(verts, key=lambda i: WEIGHT[i])
    # Branch include first
    new_cand = cand_mask & ~CONF[v] & ~(1 << v)
    _search(new_cand, cur_val + WEIGHT[v], chosen + [v])
    if not _time_left():
        return
    # Branch exclude
    _search(cand_mask & ~(1 << v), cur_val, chosen)

def construct():
    global _start_time, _best_set, _best_val
    _start_time = time.time()
    _best_val = 0.0
    _best_set = None
    full = (1 << N) - 1
    # seed with greedy
    g = _greedy_solution(full)
    _best_set = g
    _best_val = sum(WEIGHT[i] for i in g)
    _search(full, 0.0, [])
    if _best_set is None:
        _best_set = [0]
    # return numbers 1..256
    return [i + 1 for i in _best_set]
