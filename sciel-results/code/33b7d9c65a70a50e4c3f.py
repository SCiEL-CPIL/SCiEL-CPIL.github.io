import time
import random

_N = 16
_NUM_EDGES = _N * (_N - 1) // 2  # 120
_NUM_K4 = 1820

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _precompute():
    k4s = []
    for a in range(_N):
        for b in range(a + 1, _N):
            for c in range(b + 1, _N):
                for d in range(c + 1, _N):
                    k4s.append((a, b, c, d))
    edge_k4 = [[] for _ in range(_NUM_EDGES)]
    for k4i, (a, b, c, d) in enumerate(k4s):
        e01 = _edge_idx(a, b)
        e02 = _edge_idx(a, c)
        e03 = _edge_idx(a, d)
        e12 = _edge_idx(b, c)
        e13 = _edge_idx(b, d)
        e23 = _edge_idx(c, d)
        edges = [e01, e02, e03, e12, e13, e23]
        for e in edges:
            for other in edges:
                if other != e:
                    edge_k4[e].append((k4i, other))
    return k4s, edge_k4

_K4S, _EDGE_K4 = _precompute()

def _k4_edges(k4i):
    a, b, c, d = _K4S[k4i]
    return (_edge_idx(a, b), _edge_idx(a, c), _edge_idx(a, d),
            _edge_idx(b, c), _edge_idx(b, d), _edge_idx(c, d))

def _initial_colouring():
    col = [0] * _NUM_EDGES
    for i in range(_N):
        for j in range(i + 1, _N):
            col[_edge_idx(i, j)] = (i + j) % 17
    return col

def _build_state(col, maxc):
    colour_count = [0] * maxc
    for c in col:
        colour_count[c] += 1
    k4_colour_count = [[0] * maxc for _ in range(_NUM_K4)]
    k4_distinct = [0] * _NUM_K4
    for k4i in range(_NUM_K4):
        e0, e1, e2, e3, e4, e5 = _k4_edges(k4i)
        seen = 0
        for e in (e0, e1, e2, e3, e4, e5):
            c = col[e]
            if k4_colour_count[k4i][c] == 0:
                seen += 1
            k4_colour_count[k4i][c] += 1
        k4_distinct[k4i] = seen
    penalty = sum(1 for x in k4_distinct if x < 5)
    return colour_count, k4_colour_count, k4_distinct, penalty

def _used(colour_count):
    return sum(1 for x in colour_count if x > 0)

def construct():
    start = time.time()
    deadline = start + 135.0
    rng = random.Random(12345)
    col = _initial_colouring()
    maxc = max(col) + 1
    colour_count, k4_colour_count, k4_distinct, penalty = _build_state(col, maxc)
    penalty_weight = 10000
    best_score = _used(colour_count) + penalty_weight * penalty
    best_col = col[:]
    best_valid = (penalty == 0)
    best_used = _used(colour_count)

    no_improve = 0
    max_no_improve = 1500
    while time.time() < deadline:
        e = rng.randrange(_NUM_EDGES)
        oldc = col[e]
        cands = [c for c in range(len(colour_count)) if colour_count[c] > 0]
        if oldc not in cands:
            cands.append(oldc)
        best_newc = oldc
        best_delta = 0
        for nc in cands:
            if nc == oldc:
                continue
            dc_used = 0
            if colour_count[oldc] == 1:
                dc_used -= 1
            if colour_count[nc] == 0:
                dc_used += 1
            dp = 0
            seen_k4 = set()
            for k4i, other in _EDGE_K4[e]:
                if k4i in seen_k4:
                    continue
                seen_k4.add(k4i)
                cc = k4_colour_count[k4i]
                d_before = k4_distinct[k4i]
                d_after = d_before
                if cc[oldc] == 1:
                    d_after -= 1
                if cc[nc] == 0:
                    d_after += 1
                bad_before = 1 if d_before < 5 else 0
                bad_after = 1 if d_after < 5 else 0
                dp += (bad_after - bad_before)
            delta = dc_used + penalty_weight * dp
            if delta < best_delta:
                best_delta = delta
                best_newc = nc
        if best_newc != oldc and best_delta < 0:
            nc = best_newc
            col[e] = nc
            colour_count[oldc] -= 1
            if nc >= len(colour_count):
                colour_count.extend([0] * (nc + 1 - len(colour_count)))
            colour_count[nc] += 1
            seen_k4 = set()
            for k4i, other in _EDGE_K4[e]:
                if k4i in seen_k4:
                    continue
                seen_k4.add(k4i)
                cc = k4_colour_count[k4i]
                cc[oldc] -= 1
                cc[nc] += 1
                d = 0
                for c in range(len(cc)):
                    if cc[c] > 0:
                        d += 1
                k4_distinct[k4i] = d
            # update penalty incrementally: recompute only affected
            # simpler: recompute penalty over all (1820) is cheap
            penalty = sum(1 for x in k4_distinct if x < 5)
            cur_score = _used(colour_count) + penalty_weight * penalty
            if cur_score < best_score:
                best_score = cur_score
                best_col = col[:]
                best_valid = (penalty == 0)
                best_used = _used(colour_count)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                for _ in range(8):
                    e2 = rng.randrange(_NUM_EDGES)
                    nc = rng.randrange(0, 25)
                    oldc2 = col[e2]
                    col[e2] = nc
                    colour_count[oldc2] -= 1
                    if nc >= len(colour_count):
                        colour_count.extend([0] * (nc + 1 - len(colour_count)))
                    colour_count[nc] += 1
                    seen_k4 = set()
                    for k4i, other in _EDGE_K4[e2]:
                        if k4i in seen_k4:
                            continue
                        seen_k4.add(k4i)
                        cc = k4_colour_count[k4i]
                        cc[oldc2] -= 1
                        cc[nc] += 1
                        d = 0
                        for c in range(len(cc)):
                            if cc[c] > 0:
                                d += 1
                        k4_distinct[k4i] = d
                    penalty = sum(1 for x in k4_distinct if x < 5)
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.4)

    return [best_col[_edge_idx(i, j)] for i in range(_N) for j in range(i + 1, _N)]
