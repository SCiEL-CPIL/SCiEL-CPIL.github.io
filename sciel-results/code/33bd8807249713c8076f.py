"""C4-free edge set construction for Q7: vectorized greedy + 2-swap local search."""
import time
import random
import numpy as np

_D = 7
_N = 1 << _D
_NEDGES = _D * _N // 2  # 448

_edges = None
_edge_faces = None
_face_ids = None


def _precompute():
    d = _D
    n = _N
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
    n_edges = len(edges)
    # faces: for each edge, 6 faces, each a 3-tuple of neighbor edge indices
    faces = [None] * n_edges
    for idx, (x, y) in enumerate(edges):
        i = (x ^ y).bit_length() - 1
        fl = []
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            a, b = x, c
            if a > b:
                a, b = b, a
            f1 = edge_idx[(a, b)]
            a, b = y, e
            if a > b:
                a, b = b, a
            f2 = edge_idx[(a, b)]
            a, b = c, e
            if a > b:
                a, b = b, a
            f3 = edge_idx[(a, b)]
            fl.append((f1, f2, f3))
        faces[idx] = fl
    return edges, faces


def _get_precomputed():
    global _edges, _edge_faces, _face_ids
    if _edges is None:
        _edges, _edge_faces = _precompute()
        # face_ids: each edge -> list of face ids it belongs to (edge is the 4th edge)
        _face_ids = [None] * _NEDGES
        for idx, fl in enumerate(_edge_faces):
            _face_ids[idx] = [idx + _NEDGES * k for k in range(len(fl))]
    return _edges, _edge_faces, _face_ids


def _face_count_update(sel, faces, idx, sign):
    """Update face_counts for adding(+1)/removing(-1) edge idx. sel is np bool array."""
    for a, b, c in faces[idx]:
        if sel[a] and sel[b]:
            _fc[c] += sign
        if sel[a] and sel[c]:
            _fc[b] += sign
        if sel[b] and sel[c]:
            _fc[a] += sign


_fc = None  # global face-count array, sized NEDGES + 6*NEDGES


def _greedy(order, deadline):
    """Vectorized greedy. Returns (selected bool array, face_count array)."""
    global _fc
    n_edges = _NEDGES
    faces = _edge_faces
    sel = np.zeros(n_edges, dtype=bool)
    # face_counts: index = edge_idx + NEDGES*face_pos
    fc = np.zeros(n_edges + 6 * n_edges, dtype=np.int8)
    _fc = fc
    order = list(order)
    for idx in order:
        if time.time() > deadline:
            break
        fl = faces[idx]
        # check: for each of 6 faces, are all 3 others selected?
        ok = True
        for a, b, c in fl:
            if sel[a] and sel[b] and sel[c]:
                ok = False
                break
        if ok:
            sel[idx] = True
            _face_count_update(sel, faces, idx, 1)
    return sel, fc


def _addable(sel, fc, faces, idx):
    fl = faces[idx]
    for a, b, c in fl:
        if sel[a] and sel[b] and sel[c]:
            return False
    return True


def _local_search(sel, fc, deadline, rng):
    """Local search: add free edges, 1-swap, 2-swap. mutates sel, fc in place."""
    n_edges = _NEDGES
    faces = _edge_faces

    def _add(idx):
        sel[idx] = True
        _face_count_update(sel, faces, idx, 1)

    def _rem(idx):
        sel[idx] = False
        _face_count_update(sel, faces, idx, -1)

    while time.time() < deadline:
        improved = False

        # Pass A: add any currently addable (fc==0) edge
        for idx in range(n_edges):
            if not sel[idx] and fc[idx] == 0:
                _add(idx)
                improved = True

        # Pass B: 1-swap. For each selected edge with an open face, try to remove it
        # and add a currently-blocked neighbor that was closing that face.
        for idx in range(n_edges):
            if time.time() > deadline:
                break
            if not sel[idx] or fc[idx] == 0:
                continue
            fl = faces[idx]
            open_face = None
            for (a, b, c) in fl:
                if not (sel[a] and sel[b] and sel[c]):
                    open_face = (a, b, c)
                    break
            if open_face is None:
                continue
            a, b, c = open_face
            for n in (a, b, c):
                if sel[n]:
                    continue  # must be an unselected neighbor to add
                # try remove idx, add n
                _rem(idx)
                if _addable(sel, fc, faces, n):
                    _add(n)
                    improved = True
                else:
                    _add(idx)
                break

        # Pass C: 2-swap. remove two selected, add two unselected, if net gain.
        for idx in range(n_edges):
            if time.time() > deadline:
                break
            if not sel[idx] or fc[idx] == 0:
                continue
            fl = faces[idx]
            open_face = None
            for (a, b, c) in fl:
                if not (sel[a] and sel[b] and sel[c]):
                    open_face = (a, b, c)
                    break
            if open_face is None:
                continue
            a, b, c = open_face
            # candidate unselected neighbors in this face
            cands = [n for n in (a, b, c) if not sel[n]]
            if len(cands) < 2:
                continue
            # try removing idx and one other selected edge e2, adding two candidates
            for e2 in range(n_edges):
                if e2 == idx or not sel[e2]:
                    continue
                # quick: only consider e2 that shares a face with idx or is in cands' faces
                # (heuristic to limit search); skip if too far
                _rem(idx)
                _rem(e2)
                added = 0
                for n in cands:
                    if _addable(sel, fc, faces, n):
                        _add(n)
                        added += 1
                if added >= 2:
                    improved = True
                else:
                    for n in cands:
                        if sel[n]:
                            _rem(n)
                    _add(e2)
                    _add(idx)
                break  # only one e2 per idx to keep cost down

        if not improved:
            break

    return sel


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _is_c4_free(edge_list):
    """Fast validation: no 2D face has all 4 edges selected."""
    n = _N
    sel = np.zeros(_NEDGES, dtype=bool)
    idx_of = {}
    for x in range(n):
        for i in range(_D):
            y = x ^ (1 << i)
            if x < y:
                idx_of[(x, y)] = len(idx_of)
    for u, v in edge_list:
        key = (u, v) if u < v else (v, u)
        sel[idx_of[key]] = True
    faces = _edge_faces
    for idx in range(_NEDGES):
        if not sel[idx]:
            continue
        for a, b, c in faces[idx]:
            if sel[a] and sel[b] and sel[c]:
                return False
    return True


def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s limit
    edges, faces, face_ids = _get_precomputed()
    n_edges = _NEDGES
    rng = random.Random(42)

    best_sel = None
    best_size = 0

    def _record(sel):
        nonlocal best_sel, best_size
        cnt = int(np.sum(sel))
        if cnt > best_size:
            best_size = cnt
            best_sel = sel.copy()

    def _order_biased(rng):
        dim_order = list(range(_D))
        rng.shuffle(dim_order)
        order = []
        for i in dim_order:
            col = []
            for x in range(_N):
                y = x ^ (1 << i)
                if x < y:
                    idx = x * _D // 2 + (i // 2)
                    col.append(idx)
            rng.shuffle(col)
            order.extend(col)
        return order

    def _order_dim_priority(rng):
        # process dimensions in random order, within dim random vertices
        dim_order = list(range(_D))
        rng.shuffle(dim_order)
        order = []
        for i in dim_order:
            col = []
            for x in range(_N):
                y = x ^ (1 << i)
                if x < y:
                    idx = x * _D // 2 + (i // 2)
                    col.append(idx)
            rng.shuffle(col)
            order.extend(col)
        return order

    def _order_full(rng):
        order = list(range(n_edges))
        rng.shuffle(order)
        return order

    strategies = [_order_biased, _order_dim_priority, _order_full]

    # Phase 1: deterministic greedy baseline
    sel, fc = _greedy(list(range(n_edges)), deadline)
    sel = _local_search(sel, fc, deadline, rng)
    _record(sel)

    # Phase 2: randomized restarts with varied orderings + local search
    it = 0
    while time.time() < deadline:
        strat = strategies[it % len(strategies)]
        order = strat(rng)
        sel, fc = _greedy(order, deadline)
        sel = _local_search(sel, fc, deadline, rng)
        _record(sel)
        it += 1
        if best_size >= 308:
            break

    # Build result from best_sel
    if best_sel is None:
        return []
    result = []
    for idx in range(n_edges):
        if best_sel[idx]:
            u, v = edges[idx]
            result.append([u, v])

    # Validate; if somehow invalid, fall back to a clean deterministic greedy
    if result and not _is_c4_free(result):
        sel, fc = _greedy(list(range(n_edges)), deadline)
        result = []
        for idx in range(n_edges):
            if sel[idx]:
                u, v = edges[idx]
                result.append([u, v])

    return result
