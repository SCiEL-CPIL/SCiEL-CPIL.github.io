"""C4-free edge selection in Q7 via fast local search."""
import time
import numpy as np

_D = 7
_N = 1 << _D          # 128 vertices
_NE = _D * (1 << (_D - 1))   # 448 edges
_NF = (_D * (_D - 1) // 2) * (1 << (_D - 2))  # 21 * 32 = 672 faces

# ---- Precompute edge/face structure (cached) ----
_EDGE_CACHE = None

def _precompute():
    global _EDGE_CACHE
    if _EDGE_CACHE is not None:
        return _EDGE_CACHE

    # Edge list: for each vertex x and axis i with x < x^(1<<i)
    edges = []
    edge_idx = {}
    for x in range(_N):
        for i in range(_D):
            y = x ^ (1 << i)
            if x < y:
                edge_idx[(x, y)] = len(edges)
                edges.append((x, y, i))  # axis i
    assert len(edges) == _NE

    # Faces: for each pair of axes (i, j), i < j, and each fixed pattern of the other bits.
    # Represent a face by (i, j, mask) where mask encodes the other 5 bits' values.
    # We'll just enumerate: for each pair (i,j), for each "base" vertex, the face is the
    # 4-cycle in axes i,j. But each face is counted 4 times if we iterate all vertices;
    # instead iterate over base vertices that are "minimal" in the face: base has bit i=0 and bit j=0.
    face_edges = []   # list of tuples of 4 edge indices
    face_of = {}
    for i in range(_D):
        for j in range(i + 1, _D):
            # base vertices: bit i = 0 and bit j = 0
            for base in range(_N):
                if (base >> i) & 1 or (base >> j) & 1:
                    continue
                v0 = base
                v1 = base ^ (1 << i)
                v2 = base ^ (1 << j)
                v3 = base ^ (1 << i | 1 << j)
                e01 = edge_idx[(min(v0, v1), max(v0, v1))]
                e23 = edge_idx[(min(v2, v3), max(v2, v3))]
                e02 = edge_idx[(min(v0, v2), max(v0, v2))]
                e13 = edge_idx[(min(v1, v3), max(v1, v3))]
                f = len(face_edges)
                face_edges.append((e01, e23, e02, e13))
                for e in (e01, e23, e02, e13):
                    face_of.setdefault(e, []).append(f)
    assert len(face_edges) == _NF

    # edge_faces as array of shape (NE, D-1)
    edge_faces = np.full((_NE, _D - 1), -1, dtype=np.int32)
    for e, fl in face_of.items():
        assert len(fl) == _D - 1
        for k, f in enumerate(fl):
            edge_faces[e, k] = f

    _EDGE_CACHE = (edges, edge_idx, face_edges, edge_faces)
    return _EDGE_CACHE


def _face_count_from_selection(sel):
    """sel: bool array of length NE. Returns face_count array of length NF."""
    edges, edge_idx, face_edges, edge_faces = _precompute()
    fc = np.zeros(_NF, dtype=np.int8)
    # for each selected edge, increment its faces
    sel_idx = np.nonzero(sel)[0]
    if sel_idx.size:
        for e in sel_idx:
            for f in edge_faces[e]:
                if f >= 0:
                    fc[f] += 1
    return fc


def _is_addable(e, fc, edge_faces):
    """True if adding edge e keeps every face < 4."""
    for f in edge_faces[e]:
        if f >= 0 and fc[f] >= 3:
            return False
    return True


def _greedy_init(rng, edges, edge_idx, face_edges, edge_faces):
    """Randomized greedy: repeatedly add a random addable edge."""
    sel = np.zeros(_NE, dtype=bool)
    fc = np.zeros(_NF, dtype=np.int8)
    # candidate pool
    remaining = list(range(_NE))
    rng.shuffle(remaining)
    # We'll iterate; maintain addable set lazily.
    # Simple: loop over shuffled order, add if addable.
    for e in remaining:
        if _is_addable(e, fc, edge_faces):
            sel[e] = True
            for f in edge_faces[e]:
                if f >= 0:
                    fc[f] += 1
    return sel, fc


def _local_search(sel, fc, edge_faces, edge_idx, rng, deadline, max_rounds=200):
    """1-flip local search: add any addable edge; if stuck, remove a blocking edge then re-add."""
    edges, edge_idx2, face_edges, edge_faces2 = _precompute()
    # Precompute, for each edge, the set of faces it touches (as a frozenset) for overlap tests.
    edge_face_sets = [set(int(f) for f in edge_faces[e] if f >= 0) for e in range(_NE)]

    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        # Phase 1: add all currently addable edges (in random order)
        order = np.nonzero(~sel)[0]
        rng.shuffle(order)
        added_any = False
        for e in order:
            if time.time() >= deadline:
                break
            if _is_addable(e, fc, edge_faces):
                sel[e] = True
                for f in edge_faces[e]:
                    if f >= 0:
                        fc[f] += 1
                added_any = True
                improved = True
        if not added_any:
            # Stuck: find a selected edge whose removal enables the most additions.
            # Approximate: for each selected edge e, count unselected edges g that are
            # blocked ONLY by faces containing e.
            best_e = -1
            best_gain = 0
            sel_idx = np.nonzero(sel)[0]
            # sample to keep it fast
            if sel_idx.size > 200:
                sel_sample = rng.choice(sel_idx, size=200, replace=False)
            else:
                sel_sample = sel_idx
            for e in sel_sample:
                ef = edge_face_sets[e]
                # faces of e that are currently at 3 (these are the "blocking" faces)
                block_faces = set(int(f) for f in edge_faces[e] if f >= 0 and fc[f] >= 3)
                if not block_faces:
                    continue
                gain = 0
                for g in np.nonzero(~sel)[0]:
                    # g is addable now?
                    if _is_addable(g, fc, edge_faces):
                        continue
                    # g is blocked iff there exists a face f in edge_faces[g] with fc[f]==3.
                    # After removing e, fc[f] decreases by 1 for f in ef.
                    # g becomes addable iff every face f in edge_faces[g] with fc[f]==3 is in ef.
                    ok = True
                    for f in edge_faces[g]:
                        if f >= 0 and fc[f] == 3:
                            if f not in ef:
                                ok = False
                                break
                    if ok:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_e = e
            if best_e < 0:
                # No single removal helps; try removing a random selected edge to escape.
                if sel_idx.size == 0:
                    break
                best_e = int(rng.choice(sel_idx))
                best_gain = 0
            # Remove best_e
            sel[best_e] = False
            for f in edge_faces[best_e]:
                if f >= 0:
                    fc[f] -= 1
            improved = True
    return sel, fc


def construct():
    start = time.time()
    deadline = start + 110.0   # safety margin under 120s
    edges, edge_idx, face_edges, edge_faces = _precompute()

    best_sel = None
    best_count = -1
    rng = np.random.default_rng(12345)

    # Try several randomized greedy + local search restarts.
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        sel, fc = _greedy_init(rng, edges, edge_idx, face_edges, edge_faces)
        sel, fc = _local_search(sel, fc, edge_faces, edge_idx, rng, deadline)
        cnt = int(sel.sum())
        if cnt > best_count:
            best_count = cnt
            best_sel = sel.copy()
        # Early stop if we hit the upper bound.
        if best_count >= 308:
            break
        # Limit restarts to avoid pathological loops; each restart is cheap.
        if restarts >= 60 and time.time() < deadline:
            # continue but with fresh rng
            pass

    if best_sel is None:
        # Fallback: deterministic greedy (should not happen).
        sel = np.zeros(_NE, dtype=bool)
        fc = np.zeros(_NF, dtype=np.int8)
        for e in range(_NE):
            if _is_addable(e, fc, edge_faces):
                sel[e] = True
                for f in edge_faces[e]:
                    if f >= 0:
                        fc[f] += 1
        best_sel = sel

    out = []
    for e in range(_NE):
        if best_sel[e]:
            x, y, i = edges[e]
            out.append([x, y])
    return out
