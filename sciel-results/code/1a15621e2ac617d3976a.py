import random
import time
import numpy as np

def construct():
    d = 7
    N = 1 << d            # 128 vertices
    E = N * d             # 448 edges (each vertex has d incident)
    start = time.time()
    DEADLINE = start + 110.0

    # Build edge list: edge e connects v to v ^ bit(e)
    # We index edges by (v, i) for v in 0..N-1, i in 0..d-1, but each undirected
    # edge appears twice. To make unique edges, use only those with v < v^bit.
    # Instead, simpler: edge id = v*d + i, but we treat (v,i) and (v^bit, i) as same.
    # For SA we work with the 448 unique undirected edges.
    edges = []            # list of (u, v) with u < v
    edge_idx = {}         # (u,v) -> edge id
    for v in range(N):
        for i in range(d):
            w = v ^ (1 << i)
            if v < w:
                edge_idx[(v, w)] = len(edges)
                edges.append((v, w))
    M = len(edges)        # 448

    # Build faces: a 2D face is defined by (corner v, bits i, j) with i < j
    # The four edges of the face: (v, v^i), (v, v^j), (v^i, v^i^j), (v^j, v^i^j)
    # Each undirected edge belongs to exactly (d-2) = 5 faces.
    # We build face -> list of 4 edge-ids.
    faces = []            # list of list of 4 edge ids
    edge_faces = [[] for _ in range(M)]  # edge id -> list of face ids
    for v in range(N):
        for i in range(d):
            for j in range(i + 1, d):
                a = v ^ (1 << i)
                b = v ^ (1 << j)
                c = v ^ (1 << i) ^ (1 << j)
                e1 = edge_idx[(v, a)]
                e2 = edge_idx[(v, b)]
                e3 = edge_idx[(a, c)]
                e4 = edge_idx[(b, c)]
                fid = len(faces)
                faces.append([e1, e2, e3, e4])
                for e in (e1, e2, e3, e4):
                    edge_faces[e].append(fid)
    F = len(faces)        # 672

    # face_count[f] = number of selected edges in face f. Valid iff all <= 3.
    # selected[e] boolean.
    # Greedy init
    selected = np.zeros(M, dtype=np.bool_)
    face_count = np.zeros(F, dtype=np.int32)

    # Fast validity check for adding/removing edge e given current face_count
    def can_add(e, fc, sel):
        if sel[e]:
            return True
        for f in edge_faces[e]:
            if fc[f] == 3:
                return False
        return True

    # Greedy fill: iterate edges, add if possible. Do a few passes.
    order = list(range(M))
    random.shuffle(order)
    for _ in range(3):
        for e in order:
            if not selected[e] and can_add(e, face_count, selected):
                selected[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1

    def current_count():
        return int(selected.sum())

    best_selected = selected.copy()
    best_fc = face_count.copy()
    best_val = current_count()
    cur_val = best_val

    rng = random
    rng.seed()

    T0 = 1.0
    Tend = 0.02
    # We'll anneal over wall-clock time.

    def accept(delta, T):
        if delta >= 0:
            return True
        return rng.random() < (1.0 - np.exp(delta / T))

    # Precompute move structure: for each edge, its faces and the 3 other edges per face
    # For a swap we need to be efficient. We'll do simple add/remove/swap.

    while time.time() < DEADLINE:
        T = max(Tend, T0 * 0.995 ** ( (time.time() - start) * 30 ))
        # Actually use a time-based cooling: T = Tend + (T0-Tend)*exp(-k*(t-start))
        elapsed = time.time() - start
        T = Tend + (T0 - Tend) * np.exp(-0.05 * elapsed)
        T = max(T, Tend)

        move = rng.random()
        if move < 0.45:
            # Add a random edge
            e = rng.randrange(M)
            if selected[e]:
                continue
            if not can_add(e, face_count, selected):
                continue
            # Apply
            selected[e] = True
            for f in edge_faces[e]:
                face_count[f] += 1
            new_val = cur_val + 1
            if new_val > best_val:
                best_val = new_val
                best_selected = selected.copy()
                best_fc = face_count.copy()
            cur_val = new_val
            # Always accept (delta +1)
        elif move < 0.90:
            # Remove a random selected edge
            # Build list of selected edges? Use a reservoir-free approach: try random.
            if cur_val == 0:
                continue
            for _ in range(50):
                e = rng.randrange(M)
                if selected[e]:
                    break
            else:
                continue
            selected[e] = False
            for f in edge_faces[e]:
                face_count[f] -= 1
            new_val = cur_val - 1
            if accept(new_val - cur_val, T):
                cur_val = new_val
            else:
                # revert
                selected[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
        else:
            # Swap: remove selected e1, add non-selected e2
            # Try random pair
            for _ in range(30):
                e1 = rng.randrange(M)
                if not selected[e1]:
                    continue
                # remove e1 tentatively
                selected[e1] = False
                for f in edge_faces[e1]:
                    face_count[f] -= 1
                # now try to add a random e2
                for __ in range(20):
                    e2 = rng.randrange(M)
                    if selected[e2]:
                        continue
                    if can_add(e2, face_count, selected):
                        break
                else:
                    # revert removal
                    selected[e1] = True
                    for f in edge_faces[e1]:
                        face_count[f] += 1
                    continue
                selected[e2] = True
                for f in edge_faces[e2]:
                    face_count[f] += 1
                new_val = cur_val  # same count
                # delta 0, always accept for diversity
                cur_val = new_val
                break

    # Final: return best
    out = []
    for e in range(M):
        if best_selected[e]:
            u, v = edges[e]
            out.append([u, v])
    return out
