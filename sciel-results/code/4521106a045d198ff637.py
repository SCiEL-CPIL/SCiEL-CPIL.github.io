import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 110.0

    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx

    m = len(edges)

    face_edges = []
    edge_faces = [[] for _ in range(m)]

    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)
                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v0, v2), max(v0, v2))]
                e2 = edge_idx[(min(v1, v3), max(v1, v3))]
                e3 = edge_idx[(min(v2, v3), max(v2, v3))]
                fid = len(face_edges)
                face_edges.append((e0, e1, e2, e3))
                for ei in (e0, e1, e2, e3):
                    edge_faces[ei].append(fid)

    num_faces = len(face_edges)
    edge_faces_arr = [np.array(f, dtype=np.int32) for f in edge_faces]

    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        selected = np.zeros(m, dtype=bool)
        face_count = np.zeros(num_faces, dtype=np.int32)
        for ei in order:
            fids = edge_faces_arr[ei]
            if np.all(face_count[fids] < 3):
                selected[ei] = True
                face_count[fids] += 1
        return selected, face_count

    def complete_1flip(selected, face_count):
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                if selected[ei]:
                    continue
                fids = edge_faces_arr[ei]
                if np.all(face_count[fids] < 3):
                    selected[ei] = True
                    face_count[fids] += 1
                    changed = True
        return selected, face_count

    def try_2swap(selected, face_count, seed, max_attempts=100):
        rng = random.Random(seed)
        sel_list = np.where(selected)[0].tolist()
        unsel_list = np.where(~selected)[0].tolist()
        if not sel_list or not unsel_list:
            return selected, face_count, 0

        rng.shuffle(sel_list)
        rng.shuffle(unsel_list)

        best_gain = 0
        best_state = None

        for ri in sel_list[:max_attempts]:
            if time.time() >= deadline:
                break
            selected[ri] = False
            fids_ri = edge_faces_arr[ri]
            face_count[fids_ri] -= 1

            added = []
            for ai in unsel_list:
                if ai == ri:
                    continue
                fids_ai = edge_faces_arr[ai]
                if np.all(face_count[fids_ai] < 3):
                    selected[ai] = True
                    face_count[fids_ai] += 1
                    added.append(ai)

            gain = len(added) - 1
            if gain > best_gain:
                best_gain = gain
                best_state = (ri, added)

            selected[ri] = True
            face_count[fids_ri] += 1
            for ai in added:
                selected[ai] = False
                face_count[edge_faces_arr[ai]] -= 1

        if best_state is not None:
            ri, added = best_state
            selected[ri] = False
            face_count[edge_faces_arr[ri]] -= 1
            for ai in added:
                selected[ai] = True
                face_count[edge_faces_arr[ai]] += 1

        return selected, face_count, best_gain

    def count_sel(selected):
        return int(np.sum(selected))

    best_selected = None
    best_count = 0

    for seed in range(5000):
        if time.time() >= deadline - 5.0:
            break

        selected, face_count = greedy_random(seed)
        selected, face_count = complete_1flip(selected, face_count)
        cnt = count_sel(selected)

        if cnt > best_count:
            best_selected = selected.copy()
            best_count = cnt

        if time.time() < deadline - 10.0 and cnt < 304:
            selected, face_count, gain = try_2swap(selected, face_count, seed + 100000)
            if gain > 0:
                selected, face_count = complete_1flip(selected, face_count)
                cnt = count_sel(selected)
                if cnt > best_count:
                    best_selected = selected.copy()
                    best_count = cnt

        if best_count >= 304:
            break

    out = []
    for ei in range(m):
        if best_selected[ei]:
            x, y = edges[ei]
            out.append([x, y])

    return out
# EVOLVE-BLOCK-END
