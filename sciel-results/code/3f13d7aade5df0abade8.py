import time
import numpy as np


def construct():
    start = time.time()
    deadline = start + 118.0

    d = 7
    n = 1 << d  # 128
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
    E = len(edges)

    u_arr = np.array([e[0] for e in edges], dtype=np.int32)
    v_arr = np.array([e[1] for e in edges], dtype=np.int32)

    edge_map = {}
    for idx, (a, b) in enumerate(edges):
        edge_map[(a, b)] = idx
    face_tuples = []
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                if (u >> i) & 1 or (u >> j) & 1:
                    continue
                u1 = u ^ (1 << i)
                u2 = u ^ (1 << j)
                u12 = u ^ (1 << i) ^ (1 << j)
                e1 = (min(u, u1), max(u, u1))
                e2 = (min(u1, u12), max(u1, u12))
                e3 = (min(u12, u2), max(u12, u2))
                e4 = (min(u2, u), max(u2, u))
                face_tuples.append((e1, e2, e3, e4))
    F = len(face_tuples)
    face_eidx = np.zeros((F, 4), dtype=np.int32)
    for f, (e1, e2, e3, e4) in enumerate(face_tuples):
        face_eidx[f] = [edge_map[e1], edge_map[e2], edge_map[e3], edge_map[e4]]
    edge_faces = [[] for _ in range(E)]
    for f in range(F):
        for e in face_eidx[f]:
            edge_faces[int(e)].append(f)
    face_deg = np.array([len(edge_faces[e]) for e in range(E)], dtype=np.int32)

    def unhit_count(blue):
        b4 = blue[face_eidx]
        return int(np.count_nonzero((~b4).all(axis=1)))

    def fill_red(red):
        changed = True
        while changed:
            changed = False
            frc = np.bincount(face_eidx.ravel(), minlength=E).reshape(F, 4).sum(axis=1)
            bad = set()
            for f in np.where(frc == 3)[0]:
                for e in face_eidx[f]:
                    bad.add(int(e))
            for e in range(E):
                if not red[e] and e not in bad:
                    red[e] = True
                    changed = True
        return red

    # ---- structural seed: valid red set, then fill ----
    red = np.ones(E, dtype=bool)
    for f in range(F):
        es = face_eidx[f]
        if red[es].all():
            order = es[np.argsort(face_deg[es])]
            for e in order:
                if red[e]:
                    red[e] = False
                    break
    red = fill_red(red)
    blue = ~red
    best_blue = blue.copy()
    best_size = int(np.count_nonzero(red))

    # ---- simulated annealing over blue set ----
    rng = np.random.default_rng(20240711)
    blue = blue.copy()
    blue_count = int(np.count_nonzero(blue))
    cur_unhit = unhit_count(blue)
    big = 1000.0
    T = 2.0
    T_min = 0.02
    cool = 0.999
    pool = np.argsort(-face_deg)[:160]
    t_counter = 0
    while time.time() < deadline - 16.0:
        e = int(pool[rng.integers(len(pool))])
        new_blue = blue.copy()
        new_blue[e] = not blue[e]
        new_blue_count = blue_count + (1 if not blue[e] else -1)
        new_unhit = unhit_count(new_blue)
        d_obj = (new_blue_count + big * new_unhit) - (blue_count + big * cur_unhit)
        if d_obj <= 0 or rng.random() < np.exp(-d_obj / T):
            blue = new_blue
            blue_count = new_blue_count
            cur_unhit = new_unhit
            cur_size = E - blue_count
            if cur_size > best_size:
                best_size = cur_size
                best_blue = blue.copy()
        T = max(T * cool, T_min)
        t_counter += 1
        if t_counter % 30 == 0:
            red_now = ~blue
            red_now = fill_red(red_now)
            blue_now = ~red_now
            blue_count = int(np.count_nonzero(blue_now))
            cur_unhit = unhit_count(blue_now)
            cur_size = E - blue_count
            if cur_size > best_size:
                best_size = cur_size
                best_blue = blue_now.copy()
            blue = blue_now

    # ---- final polish on best_blue ----
    best_blue = best_blue.copy()
    improved = True
    while improved:
        improved = False
        for e in np.where(best_blue)[0]:
            trial = best_blue.copy()
            trial[e] = False
            if unhit_count(trial) == 0:
                best_blue[e] = False
                improved = True
    red = ~best_blue
    red = fill_red(red)
    frc = np.bincount(face_eidx.ravel(), minlength=E).reshape(F, 4).sum(axis=1)
    assert (frc < 4).all(), "invalid: a face has all 4 red edges"

    out = []
    for idx in range(E):
        if red[idx]:
            out.append([int(u_arr[idx]), int(v_arr[idx])])
    return out
