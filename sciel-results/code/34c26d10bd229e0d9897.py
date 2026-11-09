import time
import random
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 88.0

    def elapsed():
        return time.time() - t0

    N = 30
    pair_i = []
    pair_j = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = np.array(pair_i, dtype=np.int32)
    pair_j = np.array(pair_j, dtype=np.int32)
    M = len(pair_i)

    point_to_pairs = defaultdict(list)
    for k in range(M):
        point_to_pairs[pair_i[k]].append(k)
        point_to_pairs[pair_j[k]].append(k)

    uv_pairs = []
    for u in range(M):
        a = pair_i[u]
        b = pair_j[u]
        for v in point_to_pairs[a]:
            if v != u and v > u:
                uv_pairs.append((u, v))
        for v in point_to_pairs[b]:
            if v != u and v > u:
                uv_pairs.append((u, v))

    uv_u = np.array([x[0] for x in uv_pairs], dtype=np.int32)
    uv_v = np.array([x[1] for x in uv_pairs], dtype=np.int32)
    K = len(uv_u)

    def compute_cross(P):
        pair_diff = P[pair_j] - P[pair_i]
        dx = pair_diff[:, 0].astype(np.int64)
        dy = pair_diff[:, 1].astype(np.int64)
        cu = dx[uv_u]
        cv = dy[uv_u]
        du = dx[uv_v]
        dv = dy[uv_v]
        return cu * dv - cv * du

    def build_counter(cross_vals):
        nz = cross_vals[cross_vals != 0]
        if len(nz) == 0:
            return {}
        abs_nz = np.abs(nz)
        vals, cnts = np.unique(abs_nz, return_counts=True)
        return {int(v): int(c) // 3 for v, c in zip(vals, cnts)}

    def score_of(counter):
        return max(counter.values()) if counter else 0

    candidates = []
    c1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    candidates.append(c1)
    c2 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(c2)
    c3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    candidates.append(c3)
    c5 = [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + [(x, 4) for x in range(6)]
    candidates.append(c5)
    c6 = [(x, 0) for x in range(15)] + [(x, 2) for x in range(15)]
    candidates.append(c6)
    c7 = [(x, 0) for x in range(10)] + [(x, 2) for x in range(10)] + [(x, 4) for x in range(10)]
    candidates.append(c7)
    random.seed(42)
    c4 = []
    used = set()
    while len(c4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            c4.append((x, y))
    candidates.append(c4)

    best_points = None
    best_score = 0
    best_cross = None

    for cand in candidates:
        P = np.array(cand, dtype=np.int64)
        cross = compute_cross(P)
        counter = build_counter(cross)
        s = score_of(counter)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            best_cross = cross

    current_P = np.array(best_points, dtype=np.int64)
    current_cross = best_cross.copy()
    current_score = best_score

    random.seed(123)
    no_improve = 0
    step_size = 4

    while elapsed() < DEADLINE - 3.0:
        r = random.random()
        if r < 0.7:
            idx = random.randint(0, 29)
            old_x, old_y = int(current_P[idx, 0]), int(current_P[idx, 1])
            if random.random() < 0.05:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            elif random.random() < 0.2:
                new_x = old_x + random.randint(-6, 6)
                new_y = old_y + random.randint(-6, 6)
            else:
                new_x = old_x + random.randint(-step_size, step_size)
                new_y = old_y + random.randint(-step_size, step_size)
            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue
            new_pt = (new_x, new_y)
            if new_pt == (old_x, old_y):
                continue
            is_valid = True
            for j in range(30):
                if j != idx and current_P[j, 0] == new_x and current_P[j, 1] == new_y:
                    is_valid = False
                    break
            if not is_valid:
                continue
            new_P = current_P.copy()
            new_P[idx] = [new_x, new_y]
            new_cross = compute_cross(new_P)
            counter = build_counter(new_cross)
            new_score = score_of(counter)
            if new_score > current_score:
                current_P = new_P
                current_cross = new_cross
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = [tuple(current_P[i]) for i in range(30)]
                    best_cross = new_cross.copy()
                if step_size > 1 and random.random() < 0.15:
                    step_size = max(1, step_size - 1)
            else:
                no_improve += 1
        else:
            idx1 = random.randint(0, 29)
            idx2 = random.randint(0, 29)
            if idx1 == idx2:
                continue
            new_P = current_P.copy()
            new_P[idx1] = current_P[idx2]
            new_P[idx2] = current_P[idx1]
            new_cross = compute_cross(new_P)
            counter = build_counter(new_cross)
            new_score = score_of(counter)
            if new_score > current_score:
                current_P = new_P
                current_cross = new_cross
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = [tuple(current_P[i]) for i in range(30)]
                    best_cross = new_cross.copy()
            else:
                no_improve += 1

        if no_improve > 2000:
            current_P = np.array(best_points, dtype=np.int64)
            current_cross = best_cross.copy()
            current_score = best_score
            no_improve = 0
            step_size = random.randint(3, 8)
        if random.random() < 0.003:
            step_size = random.randint(1, 8)

    return best_points
