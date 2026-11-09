import time
import random
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    def elapsed():
        return time.time() - t0

    N = 30
    BOUND = 30

    def doubled_area(p, q, r):
        return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

    def full_area_counts(pts):
        counts = defaultdict(int)
        n = len(pts)
        for i in range(n):
            pi = pts[i]
            for j in range(i+1, n):
                pj = pts[j]
                for k in range(j+1, n):
                    a = doubled_area(pi, pj, pts[k])
                    if a > 0:
                        counts[a] += 1
        return counts

    def max_count(counts):
        return max(counts.values()) if counts else 0

    def dominant_area(counts):
        if not counts:
            return 1, 0
        return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))

    # ---------- Structured seeds ----------
    def seed_two_lines():
        return [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]

    def seed_grid(a, b):
        return [(x, y) for x in range(a) for y in range(b)]

    def seed_three_lines():
        return [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]

    def seed_five_lines():
        return [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + \
               [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + \
               [(x, 4) for x in range(6)]

    def seed_random(s):
        rng = random.Random(s)
        pts = []
        used = set()
        while len(pts) < N:
            x = rng.randint(-BOUND, BOUND)
            y = rng.randint(-BOUND, BOUND)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        return pts

    def seed_two_lines_spaced():
        return [(x, 0) for x in range(15)] + [(x, 2) for x in range(15)]

    seeds = [
        seed_two_lines(),
        seed_grid(6, 5),
        seed_grid(5, 6),
        seed_grid(10, 3),
        seed_grid(3, 10),
        seed_grid(15, 2),
        seed_grid(2, 15),
        seed_three_lines(),
        seed_five_lines(),
        seed_two_lines_spaced(),
        seed_random(1),
        seed_random(2),
        seed_random(3),
        seed_random(4),
    ]

    best_points = None
    best_score = 0

    for s in seeds:
        c = full_area_counts(s)
        sc = max_count(c)
        if sc > best_score:
            best_score = sc
            best_points = list(s)

    # ---------- Local search ----------
    def local_search(start_pts, stop_time, seed):
        rng = random.Random(seed)
        pts = list(start_pts)
        counts = full_area_counts(pts)
        cur_max = max_count(counts)
        target, target_count = dominant_area(counts)

        best_local = cur_max
        best_local_pts = list(pts)

        pairs_for = [[] for _ in range(N)]
        for idx in range(N):
            others = [j for j in range(N) if j != idx]
            for a in range(len(others)):
                for b in range(a+1, len(others)):
                    pairs_for[idx].append((others[a], others[b]))

        def apply_move(idx, new_pt):
            old_pt = pts[idx]
            if new_pt == old_pt:
                return False
            for j in range(N):
                if j != idx and pts[j] == new_pt:
                    return False
            if not (-BOUND <= new_pt[0] <= BOUND and -BOUND <= new_pt[1] <= BOUND):
                return False
            for (i, j) in pairs_for[idx]:
                a_old = doubled_area(pts[i], old_pt, pts[j])
                a_new = doubled_area(pts[i], new_pt, pts[j])
                if a_old > 0:
                    counts[a_old] -= 1
                    if counts[a_old] == 0:
                        del counts[a_old]
                if a_new > 0:
                    counts[a_new] = counts.get(a_new, 0) + 1
            pts[idx] = new_pt
            return True

        temp = 2.0
        cool = 0.99995
        while time.time() < stop_time:
            if rng.random() < 0.7:
                idx = rng.randrange(N)
                ox, oy = pts[idx]
                step = rng.randint(1, 4)
                nx = ox + rng.randint(-step, step)
                ny = oy + rng.randint(-step, step)
                new_pt = (nx, ny)
                snap_pts = list(pts)
                snap_counts = dict(counts)
                snap_tc = target_count
                snap_max = cur_max
                snap_target = target
                if apply_move(idx, new_pt):
                    target_count = counts.get(target, 0)
                    new_max = max_count(counts)
                    if new_max > cur_max:
                        cur_max = new_max
                        target, target_count = dominant_area(counts)
                    delta = target_count - snap_tc
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.4
                    else:
                        accept = rng.random() < np.exp(delta / temp)
                    if not accept:
                        pts = snap_pts
                        counts = snap_counts
                        target_count = snap_tc
                        cur_max = snap_max
                        target = snap_target
            else:
                i, j = rng.sample(range(N), 2)
                snap_pts = list(pts)
                snap_counts = dict(counts)
                snap_tc = target_count
                snap_max = cur_max
                snap_target = target
                ox, oy = pts[i]
                px, py = pts[j]
                step = rng.randint(1, 3)
                ni = (ox + rng.randint(-step, step), oy + rng.randint(-step, step))
                nj = (px + rng.randint(-step, step), py + rng.randint(-step, step))
                if apply_move(i, ni) and apply_move(j, nj):
                    target_count = counts.get(target, 0)
                    new_max = max_count(counts)
                    if new_max > cur_max:
                        cur_max = new_max
                        target, target_count = dominant_area(counts)
                    delta = target_count - snap_tc
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.4
                    else:
                        accept = rng.random() < np.exp(delta / temp)
                else:
                    accept = False
                if not accept:
                    pts = snap_pts
                    counts = snap_counts
                    target_count = snap_tc
                    cur_max = snap_max
                    target = snap_target

            temp *= cool
            if temp < 0.05:
                temp = 2.0
            if cur_max > best_local:
                best_local = cur_max
                best_local_pts = list(pts)
        return best_local_pts, best_local

    # Multiple restarts
    n_restarts = 8
    for r in range(n_restarts):
        if elapsed() > DEADLINE - 8:
            break
        if r == 0:
            start_pts = list(best_points)
        else:
            rngj = random.Random(1000 + r)
            start_pts = list(best_points)
            for _ in range(8):
                idx = rngj.randrange(N)
                ox, oy = start_pts[idx]
                step = rngj.randint(1, 4)
                nx = ox + rngj.randint(-step, step)
                ny = oy + rngj.randint(-step, step)
                if -BOUND <= nx <= BOUND and -BOUND <= ny <= BOUND and (nx, ny) not in start_pts:
                    start_pts[idx] = (nx, ny)

        stop_time = t0 + (DEADLINE - 8 - elapsed())
        new_pts, new_score = local_search(start_pts, stop_time, seed=2000 + r)
        if new_score > best_score:
            best_score = new_score
            best_points = new_pts

    # Final verification
    final_counts = full_area_counts(best_points)
    final_score = max_count(final_counts)
    if final_score > best_score:
        best_score = final_score

    return best_points
