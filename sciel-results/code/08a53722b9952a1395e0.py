import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_cycles(n):
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles.append(cycle_edges)
        return cycles

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)

        cycles = build_cycles(n)
        num_cycles = len(cycles)

        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        # Build edge-to-cycles mapping
        edge_to_cycles = [[] for _ in range(m)]
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)

        # Represent cycles as numpy array
        cycles_np = np.array(cycles, dtype=np.int32)

        # Initial random coloring
        colors = np.array([random.randint(0, 3) for _ in range(m)], dtype=np.int8)

        def count_bad():
            c0 = cycles_np[:, 0]
            mono = (colors[c0] == colors[cycles_np[:, 1]]) & \
                   (colors[c0] == colors[cycles_np[:, 2]]) & \
                   (colors[c0] == colors[cycles_np[:, 3]]) & \
                   (colors[c0] == colors[cycles_np[:, 4]]) & \
                   (colors[c0] == colors[cycles_np[:, 5]])
            return int(mono.sum())

        bad_count = count_bad()
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        def compute_edge_weights(colors_arr):
            weights = np.zeros(m, dtype=np.int32)
            for ci in range(num_cycles):
                ce = cycles_np[ci]
                c0 = colors_arr[ce[0]]
                cnt = 0
                for k in range(1, 6):
                    if colors_arr[ce[k]] == c0:
                        cnt += 1
                if cnt >= 5:
                    for e in ce:
                        weights[e] += 1
            return weights

        max_iter = 300000
        start_iter_time = time.time()
        stuck_counter = 0
        kick_threshold = 5000

        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break

            if stuck_counter >= kick_threshold:
                kick_size = random.randint(5, 10)
                for _ in range(kick_size):
                    e = random.randint(0, m - 1)
                    colors[e] = random.randint(0, 3)
                bad_count = count_bad()
                stuck_counter = 0
                continue

            weights = compute_edge_weights(colors)
            max_w = weights.max()

            if max_w == 0:
                e = random.randint(0, m - 1)
            else:
                top_edges = np.where(weights == max_w)[0]
                e = int(top_edges[random.randint(0, len(top_edges) - 1)])

            c_old = colors[e]
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)

            colors[e] = c_new
            new_bad = count_bad()

            if new_bad < bad_count:
                bad_count = new_bad
                stuck_counter = 0
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors.tolist()}
            else:
                if random.random() < 0.1:
                    bad_count = new_bad
                    stuck_counter += 1
                else:
                    colors[e] = c_old
                    stuck_counter += 1

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    time_alloc = {17: 25, 16: 20, 15: 15, 14: 12, 13: 10, 12: 8, 11: 7, 10: 5, 9: 3, 8: 3, 7: 2, 6: 2}

    for n in targets:
        if time.time() > deadline - 8:
            break

        tb = time_alloc.get(n, 5)
        num_restarts = 2 if n >= 15 else 1
        restart_budget = tb / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 8:
                break
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                best_result = result
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
