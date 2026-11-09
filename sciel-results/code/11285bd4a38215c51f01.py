import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 85

    edge_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = len(edge_idx)

    four_cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    four_cycles.append([edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(d,a)]])
                    four_cycles.append([edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(d,c)], edge_idx[(c,a)]])
                    four_cycles.append([edge_idx[(a,c)], edge_idx[(c,b)], edge_idx[(b,d)], edge_idx[(d,a)]])

    four_cycles = np.array(four_cycles, dtype=np.int32)
    num_cycles = len(four_cycles)

    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(num_cycles):
        for e in four_cycles[ci]:
            edge_to_cycles[int(e)].append(ci)

    best_colors = None
    best_count = 0

    for restart in range(300):
        if time.time() > deadline:
            break

        random.seed(1234 + restart)
        np.random.seed(1234 + restart)
        colors = np.full(NUM_EDGES, -1, dtype=np.int32)
        used_colors = set()

        edge_order = np.random.permutation(NUM_EDGES)

        success = True
        for e in edge_order:
            e = int(e)
            if time.time() > deadline:
                success = False
                break

            assigned = False

            for c in range(119, -1, -1):
                if c in used_colors:
                    continue
                colors[e] = c
                valid = True
                for ci in edge_to_cycles[e]:
                    cyc = four_cycles[ci]
                    if np.any(colors[cyc] == -1):
                        continue
                    if len(set(colors[cyc].tolist())) == 4:
                        valid = False
                        break
                if valid:
                    used_colors.add(c)
                    assigned = True
                    break

            if not assigned:
                for c in sorted(used_colors, reverse=True):
                    colors[e] = c
                    valid = True
                    for ci in edge_to_cycles[e]:
                        cyc = four_cycles[ci]
                        if np.any(colors[cyc] == -1):
                            continue
                        if len(set(colors[cyc].tolist())) == 4:
                            valid = False
                            break
                    if valid:
                        assigned = True
                        break

            if not assigned:
                for c in range(120):
                    colors[e] = c
                    valid = True
                    for ci in edge_to_cycles[e]:
                        cyc = four_cycles[ci]
                        if np.any(colors[cyc] == -1):
                            continue
                        if len(set(colors[cyc].tolist())) == 4:
                            valid = False
                            break
                    if valid:
                        assigned = True
                        break

            if not assigned:
                success = False
                break

        if success and not np.any(colors == -1):
            cnt = len(set(colors.tolist()))
            if cnt > best_count:
                best_count = cnt
                best_colors = colors.copy()

    if best_colors is None:
        return [0] * NUM_EDGES

    return list(best_colors)

