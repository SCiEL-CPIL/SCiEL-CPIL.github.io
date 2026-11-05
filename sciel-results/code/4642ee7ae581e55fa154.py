N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    import time
    import random
    start = time.time()
    DEADLINE = 85.0

    # Build edges in lexicographic order
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    assert len(edges) == NUM_EDGES

    edge_to_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx

    # Precompute for each edge, the list of (ek, el, ei) triples forming 4-cycles
    cycle_constraints = [[] for _ in range(NUM_EDGES)]
    for idx, (i, j) in enumerate(edges):
        others = [v for v in range(N) if v != i and v != j]
        for k in others:
            for l in others:
                if k == l:
                    continue
                ek = edge_to_idx.get((j, k))
                el = edge_to_idx.get((k, l))
                ei = edge_to_idx.get((l, i))
                if ek is not None and el is not None and ei is not None:
                    cycle_constraints[idx].append((ek, el, ei))

    def count_distinct(colors):
        return len(set(colors))

    def is_valid(colors):
        for idx in range(NUM_EDGES):
            for (ek, el, ei) in cycle_constraints[idx]:
                if ek < idx and el < idx and ei < idx:
                    if len({colors[idx], colors[ek], colors[el], colors[ei]}) == 4:
                        return False
        return True

    def greedy_color(prefer_new_prob=0.5, rng=None):
        colors = [0] * NUM_EDGES
        used_colors = set()
        next_new = 0
        if rng is None:
            rng = random

        for idx in range(NUM_EDGES):
            if rng.random() < prefer_new_prob:
                c = next_new
                valid = True
                for (ek, el, ei) in cycle_constraints[idx]:
                    if ek < idx and el < idx and ei < idx:
                        if len({c, colors[ek], colors[el], colors[ei]}) == 4:
                            valid = False
                            break
                if valid:
                    colors[idx] = c
                    used_colors.add(c)
                    next_new += 1
                    continue

            candidates = list(used_colors)
            rng.shuffle(candidates)
            for c in candidates:
                valid = True
                for (ek, el, ei) in cycle_constraints[idx]:
                    if ek < idx and el < idx and ei < idx:
                        if len({c, colors[ek], colors[el], colors[ei]}) == 4:
                            valid = False
                            break
                if valid:
                    colors[idx] = c
                    break
            else:
                colors[idx] = 0
        return colors

    def local_improve(colors):
        improved = True
        while improved:
            improved = False
            for idx in range(NUM_EDGES):
                used = set(colors)
                for new_c in range(120):
                    if new_c in used:
                        continue
                    valid = True
                    for (ek, el, ei) in cycle_constraints[idx]:
                        if ek < idx and el < idx and ei < idx:
                            if len({new_c, colors[ek], colors[el], colors[ei]}) == 4:
                                valid = False
                                break
                    if valid:
                        colors[idx] = new_c
                        improved = True
                        break
        return colors

    # First greedy with high new-color preference
    best_colors = greedy_color(prefer_new_prob=1.0)
    best_colors = local_improve(best_colors)
    best_count = count_distinct(best_colors)

    # Random restarts
    random.seed(12345)
    while time.time() < DEADLINE:
        colors = greedy_color(prefer_new_prob=0.3, rng=random)
        if is_valid(colors):
            colors = local_improve(colors)
            cnt = count_distinct(colors)
            if cnt > best_count:
                best_count = cnt
                best_colors = colors

    return best_colors
