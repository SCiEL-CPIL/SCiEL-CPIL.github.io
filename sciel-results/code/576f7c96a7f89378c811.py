import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute K(2,2,2) configurations
    configs = []
    triple_to_configs = [[] for _ in range(num_triples)]

    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]

        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append(tuple(sorted((x, y, z))))

            trans_indices = set()
            for t in transversals:
                trans_indices.add(triple_to_idx[t])

            config_idx = len(configs)
            configs.append(trans_indices)

            for idx in trans_indices:
                triple_to_configs[idx].append(config_idx)

    num_configs = len(configs)

    # Greedy hitting set
    def greedy_hitting_set(seed=None):
        if seed is not None:
            random.seed(seed)

        config_active = [True] * num_configs
        triple_hits = [0] * num_triples

        for i in range(num_triples):
            for c in triple_to_configs[i]:
                triple_hits[i] += 1

        removed = [False] * num_triples
        removed_count = 0

        while True:
            # Find triple with max hits among non-removed
            max_hits = 0
            candidates = []
            for i in range(num_triples):
                if not removed[i] and triple_hits[i] > max_hits:
                    max_hits = triple_hits[i]
                    candidates = [i]
                elif not removed[i] and triple_hits[i] == max_hits:
                    candidates.append(i)

            if max_hits == 0:
                break

            if seed is not None:
                best_triple = random.choice(candidates)
            else:
                best_triple = candidates[0]

            removed[best_triple] = True
            removed_count += 1

            for c in triple_to_configs[best_triple]:
                if config_active[c]:
                    config_active[c] = False
                    for t in configs[c]:
                        if t != best_triple and not removed[t]:
                            triple_hits[t] -= 1

        selected = [not r for r in removed]
        return selected, num_triples - removed_count

    # Local search: try to add back removed triples if safe
    def local_search(selected, size):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False

            # Try to add removed triples
            for t in range(num_triples):
                if not selected[t]:
                    # Check if adding t creates any violation
                    safe = True
                    for c in triple_to_configs[t]:
                        cnt = 0
                        for u in configs[c]:
                            if u != t and selected[u]:
                                cnt += 1
                        if cnt == 7:
                            safe = False
                            break
                    if safe:
                        selected[t] = True
                        size += 1
                        improved = True

        return selected, size

    # Run greedy
    selected, size = greedy_hitting_set()
    selected, size = local_search(selected, size)

    best_selected, best_size = selected[:], size

    # Try multiple random restarts
    while time.time() < deadline - 10:
        seed = random.randint(0, 1000000)
        sel, sz = greedy_hitting_set(seed)
        sel, sz = local_search(sel, sz)

        if sz > best_size:
            best_size = sz
            best_selected = sel[:]

        if time.time() > deadline - 15:
            break

    result = []
    for i in range(num_triples):
        if best_selected[i]:
            result.append(list(all_triples[i]))

    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
