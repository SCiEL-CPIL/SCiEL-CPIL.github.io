"""
Maximize number of triples in a 3-uniform hypergraph on 16 vertices
such that no three disjoint pairs have all 8 transversal triples present.

Strategy (Max-SAT / WalkSAT-style with conflict-driven moves):
1. Precompute all K(2,2,2) configurations and for each triple, the list of
   configs it participates in.
2. Greedy random initialization.
3. WalkSAT-style local search with:
   - Conflict-driven move selection: when no add is possible, prefer to
     remove a triple that participates in many tight (count==7) configs,
     then try to add a triple that was previously blocked.
   - Fluctuation phases: periodically remove random triples to escape local optima.
   - 2-for-1 and 1-for-2 swaps when available to increase count.
4. Multiple restarts within a 115s deadline.
"""

import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560

    # Precompute all K(2,2,2) configs
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_total)]

    for six_vertices in itertools.combinations(range(N), 6):
        a, b, c, d, e, f = six_vertices
        partitions = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            trans_indices = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(trans_indices)
            config_idx = len(k222_configs) - 1
            for ti in trans_indices:
                triple_to_k222[ti].append(config_idx)

    num_k222 = len(k222_configs)

    def time_check():
        return time.time() >= deadline

    best_result = []
    best_count = 0

    def can_add_fast(ti, k222_count):
        for ci in triple_to_k222[ti]:
            if k222_count[ci] >= 7:
                return False
        return True

    def do_add(ti, present, k222_count):
        present[ti] = True
        for ci in triple_to_k222[ti]:
            k222_count[ci] += 1

    def do_remove(ti, present, k222_count):
        present[ti] = False
        for ci in triple_to_k222[ti]:
            k222_count[ci] -= 1

    def greedy_init(rng):
        present = [False] * num_total
        k222_count = [0] * num_k222
        count = 0
        order = list(range(num_total))
        rng.shuffle(order)
        for ti in order:
            if can_add_fast(ti, k222_count):
                do_add(ti, present, k222_count)
                count += 1
        return present, k222_count, count

    def local_search(present, k222_count, count, rng, max_steps=200000):
        steps = 0
        no_improve = 0
        max_no_improve = 12000
        fluctuate_every = 4000

        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break

            # Periodic fluctuation
            if steps > 0 and steps % fluctuate_every == 0:
                num_remove = rng.randint(4, 10)
                present_indices = [i for i in range(num_total) if present[i]]
                if len(present_indices) > num_remove:
                    to_remove = rng.sample(present_indices, num_remove)
                    for ti in to_remove:
                        do_remove(ti, present, k222_count)
                        count -= 1
                no_improve = 0
                steps += 1
                continue

            moved = False

            # 1. Try to add a triple
            for _ in range(60):
                ti = rng.randint(0, num_total - 1)
                if present[ti]:
                    continue
                if can_add_fast(ti, k222_count):
                    do_add(ti, present, k222_count)
                    count += 1
                    moved = True
                    no_improve = 0
                    break

            if moved:
                steps += 1
                continue

            # 2. Conflict-driven swap: remove a triple that blocks many adds
            #    Prefer removing triples that are in many tight configs
            swapped = False
            for _ in range(80):
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue

                # Score: number of tight configs this triple is in
                tight_count = sum(1 for ci in triple_to_k222[ti_remove] if k222_count[ci] == 7)
                if tight_count == 0 and rng.random() < 0.5:
                    continue  # skip if not helpful

                do_remove(ti_remove, present, k222_count)
                count -= 1

                # Try to add up to 2 triples (1-for-1 or 1-for-2)
                added_count = 0
                added_list = []
                for _ in range(200):
                    ti_add = rng.randint(0, num_total - 1)
                    if present[ti_add]:
                        continue
                    if can_add_fast(ti_add, k222_count):
                        do_add(ti_add, present, k222_count)
                        count += 1
                        added_count += 1
                        added_list.append(ti_add)
                        if added_count >= 2:
                            break

                if added_count >= 1:
                    swapped = True
                    no_improve = 0
                else:
                    do_add(ti_remove, present, k222_count)
                    count += 1

            if swapped:
                steps += 1
                continue

            # 3. Try 2-for-1: remove two, add one (net -1, but may unblock future)
            #    Actually this decreases count, so only do it during fluctuation
            #    Skip for now; rely on fluctuation instead

            no_improve += 1
            steps += 1

        return present, count

    rng = random.Random(42)

    for restart in range(500):
        if time_check():
            break

        present, k222_count, count = greedy_init(rng)

        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]

        present, count = local_search(present, k222_count, count, rng)

        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]

        if time_check():
            break

    if not best_result:
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]

    return best_result
