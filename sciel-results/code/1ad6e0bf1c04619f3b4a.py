import itertools
import random
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560

    # Build the forbidden transversals as lists of triple indices.
    # A transversal = 3 disjoint pairs on 6 points -> 8 transversal triples.
    transversals = []
    seen = set()
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        matchings = set()
        # Generate all 15 perfect matchings on 6 points
        remaining = [1, 2, 3, 4, 5]
        for j in remaining:
            # First pair: (0, j)
            rest = [x for x in remaining if x != j]
            # Pair up the remaining 4: 3 ways
            for k in range(3):
                if k == 0:
                    p2, p3 = (rest[0], rest[1]), (rest[2], rest[3])
                elif k == 1:
                    p2, p3 = (rest[0], rest[2]), (rest[1], rest[3])
                else:
                    p2, p3 = (rest[0], rest[3]), (rest[1], rest[2])
                matching = tuple(sorted(((sp[0], sp[j]), (sp[p2[0]], sp[p2[1]]), (sp[p3[0]], sp[p3[1]]))))
                matchings.add(matching)

        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            triple_indices = []
            for (x, y, z) in ((a, c, e), (a, c, f), (a, d, e), (a, d, f),
                              (b, c, e), (b, c, f), (b, d, e), (b, d, f)):
                triple_indices.append(triple_to_idx[tuple(sorted((x, y, z)))])
            key = tuple(sorted(triple_indices))
            if key not in seen:
                seen.add(key)
                transversals.append(triple_indices)
    num_trans = len(transversals)

    # For each triple, the list of transversals it participates in.
    triple_trans = [[] for _ in range(num_triples)]
    for ti, tlist in enumerate(transversals):
        for idx in tlist:
            triple_trans[idx].append(ti)

    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_trans
        candidates = list(range(num_triples))
        random.shuffle(candidates)

        for idx in candidates:
            safe = True
            for ci in triple_trans[idx]:
                if counts[ci] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for ci in triple_trans[idx]:
                    counts[ci] += 1
        return sol, counts, sum(sol)

    def try_add_candidates(sol, counts, candidates):
        added = 0
        new_candidates = []
        for idx in candidates:
            if sol[idx]:
                continue
            safe = True
            for ci in triple_trans[idx]:
                if counts[ci] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for ci in triple_trans[idx]:
                    counts[ci] += 1
                added += 1
            else:
                new_candidates.append(idx)
        return added, new_candidates

    best_solution = None
    best_count = -1
    best_counts = None

    def polish(sol, counts, current_count):
        """Final high-precision polish: try to add any remaining safe triples,
        then do local swaps to improve count."""
        # Phase 1: Add all safe triples
        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)
        added, _ = try_add_candidates(sol, counts, candidates)
        current_count += added

        # Phase 2: Local search with swaps
        if time.time() > deadline - 2.0:
            return current_count

        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            present = [i for i in range(num_triples) if sol[i]]
            absent = [i for i in range(num_triples) if not sol[i]]
            if not present or not absent:
                break
            random.shuffle(present)
            random.shuffle(absent)

            for rem in present[:30]:
                if time.time() > deadline - 1.0:
                    break
                # Check what we'd free up by removing rem
                freed = []
                for ci in triple_trans[rem]:
                    if counts[ci] == 7:
                        freed.append(ci)
                if not freed:
                    continue

                for add in absent[:30]:
                    # Check if adding add is safe after removing rem
                    safe = True
                    for ci in triple_trans[add]:
                        if counts[ci] == 7 and ci not in freed:
                            safe = False
                            break
                    if safe:
                        # Perform swap
                        sol[rem] = False
                        sol[add] = True
                        for ci in triple_trans[rem]:
                            counts[ci] -= 1
                        for ci in triple_trans[add]:
                            counts[ci] += 1
                        # After swap, try to add more
                        added2, _ = try_add_candidates(sol, counts, absent[:50])
                        current_count += added2
                        improved = True
                        break
                if improved:
                    break

        return current_count

    # Main search loop with restarts
    num_restarts = 60
    for restart in range(num_restarts):
        if time.time() > deadline - 1.0:
            break

        sol, counts, current_count = greedy_init()

        # Polish the initial solution
        current_count = polish(sol, counts, current_count)

        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            best_counts = counts[:]

        # Continue with local search from this solution
        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)

        stagnation = 0
        while time.time() < deadline - 0.5:
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added

            if added > 0:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
                    best_counts = counts[:]
                continue

            # Find tight configurations (count == 7)
            tight_triples = []
            for ci, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in transversals[ci]:
                        if sol[t_idx]:
                            tight_triples.append(t_idx)

            if tight_triples:
                kick_idx = random.choice(tight_triples)
            else:
                active = [i for i in range(num_triples) if sol[i]]
                if not active:
                    break
                kick_idx = random.choice(active)

            sol[kick_idx] = False
            for ci in triple_trans[kick_idx]:
                counts[ci] -= 1
            current_count -= 1
            candidates.append(kick_idx)

            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added

            if added > 0:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
                    best_counts = counts[:]
            else:
                stagnation += 1
                if stagnation > 40:
                    break

    if best_solution is None:
        best_solution, best_counts, _ = greedy_init()

    # Final validation: ensure no transversal is fully present
    final_sol = best_solution[:]
    final_counts = best_counts[:]
    for ci in range(num_trans):
        while final_counts[ci] == 8:
            # Find a triple in this transversal that is present
            for t_idx in transversals[ci]:
                if final_sol[t_idx]:
                    final_sol[t_idx] = False
                    for ci2 in triple_trans[t_idx]:
                        final_counts[ci2] -= 1
                    break

    result = [list(all_triples[i]) for i in range(num_triples) if final_sol[i]]
    return result
