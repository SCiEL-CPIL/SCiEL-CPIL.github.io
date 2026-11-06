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

    # Precompute for each transversal, which triples are in it (for fast lookup)
    # and build a conflict structure: for each triple, which other triples conflict with it
    # Two triples conflict if they are in the same transversal (i.e., they share a transversal constraint)
    # Actually, the constraint is: no transversal can have all 8 triples present.
    # So the "conflict" is at the transversal level, not pairwise.

    # Build adjacency: for each transversal, store the set of triple indices
    trans_triple_set = [set(t) for t in transversals]

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

    def is_safe_to_add(idx, counts, sol):
        """Check if triple idx can be added without violating any constraint."""
        if sol[idx]:
            return True
        for ci in triple_trans[idx]:
            if counts[ci] == 7:
                return False
        return True

    def add_triple(idx, sol, counts):
        """Add triple idx to solution, update counts."""
        sol[idx] = True
        for ci in triple_trans[idx]:
            counts[ci] += 1

    def remove_triple(idx, sol, counts):
        """Remove triple idx from solution, update counts."""
        sol[idx] = False
        for ci in triple_trans[idx]:
            counts[ci] -= 1

    def polish(sol, counts, current_count):
        """High-precision polish: try to add safe triples, then do local swaps."""
        # Phase 1: Add all safe triples
        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)
        added, _ = try_add_candidates(sol, counts, candidates)
        current_count += added

        # Phase 2: Local search with 1-out-1-in swaps
        if time.time() > deadline - 2.0:
            return current_count

        improved = True
        max_swap_iters = 20
        for _ in range(max_swap_iters):
            if time.time() > deadline - 1.5:
                break
            if not improved:
                break
            improved = False

            present = [i for i in range(num_triples) if sol[i]]
            absent = [i for i in range(num_triples) if not sol[i]]
            if not present or not absent:
                break
            random.shuffle(present)
            random.shuffle(absent)

            # Try 1-out-1-in swaps
            found_swap = False
            for rem in present:
                if time.time() > deadline - 1.5:
                    break
                # What transversals does removing rem free up?
                freed_trans = []
                for ci in triple_trans[rem]:
                    if counts[ci] == 7:
                        freed_trans.append(ci)
                if not freed_trans:
                    continue

                freed_set = set(freed_trans)

                for add in absent:
                    # Check if adding add is safe after removing rem
                    safe = True
                    for ci in triple_trans[add]:
                        if counts[ci] == 7 and ci not in freed_set:
                            safe = False
                            break
                    if safe:
                        # Perform swap
                        remove_triple(rem, sol, counts)
                        add_triple(add, sol, counts)
                        current_count += 0  # net zero from swap

                        # Try to add more after swap
                        remaining_absent = [i for i in range(num_triples) if not sol[i]]
                        random.shuffle(remaining_absent)
                        added2, _ = try_add_candidates(sol, counts, remaining_absent)
                        current_count += added2
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break

            # Try 2-out-1-in swaps if 1-out-1-in didn't help
            if not improved and time.time() < deadline - 1.0:
                present = [i for i in range(num_triples) if sol[i]]
                absent = [i for i in range(num_triples) if not sol[i]]
                if len(present) >= 2 and absent:
                    random.shuffle(present)
                    random.shuffle(absent)
                    for i in range(min(len(present), 50)):
                        for j in range(i + 1, min(len(present), 50)):
                            if time.time() > deadline - 1.0:
                                break
                            rem1, rem2 = present[i], present[j]
                            # What transversals do we free up?
                            freed_trans = set()
                            for ci in triple_trans[rem1]:
                                if counts[ci] == 7:
                                    freed_trans.add(ci)
                            for ci in triple_trans[rem2]:
                                if counts[ci] == 7:
                                    freed_trans.add(ci)
                            if not freed_trans:
                                continue

                            for add in absent[:30]:
                                safe = True
                                for ci in triple_trans[add]:
                                    if counts[ci] == 7 and ci not in freed_trans:
                                        safe = False
                                        break
                                if safe:
                                    remove_triple(rem1, sol, counts)
                                    remove_triple(rem2, sol, counts)
                                    add_triple(add, sol, counts)
                                    current_count -= 1  # net -1

                                    # Try to add more
                                    remaining_absent = [i for i in range(num_triples) if not sol[i]]
                                    random.shuffle(remaining_absent)
                                    added2, _ = try_add_candidates(sol, counts, remaining_absent)
                                    current_count += added2
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break

        return current_count

    best_solution = None
    best_count = -1
    best_counts = None

    # Main search loop with restarts
    num_restarts = 40
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
        max_local_iters = 100
        for _ in range(max_local_iters):
            if time.time() > deadline - 0.5:
                break

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

            remove_triple(kick_idx, sol, counts)
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
                if stagnation > 30:
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
