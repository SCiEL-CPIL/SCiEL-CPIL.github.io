import itertools
import random
import time
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

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
        for perm in itertools.permutations(range(6)):
            p1 = tuple(sorted((sp[perm[0]], sp[perm[1]])))
            p2 = tuple(sorted((sp[perm[2]], sp[perm[3]])))
            p3 = tuple(sorted((sp[perm[4]], sp[perm[5]])))
            matchings.add(tuple(sorted((p1, p2, p3))))
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

    # Precompute: for each transversal, a numpy array of its 8 triple indices
    trans_arrays = [np.array(t, dtype=np.int32) for t in transversals]

    # Precompute: for each triple, which transversals it belongs to (as array)
    triple_trans_arrays = [np.array(t, dtype=np.int32) for t in triple_trans]

    def greedy_init():
        sol = np.zeros(num_triples, dtype=bool)
        counts = np.zeros(num_trans, dtype=np.int32)
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            # Check if adding this triple would complete any transversal
            tts = triple_trans_arrays[idx]
            if len(tts) > 0:
                if np.any(counts[tts] == 7):
                    continue
            sol[idx] = True
            if len(tts) > 0:
                counts[tts] += 1
        return sol, counts, int(sol.sum())

    def try_add_candidates(sol, counts, candidates):
        added = 0
        new_candidates = []
        for idx in candidates:
            if sol[idx]:
                continue
            tts = triple_trans_arrays[idx]
            if len(tts) > 0:
                if np.any(counts[tts] == 7):
                    new_candidates.append(idx)
                    continue
            sol[idx] = True
            if len(tts) > 0:
                counts[tts] += 1
            added += 1
        return added, new_candidates

    def walk_sat_step(sol, counts):
        """
        WalkSAT-style step: pick a variable (triple) to flip.
        If sol[idx] is True, flip to False (remove).
        If sol[idx] is False, flip to True (add) only if it doesn't create a conflict.
        
        We prefer flipping variables that are in violated clauses (counts == 8)
        or near-violated clauses (counts == 7).
        """
        # Find violated clauses (count == 8)
        violated = np.where(counts == 8)[0]
        
        if len(violated) > 0:
            # Pick a random violated clause
            ci = int(random.choice(violated))
            t_arr = trans_arrays[ci]
            # All 8 triples in this clause are selected
            # Pick one to remove
            idx = int(random.choice(t_arr))
            # Remove it
            sol[idx] = False
            tts = triple_trans_arrays[idx]
            if len(tts) > 0:
                counts[tts] -= 1
            return idx, True  # flipped from True to False
        else:
            # No violations. Try to add a triple.
            # Find triples that can be added (don't complete any clause)
            # For efficiency, sample a few random unselected triples
            unselected = np.where(~sol)[0]
            if len(unselected) == 0:
                return None, False
            
            # Try up to 10 random candidates
            for _ in range(10):
                idx = int(random.choice(unselected))
                tts = triple_trans_arrays[idx]
                if len(tts) > 0:
                    if np.any(counts[tts] == 7):
                        continue
                # Can add
                sol[idx] = True
                if len(tts) > 0:
                    counts[tts] += 1
                return idx, False  # flipped from False to True
            return None, False

    def local_search(sol, counts, current_count, max_iterations=1000):
        """
        WalkSAT-style local search to maximize number of selected triples
        while keeping all clauses unsatisfied (counts < 8).
        """
        best_sol = sol.copy()
        best_counts = counts.copy()
        best_count = current_count
        stagnant = 0
        
        for it in range(max_iterations):
            if time.time() > deadline - 1.0:
                break
            
            # If there are violations, fix them first
            violated = np.where(counts == 8)[0]
            if len(violated) > 0:
                # Pick a random violated clause and remove one of its triples
                ci = int(random.choice(violated))
                t_arr = trans_arrays[ci]
                idx = int(random.choice(t_arr))
                sol[idx] = False
                tts = triple_trans_arrays[idx]
                if len(tts) > 0:
                    counts[tts] -= 1
                current_count -= 1
                stagnant = 0
                continue
            
            # No violations. Try to add a triple.
            # Find all addable triples (those that don't complete any clause)
            # For efficiency, try random sampling
            unselected = np.where(~sol)[0]
            if len(unselected) == 0:
                break
            
            # Try up to 20 random candidates
            added = False
            for _ in range(20):
                idx = int(random.choice(unselected))
                tts = triple_trans_arrays[idx]
                if len(tts) > 0:
                    if np.any(counts[tts] == 7):
                        continue
                # Can add
                sol[idx] = True
                if len(tts) > 0:
                    counts[tts] += 1
                current_count += 1
                added = True
                stagnant = 0
                if current_count > best_count:
                    best_count = current_count
                    best_sol = sol.copy()
                    best_counts = counts.copy()
                break
            
            if not added:
                # Try a swap: remove one triple, then add another
                # Pick a random selected triple to remove
                selected = np.where(sol)[0]
                if len(selected) == 0:
                    break
                remove_idx = int(random.choice(selected))
                sol[remove_idx] = False
                tts = triple_trans_arrays[remove_idx]
                if len(tts) > 0:
                    counts[tts] -= 1
                current_count -= 1
                
                # Now try to add a different triple
                unselected = np.where(~sol)[0]
                if len(unselected) > 0:
                    for _ in range(30):
                        add_idx = int(random.choice(unselected))
                        tts2 = triple_trans_arrays[add_idx]
                        if len(tts2) > 0:
                            if np.any(counts[tts2] == 7):
                                continue
                        # Can add
                        sol[add_idx] = True
                        if len(tts2) > 0:
                            counts[tts2] += 1
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_sol = sol.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        # Couldn't add, put back the removed triple
                        sol[remove_idx] = True
                        if len(tts) > 0:
                            counts[tts] += 1
                        current_count += 1
                
                stagnant += 1
                if stagnant > 100:
                    break
        
        return best_sol, best_counts, best_count

    best_solution = None
    best_count = -1

    for restart in range(150):
        if time.time() > deadline - 1.0:
            break

        sol, counts, current_count = greedy_init()
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol.copy()
            best_counts = counts.copy()
        
        # Run local search
        sol, counts, current_count = local_search(sol, counts, current_count)
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol.copy()
            best_counts = counts.copy()

    if best_solution is None:
        best_solution, best_counts, _ = greedy_init()

    # Final validation: ensure no transversal is fully present
    final_sol = best_solution.copy()
    final_counts = best_counts.copy()
    # Remove any remaining conflicts
    for ci in range(num_trans):
        while final_counts[ci] == 8:
            t_arr = trans_arrays[ci]
            selected = t_arr[final_sol[t_arr]]
            if len(selected) == 0:
                break
            t_idx = int(selected[0])
            final_sol[t_idx] = False
            tts = triple_trans_arrays[t_idx]
            if len(tts) > 0:
                final_counts[tts] -= 1
            final_counts[ci] -= 1

    result = [list(all_triples[i]) for i in range(num_triples) if final_sol[i]]
    return result
