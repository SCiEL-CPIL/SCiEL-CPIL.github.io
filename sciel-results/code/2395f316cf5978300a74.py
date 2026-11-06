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

    # Build transversals as lists of triple indices (faster iteration than bitmasks for small sets)
    transversals = []
    seen_masks = set()
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
            idxs = []
            for (x, y, z) in ((a, c, e), (a, c, f), (a, d, e), (a, d, f),
                              (b, c, e), (b, c, f), (b, d, e), (b, d, f)):
                idxs.append(triple_to_idx[tuple(sorted((x, y, z)))])
            key = tuple(sorted(idxs))
            if key not in seen_masks:
                seen_masks.add(key)
                transversals.append(idxs)
    num_trans = len(transversals)

    # For each triple, list of transversals it belongs to
    triple_trans = [[] for _ in range(num_triples)]
    for ti, tm in enumerate(transversals):
        for idx in tm:
            triple_trans[idx].append(ti)

    # ---- Phase 1: Greedy construction from empty set ----
    # Add triples in order of lowest conflict potential
    # Score each triple by how many transversals it participates in (lower = safer)
    triple_score = [len(triple_trans[i]) for i in range(num_triples)]
    order = sorted(range(num_triples), key=lambda i: (triple_score[i], random.random()))

    state = set()  # set of triple indices present
    # Track which transversals are "satisfied" (all 8 triples present)
    trans_counts = [0] * num_trans  # how many of the 8 triples are in state
    trans_full = [False] * num_trans

    def add_triple(idx):
        state.add(idx)
        for ti in triple_trans[idx]:
            trans_counts[ti] += 1
            if trans_counts[ti] == 8:
                trans_full[ti] = True

    def remove_triple(idx):
        state.discard(idx)
        for ti in triple_trans[idx]:
            if trans_counts[ti] > 0:
                trans_counts[ti] -= 1
                if trans_counts[ti] < 8:
                    trans_full[ti] = False

    # Greedy: try to add each triple; if it creates a conflict, skip it
    # But first, try to add all "safe" triples (those not in any full transversal yet)
    for idx in order:
        if idx in state:
            continue
        # Check if adding idx would complete any transversal
        would_complete = False
        for ti in triple_trans[idx]:
            if trans_counts[ti] == 7:
                would_complete = True
                break
        if not would_complete:
            add_triple(idx)

    # Now try to add remaining triples one by one, removing a conflicting triple if needed
    remaining = [i for i in order if i not in state]
    random.shuffle(remaining)
    for idx in remaining:
        if idx in state:
            continue
        # Check conflicts
        conflict_trans = [ti for ti in triple_trans[idx] if trans_counts[ti] == 7]
        if not conflict_trans:
            add_triple(idx)
        else:
            # Try to remove one triple from each conflict to make room
            # Only proceed if we can fix all conflicts by removing at most 1 triple
            # Find a triple whose removal fixes all conflicts
            candidates_to_remove = []
            for ti in conflict_trans:
                # The 7 present triples in this transversal
                for j in transversals[ti]:
                    if j != idx and j in state:
                        candidates_to_remove.append(j)
            # Try removing each candidate and check if all conflicts are resolved
            for rem in candidates_to_remove:
                # Check if removing rem resolves all conflict_trans
                ok = True
                for ti in conflict_trans:
                    if rem not in transversals[ti]:
                        ok = False
                        break
                if ok:
                    remove_triple(rem)
                    add_triple(idx)
                    break

    best_state = set(state)
    best_count = len(best_state)
    num_conflicts = sum(1 for tf in trans_full if tf)

    # ---- Phase 2: Local search to add more triples by resolving conflicts ----
    def try_local_search(state_set, trans_counts_arr, trans_full_arr, max_iters, time_limit):
        """Local search: try to add absent triples by removing conflicting ones."""
        local_state = set(state_set)
        local_counts = list(trans_counts_arr)
        local_full = list(trans_full_arr)
        local_best = set(local_state)
        local_best_count = len(local_best)
        local_best_conf = sum(1 for tf in local_full if tf)

        # Precompute for speed: for each triple, which transversals it's in
        # and the indices in those transversals

        def l_add(idx):
            local_state.add(idx)
            for ti in triple_trans[idx]:
                local_counts[ti] += 1
                if local_counts[ti] == 8:
                    local_full[ti] = True

        def l_remove(idx):
            local_state.discard(idx)
            for ti in triple_trans[idx]:
                if local_counts[ti] > 0:
                    local_counts[ti] -= 1
                    if local_counts[ti] < 8:
                        local_full[ti] = False

        # Build list of absent triples sorted by "desirability"
        # A triple is desirable if it participates in few transversals
        absent = [i for i in range(num_triples) if i not in local_state]
        random.shuffle(absent)

        for _ in range(max_iters):
            if time.time() > time_limit:
                break

            # Find current conflicts
            conflicts = [ti for ti in range(num_trans) if local_full[ti]]
            if not conflicts:
                # No conflicts: try to add more triples
                added_any = False
                for idx in absent[:]:
                    if idx in local_state:
                        continue
                    # Check if adding would create conflict
                    would_conflict = False
                    for ti in triple_trans[idx]:
                        if local_counts[ti] == 7:
                            would_conflict = True
                            break
                    if not would_conflict:
                        l_add(idx)
                        if idx in absent:
                            absent.remove(idx)
                        added_any = True
                        if len(local_state) > local_best_count:
                            local_best = set(local_state)
                            local_best_count = len(local_state)
                            local_best_conf = sum(1 for tf in local_full if tf)
                if not added_any:
                    break
                continue

            # Try to resolve conflicts by adding absent triples that help
            # or by removing present triples to free up capacity
            # Strategy: for each conflict, find an absent triple that, if added,
            # requires removing exactly one present triple
            resolved = False
            for ti in conflicts:
                # The 8 triples in this transversal
                t8 = transversals[ti]
                present_in_t8 = [j for j in t8 if j in local_state]
                absent_in_t8 = [j for j in t8 if j not in local_state]
                if len(present_in_t8) == 8:
                    # Full conflict: need to remove at least one
                    # Remove the one that appears in the fewest other conflicts
                    best_rem = -1
                    best_score = float('inf')
                    for j in present_in_t8:
                        # Count how many other conflicts j participates in
                        sc = 0
                        for ti2 in triple_trans[j]:
                            if ti2 != ti and local_full[ti2]:
                                sc += 1
                        if sc < best_score:
                            best_score = sc
                            best_rem = j
                    if best_rem >= 0:
                        l_remove(best_rem)
                        resolved = True
                        break
            if resolved:
                # Update best if we made progress (fewer conflicts or same conflicts but more triples)
                cur_conf = sum(1 for tf in local_full if tf)
                if cur_conf < local_best_conf or (cur_conf == local_best_conf and len(local_state) > local_best_count):
                    local_best = set(local_state)
                    local_best_count = len(local_state)
                    local_best_conf = cur_conf
                continue

            # Try: find an absent triple that, if we remove one present triple from each of its
            # near-full transversals, allows it to be added
            found_improvement = False
            for idx in absent[:20]:  # limit candidates for speed
                if idx in local_state:
                    continue
                near_full = [ti for ti in triple_trans[idx] if local_counts[ti] == 7]
                if not near_full:
                    # Can add directly
                    l_add(idx)
                    if idx in absent:
                        absent.remove(idx)
                    cur_conf = sum(1 for tf in local_full if tf)
                    if len(local_state) > local_best_count:
                        local_best = set(local_state)
                        local_best_count = len(local_state)
                        local_best_conf = cur_conf
                    found_improvement = True
                    break
                elif len(near_full) == 1:
                    # Need to remove one triple from that one transversal
                    ti = near_full[0]
                    t8 = transversals[ti]
                    # Find a present triple in t8 (not idx) that's in fewest other conflicts
                    best_rem = -1
                    best_score = float('inf')
                    for j in t8:
                        if j != idx and j in local_state:
                            sc = 0
                            for ti2 in triple_trans[j]:
                                if ti2 != ti and local_full[ti2]:
                                    sc += 1
                            if sc < best_score:
                                best_score = sc
                                best_rem = j
                    if best_rem >= 0 and best_score == 0:
                        # Removing best_rem only affects this one conflict
                        l_remove(best_rem)
                        l_add(idx)
                        if idx in absent:
                            absent.remove(idx)
                        cur_conf = sum(1 for tf in local_full if tf)
                        if len(local_state) > local_best_count:
                            local_best = set(local_state)
                            local_best_count = len(local_state)
                            local_best_conf = cur_conf
                        found_improvement = True
                        break
            if not found_improvement:
                # Try random swap: remove a random present triple, try to add a random absent one
                if local_state and absent:
                    rem = random.choice(list(local_state))
                    l_remove(rem)
                    # Now try to add some absent triples
                    for idx in absent[:5]:
                        if idx in local_state:
                            continue
                        would_conflict = False
                        for ti in triple_trans[idx]:
                            if local_counts[ti] == 7:
                                would_conflict = True
                                break
                        if not would_conflict:
                            l_add(idx)
                            if idx in absent:
                                absent.remove(idx)
                            break
                    cur_conf = sum(1 for tf in local_full if tf)
                    if len(local_state) > local_best_count:
                        local_best = set(local_state)
                        local_best_count = len(local_state)
                        local_best_conf = cur_conf

        return local_best, local_best_count, local_best_conf

    # Run multiple rounds of local search
    for rnd in range(10):
        if time.time() > deadline - 2.0:
            break
        cur_state, cur_count, cur_conf = try_local_search(
            best_state, trans_counts, trans_full, 5000, deadline - 1.0)
        if cur_count > best_count:
            best_state = cur_state
            best_count = cur_count
            # Rebuild trans_counts and trans_full
            trans_counts = [0] * num_trans
            trans_full = [False] * num_trans
            for idx in best_state:
                for ti in triple_trans[idx]:
                    trans_counts[ti] += 1
                    if trans_counts[ti] == 8:
                        trans_full[ti] = True

    # ---- Phase 3: Ensure validity by removing triples from any remaining full transversals ----
    final_state = set(best_state)
    # Rebuild counts
    fc = [0] * num_trans
    ff = [False] * num_trans
    for idx in final_state:
        for ti in triple_trans[idx]:
            fc[ti] += 1
            if fc[ti] == 8:
                ff[ti] = True

    random.shuffle(transversals)
    while True:
        if time.time() > deadline - 0.5:
            break
        conflict_found = False
        for ti in range(num_trans):
            if ff[ti]:
                conflict_found = True
                # Remove a random triple from this transversal
                t8 = transversals[ti]
                present = [j for j in t8 if j in final_state]
                if present:
                    rem = random.choice(present)
                    final_state.discard(rem)
                    for ti2 in triple_trans[rem]:
                        if fc[ti2] > 0:
                            fc[ti2] -= 1
                            if fc[ti2] < 8:
                                ff[ti2] = False
                break
        if not conflict_found:
            break

    result = [list(all_triples[i]) for i in sorted(final_state)]
    return result
