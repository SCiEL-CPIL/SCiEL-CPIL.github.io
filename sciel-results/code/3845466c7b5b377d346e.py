import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    all_vals = list(range(1, max_val + 1))

    # Build a conflict structure: for each candidate x, the set of sums it would create
    # with each other element. We need: for a Sidon set S, all pairwise sums a+b (a<=b) are distinct.

    # Precompute: for each x in 1..284, the set of sums {x+y : y in 1..284, y>=x} and {2x}
    # Actually we need: for each pair (x, y) with x<=y, the sum x+y.
    # Two elements x1, x2 in S conflict if there exist pairs (a,b), (c,d) in S x S (a<=b, c<=d)
    # with a+b = c+d but {a,b} != {c,d}.

    # More efficiently: when we add x to S, we create sums {x+y for y in S} and {2x}.
    # These must all be new (not in the existing sum set).

    def init_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def can_add(x, S, Sums):
        if 2 * x in Sums:
            return False
        for y in S:
            if x + y in Sums:
                return False
        return True

    def add_element(x, S, Sums):
        Sums.add(2 * x)
        for y in S:
            Sums.add(x + y)
        S.append(x)
        S.sort()

    def remove_element(x, S, Sums):
        S.remove(x)
        Sums.discard(2 * x)
        for y in S:
            Sums.discard(x + y)

    def greedy_fill(S, Sums, order='ascending'):
        S_set = set(S)
        if order == 'ascending':
            candidates = list(range(1, max_val + 1))
        elif order == 'descending':
            candidates = list(range(max_val, 0, -1))
        else:
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)

        for x in candidates:
            if x in S_set:
                continue
            if can_add(x, S, Sums):
                add_element(x, S, Sums)
                S_set.add(x)
        return S, Sums

    def local_search(S, Sums, max_iters=300, aggressive=False, deadline_local=None):
        if deadline_local is None:
            deadline_local = deadline
        S_set = set(S)
        improved = False
        iters = 0
        stuck_count = 0

        while iters < max_iters and time.time() < deadline_local:
            iters += 1

            # Try to add an element
            candidates = [x for x in all_vals if x not in S_set]
            if not candidates:
                break

            random.shuffle(candidates)
            added = False
            for x in candidates[:min(60, len(candidates))]:
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    S_set.add(x)
                    improved = True
                    added = True
                    stuck_count = 0
                    break

            if added:
                continue

            stuck_count += 1

            removable = [x for x in S if x not in prescribed_set]
            if not removable:
                break

            random.shuffle(removable)
            swap_done = False

            # Try single swaps
            for r in removable[:min(12, len(removable))]:
                remove_element(r, S, Sums)
                S_set.discard(r)

                candidates = [x for x in all_vals if x not in S_set]
                random.shuffle(candidates)

                for x in candidates[:min(60, len(candidates))]:
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        swap_done = True
                        improved = True
                        stuck_count = 0
                        break

                if swap_done:
                    break
                else:
                    add_element(r, S, Sums)
                    S_set.add(r)

            if swap_done:
                continue

            # Try double swaps if aggressive
            if aggressive and len(removable) >= 2:
                random.shuffle(removable)
                r1, r2 = removable[0], removable[1]

                remove_element(r1, S, Sums)
                S_set.discard(r1)
                remove_element(r2, S, Sums)
                S_set.discard(r2)

                candidates = [x for x in all_vals if x not in S_set]
                random.shuffle(candidates)

                added_count = 0
                added_elems = []
                for x in candidates[:min(100, len(candidates))]:
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        added_elems.append(x)
                        added_count += 1
                        if added_count >= 2:
                            break

                if added_count >= 2:
                    improved = True
                    stuck_count = 0
                else:
                    for x in added_elems:
                        remove_element(x, S, Sums)
                        S_set.discard(x)
                    add_element(r1, S, Sums)
                    S_set.add(r1)
                    add_element(r2, S, Sums)
                    S_set.add(r2)

            if stuck_count > 60:
                break

        return S, Sums, improved

    def hill_climbing_polish(S, Sums):
        S_set = set(S)
        improved = True
        polish_iters = 0

        while improved and polish_iters < 150 and time.time() < deadline:
            improved = False
            polish_iters += 1

            # Try all adds
            for x in all_vals:
                if x in S_set:
                    continue
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    S_set.add(x)
                    improved = True
                    break

            if improved:
                continue

            # Try all single swaps
            removable = [x for x in S if x not in prescribed_set]
            for r in removable:
                remove_element(r, S, Sums)
                S_set.discard(r)

                for x in all_vals:
                    if x in S_set:
                        continue
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        improved = True
                        break

                if improved:
                    break
                else:
                    add_element(r, S, Sums)
                    S_set.add(r)

        return S, Sums

    # Branch and bound on conflict graph
    # For a Sidon set, the constraint is: all pairwise sums are distinct.
    # We can model this as: build a graph where vertices are candidates 1..284,
    # and we want a maximum subset such that no two pairs have the same sum.

    # Precompute for each candidate x, the set of "sums" it participates in with every other candidate.
    # When we have a current set S and sum set Sums, a candidate x is "available" if can_add(x, S, Sums).

    # For branch-and-bound, we'll use the following approach:
    # - Maintain a set of "available" candidates (those that can be added)
    # - At each step, branch on including or excluding a candidate
    # - Use upper bound: current_size + number of remaining available candidates (rough)
    # - Better upper bound: solve a relaxation

    # Actually, let's use a smarter approach. The key insight is that the Sidon constraint
    # is local: adding x only conflicts with existing elements. So we can use a
    # backtracking search with good pruning.

    def branch_and_bound(S, Sums, available, best_size, depth=0):
        """Backtracking search with pruning. Returns best set found."""
        if time.time() > deadline:
            return S.copy()

        if len(S) + len(available) <= best_size:
            return S.copy()

        if not available:
            return S.copy()

        # Try adding each available element
        # Heuristic: try elements that leave the most future options
        avail_list = list(available)

        # Sort by a heuristic: elements that create fewer conflicts
        # (i.e., elements with fewer sums that are already "close" to being used)
        # For simplicity, just try in sorted order or random

        best_S = S.copy()

        for i, x in enumerate(avail_list):
            if time.time() > deadline:
                break

            new_S = S + [x]
            new_Sums = set(Sums)
            new_Sums.add(2 * x)
            for y in S:
                new_Sums.add(x + y)

            new_available = set()
            for z in available:
                if z == x:
                    continue
                if 2 * z not in new_Sums:
                    conflict = False
                    for y in new_S:
                        if z + y in new_Sums:
                            conflict = True
                            break
                    if not conflict:
                        new_available.add(z)

            result = branch_and_bound(new_S, new_Sums, new_available, best_size, depth + 1)
            if len(result) > len(best_S):
                best_S = result

            # Pruning: if we can't beat best_size even with all remaining
            if len(S) + len(available) - 1 <= len(best_S):
                break

        return best_S

    def improved_branch_and_bound(S, Sums, deadline_local=None):
        """Branch and bound with better pruning."""
        if deadline_local is None:
            deadline_local = deadline

        S_set = set(S)
        available = set(x for x in all_vals if x not in S_set and can_add(x, S, Sums))

        if not available:
            return S.copy()

        best_S = S.copy()
        best_size = len(S)

        # Recursive search with memoization of failed states
        # State: (frozen Sums, frozenset of available) -> whether we found a better solution
        # This might be too memory-heavy, so use a simpler approach

        # Instead, use iterative deepening: try to find solutions of size best_size+1, +2, etc.

        def search_for_size(target_size, S, Sums, available, depth=0):
            if time.time() > deadline_local:
                return None

            if len(S) == target_size:
                return S.copy()

            if not available:
                return None

            # Pruning: if we can't reach target_size
            if len(S) + len(available) < target_size:
                return None

            avail_list = list(available)

            # Heuristic ordering: try elements that are "central" (create fewer conflicts)
            # Sort by number of conflicts with other available elements
            def conflict_count(x):
                count = 0
                S_temp = S + [x]
                Sums_temp = set(Sums)
                Sums_temp.add(2 * x)
                for y in S:
                    Sums_temp.add(x + y)
                for z in available:
                    if z == x:
                        continue
                    if 2 * z in Sums_temp or any(z + y in Sums_temp for y in S_temp):
                        count += 1
                return count

            avail_list.sort(key=conflict_count)

            for x in avail_list:
                if time.time() > deadline_local:
                    return None

                new_S = S + [x]
                new_Sums = set(Sums)
                new_Sums.add(2 * x)
                for y in S:
                    new_Sums.add(x + y)

                new_available = set()
                for z in available:
                    if z == x:
                        continue
                    if 2 * z not in new_Sums:
                        conflict = False
                        for y in new_S:
                            if z + y in new_Sums:
                                conflict = True
                                break
                        if not conflict:
                            new_available.add(z)

                result = search_for_size(target_size, new_S, new_Sums, new_available, depth + 1)
                if result is not None:
                    return result

            return None

        # Try to increase size by 1, 2, 3, ...
        for extra in range(1, 6):
            target = best_size + extra
            result = search_for_size(target, S, Sums, available)
            if result is not None:
                best_S = result
                best_size = target
                # Rebuild available
                S_set_new = set(best_S)
                available = set(x for x in all_vals if x not in S_set_new and can_add(x, best_S, init_sums(best_S)))
                Sums = init_sums(best_S)
            else:
                break

        return best_S

    best_size = 0
    best_S = []

    # Strategy: multiple restarts with different initializations, then local search, then B&B polish
    restart_count = 0
    found_20 = False

    while time.time() < deadline - 20:
        restart_count += 1

        # Choose initialization strategy
        if restart_count % 4 == 1:
            order = 'ascending'
        elif restart_count % 4 == 2:
            order = 'descending'
        elif restart_count % 4 == 3:
            order = 'random'
        else:
            # Lookahead greedy: at each step, pick the element that allows the most future additions
            order = 'lookahead'

        S = list(prescribed)
        S.sort()
        Sums = init_sums(S)

        if order == 'lookahead':
            S_set = set(S)
            available = set(x for x in all_vals if x not in S_set and can_add(x, S, Sums))
            while available and time.time() < deadline - 10:
                # Pick element with fewest conflicts (leaves most options)
                best_x = None
                best_score = -1
                for x in list(available):
                    new_S = S + [x]
                    new_Sums = set(Sums)
                    new_Sums.add(2 * x)
                    for y in S:
                        new_Sums.add(x + y)
                    new_avail = 0
                    for z in available:
                        if z == x:
                            continue
                        if 2 * z not in new_Sums:
                            conflict = False
                            for y in new_S:
                                if z + y in new_Sums:
                                    conflict = True
                                    break
                            if not conflict:
                                new_avail += 1
                    if new_avail > best_score:
                        best_score = new_avail
                        best_x = x
                if best_x is None:
                    break
                add_element(best_x, S, Sums)
                S_set.add(best_x)
                available = set(x for x in all_vals if x not in S_set and can_add(x, S, Sums))
        else:
            S, Sums = greedy_fill(S, Sums, order=order)

        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()

        if best_size >= 20:
            found_20 = True
            break

        # Local search
        aggressive = restart_count % 2 == 0
        S, Sums, improved = local_search(S, Sums, max_iters=200, aggressive=aggressive)
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()

        if best_size >= 20:
            found_20 = True
            break

        # B&B polish on best solution
        if time.time() < deadline - 15:
            S_bb = best_S.copy()
            Sums_bb = init_sums(S_bb)
            S_bb = improved_branch_and_bound(S_bb, Sums_bb)
            if len(S_bb) > best_size:
                best_size = len(S_bb)
                best_S = S_bb.copy()

            if best_size >= 20:
                found_20 = True
                break

    # Final polish
    if best_S and time.time() < deadline - 5:
        S = best_S.copy()
        Sums = init_sums(S)
        S, Sums = hill_climbing_polish(S, Sums)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()

    return best_S
