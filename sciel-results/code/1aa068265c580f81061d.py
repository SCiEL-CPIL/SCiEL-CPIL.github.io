# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from collections import defaultdict

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find the largest subset of Z/205Z with no two elements differing by a
    nonzero square mod 205.

    Uses:
      - Bitset-based conflict graph for fast compatibility checks
      - Branch-and-bound maximum independent set (with greedy coloring bound)
      - Systematic local search (1-1 swaps, 2-2 swaps)
      - Multiple random restarts
    """
    m = 205
    deadline = time.time() + 590  # safety margin before 630s

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_set = forbidden

    # Build conflict graph as bitsets: conflict_mask[v] is a 205-bit int
    # where bit u is set iff (v - u) % m is in forbidden_set
    conflict_mask = [0] * m
    for v in range(m):
        mask = 0
        for u in range(m):
            if u == v:
                continue
            d = (v - u) % m
            if d in forbidden_set:
                mask |= (1 << u)
        conflict_mask[v] = mask

    # Also build adjacency list for branch-and-bound
    adj = [[] for _ in range(m)]
    for v in range(m):
        mask = conflict_mask[v]
        for u in range(m):
            if mask & (1 << u):
                adj[v].append(u)

    def is_compatible(candidate, selected_bits):
        """Check if candidate is compatible with all elements in selected_bits (a bitmask)."""
        return (conflict_mask[candidate] & selected_bits) == 0

    def greedy_color_bound(candidates_bits):
        """
        Greedy coloring bound for branch-and-bound.
        Returns an upper bound on the size of an independent set in candidates.
        """
        # Extract candidate list
        cands = []
        temp = candidates_bits
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            cands.append(idx)
            temp ^= low

        if not cands:
            return 0

        # Greedy coloring: assign colors to vertices
        # color[v] = number of the color class v belongs to
        color = {}
        color_classes = []  # list of lists of vertices in each color class

        for v in cands:
            # Find the smallest color that v can take (no neighbor in that color class)
            used_colors = set()
            for u in adj[v]:
                if u in color:
                    used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(v)

        return len(color_classes)

    def branch_and_bound(candidates_bits, current_size, current_set_bits):
        """
        Branch-and-bound for maximum independent set.
        candidates_bits: bitmask of remaining candidate vertices
        current_size: size of the independent set built so far
        current_set_bits: bitmask of vertices in the current independent set
        Returns (best_size, best_set_bits)
        """
        best_size = current_size
        best_set = current_set_bits

        if candidates_bits == 0:
            return best_size, best_set

        # Upper bound: current_size + coloring bound on the *current* candidate set
        ub = current_size + greedy_color_bound(candidates_bits)
        if ub <= best_size:
            return best_size, best_set

        # Extract candidate list, sorted by degree in the remaining subgraph (descending)
        cands = []
        temp = candidates_bits
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            cands.append(idx)
            temp ^= low

        cand_set = set(cands)
        cands.sort(key=lambda v: -len([u for u in adj[v] if u in cand_set]))

        # Work on a local copy; the loop implicitly excludes each vertex after branching
        local_cands = candidates_bits

        for v in cands:
            if time.time() > deadline:
                break

            # Branch 1: include v
            new_set_bits = current_set_bits | (1 << v)
            # Remove v and all its neighbors from the candidate pool
            new_candidates = local_cands & ~(1 << v)
            for u in adj[v]:
                new_candidates &= ~(1 << u)

            sub_size, sub_set = branch_and_bound(new_candidates, current_size + 1, new_set_bits)
            if sub_size > best_size:
                best_size = sub_size
                best_set = sub_set

            # Branch 2: exclude v
            local_cands &= ~(1 << v)

            # Tighten the upper bound using the *reduced* candidate set
            ub = current_size + greedy_color_bound(local_cands)
            if ub <= best_size:
                break

        return best_size, best_set

    def local_search(selected_bits, max_iters=2000):
        """
        Local search: try 1-1 swaps and direct additions.
        selected_bits: bitmask of the current independent set
        Returns improved bitmask.
        """
        current = selected_bits
        current_size = bin(current).count('1')
        best = current
        best_size = current_size

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Try to add any compatible vertex
            temp = ~current & ((1 << m) - 1)
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if is_compatible(v, current):
                    new_bits = current | (1 << v)
                    current = new_bits
                    current_size += 1
                    improved = True
                    # Restart the scan since adding one may enable more
                    break
                temp ^= low
            if improved:
                if current_size > best_size:
                    best_size = current_size
                    best = current
                continue

            # Try 1-1 swaps: remove one, add another
            cands_in = []
            temp = current
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                cands_in.append(v)
                temp ^= low

            random.shuffle(cands_in)
            swap_done = False
            for rem in cands_in:
                if time.time() > deadline:
                    break
                # Try to add some vertex after removing rem
                trial = current & ~(1 << rem)
                # Find any compatible vertex not in trial
                temp2 = ~trial & ((1 << m) - 1)
                # Limit the search for speed
                count = 0
                while temp2 and count < 50:
                    low2 = temp2 & (-temp2)
                    add_v = low2.bit_length() - 1
                    if is_compatible(add_v, trial):
                        new_bits = trial | (1 << add_v)
                        current = new_bits
                        # current_size stays the same
                        swap_done = True
                        break
                    temp2 ^= low2
                    count += 1
                if swap_done:
                    break

            if not swap_done and current_size > best_size:
                best_size = current_size
                best = current

        return best, best_size

    def local_search_2swap(selected_bits, max_iters=500):
        """
        Enhanced local search with 2-2 swaps.
        selected_bits: bitmask of the current independent set
        Returns improved bitmask.
        """
        current = selected_bits
        current_size = bin(current).count('1')
        best = current
        best_size = current_size

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Try to add any compatible vertex
            temp = ~current & ((1 << m) - 1)
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if is_compatible(v, current):
                    new_bits = current | (1 << v)
                    current = new_bits
                    current_size += 1
                    improved = True
                    break
                temp ^= low
            if improved:
                if current_size > best_size:
                    best_size = current_size
                    best = current
                continue

            # Try 1-1 swaps: remove one, add another
            cands_in = []
            temp = current
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                cands_in.append(v)
                temp ^= low

            random.shuffle(cands_in)
            swap_done = False
            for rem in cands_in:
                if time.time() > deadline:
                    break
                trial = current & ~(1 << rem)
                temp2 = ~trial & ((1 << m) - 1)
                count = 0
                while temp2 and count < 30:
                    low2 = temp2 & (-temp2)
                    add_v = low2.bit_length() - 1
                    if is_compatible(add_v, trial):
                        new_bits = trial | (1 << add_v)
                        current = new_bits
                        swap_done = True
                        break
                    temp2 ^= low2
                    count += 1
                if swap_done:
                    break

            if swap_done:
                if current_size > best_size:
                    best_size = current_size
                    best = current
                continue

            # Try 2-2 swaps: remove two, add two
            if current_size >= 2:
                cands_in2 = cands_in[:]
                random.shuffle(cands_in2)
                two_swap_done = False
                for i in range(len(cands_in2)):
                    if time.time() > deadline:
                        break
                    rem1 = cands_in2[i]
                    for j in range(i + 1, len(cands_in2)):
                        rem2 = cands_in2[j]
                        trial = current & ~(1 << rem1) & ~(1 << rem2)
                        # Find two compatible vertices to add
                        temp3 = ~trial & ((1 << m) - 1)
                        found = []
                        count3 = 0
                        while temp3 and count3 < 40 and len(found) < 2:
                            low3 = temp3 & (-temp3)
                            add_v = low3.bit_length() - 1
                            if is_compatible(add_v, trial):
                                # Check if compatible with already-found additions
                                ok = True
                                for f in found:
                                    if not is_compatible(add_v, trial | (1 << f)):
                                        ok = False
                                        break
                                if ok:
                                    found.append(add_v)
                                    trial |= (1 << add_v)
                            temp3 ^= low3
                            count3 += 1
                        if len(found) == 2:
                            new_bits = current & ~(1 << rem1) & ~(1 << rem2) | (1 << found[0]) | (1 << found[1])
                            current = new_bits
                            two_swap_done = True
                            break
                    if two_swap_done:
                        break

                if two_swap_done:
                    if current_size > best_size:
                        best_size = current_size
                        best = current
                    continue

            if current_size > best_size:
                best_size = current_size
                best = current

        return best, best_size

    def high_precision_polish(selected_bits, max_iters=2000):
        """
        High-precision polish pass: exhaustive 1-1 swaps with full scan,
        plus targeted 2-2 swaps for the final improvement.
        """
        current = selected_bits
        current_size = bin(current).count('1')
        best = current
        best_size = current_size

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Exhaustive 1-1 swap search
            cands_in = []
            temp = current
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                cands_in.append(v)
                temp ^= low

            best_swap = None
            for rem in cands_in:
                if time.time() > deadline:
                    break
                trial = current & ~(1 << rem)
                temp2 = ~trial & ((1 << m) - 1)
                # Full scan for best addition (prefer lower index for determinism)
                best_add = None
                while temp2:
                    low2 = temp2 & (-temp2)
                    add_v = low2.bit_length() - 1
                    if is_compatible(add_v, trial):
                        best_add = add_v
                        break
                    temp2 ^= low2
                if best_add is not None:
                    new_bits = trial | (1 << best_add)
                    # Accept the swap (size stays same, but may enable future additions)
                    current = new_bits
                    improved = True
                    break

            if improved:
                # After a swap, try to add more
                temp_add = ~current & ((1 << m) - 1)
                while temp_add:
                    low_a = temp_add & (-temp_add)
                    v_a = low_a.bit_length() - 1
                    if is_compatible(v_a, current):
                        current |= (1 << v_a)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best = current
                        break
                    temp_add ^= low_a
                continue

            # Try 2-2 swaps with full scan
            if current_size >= 2:
                cands_in2 = cands_in[:]
                random.shuffle(cands_in2)
                two_swap_done = False
                for i in range(len(cands_in2)):
                    if time.time() > deadline:
                        break
                    rem1 = cands_in2[i]
                    for j in range(i + 1, len(cands_in2)):
                        rem2 = cands_in2[j]
                        trial = current & ~(1 << rem1) & ~(1 << rem2)
                        # Find two compatible additions with full scan
                        temp3 = ~trial & ((1 << m) - 1)
                        found = []
                        while temp3 and len(found) < 2:
                            low3 = temp3 & (-temp3)
                            add_v = low3.bit_length() - 1
                            if is_compatible(add_v, trial):
                                ok = True
                                for f in found:
                                    if not is_compatible(add_v, trial | (1 << f)):
                                        ok = False
                                        break
                                if ok:
                                    found.append(add_v)
                                    trial |= (1 << add_v)
                            temp3 ^= low3
                        if len(found) == 2:
                            new_bits = current & ~(1 << rem1) & ~(1 << rem2) | (1 << found[0]) | (1 << found[1])
                            current = new_bits
                            two_swap_done = True
                            break
                    if two_swap_done:
                        break

                if two_swap_done:
                    # Try to add more after 2-2 swap
                    temp_add = ~current & ((1 << m) - 1)
                    while temp_add:
                        low_a = temp_add & (-temp_add)
                        v_a = low_a.bit_length() - 1
                        if is_compatible(v_a, current):
                            current |= (1 << v_a)
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best = current
                            break
                        temp_add ^= low_a
                    continue

            if current_size > best_size:
                best_size = current_size
                best = current

        return best, best_size

    def random_greedy():
        """Randomized greedy construction."""
        order = list(range(m))
        random.shuffle(order)
        selected_bits = 0
        for v in order:
            if is_compatible(v, selected_bits):
                selected_bits |= (1 << v)
        return selected_bits

    def greedy_with_exhaustive_add():
        """
        Deterministic construction: greedily add vertices in a fixed order,
        then repeatedly scan for any vertex that can still be added.
        This yields a maximal (not just greedy) independent set and is very fast.
        """
        bits = 0
        # First pass: add in natural order
        for v in range(m):
            if is_compatible(v, bits):
                bits |= (1 << v)
        # Second pass: exhaustive add (may enable more via different orderings)
        changed = True
        while changed:
            changed = False
            for v in range(m):
                if not (bits >> v) & 1 and is_compatible(v, bits):
                    bits |= (1 << v)
                    changed = True
        return bits

    # Main search
    best_bits = 0
    best_size = 0

    # Initial deterministic construction (greedy + exhaustive add)
    best_bits = greedy_with_exhaustive_add()
    best_size = bin(best_bits).count('1')

    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 60:
        restart_count += 1

        # Random greedy
        g_bits = random_greedy()
        g_size = bin(g_bits).count('1')

        # Local search polish with 2-2 swaps
        ls_bits, ls_size = local_search_2swap(g_bits, max_iters=800)

        if ls_size > best_size:
            best_size = ls_size
            best_bits = ls_bits

        # Occasionally run B&B on the vertices *not* in the current best,
        # to see if we can find a larger independent set among the excluded vertices
        # that could replace some current members via swaps.
        if restart_count % 5 == 0 and time.time() < deadline - 90:
            # Consider the complement of the current best as a candidate pool
            complement = ((1 << m) - 1) & ~best_bits
            # Only run B&B if the complement is small enough to be tractable
            comp_size = bin(complement).count('1')
            if comp_size <= 60:
                bb_size, bb_set = branch_and_bound(complement, 0, 0)
                if bb_size > 0:
                    # Try to merge: for each vertex in bb_set, see if it can replace
                    # a vertex in best_bits. This is a heuristic merge.
                    # Simple approach: try adding bb_set vertices one by one via swaps
                    trial = best_bits
                    trial_size = best_size
                    for v in range(m):
                        if (bb_set >> v) & 1:
                            # Try to add v by removing a conflicting vertex
                            conflicts_in_best = conflict_mask[v] & trial
                            if conflicts_in_best:
                                # Remove one conflicting vertex
                                low_c = conflicts_in_best & (-conflicts_in_best)
                                rem_v = low_c.bit_length() - 1
                                trial = (trial & ~(1 << rem_v)) | (1 << v)
                                # trial_size stays the same
                            else:
                                trial |= (1 << v)
                                trial_size += 1
                    if trial_size > best_size:
                        best_size = trial_size
                        best_bits = trial

        # Try high-precision polish on the best found so far
        if restart_count % 3 == 0 and time.time() < deadline - 30:
            hp_bits, hp_size = high_precision_polish(best_bits, max_iters=500)
            if hp_size > best_size:
                best_size = hp_size
                best_bits = hp_bits

    # Final high-precision polish on the best result
    if time.time() < deadline - 5:
        final_bits, final_size = high_precision_polish(best_bits, max_iters=1000)
        if final_size > best_size:
            best_size = final_size
            best_bits = final_bits

    # Convert bitmask to array
    result = []
    for v in range(m):
        if best_bits & (1 << v):
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
