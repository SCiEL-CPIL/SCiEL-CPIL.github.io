# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 580.0

    # Compute forbidden set and build bitset conflict masks
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # conflict_mask[i] is a bitmask where bit j is set if (i-j) % m is in forbidden
    # i.e., i and j cannot both be in the independent set
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Also precompute: for a candidate c and current set S (as bitmask),
    # c conflicts with S if conflict_mask[c] & S != 0

    best_size = 0
    best_set = 0  # bitmask of best independent set found

    def _branch_and_bound(remaining, current_mask, current_size):
        """Branch and bound for maximum independent set.
        remaining: sorted list of candidate vertices (as bitmask or list)
        current_mask: bitmask of currently selected vertices
        current_size: number of currently selected vertices
        Returns (best_size, best_mask)
        """
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if not remaining:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return

        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best_size:
            return

        # Order remaining by degree (fewest conflicts first for better pruning)
        # Actually, for independent set, we want vertices with fewer conflicts
        # to be considered first (they're more likely to be addable)
        # Sort by number of conflicts within remaining
        remaining_sorted = sorted(remaining, key=lambda v: bin(conflict_mask[v] & remaining_mask).count('1'))

        remaining_mask = 0
        for v in remaining:
            remaining_mask |= (1 << v)

        # Try including the first vertex
        v = remaining_sorted[0]
        v_bit = 1 << v

        # Check if v can be added (no conflict with current)
        if conflict_mask[v] & current_mask == 0:
            # Include v
            new_remaining = [u for u in remaining_sorted[1:] if (conflict_mask[u] & v_bit) == 0]
            _branch_and_bound(new_remaining, current_mask | v_bit, current_size + 1)

            # Exclude v
            remaining_exclude = [u for u in remaining_sorted[1:]]
            if time.time() > deadline:
                return
            _branch_and_bound(remaining_exclude, current_mask, current_size)
        else:
            # v conflicts with current, must exclude
            remaining_exclude = [u for u in remaining_sorted[1:]]
            _branch_and_bound(remaining_exclude, current_mask, current_size)

    # Use a smarter approach: iterative deepening or just branch-and-bound with good ordering
    # For m=205, we need a more efficient approach.
    # Let's use a different strategy: greedy coloring bound + branch and bound

    def _greedy_color_bound(candidates_mask):
        """Compute a greedy coloring upper bound for the subgraph induced by candidates.
        Returns an upper bound on the size of independent set in candidates."""
        if candidates_mask == 0:
            return 0

        # Greedy coloring: assign colors to vertices such that adjacent vertices get different colors
        # The number of colors used is an upper bound on the independence number
        uncolored = candidates_mask
        color = 0
        while uncolored:
            color += 1
            # Find a maximal independent set in uncolored (greedy)
            color_class = 0
            remaining = uncolored
            while remaining:
                # Pick vertex with minimum degree in remaining
                best_v = -1
                best_deg = float('inf')
                temp = remaining
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    deg = bin(conflict_mask[v] & remaining).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                if best_v >= 0:
                    v_bit = 1 << best_v
                    color_class |= v_bit
                    remaining &= ~v_bit
                    # Remove all neighbors of best_v from remaining
                    remaining &= ~conflict_mask[best_v]
            uncolored &= ~color_class

        return color

    # Branch and bound with coloring bound
    def _bnb(candidates, current_mask, current_size):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return

        # Upper bound using coloring
        cand_mask = 0
        for v in candidates:
            cand_mask |= (1 << v)
        ub = _greedy_color_bound(cand_mask)
        if current_size + ub <= best_size:
            return

        # Order candidates by degree (ascending) for better branching
        candidates_sorted = sorted(candidates, key=lambda v: bin(conflict_mask[v] & cand_mask).count('1'))

        v = candidates_sorted[0]
        v_bit = 1 << v
        rest = candidates_sorted[1:]

        # Branch 1: include v
        if conflict_mask[v] & current_mask == 0:
            new_candidates = [u for u in rest if (conflict_mask[u] & v_bit) == 0]
            _bnb(new_candidates, current_mask | v_bit, current_size + 1)

        # Branch 2: exclude v
        if time.time() > deadline:
            return
        _bnb(rest, current_mask, current_size)

    # Initialize with a greedy solution
    def _greedy_solve(order):
        selected = 0
        size = 0
        for v in order:
            if conflict_mask[v] & selected == 0:
                selected |= (1 << v)
                size += 1
        return selected, size

    # Get initial solution from greedy
    random.seed(42)
    best_initial = 0
    best_initial_size = 0
    for _ in range(200):
        order = list(range(m))
        random.shuffle(order)
        sol, sz = _greedy_solve(order)
        if sz > best_initial_size:
            best_initial_size = sz
            best_initial = sol

    best_size = best_initial_size
    best_set = best_initial

    # Now run branch and bound starting from full set
    # But m=205 is too large for naive B&B. We need to be strategic.
    # Let's try: start with the greedy solution, then try to improve via B&B on the "gap"
    # Or use a time-limited B&B that explores promising branches

    # Alternative: use the B&B but with aggressive pruning and time limit
    # Start with all vertices as candidates
    all_candidates = list(range(m))

    # Run B&B with time limit
    _bnb(all_candidates, 0, 0)

    # If we still have time, try local search improvements
    if time.time() < deadline - 10:
        # Convert best_set to list
        best_list = [i for i in range(m) if (best_set >> i) & 1]
        best_set_list = set(best_list)

        # Try to add vertices
        for v in range(m):
            if v not in best_set_list and conflict_mask[v] & best_set == 0:
                best_set |= (1 << v)
                best_size += 1
                best_set_list.add(v)

        # Try swap: remove one, add two
        for s in list(best_set_list):
            s_bit = 1 << s
            new_set = best_set & ~s_bit
            new_size = best_size - 1
            added = 0
            for v in range(m):
                if v == s or (new_set >> v) & 1:
                    continue
                if conflict_mask[v] & new_set == 0:
                    new_set |= (1 << v)
                    new_size += 1
                    added += 1
                    if added == 2:
                        break
            if new_size > best_size:
                best_set = new_set
                best_size = new_size
                best_set_list = set(i for i in range(m) if (best_set >> i) & 1)

    # Extract result
    result = [i for i in range(m) if (best_set >> i) & 1]
    return np.array(sorted(result), dtype=np.int64)


# EVOLVE-BLOCK-END
