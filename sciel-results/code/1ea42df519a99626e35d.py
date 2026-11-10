import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50
    n = 256

    # Precompute conflict masks for all pairs
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    # Odds are pairwise compatible: for distinct odd a,b, a+b is even, a*b is odd,
    # so (a*b) % (a+b) != 0. Fix all odds in the solution.
    odds_mask = 0
    for x in range(1, n + 1, 2):
        odds_mask |= (1 << x)

    evens = list(range(2, n + 1, 2))

    # Conflict count among all vertices (for ordering)
    conflict_count = [0] * (n + 1)
    for a in range(1, n + 1):
        m = conflict_mask[a]
        while m:
            conflict_count[a] += 1
            m &= m - 1

    # Greedy helper: add evens in given order to a base mask
    def greedy_add(base_mask, order):
        cur = base_mask
        for e in order:
            if (conflict_mask[e] & cur) == 0:
                cur |= (1 << e)
        return cur

    # Local search: try to add any compatible vertex (mainly evens)
    def local_search_add(mask):
        improved = True
        while improved:
            improved = False
            for x in evens:
                if not (mask >> x) & 1:
                    if (conflict_mask[x] & mask) == 0:
                        mask |= (1 << x)
                        improved = True
                        break
        return mask

    # Build a strong initial incumbent: all odds + greedy evens by conflict count
    evens_sorted = sorted(evens, key=lambda x: conflict_count[x])
    best_mask = local_search_add(greedy_add(odds_mask, evens_sorted))
    best_size = best_mask.bit_count()

    # Try a few random greedy orders to improve incumbent
    for _ in range(200):
        order = evens[:]
        random.shuffle(order)
        cand = local_search_add(greedy_add(odds_mask, order))
        sz = cand.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = cand

    # Branch and bound over evens only.
    # Represent candidate evens as a bitmask over even vertices.
    # We keep the full vertex mask for conflict checks, but only branch on evens.

    # Map even value -> bit position in a compact 128-bit mask for speed? 
    # Simpler: use the same 256-bit mask but only set even bits in candidates.
    all_evens_mask = 0
    for e in evens:
        all_evens_mask |= (1 << e)

    def greedy_color_bound(cand_mask):
        """Upper bound on independent set size in cand_mask using greedy coloring."""
        if not cand_mask:
            return 0
        remaining = cand_mask
        color_count = 0
        while remaining:
            color_count += 1
            # Find vertex with max degree in remaining
            best_v = -1
            max_deg = -1
            temp = remaining
            while temp:
                bit = temp & (-temp)
                v = bit.bit_length() - 1
                deg = (conflict_mask[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= bit
            if best_v == -1:
                break
            # Remove best_v and its neighbors
            remaining &= ~(1 << best_v)
            remaining &= ~conflict_mask[best_v]
        return color_count

    def bb(candidates, current_mask, current_size):
        nonlocal best_size, best_mask

        if time.time() > deadline:
            return

        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        # Upper bound
        ub = current_size + greedy_color_bound(candidates)
        if ub <= best_size:
            return

        # Select pivot: max degree in candidates
        pivot = -1
        max_deg = -1
        temp = candidates
        while temp:
            bit = temp & (-temp)
            v = bit.bit_length() - 1
            deg = (conflict_mask[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp ^= bit

        if pivot == -1:
            return

        # Branch 1: Include pivot
        new_cand = candidates & ~(1 << pivot) & ~conflict_mask[pivot]
        bb(new_cand, current_mask | (1 << pivot), current_size + 1)

        if time.time() > deadline:
            return

        # Branch 2: Exclude pivot
        new_cand = candidates & ~(1 << pivot)
        bb(new_cand, current_mask, current_size)

    # Start B&B from the odds base mask
    bb(all_evens_mask, odds_mask, odds_mask.bit_count())

    # Final local search on best
    best_mask = local_search_add(best_mask)
    best_size = best_mask.bit_count()

    # Convert to list
    result = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    return result
