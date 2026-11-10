"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    n = 128
    numbers = list(range(1, n + 1))

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # Build forbidden pairs: for each x, list of (u,v) such that {x,u,v} is a bad triple
    forbidden_pairs = {i: [] for i in range(1, n + 1)}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    forbidden_pairs[a].append((b, c))
                    forbidden_pairs[b].append((a, c))
                    forbidden_pairs[c].append((a, b))

    # Degree of each element (number of forbidden pairs)
    deg = {i: len(forbidden_pairs[i]) for i in numbers}

    def can_add(x, mask):
        for u, v in forbidden_pairs[x]:
            if ((mask >> (u - 1)) & 1) and ((mask >> (v - 1)) & 1):
                return False
        return True

    def greedy_fill(mask, order):
        for x in order:
            if (mask >> (x - 1)) & 1:
                continue
            if can_add(x, mask):
                mask |= (1 << (x - 1))
        return mask

    def try_additions(mask, s_set):
        added = False
        for x in greedy_order:
            if x in s_set:
                continue
            if can_add(x, mask):
                mask |= (1 << (x - 1))
                s_set.add(x)
                added = True
        return mask, s_set, added

    # Multiple orderings for diversity
    deg_asc = sorted(numbers, key=lambda x: deg[x])
    deg_desc = sorted(numbers, key=lambda x: -deg[x])
    natural = list(range(1, n + 1))
    reverse_natural = list(range(n, 0, -1))

    # Best solution tracking
    best_size = 0
    best_mask = 0

    # Phase 1: Multiple greedy initializations with different orderings
    initial_orders = [deg_asc, deg_desc, natural, reverse_natural]
    # Add a few random orders
    for _ in range(4):
        r = numbers[:]
        random.shuffle(r)
        initial_orders.append(r)

    for order in initial_orders:
        if time.time() > deadline - 20:
            break
        m = 0
        for x in order:
            if can_add(x, m):
                m |= (1 << (x - 1))
        # Refill with degree-ascending order to fill gaps
        m = greedy_fill(m, deg_asc)
        m = greedy_fill(m, natural)
        if m.bit_count() > best_size:
            best_size = m.bit_count()
            best_mask = m

    # Phase 2: Local search with swaps and perturbations
    # Start from best_mask
    current_mask = best_mask
    current_set = set(i + 1 for i in range(n) if (current_mask >> i) & 1)
    best_mask = current_mask
    best_size = current_mask.bit_count()

    improved = True
    no_improve_count = 0

    while improved and time.time() < deadline - 5:
        improved = False

        # Try all additions
        current_mask, current_set, added = try_additions(current_mask, current_set)
        if added:
            if current_mask.bit_count() > best_size:
                best_size = current_mask.bit_count()
                best_mask = current_mask
            improved = True
            no_improve_count = 0
            continue

        # Try random swaps (up to 30 per outer iteration)
        swaps_tried = 0
        max_swaps = 30
        while swaps_tried < max_swaps and time.time() < deadline - 5:
            x = random.choice(list(current_set))
            valid_y = None
            for y in deg_asc:
                if y in current_set:
                    continue
                temp_mask = current_mask & ~(1 << (x - 1))
                if can_add(y, temp_mask):
                    valid_y = y
                    break
            if valid_y is None:
                # Try a few random y's
                for _ in range(5):
                    y = random.choice(numbers)
                    if y in current_set:
                        continue
                    temp_mask = current_mask & ~(1 << (x - 1))
                    if can_add(y, temp_mask):
                        valid_y = y
                        break
            if valid_y is None:
                swaps_tried += 1
                continue

            # Perform swap
            current_mask = (current_mask & ~(1 << (x - 1))) | (1 << (valid_y - 1))
            current_set.remove(x)
            current_set.add(valid_y)
            swaps_tried += 1

            # Try to add more
            current_mask, current_set, added = try_additions(current_mask, current_set)
            if current_mask.bit_count() > best_size:
                best_size = current_mask.bit_count()
                best_mask = current_mask
                improved = True
                no_improve_count = 0
                break

        # If no improvement, try perturbation: remove k elements and refill
        if not improved:
            no_improve_count += 1
            # Remove 2-4 random elements
            removals = min(random.randint(2, 4), len(current_set) // 3 + 1)
            to_remove = random.sample(list(current_set), removals)
            for x in to_remove:
                current_mask &= ~(1 << (x - 1))
                current_set.remove(x)

            # Greedy refill with a random order
            refill_order = deg_asc[:]
            random.shuffle(refill_order)
            current_mask = greedy_fill(current_mask, refill_order)
            current_mask = greedy_fill(current_mask, deg_asc)
            current_set = set(i + 1 for i in range(n) if (current_mask >> i) & 1)

            if current_mask.bit_count() > best_size:
                best_size = current_mask.bit_count()
                best_mask = current_mask
                improved = True
                no_improve_count = 0

            # If no improvement after several perturbations, restart from a new greedy
            if no_improve_count > 8:
                # New random greedy initialization
                r = numbers[:]
                random.shuffle(r)
                m = 0
                for x in r:
                    if can_add(x, m):
                        m |= (1 << (x - 1))
                m = greedy_fill(m, deg_asc)
                m = greedy_fill(m, natural)
                current_mask = m
                current_set = set(i + 1 for i in range(n) if (m >> i) & 1)
                no_improve_count = 0

        if time.time() > deadline - 5:
            break

    # Final greedy fill on best_mask
    best_mask = greedy_fill(best_mask, deg_asc)
    best_mask = greedy_fill(best_mask, natural)

    # Extract result
    result = [i + 1 for i in range(n) if (best_mask >> i) & 1]

    # Sanity Check
    is_set = set(result)
    valid = True
    for a in result:
        for b in result:
            if a >= b:
                continue
            for c in result:
                if b >= c:
                    continue
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to a simple greedy
        current_mask = 0
        for x in deg_asc:
            if can_add(x, current_mask):
                current_mask |= (1 << (x - 1))
        result = sorted(i + 1 for i in range(n) if (current_mask >> i) & 1)

    return result

# EVOLVE-BLOCK-END
