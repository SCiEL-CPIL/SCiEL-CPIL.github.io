import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin from 90s limit

    # Bitset: bit i set means sum i is reachable (we track 0..200)
    # Adding x: new = b | (b << x); valid iff bit 200 stays clear
    # Removing x from set S (bits b): we must recompute by processing S\{x}
    #   (but we can cache per-element contribution for fast swap checks)

    def can_add(b, x):
        # O(1) check
        return ((b | (b << x)) >> 200) & 1 == 0

    def recompute_bits(elements, mask):
        b = 1
        for e in elements:
            b |= (b << e)
            b &= mask
        return b

    mask = (1 << 201) - 1

    # ---- Phase 1: Greedy initialization with multiple orders ----
    best_set = set()
    best_bits = 1
    best_size = 0

    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        sorted(range(1, 101), key=lambda x: random.random()),
        sorted(range(1, 101), key=lambda x: random.random()),
    ]

    for order in orders:
        s = set()
        b = 1
        for x in order:
            if can_add(b, x):
                s.add(x)
                b = (b | (b << x)) & mask
        if len(s) > best_size:
            best_set = s
            best_bits = b
            best_size = len(s)

    # ---- Phase 2: Aggressive local search ----
    s = set(best_set)
    b = best_bits
    size = best_size

    # Precompute for each element its "contribution" to the bitset
    # For removal: recompute_bits is O(|S|) but we can do it smarter.
    # For a swap (remove y, add x): recompute S\{y} then check x.

    # Use a list for the set to enable faster iteration
    def set_to_list(st):
        return sorted(st)

    # Track best
    global_best_set = set(s)
    global_best_bits = b
    global_best_size = size

    def try_add(s, b, size):
        """Try to add any element. Returns (new_s, new_b, new_size) or None."""
        for x in range(1, 101):
            if x not in s and can_add(b, x):
                ns = s | {x}
                nb = (b | (b << x)) & mask
                return ns, nb, size + 1
        return None

    def try_multi_add(s, b, size):
        """Try to add multiple elements greedily. Returns (new_s, new_b, new_size) or None."""
        ns = set(s)
        nb = b
        nsize = size
        changed = True
        while changed:
            changed = False
            for x in range(1, 101):
                if x not in ns and can_add(nb, x):
                    ns.add(x)
                    nb = (nb | (nb << x)) & mask
                    nsize += 1
                    changed = True
        if nsize > size:
            return ns, nb, nsize
        return None

    def try_swap_add(s, b, size):
        """Try 1-swap + 1-add. Returns (new_s, new_b, new_size) or None."""
        elements = list(s)
        random.shuffle(elements)
        candidates = [x for x in range(1, 101) if x not in s]
        random.shuffle(candidates)
        
        # Limit candidates to speed up search
        candidates = candidates[:30]

        for y in elements:
            # Remove y: recompute bits
            s_minus = s - {y}
            b_minus = recompute_bits(list(s_minus), mask)

            # Try to add any x to s_minus
            for x in candidates:
                if can_add(b_minus, x):
                    ns = s_minus | {x}
                    nb = (b_minus | (b_minus << x)) & mask
                    # Now try to add a z to this new set
                    for z in range(1, 101):
                        if z not in ns and can_add(nb, z):
                            nns = ns | {z}
                            nnb = (nb | (nb << z)) & mask
                            return nns, nnb, size + 1
                    # Even without extra add, accept if we can add at least one
                    # Actually we need net gain, so just the swap+add gives size+1
                    # Wait: swap removes y, adds x -> same size. Then add z -> size+1
                    # If we can't add z, the swap alone doesn't help.
                    # So we only accept if we can add z.
                    # Let's also try: swap y->x, then add z
                    # That's what we're doing.
        return None

    def try_double_swap(s, b, size):
        """Try removing 2 elements and adding 2 elements (net 0), then add 1."""
        elements = list(s)
        random.shuffle(elements)
        candidates = [x for x in range(1, 101) if x not in s]
        random.shuffle(candidates)
        
        # Limit candidates to speed up search
        candidates = candidates[:20]

        # Try pairs of removals
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() - start_time > DEADLINE:
                    return None
                y1, y2 = elements[i], elements[j]
                s_minus = s - {y1, y2}
                b_minus = recompute_bits(list(s_minus), mask)

                # Try adding two candidates
                found = False
                for x in candidates:
                    if found:
                        break
                    if can_add(b_minus, x):
                        b1 = (b_minus | (b_minus << x)) & mask
                        s1 = s_minus | {x}
                        for x2 in candidates:
                            if x2 == x:
                                continue
                            if can_add(b1, x2):
                                b2 = (b1 | (b1 << x2)) & mask
                                s2 = s1 | {x2}
                                # Now try to add a third
                                for z in range(1, 101):
                                    if z not in s2 and can_add(b2, z):
                                        ns = s2 | {z}
                                        nb = (b2 | (b2 << z)) & mask
                                        return ns, nb, size + 1
                                # Even without third, we have size+1 (removed 2, added 2, but that's net 0)
                                # Wait: removed 2, added 2 -> same size. Need +1 more.
                                # So we need the third add.
                                found = True
                                break
        return None

    # Main search loop
    improved = True
    stagnation = 0
    max_stagnation = 20

    while improved and (time.time() - start_time < DEADLINE):
        improved = False

        # Try simple additions first
        result = try_add(s, b, size)
        if result is not None:
            s, b, size = result
            improved = True
            stagnation = 0
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size
            continue

        # Try multiple additions
        result = try_multi_add(s, b, size)
        if result is not None:
            s, b, size = result
            improved = True
            stagnation = 0
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size
            continue

        # Try 1-swap + 1-add
        result = try_swap_add(s, b, size)
        if result is not None:
            s, b, size = result
            improved = True
            stagnation = 0
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size
            continue

        # Try 2-swap + 1-add
        result = try_double_swap(s, b, size)
        if result is not None:
            s, b, size = result
            improved = True
            stagnation = 0
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size
            continue

        # No improvement this round
        stagnation += 1
        if stagnation >= max_stagnation:
            break

        # Random restart from a slightly perturbed state
        if time.time() - start_time > DEADLINE:
            break

        # Perturbation: remove 3 random elements, try to re-add
        elements = list(s)
        if len(elements) > 3:
            to_remove = random.sample(elements, 3)
            s_pert = s - set(to_remove)
            b_pert = recompute_bits(list(s_pert), mask)
            s = s_pert
            b = b_pert
            size = len(s)

            # Try to re-add
            for _ in range(10):
                result = try_add(s, b, size)
                if result is None:
                    break
                s, b, size = result
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size

    # Return the best found
    if len(global_best_set) > 0:
        return list(global_best_set)
    return list(s)
