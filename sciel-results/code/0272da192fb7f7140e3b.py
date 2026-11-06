import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin from 90s limit

    # Bitset: bit i set means sum i is reachable (we track 0..200)
    mask = (1 << 201) - 1
    BIT_200 = 1 << 200

    def can_add(b, x):
        # O(1) check: does adding x create a sum of 200?
        return not ((b | (b << x)) & BIT_200)

    def recompute_bits(elements, mask):
        b = 1
        for e in elements:
            b |= (b << e)
            b &= mask
        return b

    def verify_no_sum_200(s):
        b = 1
        for e in s:
            b |= (b << e)
            b &= mask
        return ((b >> 200) & 1) == 0

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
        sorted(range(1, 101), key=lambda x: -x / random.random()),
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

    def try_swap_add(s, b, size, removal_cache):
        """Try 1-swap + 1-add using precomputed removal bitsets."""
        elements = list(s)
        random.shuffle(elements)
        candidates = [x for x in range(1, 101) if x not in s]
        random.shuffle(candidates)

        for y in elements:
            b_minus = removal_cache.get(y)
            if b_minus is None:
                continue

            for x in candidates:
                if can_add(b_minus, x):
                    ns = s - {y} | {x}
                    nb = (b_minus | (b_minus << x)) & mask
                    # Now try to add a z to this new set
                    for z in range(1, 101):
                        if z not in ns and can_add(nb, z):
                            nns = ns | {z}
                            nnb = (nb | (nb << z)) & mask
                            return nns, nnb, size + 1
        return None

    def try_double_swap(s, b, size, removal_cache):
        """Try removing 2 elements and adding 2 elements (net 0), then add 1."""
        elements = list(s)
        random.shuffle(elements)
        candidates = [x for x in range(1, 101) if x not in s]
        random.shuffle(candidates)

        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() - start_time > DEADLINE:
                    return None
                y1, y2 = elements[i], elements[j]
                
                s_minus = s - {y1, y2}
                b_minus = recompute_bits(list(s_minus), mask)

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
                                for z in range(1, 101):
                                    if z not in s2 and can_add(b2, z):
                                        ns = s2 | {z}
                                        nb = (b2 | (b2 << z)) & mask
                                        return ns, nb, size + 1
                                found = True
                                break
        return None

    # Main search loop
    improved = True
    stagnation = 0
    max_stagnation = 20

    while improved and (time.time() - start_time < DEADLINE):
        improved = False

        # Precompute removal bitsets for current set
        removal_cache = {}
        s_list = list(s)
        for y in s_list:
            s_minus = s - {y}
            removal_cache[y] = recompute_bits(s_minus, mask)

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

        # Try 1-swap + 1-add
        result = try_swap_add(s, b, size, removal_cache)
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
        result = try_double_swap(s, b, size, removal_cache)
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

            for _ in range(10):
                result = try_add(s, b, size)
                if result is None:
                    break
                s, b, size = result
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size

    # Final verification
    if not verify_no_sum_200(global_best_set):
        if verify_no_sum_200(best_set):
            return list(best_set)
        return [100]

    if len(global_best_set) > 0:
        return list(global_best_set)
    return list(s)
