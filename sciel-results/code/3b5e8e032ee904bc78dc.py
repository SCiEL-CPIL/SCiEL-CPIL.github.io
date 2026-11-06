import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin from 90s limit

    # Bitset: bit i set means sum i is reachable (we track 0..200)
    # Adding x: new = b | (b << x); valid iff bit 200 stays clear
    mask = (1 << 201) - 1
    TARGET = 200

    def can_add(b, x):
        # O(1) check
        return ((b | (b << x)) >> TARGET) & 1 == 0

    def recompute_bits(elements):
        b = 1
        for e in elements:
            b |= (b << e)
            b &= mask
        return b

    # Precompute: for each element x, which sums does it "reach" when added?
    # Actually, let's precompute a faster removal method using prefix/suffix bitsets
    # For a sorted list, we can compute prefix and suffix reachable sums
    # Then removing element at position i: combine prefix[i-1] and suffix[i+1]
    
    def build_prefix_suffix(elements):
        """Build prefix and suffix bitset arrays for fast removal."""
        n = len(elements)
        if n == 0:
            return [], []
        prefix = [0] * n
        suffix = [0] * n
        # prefix[i] = bitset of sums reachable using elements[0..i]
        b = 1
        for i in range(n):
            b |= (b << elements[i])
            b &= mask
            prefix[i] = b
        # suffix[i] = bitset of sums reachable using elements[i..n-1]
        b = 1
        for i in range(n-1, -1, -1):
            b |= (b << elements[i])
            b &= mask
            suffix[i] = b
        return prefix, suffix

    def bits_without(sorted_elements, idx):
        """Get bitset of sums reachable without element at index idx in sorted list."""
        n = len(sorted_elements)
        if n == 1:
            return 1
        # Build prefix and suffix on the fly (cached per call pattern)
        # For efficiency, we'll compute incrementally
        prefix = [0] * n
        b = 1
        for i in range(n):
            b |= (b << sorted_elements[i])
            b &= mask
            prefix[i] = b
        
        # Now compute the bitset without element idx
        # Combine prefix[idx-1] with the "rest" after idx
        if idx == 0:
            # Only elements[1..n-1]
            b = 1
            for i in range(1, n):
                b |= (b << sorted_elements[i])
                b &= mask
            return b
        elif idx == n - 1:
            return prefix[idx - 1]
        else:
            # prefix[idx-1] combined with elements[idx+1..n-1]
            b = prefix[idx - 1]
            for i in range(idx + 1, n):
                b |= (b << sorted_elements[i])
                b &= mask
            return b

    # ---- Phase 1: Greedy initialization with multiple orders ----
    best_set = set()
    best_bits = 1
    best_size = 0

    # Key insight: large elements are "safer" to include since they contribute
    # more to reaching 200, so they block fewer combinations.
    # But small elements create many subset sums. Let's try multiple strategies.
    
    orders = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),      # ascending
        list(range(50, 0, -1)) + list(range(100, 50, -1)),  # middle-out
        list(range(100, 49, -1)) + list(range(50, 0, -1)),  # high-first then low
        list(range(1, 51)) + list(range(100, 49, -1)),      # low-first then high
    ]
    
    # Add a few random orders
    for _ in range(3):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)

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
        """Try to add multiple elements greedily until no more can be added."""
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
        """Try 1-swap + 1-add: remove y, add x, then add z. Net gain +1."""
        elements = sorted(s)
        candidates = [x for x in range(1, 101) if x not in s]
        random.shuffle(candidates)
        candidates = candidates[:25]  # Limit for speed
        
        for y in elements:
            if time.time() - start_time > DEADLINE:
                return None
            # Remove y and recompute
            s_minus = s - {y}
            b_minus = recompute_bits(list(s_minus))

            for x in candidates:
                if can_add(b_minus, x):
                    ns = s_minus | {x}
                    nb = (b_minus | (b_minus << x)) & mask
                    # Try to add one more element z
                    for z in range(1, 101):
                        if z not in ns and can_add(nb, z):
                            return ns | {z}, (nb | (nb << z)) & mask, size + 1
        return None

    def try_double_swap(s, b, size):
        """Try removing 2, adding 2, then adding 1 more. Net gain +1."""
        elements = sorted(s)
        candidates = [x for x in range(1, 101) if x not in s]
        random.shuffle(candidates)
        candidates = candidates[:15]  # Limit for speed
        
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() - start_time > DEADLINE:
                    return None
                y1, y2 = elements[i], elements[j]
                s_minus = s - {y1, y2}
                b_minus = recompute_bits(list(s_minus))

                for x in candidates:
                    if can_add(b_minus, x):
                        b1 = (b_minus | (b_minus << x)) & mask
                        s1 = s_minus | {x}
                        for x2 in candidates:
                            if x2 == x:
                                continue
                            if can_add(b1, x2):
                                b2 = (b1 | (b1 << x2)) & mask
                                s2 = s1 | {x2}
                                # Try to add one more
                                for z in range(1, 101):
                                    if z not in s2 and can_add(b2, z):
                                        return s2 | {z}, (b2 | (b2 << z)) & mask, size + 1
        return None

    # Main search loop with improved perturbation strategy
    improved = True
    stagnation = 0
    max_stagnation = 15

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

        if time.time() - start_time > DEADLINE:
            break

        # Perturbation: try different strategies based on stagnation level
        elements = list(s)
        if len(elements) > 3:
            # For mild stagnation: remove 2-3 random elements
            # For heavy stagnation: remove 4-5 elements for bigger jump
            if stagnation < 5:
                n_remove = random.choice([2, 3])
            else:
                n_remove = random.choice([3, 4, 5])
            
            n_remove = min(n_remove, len(elements) - 1)
            to_remove = random.sample(elements, n_remove)
            s_pert = s - set(to_remove)
            b_pert = recompute_bits(list(s_pert))
            s = s_pert
            b = b_pert
            size = len(s)

            # Aggressively try to re-add
            result = try_multi_add(s, b, size)
            if result is not None:
                s, b, size = result
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size

    # Final: try one more round of multi-add on the best set
    if time.time() - start_time < DEADLINE:
        result = try_multi_add(global_best_set, global_best_bits, global_best_size)
        if result is not None:
            global_best_set, global_best_bits, global_best_size = result

    # Return the best found
    if len(global_best_set) > 0:
        return list(global_best_set)
    return list(s)
