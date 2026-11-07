"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    deadline = time.time() + 85
    best = [31, 32, 34]
    best_len = 3

    # Precompute divisibility multiples mask for each sum value
    # sums can range from 1 to 256*30 = 7680, but realistically with n<=10 it's at most 2560
    # Let's use max_sum = 7680 to be safe
    MAX_SUM = 7680
    # multiples_mask[s] is a bitset where bit m is set if m is a multiple of s and m > s
    # We'll represent bitsets as Python ints (arbitrary precision)
    # Bit i represents value i+1 (1-indexed) or just use bit i for value i
    # Let's use bit i for value i (0-indexed, but we ignore 0)
    # Actually, let's use bit (value-1) for value, so value 1 is bit 0, value 7680 is bit 7679
    
    # Precompute for each s from 1 to MAX_SUM:
    # multiples_mask[s] has bits set at positions (k*s - 1) for k >= 2, k*s <= MAX_SUM
    multiples_mask = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        for m in range(2 * s, MAX_SUM + 1, s):
            mask |= (1 << (m - 1))
        multiples_mask[s] = mask

    def _check_add(sums_bitset, candidate, n_elements):
        """
        Check if we can add candidate to the current set.
        sums_bitset: bitset of current subset sums (bit s-1 set if sum s exists)
        candidate: the value to add
        n_elements: current number of elements
        Returns: (new_sums_bitset, True) if valid, (None, False) otherwise.
        """
        # New sums = old_sums ∪ (old_sums + candidate) ∪ {candidate}
        # In bitset terms:
        # shifted = sums_bitset << candidate  (shift left by candidate positions)
        # But careful: bit i represents value i+1.
        # If we have bit (s-1) set, shifting left by candidate gives bit (s-1+candidate) which represents value s+candidate.
        # So shifted = sums_bitset << candidate
        shifted = sums_bitset << candidate
        
        # new_sums = sums_bitset | shifted | (1 << (candidate - 1))
        new_sums = sums_bitset | shifted | (1 << (candidate - 1))
        
        # Check for collisions: the number of bits set in new_sums should be 2^(n_elements+1) - 1
        # But we can also check if (sums_bitset & shifted) or (sums_bitset & (1 << (candidate-1))) or (shifted & (1 << (candidate-1))) is non-zero
        # Actually, simpler: if len(new_sums bits) != expected, there's a collision.
        # But bit_count is fast.
        expected = (1 << (n_elements + 1)) - 1
        if new_sums.bit_count() != expected:
            return None, False
        
        # Check divisibility: for each sum s in new_sums, no multiple of s (other than s) should be in new_sums.
        # Equivalently: for each s in new_sums, (multiples_mask[s] & new_sums) should be 0.
        # But iterating over all bits in new_sums might be slow.
        # Alternative: for each s in the *original* sums and candidate, check multiples.
        # Actually, we can check: for each s in new_sums, if multiples_mask[s] & new_sums != 0, invalid.
        # To speed up, we can iterate over set bits in new_sums.
        # But new_sums can have up to 2^n bits, which for n=10 is 1024, manageable.
        
        # Even faster: precompute a "forbidden" bitset for the new candidate?
        # Let's just iterate over set bits in new_sums.
        temp = new_sums
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            s = lsb.bit_length()  # This gives position + 1? No, bit_length of (1<<k) is k+1.
            # If lsb = 1 << (s-1), then lsb.bit_length() = s. So s = lsb.bit_length()
            s_val = lsb.bit_length()
            if multiples_mask[s_val] & new_sums:
                return None, False
            temp ^= lsb
        
        return new_sums, True

    def _greedy_bitset(start_elements, rng, deadline, order):
        """Greedy grow using bitset representation."""
        elements = list(start_elements)
        if not elements:
            sums_bitset = 0
        else:
            # Build initial sums bitset
            sums_bitset = 0
            n = len(elements)
            for mask in range(1, 1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += elements[i]
                sums_bitset |= (1 << (s - 1))
        
        candidates = list(order)
        # Remove already used
        used = set(elements)
        candidates = [c for c in candidates if c not in used]
        
        while time.time() < deadline:
            added = False
            for c in candidates:
                ns, ok = _check_add(sums_bitset, c, len(elements))
                if ok:
                    elements.append(c)
                    sums_bitset = ns
                    candidates.remove(c)
                    added = True
                    break
            if not added:
                break
        return elements

    def _validate(elements):
        """Full validation."""
        if not elements:
            return True
        if len(elements) != len(set(elements)):
            return False
        for x in elements:
            if not isinstance(x, int) or isinstance(x, bool) or not (1 <= x <= 256):
                return False
        n = len(elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        if len(sums) != (1 << n) - 1:
            return False
        # Check divisibility
        for s in sums:
            for m in range(2 * s, max(sums) + 1, s):
                if m in sums:
                    return False
        return True

    # Strategy 1: Greedy with different orderings
    orders = [
        list(range(256, 0, -1)),
        list(range(1, 257)),
        list(range(200, 257)) + list(range(1, 201)),
        list(range(128, 257)) + list(range(1, 129)),
        list(range(256, 0, -2)) + list(range(255, 0, -2)),
        list(range(255, 0, -2)) + list(range(256, 0, -2)),
    ]
    
    for trial in range(300):
        if time.time() > deadline:
            break
        rng_t = random.Random(10000 + trial)
        order = list(orders[trial % len(orders)])
        rng_t.shuffle(order)
        # Start with 1-2 large elements
        start_size = rng_t.randint(1, 2)
        start = rng_t.sample(range(100, 257), start_size)
        if not _validate(start):
            continue
        result = _greedy_bitset(start, rng_t, deadline, order)
        if len(result) > best_len:
            best = result
            best_len = len(result)

    # Strategy 2: Local search - swap elements to try to grow
    if time.time() < deadline - 10:
        rng_ls = random.Random(777)
        current = list(best)
        # Try to grow by removing one element and adding two
        for _ in range(5000):
            if time.time() > deadline:
                break
            if len(current) < 2:
                break
            # Remove a random element
            idx = rng_ls.randrange(len(current))
            removed = current[idx]
            new_elements = current[:idx] + current[idx+1:]
            
            # Build sums for new_elements
            n = len(new_elements)
            sums_bitset = 0
            for mask in range(1, 1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += new_elements[i]
                sums_bitset |= (1 << (s - 1))
            
            # Try to add two new elements
            used = set(new_elements)
            available = [c for c in range(1, 257) if c not in used]
            rng_ls.shuffle(available)
            
            added_any = False
            for c1 in available[:20]:
                ns1, ok1 = _check_add(sums_bitset, c1, len(new_elements))
                if not ok1:
                    continue
                # Now try to add c2 to ns1
                for c2 in available[:20]:
                    if c2 == c1:
                        continue
                    ns2, ok2 = _check_add(ns1, c2, len(new_elements) + 1)
                    if ok2:
                        new_set = new_elements + [c1, c2]
                        if len(new_set) > len(current):
                            current = new_set
                            if len(current) > best_len:
                                best = list(current)
                                best_len = len(current)
                        added_any = True
                        break
                if added_any:
                    break
            
            if not added_any:
                # Revert
                current = best[:]
    
    # Strategy 3: Backtracking for small depths
    def _backtrack(depth, current_elements, current_sums, deadline, rng):
        if time.time() > deadline:
            return None
        if depth == 0:
            return list(current_elements)
        
        candidates = list(range(256, 0, -1))
        rng.shuffle(candidates)
        
        for c in candidates:
            if c in current_elements:
                continue
            ns, ok = _check_add(current_sums, c, len(current_elements))
            if ok:
                result = _backtrack(depth - 1, current_elements + [c], ns, deadline, rng)
                if result is not None:
                    return result
        return None

    for target_depth in range(best_len + 1, best_len + 3):
        if time.time() > deadline - 5:
            break
        rng_b = random.Random(99999 + target_depth)
        # Start with empty or small seed
        start = rng_b.sample(range(100, 257), min(2, target_depth))
        if not _validate(start):
            start = []
        n = len(start)
        sums_bitset = 0
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += start[i]
            sums_bitset |= (1 << (s - 1))
        result = _backtrack(target_depth - n, start, sums_bitset, deadline - 5, rng_b)
        if result is not None and len(result) > best_len:
            best = result
            best_len = len(result)

    # Final validation
    if not _validate(best):
        best = [31, 32, 34]

    return [int(x) for x in best]
# EVOLVE-BLOCK-END
