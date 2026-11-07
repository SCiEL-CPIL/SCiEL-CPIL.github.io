"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256, 'maximum_size': 60}

def construct():
    start = time.time()
    deadline = start + 85

    def check_validity_bitset(s):
        """
        Fast validity check using bitsets.
        reachable[k] is a Python int where bit j is set iff sum j is achievable with exactly k elements.
        Returns True iff no sum appears in two different cardinality sets.
        """
        n = len(s)
        max_sum = sum(s)
        if max_sum > 256 * 60:
            max_sum = 256 * 60
        # reachable[k]: bitset of achievable sums with exactly k elements
        reachable = [0] * (n + 1)
        reachable[0] = 1  # sum 0 with 0 elements
        for x in s:
            # Update in reverse
            for k in range(n, 0, -1):
                if reachable[k - 1]:
                    reachable[k] |= (reachable[k - 1] << x)
        # Check: for each sum value, it should appear in at most one cardinality
        # Build a "used" bitset: for each sum, mark which cardinalities can produce it
        # Efficient: iterate over all sums 0..max_sum, check how many reachable[k] have that bit set
        # But that's O(n * max_sum). Alternative: for each k, check if reachable[k] & combined is empty
        combined = 0
        for k in range(n + 1):
            if reachable[k] & combined:
                return False
            combined |= reachable[k]
        return True

    def check_validity_fast(s):
        """
        Even faster: incremental check. Only check new sums added by the last element.
        """
        n = len(s)
        if n <= 1:
            return True
        # Compute reachable for s[:-1]
        prev = s[:-1]
        max_sum_prev = sum(prev)
        reachable = [0] * (n)  # indices 0..n-1 for prev set of size n-1
        reachable[0] = 1
        for x in prev:
            for k in range(len(prev), 0, -1):
                if reachable[k - 1]:
                    reachable[k] |= (reachable[k - 1] << x)
        # Now add last element x_last
        x_last = s[-1]
        # New sums at cardinality k: (reachable[k-1] << x_last) for k=1..n-1
        # Check these don't conflict with existing reachable[k] or with each other
        # Also need to check they don't conflict with each other across different k
        new_reachable = [0] * (n)
        for k in range(1, n):
            if reachable[k - 1]:
                new_reachable[k] = reachable[k - 1] << x_last
        # Check new_reachable[k] doesn't intersect reachable[k] (same cardinality is fine, different is not)
        # Actually: new sums at cardinality k must not appear in reachable[j] for j != k
        # And new sums at cardinality k must not appear in new_reachable[j] for j != k
        # But new sums at same cardinality k CAN overlap with reachable[k] (repeated sums within same cardinality allowed)
        # So: for each k, new_reachable[k] & combined_other_kardinality must be 0
        combined_other = 0
        for k in range(n):
            if k > 0:
                combined_other |= reachable[k]  # existing sums at cardinality k (k>=1)
        # For each new cardinality k:
        for k in range(1, n):
            if new_reachable[k]:
                # Must not intersect with existing sums at OTHER cardinalities
                other_existing = 0
                for j in range(n):
                    if j != k and j > 0:
                        other_existing |= reachable[j]
                if new_reachable[k] & other_existing:
                    return False
                # Must not intersect with new sums at other cardinalities
                other_new = 0
                for j in range(1, n):
                    if j != k:
                        other_new |= new_reachable[j]
                if new_reachable[k] & other_new:
                    return False
        return True

    # Start with powers of 2
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best = base[:]

    # Greedy extension
    used = set(base)
    for x in range(1, 257):
        if time.time() > deadline - 10:
            break
        if x in used:
            continue
        if check_validity_fast(base + [x]):
            best = base + [x]
            base = best
            used.add(x)

    # Simulated annealing to escape local optima
    current = best[:]
    current_set = set(current)
    best_size = len(current)
    
    # Precompute which elements are "free" (can be added) for current set
    def can_add(s, x):
        return check_validity_fast(s + [x])
    
    # SA: try swaps and additions
    temp = 1.0
    cooling = 0.995
    min_temp = 0.01
    random.seed(42)
    
    while temp > min_temp and time.time() < deadline - 15:
        # Choose a random move
        move_type = random.random()
        if move_type < 0.4 and len(current) < 60:
            # Try adding a random unused element
            unused = [x for x in range(1, 257) if x not in current_set]
            if not unused:
                temp *= cooling
                continue
            x = random.choice(unused)
            new_set = current + [x]
            if check_validity_fast(new_set):
                # Accept (improvement in size)
                current = new_set
                current_set = set(current)
                if len(current) > best_size:
                    best_size = len(current)
                    best = current[:]
                temp *= cooling
            else:
                temp *= cooling
        elif move_type < 0.8:
            # Swap: remove one, add another
            if len(current) < 2:
                temp *= cooling
                continue
            remove_idx = random.randint(0, len(current) - 1)
            removed = current[remove_idx]
            unused = [x for x in range(1, 257) if x not in current_set or x == removed]
            if not unused:
                temp *= cooling
                continue
            add_x = random.choice(unused)
            if add_x == removed:
                temp *= cooling
                continue
            new_set = [e for i, e in enumerate(current) if i != remove_idx] + [add_x]
            if check_validity_fast(new_set):
                # Same size, accept with some probability
                if random.random() < 0.3:  # Accept same-size valid swaps
                    current = new_set
                    current_set = set(current)
                temp *= cooling
            else:
                temp *= cooling
        else:
            # Remove one element (to allow future additions)
            if len(current) < 3:
                temp *= cooling
                continue
            remove_idx = random.randint(0, len(current) - 1)
            removed = current[remove_idx]
            new_set = [e for i, e in enumerate(current) if i != remove_idx]
            current = new_set
            current_set = set(current)
            temp *= cooling
        
        temp *= cooling

    # Final greedy pass
    used = set(best)
    for x in range(1, 257):
        if time.time() > deadline - 5:
            break
        if x in used:
            continue
        if check_validity_fast(best + [x]):
            best = best + [x]
            used.add(x)

    # Second SA round if time permits
    if time.time() < deadline - 20 and len(best) < 60:
        current = best[:]
        current_set = set(current)
        best_size = len(current)
        temp = 0.5
        cooling = 0.995
        min_temp = 0.01
        random.seed(43)
        
        while temp > min_temp and time.time() < deadline - 20:
            move_type = random.random()
            if move_type < 0.5 and len(current) < 60:
                unused = [x for x in range(1, 257) if x not in current_set]
                if not unused:
                    temp *= cooling
                    continue
                x = random.choice(unused)
                new_set = current + [x]
                if check_validity_fast(new_set):
                    current = new_set
                    current_set = set(current)
                    if len(current) > best_size:
                        best_size = len(current)
                        best = current[:]
                    temp *= cooling
                else:
                    temp *= cooling
            elif move_type < 0.85:
                if len(current) < 2:
                    temp *= cooling
                    continue
                remove_idx = random.randint(0, len(current) - 1)
                removed = current[remove_idx]
                unused = [x for x in range(1, 257) if x not in current_set or x == removed]
                if not unused:
                    temp *= cooling
                    continue
                add_x = random.choice(unused)
                if add_x == removed:
                    temp *= cooling
                    continue
                new_set = [e for i, e in enumerate(current) if i != remove_idx] + [add_x]
                if check_validity_fast(new_set):
                    if random.random() < 0.4:
                        current = new_set
                        current_set = set(current)
                    temp *= cooling
                else:
                    temp *= cooling
            else:
                if len(current) < 3:
                    temp *= cooling
                    continue
                remove_idx = random.randint(0, len(current) - 1)
                new_set = [e for i, e in enumerate(current) if i != remove_idx]
                current = new_set
                current_set = set(current)
                temp *= cooling
            temp *= cooling

    # Final greedy pass again
    used = set(best)
    for x in range(1, 257):
        if time.time() > deadline - 3:
            break
        if x in used:
            continue
        if check_validity_fast(best + [x]):
            best = best + [x]
            used.add(x)

    result = [int(x) for x in best]
    return result
# EVOLVE-BLOCK-END
