"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start = time.time()
    DEADLINE = 85.0

    def subset_sum_check(selected, target=200):
        """Bitset check: returns True if no subset sums to target."""
        bits = 1
        for x in selected:
            bits |= (bits << x)
            if bits & (1 << target):
                return False
        return True

    def greedy_order(order):
        """Greedy: add elements in given order if valid."""
        selected = []
        bits = 1
        for x in order:
            new_bits = bits | (bits << x)
            if not (new_bits & (1 << 200)):
                selected.append(x)
                bits = new_bits
        return selected

    def structured_block():
        return list(range(66, 101))

    candidates = []

    # Strategy 1: Descending greedy
    candidates.append(greedy_order(range(100, 0, -1)))

    # Strategy 2: Ascending greedy
    candidates.append(greedy_order(range(1, 101)))

    # Strategy 3: Structured block
    candidates.append(structured_block())

    # Strategy 4: Try to extend structured block downward
    base = list(range(66, 101))
    bits = 1
    for x in base:
        bits |= (bits << x)
    extended = list(base)
    for x in range(65, 0, -1):
        if time.time() - start > DEADLINE - 5:
            break
        new_bits = bits | (bits << x)
        if not (new_bits & (1 << 200)):
            extended.append(x)
            bits = new_bits
    candidates.append(extended)

    # Strategy 5: Randomized greedy with multiple seeds
    import random
    rng = random.Random(123)
    for seed in range(30):
        if time.time() - start > DEADLINE - 10:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        candidates.append(greedy_order(order))

    # Strategy 6: Local search from best candidate
    best = max(candidates, key=lambda c: len(c) if subset_sum_check(c) else 0)
    improved = list(best)
    improved_set = set(improved)
    bits = 1
    for x in improved:
        bits |= (bits << x)

    # Try adding elements
    for x in range(1, 101):
        if time.time() - start > DEADLINE - 5:
            break
        if x in improved_set:
            continue
        new_bits = bits | (bits << x)
        if not (new_bits & (1 << 200)):
            improved.append(x)
            improved_set.add(x)
            bits = new_bits

    # Try 1-for-1 swaps to potentially enable more additions
    for i in range(len(improved)):
        if time.time() - start > DEADLINE - 3:
            break
        removed = improved[i]
        bits_without = 1
        for x in improved:
            if x != removed:
                bits_without |= (bits_without << x)
        for a in range(1, 101):
            if time.time() - start > DEADLINE - 3:
                break
            if a == removed or a in improved_set:
                continue
            new_bits = bits_without | (bits_without << a)
            if not (new_bits & (1 << 200)):
                improved[i] = a
                improved_set.discard(removed)
                improved_set.add(a)
                bits = new_bits
                break

    # Try adding again after swaps
    for x in range(1, 101):
        if time.time() - start > DEADLINE - 2:
            break
        if x in improved_set:
            continue
        new_bits = bits | (bits << x)
        if not (new_bits & (1 << 200)):
            improved.append(x)
            improved_set.add(x)
            bits = new_bits

    candidates.append(improved)

    # Strategy 7: Try removing one element from structured block and adding multiple smaller ones
    for skip in range(66, 101):
        if time.time() - start > DEADLINE - 2:
            break
        base_no_skip = [x for x in range(66, 101) if x != skip]
        bits = 1
        for x in base_no_skip:
            bits |= (bits << x)
        trial = list(base_no_skip)
        trial_set = set(trial)
        for x in range(skip - 1, 0, -1):
            if time.time() - start > DEADLINE - 2:
                break
            if x in trial_set:
                continue
            new_bits = bits | (bits << x)
            if not (new_bits & (1 << 200)):
                trial.append(x)
                trial_set.add(x)
                bits = new_bits
        candidates.append(trial)

    # Return the best valid candidate
    best_candidate = [1]
    best_len = 0
    for c in candidates:
        if subset_sum_check(c) and len(c) > best_len:
            best_candidate = c
            best_len = len(c)

    return best_candidate
# EVOLVE-BLOCK-END
