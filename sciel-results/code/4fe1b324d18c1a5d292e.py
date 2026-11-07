import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    n = 12
    m = 40
    total_masks = 1 << n  # 4096 possible subsets

    # Precompute popcount for 16-bit chunks (for fast popcount of large ints)
    pc16 = [0] * 65536
    for i in range(1, 65536):
        pc16[i] = pc16[i >> 1] + (i & 1)

    def fast_popcount(x):
        """Fast popcount for Python integers using 16-bit lookup."""
        count = 0
        while x:
            count += pc16[x & 0xFFFF]
            x >>= 16
        return count

    # Precompute comparability bitsets
    # comp_bits[i]: 4096-bit int where bit j is set if i and j are comparable
    comp_bits = [0] * total_masks
    for i in range(total_masks):
        bits = 0
        for j in range(total_masks):
            if i != j:
                if (i & j) == i or (i & j) == j:
                    bits |= (1 << j)
        comp_bits[i] = bits

    def count_comparable(family):
        """Count comparable pairs in family."""
        k = len(family)
        count = 0
        for i in range(k):
            row = comp_bits[family[i]]
            for j in range(i + 1, k):
                if (row >> family[j]) & 1:
                    count += 1
        return count

    # Build initial solution: greedy
    random.seed(42)

    # Start with a chain
    chain = [(1 << size) - 1 for size in range(13)]
    family = list(chain)
    family_set = set(family)
    family_bits = 0
    for mask in family:
        family_bits |= (1 << mask)

    # Greedily add more masks
    candidates = [c for c in range(total_masks) if c not in family_set]
    random.shuffle(candidates)

    for _ in range(m - len(family)):
        best_mask = None
        best_gain = -1
        for c in candidates[:500]:
            gain = fast_popcount(comp_bits[c] & family_bits)
            if gain > best_gain:
                best_gain = gain
                best_mask = c
        if best_mask is not None:
            family.append(best_mask)
            family_set.add(best_mask)
            family_bits |= (1 << best_mask)
            candidates.remove(best_mask)

    best_score = count_comparable(family)
    best_family = list(family)

    # Local search with efficient single-move evaluation
    def local_search(family, family_bits, family_set, time_limit):
        """Local search: try single replacements until no improvement or time limit."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(family)
            for i in range(k):
                if time.time() > time_limit:
                    break
                old_mask = family[i]
                other_bits = family_bits ^ (1 << old_mask)
                old_degree = fast_popcount(comp_bits[old_mask] & other_bits)

                best_new_mask = None
                best_new_degree = -1

                # Try all masks not in family
                for c in range(total_masks):
                    if c in family_set:
                        continue
                    new_degree = fast_popcount(comp_bits[c] & other_bits)
                    if new_degree > best_new_degree:
                        best_new_degree = new_degree
                        best_new_mask = c

                if best_new_mask is not None and best_new_degree > old_degree:
                    family_set.remove(old_mask)
                    family_bits ^= (1 << old_mask)
                    family[i] = best_new_mask
                    family_set.add(best_new_mask)
                    family_bits |= (1 << best_new_mask)
                    improved = True

        return family, family_bits, family_set

    # Multi-start local search
    restarts = 0
    no_improve_count = 0
    max_no_improve = 8

    while time.time() < deadline - 5.0:
        restarts += 1

        if restarts == 1:
            family = list(best_family)
            family_set = set(family)
            family_bits = 0
            for mask in family:
                family_bits |= (1 << mask)
        else:
            family = list(best_family)
            family_set = set(family)
            family_bits = 0
            for mask in family:
                family_bits |= (1 << mask)

            # Perturb: randomly replace some masks
            num_replace = random.randint(2, min(5, restarts))
            indices = random.sample(range(m), num_replace)
            for idx in indices:
                old_mask = family[idx]
                family_set.remove(old_mask)
                family_bits ^= (1 << old_mask)
                while True:
                    new_mask = random.randint(0, total_masks - 1)
                    if new_mask not in family_set:
                        break
                family[idx] = new_mask
                family_set.add(new_mask)
                family_bits |= (1 << new_mask)

        # Run local search
        family, family_bits, family_set = local_search(
            family, family_bits, family_set, deadline - 5.0
        )

        score = count_comparable(family)
        if score > best_score:
            best_score = score
            best_family = list(family)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve and restarts > 10:
                break

    # Final polish: run local search on best solution
    family = list(best_family)
    family_set = set(family)
    family_bits = 0
    for mask in family:
        family_bits |= (1 << mask)

    family, family_bits, family_set = local_search(
        family, family_bits, family_set, deadline - 1.0
    )

    score = count_comparable(family)
    if score > best_score:
        best_score = score
        best_family = list(family)

    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)

    # Ensure types are correct: lists of ints
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]

    return result
