import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 84.0  # Safety margin before 90s kill

    N = 100
    target = 200

    # Helper to compute reachable sums up to target using a set
    def compute_reachable(S, target):
        reachable = {0}
        for x in S:
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= target:
                    new_sums.add(ns)
            reachable |= new_sums
        return reachable

    # Helper to check if a set has a subset summing to target
    def has_subset_sum(S, target):
        reachable = {0}
        for x in S:
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= target:
                    new_sums.add(ns)
            reachable |= new_sums
            if target in reachable:
                return True
        return False

    # Base set: {66, ..., 100}
    # Min sum of 3 elements: 66+67+68 = 201 > 200
    # Max sum of 2 elements: 99+100 = 199 < 200
    # So no subset sums to 200.
    base_S = set(range(66, 101))
    best_set = set(base_S)
    best_size = len(base_S)

    candidates = list(range(1, 66))

    # Strategy 1: Greedy addition with various orders
    orders = [
        candidates,                    # Ascending
        sorted(candidates, reverse=True), # Descending
    ]

    random.seed(42)
    for _ in range(30):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)

    for order in orders:
        if time.time() > deadline:
            break

        S_try = set(base_S)
        reachable_try = compute_reachable(base_S, target)

        for x in order:
            if x in S_try:
                continue

            # Check if adding x creates a sum of 200
            if (target - x) in reachable_try:
                continue

            S_try.add(x)
            # Update reachable sums
            new_sums = set()
            for s in reachable_try:
                ns = s + x
                if ns <= target:
                    new_sums.add(ns)
            reachable_try |= new_sums

        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = set(S_try)

    # Strategy 2: Local search / refinement
    # Try to add more elements by removing conflicting ones
    if time.time() < deadline - 10:
        # Try all pairs of elements from base_S to remove, then greedily add
        base_list = sorted(base_S)
        for i in range(len(base_list)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(base_list)):
                if time.time() > deadline:
                    break

                S_try = set(base_S)
                S_try.discard(base_list[i])
                S_try.discard(base_list[j])

                reachable_try = compute_reachable(S_try, target)

                # Now greedily add candidates
                for x in sorted(candidates):
                    if x in S_try:
                        continue
                    if (target - x) in reachable_try:
                        continue
                    S_try.add(x)
                    new_sums = set()
                    for s in reachable_try:
                        ns = s + x
                        if ns <= target:
                            new_sums.add(ns)
                    reachable_try |= new_sums

                if len(S_try) > best_size:
                    best_size = len(S_try)
                    best_set = set(S_try)

    # Strategy 3: Try different base sets
    # Maybe we can use fewer large numbers but more small ones
    for start in range(60, 66):
        if time.time() > deadline:
            break
        alt_base = set(range(start, 101))
        # Check validity
        if has_subset_sum(alt_base, target):
            continue

        S_try = set(alt_base)
        reachable_try = compute_reachable(alt_base, target)

        for x in sorted(candidates):
            if x in S_try:
                continue
            if (target - x) in reachable_try:
                continue
            S_try.add(x)
            new_sums = set()
            for s in reachable_try:
                ns = s + x
                if ns <= target:
                    new_sums.add(ns)
            reachable_try |= new_sums

        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = set(S_try)

    # Strategy 4: Randomized local search
    if time.time() < deadline - 5:
        random.seed(123)
        for _ in range(50):
            if time.time() > deadline:
                break

            S_try = set(base_S)
            reachable_try = compute_reachable(base_S, target)

            # Randomly remove a few elements
            for _ in range(random.randint(0, 3)):
                if len(S_try) > 10:
                    elem = random.choice(list(S_try))
                    S_try.discard(elem)
                    # Recompute reachable
                    reachable_try = compute_reachable(S_try, target)

            # Greedily add
            shuffled = candidates[:]
            random.shuffle(shuffled)
            for x in shuffled:
                if x in S_try:
                    continue
                if (target - x) in reachable_try:
                    continue
                S_try.add(x)
                new_sums = set()
                for s in reachable_try:
                    ns = s + x
                    if ns <= target:
                        new_sums.add(ns)
                reachable_try |= new_sums

            if len(S_try) > best_size:
                best_size = len(S_try)
                best_set = set(S_try)

    # Final verification
    if not has_subset_sum(best_set, target):
        return list(best_set)

    # Fallback: return the base set
    return list(base_S)
