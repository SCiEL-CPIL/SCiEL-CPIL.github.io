import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit
    n = 4096

    # Precompute conflict matrix: conflict[i][j] = True if including both i+1 and j+1
    # creates a triple-sum collision.
    # For elements a, b (a <= b), a collision exists if:
    #   3a = 3b (impossible for distinct)
    #   3a = 2a+b -> a=b (impossible)
    #   3a = a+2b -> a = 2b (impossible for a<b)
    #   3a = a+b+c for some c in set... but for PAIRWISE: we need to check if
    #   any triple sum with a equals any triple sum with b.
    #
    # Triple sums involving a (with a,b in set, a<b):
    #   3a, 2a+b, a+2b, a+b+c (c in set)
    # Triple sums involving b:
    #   3b, 2b+a, b+2a, a+b+c (c in set)
    #
    # For pairwise conflict (just a and b in set):
    #   3a vs 3b: no
    #   3a vs 2a+b: a=b no
    #   3a vs a+2b: a=2b no (a<b)
    #   3a vs a+b+c: c=2a-b, need c>=1, c!=a, c!=b -> c=2a-b
    #   2a+b vs 3b: a=b no
    #   2a+b vs 2a+b: same! Wait, 2a+b is a triple sum (a,a,b). And for b: 2a+b = a+2b? No, that's a+2b.
    #   Actually 2a+b (triple a,a,b) vs 2b+a (triple b,b,a) = a+2b. These are different unless a=b.
    #   2a+b vs a+2b: a=b no
    #   2a+b vs a+b+c: c=a -> triple (a,a,b) sum = 2a+b, and (a,b,a) sum = 2a+b. Same triple!
    #     Wait, combinations_with_replacement({a,b},3) gives: (a,a,a),(a,a,b),(a,b,b),(b,b,b)
    #     Sums: 3a, 2a+b, a+2b, 3b. All distinct for a<b. So NO pairwise conflict!
    #
    # So conflicts only arise with 3+ elements. This means I need a different approach.
    #
    # Better approach: incremental greedy with efficient sum checking, multiple orderings,
    # and local search (swap/add) to improve.

    def run_greedy(order, existing_set=None):
        """Greedy: try to add each element in order if it doesn't create a sum collision."""
        selected = list(existing_set) if existing_set else []
        if not selected:
            current_sums = set()
        else:
            current_sums = set(sum(t) for t in combinations_with_replacement(selected, 3))

        for x in order:
            if x in selected:
                continue
            # Check all new triple sums involving x
            new_sums = set()
            # (x,x,x)
            new_sums.add(3 * x)
            # (x,x,y) for y in selected
            for y in selected:
                new_sums.add(2 * x + y)
            # (x,y,z) for y<=z in selected
            for y, z in combinations_with_replacement(selected, 2):
                new_sums.add(x + y + z)

            if new_sums & current_sums:
                continue

            selected.append(x)
            current_sums |= new_sums
        return selected

    # Strategy 1: Multiple greedy orderings
    best_set = []
    best_size = 0

    # Deterministic baseline: ascending
    s = run_greedy(range(1, n + 1))
    if len(s) > best_size:
        best_set, best_size = s, len(s)

    # Descending
    if time.time() < deadline:
        s = run_greedy(range(n, 0, -1))
        if len(s) > best_size:
            best_set, best_size = s, len(s)

    # Random orderings
    rng = random.Random(42)
    for trial in range(200):
        if time.time() >= deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        s = run_greedy(order)
        if len(s) > best_size:
            best_set, best_size = s, len(s)

    # Strategy 2: Biased orderings (favor smaller numbers first for density)
    for trial in range(100):
        if time.time() >= deadline:
            break
        # Mix: first half random from [1, n//2], second half random from [n//2, n]
        low = list(range(1, n // 2 + 1))
        high = list(range(n // 2 + 1, n + 1))
        rng.shuffle(low)
        rng.shuffle(high)
        # Interleave
        order = []
        i, j = 0, 0
        while i < len(low) or j < len(high):
            if i < len(low) and (j >= len(high) or rng.random() < 0.5):
                order.append(low[i]); i += 1
            else:
                order.append(high[j]); j += 1
        s = run_greedy(order)
        if len(s) > best_size:
            best_set, best_size = s, len(s)

    # Strategy 3: Local search - try to add elements not in best_set
    if time.time() < deadline:
        current = sorted(best_set)
        current_sums = set(sum(t) for t in combinations_with_replacement(current, 3))
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each candidate
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                new_sums = set()
                new_sums.add(3 * x)
                for y in current:
                    new_sums.add(2 * x + y)
                for y, z in combinations_with_replacement(current, 2):
                    new_sums.add(x + y + z)
                if not (new_sums & current_sums):
                    current.append(x)
                    current_set.add(x)
                    current_sums |= new_sums
                    improved = True
                    if len(current) > best_size:
                        best_set, best_size = list(current), len(current)
                    break  # restart scan after adding

    # Strategy 4: Local search with swaps (remove one, add two)
    if time.time() < deadline:
        current = sorted(best_set)
        current_sums = set(sum(t) for t in combinations_with_replacement(current, 3))
        current_set = set(current)
        for _ in range(50):
            if time.time() >= deadline:
                break
            # Try removing one element and adding two
            found_improvement = False
            for remove_idx in range(len(current)):
                if time.time() >= deadline:
                    break
                removed = current.pop(remove_idx)
                current_set.discard(removed)
                # Recompute sums
                current_sums = set(sum(t) for t in combinations_with_replacement(current, 3))

                # Try to add two new elements
                added = []
                for x in range(1, n + 1):
                    if x in current_set or x == removed:
                        continue
                    new_sums = set()
                    new_sums.add(3 * x)
                    for y in current:
                        new_sums.add(2 * x + y)
                    for y, z in combinations_with_replacement(current, 2):
                        new_sums.add(x + y + z)
                    if not (new_sums & current_sums):
                        current.append(x)
                        current_set.add(x)
                        current_sums |= new_sums
                        added.append(x)
                        break

                for x in range(1, n + 1):
                    if x in current_set or x == removed:
                        continue
                    new_sums = set()
                    new_sums.add(3 * x)
                    for y in current:
                        new_sums.add(2 * x + y)
                    for y, z in combinations_with_replacement(current, 2):
                        new_sums.add(x + y + z)
                    if not (new_sums & current_sums):
                        current.append(x)
                        current_set.add(x)
                        current_sums |= new_sums
                        added.append(x)
                        break

                if len(added) == 2 and len(current) > best_size:
                    best_set, best_size = sorted(current), len(current)
                    found_improvement = True
                else:
                    # Undo
                    for a in added:
                        current.remove(a)
                        current_set.discard(a)
                    current.insert(remove_idx, removed)
                    current_set.add(removed)
                    current_sums = set(sum(t) for t in combinations_with_replacement(current, 3))

                if found_improvement:
                    break

    # Strategy 5: More aggressive random restarts with local improvement
    for trial in range(50):
        if time.time() >= deadline:
            break
        rng2 = random.Random(1000 + trial)
        order = list(range(1, n + 1))
        rng2.shuffle(order)
        s = run_greedy(order)
        # Local improvement
        s_sorted = sorted(s)
        s_set = set(s_sorted)
        s_sums = set(sum(t) for t in combinations_with_replacement(s_sorted, 3))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if x in s_set:
                    continue
                new_sums = set()
                new_sums.add(3 * x)
                for y in s_sorted:
                    new_sums.add(2 * x + y)
                for y, z in combinations_with_replacement(s_sorted, 2):
                    new_sums.add(x + y + z)
                if not (new_sums & s_sums):
                    s_sorted.append(x)
                    s_set.add(x)
                    s_sums |= new_sums
                    improved = True
                    if len(s_sorted) > best_size:
                        best_set, best_size = list(s_sorted), len(s_sorted)
                    break

    return best_set

# EVOLVE-BLOCK-END
