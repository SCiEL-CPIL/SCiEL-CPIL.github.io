"""Rewritten program: local-search + greedy construction for pair-sum divisibility avoidance."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 256
    # Precompute conflict matrix: conflict[i][j] = True if (i*j) % (i+j) == 0
    # Use 1-indexed: index 0 unused, indices 1..256
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True

    def count_conflicts_in_set(s):
        """Count number of conflicting pairs in set s (list of ints)."""
        c = 0
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if conflict[s[i]][s[j]]:
                    c += 1
        return c

    def is_valid(s):
        return count_conflicts_in_set(s) == 0

    # Phase 1: Greedy construction
    # Start with all numbers, then remove to minimize conflicts
    # Better: start empty, greedily add numbers that cause fewest new conflicts

    # Strategy A: Greedy by ordering
    # Try multiple orderings and pick the best

    best_set = None
    best_size = 0

    # Ordering 1: all numbers 1..256
    # Ordering 2: odds first, then evens
    # Ordering 3: random shuffles

    def greedy_add(ordering):
        """Greedily add elements in the given order, skipping if it creates a conflict."""
        s = []
        for x in ordering:
            # Check if x conflicts with any element in s
            ok = True
            for y in s:
                if conflict[x][y]:
                    ok = False
                    break
            if ok:
                s.append(x)
        return s

    # Try several orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),  # odds then evens
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),  # evens then odds
    ]

    # Add some random orderings
    rng = random.Random(42)
    for _ in range(5):
        orderings.append(rng.sample(range(1, N + 1), N))

    for order in orderings:
        s = greedy_add(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # Phase 2: Local search to improve
    # Try to add elements not in the set, or swap elements

    def local_search(s, max_iters=20000):
        """Local search: try adding, removing, or swapping to reduce conflicts / increase size."""
        s = s[:]
        in_set = [False] * (N + 1)
        for x in s:
            in_set[x] = True

        def conflicts_with_set(x, current_set):
            """Count how many elements in current_set conflict with x."""
            c = 0
            for y in current_set:
                if conflict[x][y]:
                    c += 1
            return c

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # Try adding each element not in set
            best_gain = 0
            best_add = None
            for x in range(1, N + 1):
                if not in_set[x]:
                    c = conflicts_with_set(x, s)
                    if c == 0:
                        # Can add without conflict
                        best_gain = 1
                        best_add = x
                        break
                    # If we have some conflict, we might still want to add if net beneficial
                    # For now, only add if zero conflict

            if best_add is not None:
                s.append(best_add)
                in_set[best_add] = True
                improved = True
                continue

            # Try swapping: remove one, add one
            # For each element in s, try removing it and see if we can add more
            for i in range(len(s)):
                if time.time() >= deadline:
                    return s
                rem = s[i]
                # After removing rem, try to add elements that were previously blocked
                # This is expensive; do a simpler version
                # Try: remove rem, then greedily add best candidate
                new_s = s[:i] + s[i+1:]
                new_in_set = in_set[:]
                new_in_set[rem] = False

                # Find best element to add
                best_candidate = None
                for x in range(1, N + 1):
                    if not new_in_set[x]:
                        c = conflicts_with_set(x, new_s)
                        if c == 0:
                            best_candidate = x
                            break

                if best_candidate is not None:
                    # Check if this swap is beneficial (same size but maybe better structure)
                    # Actually same size, so only do it if it enables further additions
                    # For now, skip pure swaps of same size unless we can chain
                    pass

            # Try multi-add: add 2 elements simultaneously
            # This is O(N^2) per iteration, might be too slow
            # Skip for now

        return s

    # Apply local search to best greedy result
    if best_set is not None:
        best_set = local_search(best_set)
        best_size = len(best_set)

    # Phase 3: More aggressive search - start from scratch with different strategies
    # Strategy: start with a large set, then remove conflicting elements
    def remove_conflicts(s):
        """Remove elements to make set conflict-free. Greedy removal."""
        s = s[:]
        changed = True
        while changed:
            changed = False
            # Find the element involved in the most conflicts
            conflict_count = [0] * (N + 1)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    if conflict[s[i]][s[j]]:
                        conflict_count[s[i]] += 1
                        conflict_count[s[j]] += 1
            # Remove the element with highest conflict count
            max_c = 0
            max_idx = -1
            for i, x in enumerate(s):
                if conflict_count[x] > max_c:
                    max_c = conflict_count[x]
                    max_idx = i
            if max_c > 0:
                s.pop(max_idx)
                changed = True
        return s

    # Try: start with all numbers, remove conflicts
    s_all = list(range(1, N + 1))
    s_removed = remove_conflicts(s_all)
    if len(s_removed) > best_size:
        best_size = len(s_removed)
        best_set = s_removed

    # Phase 4: Final polish - try to add any remaining elements
    if best_set is not None:
        in_set = [False] * (N + 1)
        for x in best_set:
            in_set[x] = True
        for x in range(1, N + 1):
            if not in_set[x]:
                ok = True
                for y in best_set:
                    if conflict[x][y]:
                        ok = False
                        break
                if ok:
                    best_set.append(x)
                    in_set[x] = True

    # Phase 5: Local search with swaps for further improvement
    if best_set is not None:
        best_set = local_search(best_set, max_iters=50000)

    # Ensure nonempty
    if not best_set:
        best_set = [1]

    return sorted(best_set)
# EVOLVE-BLOCK-END
