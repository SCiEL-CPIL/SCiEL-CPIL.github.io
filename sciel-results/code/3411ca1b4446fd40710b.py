"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s limit

    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    N = 284

    def init_sums(s):
        sums = set()
        for i, x in enumerate(s):
            for y in s[i:]:
                sums.add(x + y)
        return sums

    def can_add(s, sums, x):
        """Check if x can be added to Sidon set s with existing sums."""
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def greedy_with_order(s, sums, order):
        """Greedy add elements from s in the given order."""
        s = list(s)
        sums = set(sums)
        for x in order:
            if x in s:
                continue
            if can_add(s, sums, x):
                s.append(x)
                for y in s[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
        return s, sums

    # Strategy 1: Simple greedy (like seed) - fast baseline
    s1 = list(prescribed)
    sums1 = init_sums(s1)
    for x in range(12, N + 1):
        if can_add(s1, sums1, x):
            s1.append(x)
            for y in s1[:-1]:
                sums1.add(x + y)
            sums1.add(2 * x)

    best_set = s1[:]
    best_size = len(best_set)

    # Strategy 2: Randomized greedy with multiple restarts
    random.seed(42)

    for restart in range(200):
        if time.time() > DEADLINE:
            break

        # Create random candidate order
        candidates = list(range(12, N + 1))
        random.shuffle(candidates)

        s, sums = list(prescribed), init_sums(prescribed)
        for x in candidates:
            if can_add(s, sums, x):
                s.append(x)
                for y in s[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)

        if len(s) > best_size:
            best_set = s[:]
            best_size = len(best_set)

    # Strategy 3: Local search - remove one non-prescribed element, try to add a better one
    for _ in range(500):
        if time.time() > DEADLINE:
            break

        s = best_set[:]
        sums = init_sums(s)

        # Pick a random non-prescribed element to remove
        removable = [x for x in s if x not in prescribed]
        if not removable:
            break

        x_remove = random.choice(removable)
        idx = s.index(x_remove)
        s_trial = s[:idx] + s[idx + 1:]
        sums_trial = init_sums(s_trial)

        # Try to add a random candidate
        candidates = [x for x in range(12, N + 1) if x not in s_trial]
        random.shuffle(candidates)

        for y in candidates:
            if can_add(s_trial, sums_trial, y):
                s_trial.append(y)
                if len(s_trial) > best_size:
                    best_set = s_trial[:]
                    best_size = len(best_set)
                break

    # Strategy 4: More careful construction using priority
    # Elements that are "less restrictive" should be added first
    for restart in range(50):
        if time.time() > DEADLINE:
            break

        s = list(prescribed)
        sums = init_sums(s)
        remaining = list(range(12, N + 1))

        while remaining:
            if time.time() > DEADLINE:
                break

            # Score each candidate by how many conflicts it has
            scored = []
            for x in remaining:
                conflict_count = 0
                for y in s:
                    if x + y in sums:
                        conflict_count += 1
                if 2 * x in sums:
                    conflict_count += 1
                scored.append((conflict_count, x))

            # Sort by conflict count (ascending), then randomly among ties
            scored.sort(key=lambda t: t[0])

            # Pick from the least conflicting candidates
            min_conflicts = scored[0][0]
            candidates_least = [x for c, x in scored if c == min_conflicts]

            if not candidates_least:
                break

            # Randomly pick from least conflicting
            x = random.choice(candidates_least)

            # Check validity
            if can_add(s, sums, x):
                s.append(x)
                for y in s[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
                remaining.remove(x)
            else:
                # Shouldn't happen, but skip
                remaining.remove(x)

        if len(s) > best_size:
            best_set = s[:]
            best_size = len(best_set)

    return best_set

# EVOLVE-BLOCK-END
