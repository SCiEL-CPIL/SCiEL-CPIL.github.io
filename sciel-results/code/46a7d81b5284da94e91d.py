import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85.0  # 90s wall clock, 5s margin

    def is_valid(S):
        """Check validity: distinct subset sums, no divisibility."""
        n = len(S)
        if n == 0:
            return True
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
        sums.discard(0)
        L = sorted(sums)
        for i in range(len(L)):
            ai = L[i]
            for j in range(i + 1, len(L)):
                if L[j] % ai == 0:
                    return False
        return True

    def count_conflicts(S):
        """Count divisibility conflicts. Returns large number if duplicates exist."""
        n = len(S)
        if n == 0:
            return 0
        sums = {0}
        dup = False
        for x in S:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    dup = True
                new_sums.add(ns)
            sums = sums | new_sums
        if dup:
            return 10000
        sums.discard(0)
        L = sorted(sums)
        conflicts = 0
        for i in range(len(L)):
            ai = L[i]
            for j in range(i + 1, len(L)):
                if L[j] % ai == 0:
                    conflicts += 1
        return conflicts

    def local_search(S, max_iters=8000):
        """Simulated annealing / iterated local search."""
        S = list(S)
        best = list(S)
        best_conflicts = count_conflicts(S)
        best_size = len(S)

        for it in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break

            # Generate a neighbor
            r = random.random()
            if r < 0.4:
                # Replace an element
                idx = random.randint(0, len(S) - 1)
                c = random.randint(1, 256)
                if c in S and c != S[idx]:
                    continue
                new_S = S[:idx] + [c] + S[idx+1:]
            elif r < 0.7:
                # Add an element
                c = random.randint(1, 256)
                if c in S:
                    continue
                new_S = S + [c]
            else:
                # Remove an element (only if large enough)
                if len(S) > 3:
                    idx = random.randint(0, len(S) - 1)
                    new_S = S[:idx] + S[idx+1:]
                else:
                    idx = random.randint(0, len(S) - 1)
                    c = random.randint(1, 256)
                    if c in S:
                        continue
                    new_S = S[:idx] + [c] + S[idx+1:]

            nc = count_conflicts(new_S)

            # Acceptance criterion
            accept = False
            if nc < best_conflicts:
                accept = True
            elif nc == best_conflicts:
                # Prefer larger sets when conflicts are equal
                if len(new_S) > len(S):
                    accept = True
                elif len(new_S) == len(S):
                    accept = random.random() < 0.3
            else:
                # Metropolis: accept worse with decreasing probability
                delta = nc - best_conflicts
                temp = max(0.05, 1.0 * (1.0 - it / float(max_iters)))
                if delta <= 10:
                    prob = temp / (1.0 + 0.5 * delta)
                else:
                    prob = 0.0
                if random.random() < prob:
                    accept = True

            if accept:
                S = new_S
                if nc < best_conflicts or (nc == best_conflicts and len(S) > best_size):
                    best_conflicts = nc
                    best = list(S)
                    best_size = len(S)

        # Return the best valid set found, or best overall
        if best_conflicts == 0:
            return best
        # Try to clean up: remove elements until valid
        result = list(best)
        while len(result) > 1 and count_conflicts(result) > 0:
            # Remove the element that causes the most conflicts
            best_remove = 0
            best_conf = count_conflicts(result)
            for i in range(len(result)):
                trial = result[:i] + result[i+1:]
                tc = count_conflicts(trial)
                if tc < best_conf:
                    best_conf = tc
                    best_remove = i
            result = result[:best_remove] + result[best_remove+1:]
        if count_conflicts(result) == 0 and len(result) >= 1:
            return result
        return best if best_conflicts == 0 else [31, 32, 34]

    best_set = [31, 32, 34]

    # Multiple random starts with different strategies
    for trial in range(40):
        if time.time() - start_time > time_limit - 5:
            break

        # Different initialization strategies
        if trial % 4 == 0:
            # Large numbers, spread out
            S = random.sample(range(180, 257), 5)
        elif trial % 4 == 1:
            # Medium-large numbers
            S = random.sample(range(120, 257), 5)
        elif trial % 4 == 2:
            # Mixed
            S = random.sample(range(80, 257), 5)
        else:
            # Random full range
            S = random.sample(range(1, 257), 5)

        result = local_search(S, max_iters=6000)
        if result and is_valid(result):
            if len(result) > len(best_set):
                best_set = result
            elif len(result) == len(best_set) and random.random() < 0.5:
                # Keep the one we found first (or random tie-break)
                pass

        # Early exit if we found a large valid set
        if len(best_set) >= 8:
            break

    # Final verification
    if not is_valid(best_set):
        best_set = [31, 32, 34]

    return [int(x) for x in best_set]
