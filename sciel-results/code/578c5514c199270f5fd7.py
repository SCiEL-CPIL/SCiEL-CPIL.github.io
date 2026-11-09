"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))

    # Precompute conflict as a set of (min, max) tuples
    conflict_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_set.add((a, b))

    # Precompute conflict bitmask for each number (bit i set if conflicts with i+1)
    # Index 0 unused; index 1..256
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a == b:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            if (lo, hi) in conflict_set:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Precompute: for each num, list of numbers it conflicts with
    conflict_list = [[] for _ in range(N + 1)]
    for (a, b) in conflict_set:
        conflict_list[a].append(b)
        conflict_list[b].append(a)

    def can_add(num, current_set):
        """Check if num can be added to current_set without any conflict."""
        mask = conflict_mask[num]
        for x in current_set:
            if x == num:
                continue
            if mask & (1 << x):
                return False
        return True

    def greedy_fill(s, candidates):
        """Try to add each candidate to s in order; return new size added."""
        added = 0
        for c in candidates:
            if c in s:
                continue
            if can_add(c, s):
                s.add(c)
                added += 1
        return added

    # Start with all odd numbers
    odds = set(x for x in range(1, N + 1) if x % 2 == 1)
    best_set = set(odds)
    best_size = len(best_set)

    # Greedy: try adding each even number
    evens = [x for x in range(2, N + 1) if x % 2 == 0]
    random.shuffle(evens)
    current = set(odds)
    for e in evens:
        if time.time() > deadline:
            break
        if can_add(e, current):
            current.add(e)

    if len(current) > best_size:
        best_set = set(current)
        best_size = len(current)

    # 1-swap local search: remove one element, greedily refill
    current_set = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        elements = list(current_set)
        random.shuffle(elements)
        for rem in elements:
            if time.time() > deadline:
                break
            new_set = current_set - {rem}
            candidates = [x for x in range(1, N + 1) if x not in new_set]
            random.shuffle(candidates)
            added = greedy_fill(new_set, candidates)
            if len(new_set) > len(current_set):
                current_set = new_set
                best_set = set(new_set)
                best_size = len(new_set)
                improved = True
                break

    # Randomized perturbation + refill
    for restart in range(80):
        if time.time() > deadline:
            break

        trial_set = set(best_set)
        n_remove = random.randint(1, 4)
        to_remove = random.sample(list(trial_set), min(n_remove, len(trial_set)))
        for t in to_remove:
            trial_set.remove(t)

        candidates = [x for x in range(1, N + 1) if x not in trial_set]
        random.shuffle(candidates)
        greedy_fill(trial_set, candidates)

        if len(trial_set) > best_size:
            best_set = set(trial_set)
            best_size = len(trial_set)

    # Final 1-swap polish
    current_set = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        elements = list(current_set)
        random.shuffle(elements)
        for rem in elements:
            if time.time() > deadline:
                break
            new_set = current_set - {rem}
            candidates = [x for x in range(1, N + 1) if x not in new_set]
            random.shuffle(candidates)
            greedy_fill(new_set, candidates)
            if len(new_set) > len(current_set):
                current_set = new_set
                best_set = set(new_set)
                best_size = len(new_set)
                improved = True
                break

    # Final verification
    best_list = sorted(best_set)
    valid = True
    for i in range(len(best_list)):
        for j in range(i + 1, len(best_list)):
            a, b = best_list[i], best_list[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid or len(best_list) == 0:
        return list(range(1, N + 1, 2))

    return best_list
# EVOLVE-BLOCK-END
