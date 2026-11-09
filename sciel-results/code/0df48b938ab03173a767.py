"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))

    # Precompute conflict: conflict[a][b] = True if (a*b)%(a+b)==0
    # Use a set of frozensets for quick lookup, or a 2D structure
    # For efficiency, use a set of tuples (min, max)
    conflict_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_set.add((a, b))

    def has_conflict(a, b, current_set):
        """Check if adding b to current_set (which contains a) creates a conflict."""
        if (a, b) in conflict_set or (b, a) in conflict_set:
            return True
        return False

    def can_add(num, current_set):
        """Check if num can be added to current_set without any conflict."""
        for x in current_set:
            if x == num:
                continue
            lo, hi = (x, num) if x < num else (num, x)
            if (lo, hi) in conflict_set:
                return False
        return True

    # Start with all odd numbers - guaranteed valid
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

    # Local search: iterative improvement
    current_set = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing one element and adding others
        elements = list(current_set)
        random.shuffle(elements)
        for rem in elements:
            if time.time() > deadline:
                break
            new_set = current_set - {rem}
            # Try to add evens not in new_set
            candidates = [x for x in range(2, N + 1, 2) if x not in new_set]
            random.shuffle(candidates)
            added = 0
            for c in candidates:
                if time.time() > deadline:
                    break
                if can_add(c, new_set):
                    new_set.add(c)
                    added += 1
            if len(new_set) > len(current_set):
                current_set = new_set
                best_set = set(new_set)
                best_size = len(new_set)
                improved = True
                break

    # Randomized local search with multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break

        # Start from best, perturb
        trial_set = set(best_set)
        # Remove 1-4 random elements
        n_remove = random.randint(1, 4)
        to_remove = random.sample(list(trial_set), min(n_remove, len(trial_set)))
        for t in to_remove:
            trial_set.remove(t)

        # Try to add evens
        evens2 = [x for x in range(2, N + 1, 2) if x not in trial_set]
        random.shuffle(evens2)
        for e in evens2:
            if time.time() > deadline:
                break
            if can_add(e, trial_set):
                trial_set.add(e)

        # Try to add odds too (in case we removed some)
        odds2 = [x for x in range(1, N + 1, 2) if x not in trial_set]
        random.shuffle(odds2)
        for o in odds2:
            if time.time() > deadline:
                break
            if can_add(o, trial_set):
                trial_set.add(o)

        if len(trial_set) > best_size:
            best_set = set(trial_set)
            best_size = len(trial_set)

    # Final swap-based local search
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
            # Try adding all possible elements
            for c in range(1, N + 1):
                if c in new_set:
                    continue
                if can_add(c, new_set):
                    new_set.add(c)
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
