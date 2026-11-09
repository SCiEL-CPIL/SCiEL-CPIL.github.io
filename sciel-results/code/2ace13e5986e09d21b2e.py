"""B2[2] set in 1..1000: maximize size with every sum having <= 2 representations a+b, a<=b."""
import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def sum_counts(chosen):
        """Return dict of sum -> count of representations a+b, a<=b, a,b in chosen."""
        counts = {}
        n = len(chosen)
        for i in range(n):
            a = chosen[i]
            for j in range(i, n):
                b = chosen[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def can_add(x, chosen_list, counts):
        """Check if x can be added. Returns (valid, slack)."""
        slack = 0
        for y in chosen_list:
            s = x + y
            c = counts.get(s, 0) + 1
            if c > 2:
                return False, 0
            if c < 2:
                slack += 1
        s = 2 * x
        c = counts.get(s, 0) + 1
        if c > 2:
            return False, 0
        if c < 2:
            slack += 1
        return True, slack

    def add(x, chosen, chosen_set, counts):
        """Add x to chosen, update counts."""
        chosen.append(x)
        chosen_set.add(x)
        for y in chosen[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1

    def remove(x, chosen, chosen_set, counts):
        """Remove x from chosen, update counts."""
        idx = chosen.index(x)
        chosen.pop(idx)
        chosen_set.discard(x)
        for y in chosen:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]

    def verify(chosen):
        """Verify the set is valid: every sum has at most 2 representations."""
        counts = sum_counts(chosen)
        for s, c in counts.items():
            if c > 2:
                return False
        return True

    # ---- Phase 1: Constraint-aware greedy with multiple trials ----
    best_size = 0
    best_chosen = []
    best_counts = {}

    for trial in range(8):
        if elapsed() > DEADLINE - 80:
            break
        random.seed(42 + trial * 137)
        chosen = []
        chosen_set = set()
        counts = {}

        temperature = 0.2 + trial * 0.08
        for x in range(1, 1001):
            if elapsed() > DEADLINE - 80:
                break
            valid, slack = can_add(x, chosen, counts)
            if valid:
                p = min(1.0, 0.6 + slack * temperature)
                if random.random() < p:
                    add(x, chosen, chosen_set, counts)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
            best_counts = dict(counts)

    chosen = best_chosen
    chosen_set = set(chosen)
    counts = best_counts

    # ---- Phase 2: Local search with add + swap ----
    improved = True
    swap_attempts = 0
    MAX_SWAPS = 200

    while improved and elapsed() < DEADLINE - 20:
        improved = False

        for x in range(1, 1001):
            if elapsed() > DEADLINE - 20:
                break
            if x in chosen_set:
                continue
            valid, slack = can_add(x, chosen, counts)
            if valid:
                add(x, chosen, chosen_set, counts)
                improved = True
                break

        if improved:
            continue

        if swap_attempts >= MAX_SWAPS:
            break

        for _ in range(20):
            if elapsed() > DEADLINE - 20 or swap_attempts >= MAX_SWAPS:
                break
            swap_attempts += 1

            if not chosen:
                break
            a = random.choice(chosen)
            remove(a, chosen, chosen_set, counts)

            best_x = None
            best_slack = -1
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                valid, slack = can_add(x, chosen, counts)
                if valid and slack > best_slack:
                    best_slack = slack
                    best_x = x

            if best_x is not None:
                add(best_x, chosen, chosen_set, counts)
                for x in range(1, 1001):
                    if x in chosen_set:
                        continue
                    valid, _ = can_add(x, chosen, counts)
                    if valid:
                        add(x, chosen, chosen_set, counts)
                        improved = True
                        break
            else:
                add(a, chosen, chosen_set, counts)

    # ---- Phase 3: Final exhaustive add pass ----
    while elapsed() < DEADLINE - 5:
        found = False
        for x in range(1, 1001):
            if elapsed() > DEADLINE - 5:
                break
            if x in chosen_set:
                continue
            valid, _ = can_add(x, chosen, counts)
            if valid:
                add(x, chosen, chosen_set, counts)
                found = True
                break
        if not found:
            break

    # ---- Phase 4: Final verification ----
    if not verify(chosen):
        chosen = []
        chosen_set = set()
        counts = {}
        for x in range(1, 1001):
            valid, _ = can_add(x, chosen, counts)
            if valid:
                add(x, chosen, chosen_set, counts)
        if not verify(chosen):
            chosen = [1, 2, 4, 8]

    return chosen
