"""Rewritten program for large interval sets avoiding a prescribed subset sum."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    def has_sum_200(elements):
        """Check if any subset of elements sums to exactly 200."""
        reachable = {0}
        for x in elements:
            new = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new.add(ns)
            if 200 in new:
                return True
            reachable |= new
        return False

    def reachable_sums(elements):
        """Compute the set of all reachable subset sums up to 200."""
        reachable = {0}
        for x in elements:
            new = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new.add(ns)
            reachable |= new
        return reachable

    # Known-good base: [66, 100], size 35
    base = list(range(66, 101))
    best = base[:]
    best_size = len(best)

    # Greedy add from 1-65 in descending order
    current = base[:]
    for x in range(65, 0, -1):
        if time.time() > deadline:
            break
        test = current + [x]
        if not has_sum_200(test):
            current = test

    if len(current) > best_size:
        best = current
        best_size = len(best)

    # Greedy add in ascending order
    current = base[:]
    for x in range(1, 66):
        if time.time() > deadline:
            break
        test = current + [x]
        if not has_sum_200(test):
            current = test

    if len(current) > best_size:
        best = current
        best_size = len(best)

    # Try multiple random orderings with better seed diversity
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 66))
        rng.shuffle(order)
        current = base[:]
        for x in order:
            if time.time() > deadline - 5:
                break
            test = current + [x]
            if not has_sum_200(test):
                current = test
        if len(current) > best_size:
            best = current
            best_size = len(best)

    # Local search: try removing one element and adding two others
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            best_set = set(best)
            # Try removing one element and adding two others
            for remove in list(best_set):
                if time.time() > deadline - 10:
                    break
                remaining = [x for x in best if x != remove]
                avail = [x for x in range(1, 66) if x not in set(remaining)]
                for i in range(len(avail)):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i+1, len(avail)):
                        test = remaining + [avail[i], avail[j]]
                        if len(test) > len(best) and not has_sum_200(test):
                            best = test
                            best_set = set(best)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Try single element swaps to improve
    if time.time() < deadline - 5:
        best_set = set(best)
        for remove in list(best_set):
            if time.time() > deadline - 5:
                break
            remaining = [x for x in best if x != remove]
            avail = [x for x in range(1, 66) if x not in set(remaining)]
            for add in avail:
                if time.time() > deadline - 5:
                    break
                test = remaining + [add]
                if len(test) > len(best) and not has_sum_200(test):
                    best = test
                    best_set = set(best)

    # Final verification
    if has_sum_200(best):
        best = list(range(66, 101))

    return best
# EVOLVE-BLOCK-END
