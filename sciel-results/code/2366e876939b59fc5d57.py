# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses simulated annealing with multiple restarts and a deadline guard.
    """
    m = 205
    deadline = time.time() + 600  # 10 minute wall-clock limit with safety margin

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = list(forbidden)
    forbidden_set = set(forbidden_list)

    # Precompute conflict matrix: conflict[i][j] = True if (i-j) mod m is in forbidden
    # conflict is symmetric: conflict[i][j] = conflict[j][i]
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d = (i - j) % m
            if d in forbidden_set:
                conflict[i][j] = True
                conflict[j][i] = True

    def is_valid_subset(subset):
        """Check if subset is a valid independent set."""
        s = set(subset)
        for a in s:
            for b in s:
                if a != b and conflict[a][b]:
                    return False
        return True

    def greedy_seed():
        """Greedy construction: pick elements in order that maximizes available neighbors."""
        # Try multiple greedy orders and keep the best
        best_subset = []
        for trial in range(20):
            order = list(range(m))
            if trial > 0:
                random.shuffle(order)
            subset = []
            for candidate in order:
                ok = True
                for s in subset:
                    if conflict[candidate][s]:
                        ok = False
                        break
                if ok:
                    subset.append(candidate)
            if len(subset) > len(best_subset):
                best_subset = subset[:]
        return best_subset

    def can_add(subset_set, candidate):
        """Check if candidate can be added to the set."""
        for s in subset_set:
            if conflict[candidate][s]:
                return False
        return True

    def simulated_annealing(initial_subset, max_steps=50000):
        """Simulated annealing to find larger independent sets."""
        current = set(initial_subset)
        current_size = len(current)
        best = set(current)
        best_size = current_size

        T = 1.0
        T_min = 0.001
        cooling = 0.99995

        steps = 0
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 5:
                break

            # Choose a move: add, remove, or swap
            move_type = random.random()
            if move_type < 0.4:
                # Try to add an element
                candidates = [i for i in range(m) if i not in current]
                if not candidates:
                    T *= cooling
                    steps += 1
                    continue
                cand = random.choice(candidates)
                if can_add(current, cand):
                    new_set = current | {cand}
                    delta = 1
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current = new_set
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
                else:
                    # Can't add directly; try removing a conflicting element first
                    conflicts_with = [s for s in current if conflict[cand][s]]
                    if conflicts_with:
                        remove_cand = random.choice(conflicts_with)
                        new_set = (current - {remove_cand}) | {cand}
                        delta = 0  # size unchanged, but might enable future growth
                        if random.random() < np.exp(delta / T):
                            current = new_set
                            if len(current) > best_size:
                                best = set(current)
                                best_size = len(current)
            elif move_type < 0.7:
                # Try to remove an element
                if current:
                    remove_cand = random.choice(list(current))
                    new_set = current - {remove_cand}
                    delta = -1
                    if random.random() < np.exp(delta / T):
                        current = new_set
            else:
                # Swap: remove one, add one
                if current:
                    remove_cand = random.choice(list(current))
                    candidates = [i for i in range(m) if i not in current]
                    if candidates:
                        add_cand = random.choice(candidates)
                        new_set = (current - {remove_cand}) | {add_cand}
                        # Check validity
                        valid = True
                        for a in new_set:
                            for b in new_set:
                                if a != b and conflict[a][b]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            delta = 0
                            if random.random() < np.exp(delta / T):
                                current = new_set
                                if len(current) > best_size:
                                    best = set(current)
                                    best_size = len(current)

            T *= cooling
            steps += 1

        return list(best)

    def local_search_improve(subset):
        """Try to improve by adding elements or swapping to increase size."""
        current = set(subset)
        improved = True
        while improved:
            improved = False
            # Try adding any element that fits
            for cand in range(m):
                if cand not in current and can_add(current, cand):
                    current.add(cand)
                    improved = True
                    break
            if improved:
                continue
            # Try swaps: remove one, add one that fits
            for remove_cand in list(current):
                for add_cand in range(m):
                    if add_cand in current:
                        continue
                    new_set = (current - {remove_cand}) | {add_cand}
                    valid = True
                    for a in new_set:
                        for b in new_set:
                            if a != b and conflict[a][b]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
        return list(current)

    # Main search
    best_overall = []

    # Start with greedy seeds
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        seed = greedy_seed()
        improved = local_search_improve(seed)
        if len(improved) > len(best_overall):
            best_overall = improved[:]

    # Simulated annealing from multiple starts
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        seed = greedy_seed()
        result = simulated_annealing(seed, max_steps=30000)
        if len(result) > len(best_overall):
            best_overall = result[:]

    # Final local search on the best
    if best_overall:
        final = local_search_improve(best_overall)
        if len(final) > len(best_overall):
            best_overall = final

    # Ensure we have at least one element
    if not best_overall:
        best_overall = [0]

    return np.array(best_overall, dtype=np.int64)


# EVOLVE-BLOCK-END
