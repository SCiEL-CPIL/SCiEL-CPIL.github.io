# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/205Z of maximum
    size that does NOT contain any two elements differing by a nonzero square modulo 205.

    Uses a greedy construction followed by simulated annealing with a precomputed
    conflict matrix for O(1) feasibility checks.
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Build conflict matrix: conflict[i][j] = True if (i-j) mod m or (j-i) mod m is in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i != j:
                d1 = (i - j) % m
                d2 = (j - i) % m
                if d1 in forbidden or d2 in forbidden:
                    conflict[i][j] = True

    def is_valid_subset(selected):
        """Check if the subset is valid (no conflicts)."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if conflict[selected[i]][selected[j]]:
                    return False
        return True

    def greedy_construct():
        """Greedy construction: add elements in order if they don't conflict."""
        selected = []
        for candidate in range(m):
            ok = True
            for s in selected:
                if conflict[candidate][s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected

    def simulated_annealing(initial_set, max_iters=500000):
        """Simulated annealing to improve the subset."""
        # Start with a set, try to add/remove elements
        current = list(initial_set)
        current_set = set(current)
        best = list(current)
        best_set = set(current)
        best_size = len(current)

        T = 1.0
        cooling = 0.9999
        min_T = 1e-6

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break

            # Random move: try to add a random element not in current, or remove a random element
            if random.random() < 0.7:
                # Try to add
                candidates = [x for x in range(m) if x not in current_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                # Check if adding x creates conflicts
                conflicts = 0
                for s in current:
                    if conflict[x][s]:
                        conflicts += 1
                if conflicts == 0:
                    current.append(x)
                    current_set.add(x)
                    if len(current) > best_size:
                        best = list(current)
                        best_set = set(current)
                        best_size = len(current)
                else:
                    # Accept with probability exp(-conflicts / T)
                    if random.random() < np.exp(-conflicts / T):
                        current.append(x)
                        current_set.add(x)
            else:
                # Try to remove
                if len(current) > 1:
                    idx = random.randint(0, len(current) - 1)
                    x = current.pop(idx)
                    current_set.discard(x)

            T *= cooling

        return best

    # Initial greedy construction
    initial = greedy_construct()
    best = list(initial)
    best_size = len(initial)

    # Multiple restarts with different random seeds
    rng = random.Random(42)
    num_restarts = 20
    iters_per_restart = 100000

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Create a perturbed initial set
        if r == 0:
            start = list(initial)
        else:
            # Random perturbation: start with a random subset of size ~ best_size
            size = max(1, best_size - rng.randint(0, 3))
            start = rng.sample(range(m), size)
            # Remove conflicts from start
            clean = []
            for x in start:
                ok = True
                for s in clean:
                    if conflict[x][s]:
                        ok = False
                        break
                if ok:
                    clean.append(x)
            start = clean

        result = simulated_annealing(start, max_iters=iters_per_restart)
        if len(result) > best_size:
            best = result
            best_size = len(result)

    return np.array(best, dtype=np.int64)


# EVOLVE-BLOCK-END
