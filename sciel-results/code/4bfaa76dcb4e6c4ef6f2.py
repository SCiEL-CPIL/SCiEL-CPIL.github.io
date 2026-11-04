"""Construct a finite set U of nonnegative integers containing 0 to maximize the
sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)
"""
import time
import math
import random
from collections import deque

DEADLINE = 150.0  # wall-clock seconds


def _score_set(U):
    """Return the score for a set U (list of nonnegative ints containing 0)."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n < 1:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 0.0

    # Compute U+U and U-U
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)

    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _geometric_seed(ratio, n_terms, start=1):
    """Generate a geometric-like set {0, start, start*ratio, start*ratio^2, ...}."""
    U = [0]
    val = start
    for _ in range(n_terms):
        U.append(int(val))
        val *= ratio
    return U


def _local_search(U, time_budget):
    """Perform local search to improve the set.

    Moves: add a new element, remove an element, or replace an element.
    """
    start = time.time()
    best = list(U)
    best_score = _score_set(best)

    # Candidate pool for additions
    max_val = best[-1] if best else 1
    pool_size = 500
    candidates = list(range(1, max(2, max_val + 1)))
    # Also add some random candidates beyond current max
    for _ in range(pool_size // 2):
        candidates.append(max_val + random.randint(1, max(1, max_val // 2)))
    random.shuffle(candidates)

    # Keep track of visited states to avoid cycles
    visited = {tuple(sorted(best))}
    iterations = 0
    max_iterations = 50000

    while time.time() - start < time_budget and iterations < max_iterations:
        iterations += 1
        improved = False

        # Try adding a candidate
        for c in candidates[:100]:
            if c in best:
                continue
            new_set = sorted(best + [c])
            key = tuple(new_set)
            if key in visited:
                continue
            s = _score_set(new_set)
            if s > best_score + 1e-12:
                best = new_set
                best_score = s
                visited.add(key)
                improved = True
                # Regenerate candidate pool around new max
                max_val = best[-1]
                candidates = list(range(1, max(2, max_val + 1)))
                for _ in range(pool_size // 2):
                    candidates.append(max_val + random.randint(1, max(1, max_val // 2)))
                random.shuffle(candidates)
                break

        if improved:
            continue

        # Try removing an element (not 0)
        for i in range(len(best)):
            if best[i] == 0:
                continue
            new_set = best[:i] + best[i+1:]
            if not new_set or new_set[0] != 0:
                continue
            new_set = sorted(new_set)
            key = tuple(new_set)
            if key in visited:
                continue
            s = _score_set(new_set)
            if s > best_score + 1e-12:
                best = new_set
                best_score = s
                visited.add(key)
                improved = True
                break

        if improved:
            continue

        # Try replacing an element (not 0) with a candidate
        for i in range(len(best)):
            if best[i] == 0:
                continue
            for c in candidates[:50]:
                if c in best:
                    continue
                new_set = best[:i] + [c] + best[i+1:]
                new_set = sorted(new_set)
                if len(new_set) != len(best):
                    continue  # duplicate
                key = tuple(new_set)
                if key in visited:
                    continue
                s = _score_set(new_set)
                if s > best_score + 1e-12:
                    best = new_set
                    best_score = s
                    visited.add(key)
                    improved = True
                    max_val = best[-1]
                    candidates = list(range(1, max(2, max_val + 1)))
                    for _ in range(pool_size // 2):
                        candidates.append(max_val + random.randint(1, max(1, max_val // 2)))
                    random.shuffle(candidates)
                    break
            if improved:
                break

        if not improved:
            # No improvement in this round; try random perturbations
            best_set = set(best)
            for _ in range(10):
                # Random add
                c = random.randint(1, max(2, best[-1] * 2))
                if c not in best_set:
                    new_set = sorted(best_set | {c})
                    key = tuple(new_set)
                    if key not in visited:
                        s = _score_set(new_set)
                        if s > best_score + 1e-12:
                            best = new_set
                            best_score = s
                            visited.add(key)
                            break
                # Random remove (not 0)
                removable = [x for x in best if x != 0]
                if removable:
                    rm = random.choice(removable)
                    new_set = sorted(x for x in best if x != rm)
                    key = tuple(new_set)
                    if key not in visited:
                        s = _score_set(new_set)
                        if s > best_score + 1e-12:
                            best = new_set
                            best_score = s
                            visited.add(key)
                            break
            # Update candidate pool periodically
            if iterations % 100 == 0:
                max_val = best[-1]
                candidates = list(range(1, max(2, max_val + 1)))
                for _ in range(pool_size // 2):
                    candidates.append(max_val + random.randint(1, max(1, max_val // 2)))
                random.shuffle(candidates)

    return best, best_score


def construct():
    """Main entry point: construct the best set found within time budget."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 2.0  # safety margin

    random.seed(42)

    # Generate diverse initial seeds
    seeds = []

    # Small fixed candidates
    seeds.append([0, 1, 3])
    seeds.append([0, 1, 3, 7])
    seeds.append([0, 1, 3, 8])
    seeds.append([0, 1, 3, 9])
    seeds.append([0, 1, 3, 6])
    seeds.append([0, 1, 3, 5])
    seeds.append([0, 1, 2, 7])
    seeds.append([0, 1, 4, 10])

    # Geometric-like seeds
    for ratio in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for n in range(3, 8):
            seeds.append(_geometric_seed(ratio, n))

    # Evaluate all seeds
    best = None
    best_score = -1.0
    for seed in seeds:
        if time.time() > deadline:
            break
        s = _score_set(seed)
        if s > best_score:
            best_score = s
            best = list(seed)

    # Local search from the best seed
    if best is not None:
        time_remaining = max(0.0, deadline - time.time())
        if time_remaining > 1.0:
            best, best_score = _local_search(best, time_remaining)

    # If we still have time, try local search from a few other seeds
    if best is not None:
        time_remaining = max(0.0, deadline - time.time())
        if time_remaining > 2.0:
            # Try from second-best seed
            seed_scores = []
            for seed in seeds:
                s = _score_set(seed)
                seed_scores.append((s, seed))
            seed_scores.sort(key=lambda x: -x[0])
            for s, seed in seed_scores[:3]:
                if time.time() > deadline - 1.0:
                    break
                if seed == best:
                    continue
                time_left = max(0.0, deadline - time.time() - 1.0)
                if time_left < 0.5:
                    break
                candidate, cs = _local_search(list(seed), time_left)
                if cs > best_score:
                    best = candidate
                    best_score = cs

    # Final safety check
    if best is None or 0 not in best:
        best = [0, 1, 3]

    return sorted(set(best))
