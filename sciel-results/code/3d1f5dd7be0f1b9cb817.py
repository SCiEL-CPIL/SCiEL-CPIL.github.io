"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import isqrt

NMAX = 256

def _valid_addition(x, S_list):
    """Check if x can be added to set S_list without violating the constraint.
    For all distinct a,b,c in final set, a does not divide b*c.
    When adding x, check every triple that involves x and two other distinct members:
    - (x, b, c): x | b*c?
    - (b, x, c): b | x*c?
    - (c, x, b): c | x*b?
    for all ordered b,c in S_list, b != c.
    """
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True


def _conflict_degree(x, S_list):
    """Count how many pairs (b,c) in S_list would create a conflict if x were added.
    A conflict exists if any of: x|b*c, b|x*c, c|x*b for distinct b,c in S_list.
    """
    n = len(S_list)
    count = 0
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % x == 0:
                count += 1
            if (x * c) % b == 0:
                count += 1
            if (x * b) % c == 0:
                count += 1
    return count


def _conflict_based_removal(deadline, seed=42):
    """Start with all numbers, then iteratively remove the element with highest conflict degree.
    After removal, try to add back elements that can now fit.
    """
    rng = random.Random(seed)
    S = list(range(2, NMAX + 1))
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    
    # Phase 1: Iteratively remove the element with highest conflict degree
    for iteration in range(100):
        if time.time() > deadline:
            break
        if len(S) < 3:
            break
        
        # Find the element with highest conflict degree
        max_conflict = -1
        max_idx = -1
        for i, x in enumerate(S):
            cd = _conflict_degree(x, [s for j, s in enumerate(S) if j != i])
            if cd > max_conflict:
                max_conflict = cd
                max_idx = i
        
        if max_conflict == 0:
            break  # No conflicts remain
        
        # Remove the element with highest conflict
        removed = S.pop(max_idx)
        S_set.discard(removed)
        
        # Try to add back some elements
        candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
        rng.shuffle(candidates)
        for x in candidates[:30]:
            if _valid_addition(x, S):
                S.append(x)
                S_set.add(x)
        
        if len(S) > best_size:
            best = S[:]
            best_size = len(S)
    
    # Phase 2: Local search refinement
    S_final = _local_search(S, deadline, max_iters=500, seed=seed)
    if len(S_final) > best_size:
        best = S_final
        best_size = len(S_final)
    
    return best


def _greedy_largest_first(deadline):
    chosen = []
    chosen_set = set()
    for x in range(NMAX, 1, -1):
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _greedy_smallest_first(deadline):
    chosen = []
    chosen_set = set()
    for x in range(2, NMAX + 1):
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _greedy_randomized(deadline, seed=42):
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    rng.shuffle(candidates)
    chosen = []
    chosen_set = set()
    for x in candidates:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _greedy_weighted(deadline, seed=42, alpha=0.5):
    """Greedy with weighted randomization favoring larger numbers."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    weights = [(x / NMAX) ** alpha for x in candidates]
    chosen = []
    chosen_set = set()
    remaining = candidates[:]
    remaining_weights = weights[:]
    while remaining:
        if time.time() > deadline:
            break
        total_w = sum(remaining_weights)
        r = rng.random() * total_w
        cumulative = 0.0
        idx = 0
        for k in range(len(remaining)):
            cumulative += remaining_weights[k]
            if r <= cumulative:
                idx = k
                break
        x = remaining[idx]
        remaining.pop(idx)
        remaining_weights.pop(idx)
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _local_search(S, deadline, max_iters=2000, seed=99999):
    """Local search: try adding elements, removing, swapping."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)

    for _it in range(max_iters):
        if time.time() > deadline:
            break

        # Try adding a random element
        if len(S) < NMAX:
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                added = False
                for x in candidates[:20]:
                    if _valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue

        # Try swapping: remove one, add another
        if S:
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i + 1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:40]:
                    if _valid_addition(x, S_temp):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    S = S_temp + [removed]
                    S_set = set(S)

    return best


def _two_opt_search(S, deadline, max_iters=400, seed=77777):
    """Try removing pairs and adding pairs to improve."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)

    for _it in range(max_iters):
        if time.time() > deadline:
            break
        if len(S) < 2:
            break

        i, j = rng.sample(range(len(S)), 2)
        i, j = min(i, j), max(i, j)
        S_temp = S[:i] + S[i + 1:j] + S[j + 1:]
        S_temp_set = set(S_temp)

        candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
        if not candidates:
            continue

        rng.shuffle(candidates)
        found = False
        for x in candidates[:25]:
            if not _valid_addition(x, S_temp):
                continue
            S_temp2 = S_temp + [x]
            S_temp2_set = S_temp_set | {x}
            for y in candidates[:25]:
                if y == x or y in S_temp2_set:
                    continue
                if _valid_addition(y, S_temp2):
                    S = S_temp2 + [y]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    found = True
                    break
            if found:
                break
    return best


def _simulated_annealing(S, deadline, max_iters=3000, T_start=100.0, T_end=1.0, seed=55555):
    """Simulated annealing: can accept worse states to escape local optima."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)

    current_temp = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))

    for _it in range(max_iters):
        if time.time() > deadline:
            break
        current_temp *= cooling

        op = rng.random()
        if op < 0.35 and len(S) < NMAX:
            # Add
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        elif op < 0.6 and S:
            # Remove (always allowed, may shrink to allow re-growth)
            i = rng.randrange(len(S))
            removed = S[i]
            S.pop(i)
            S_set.discard(removed)
            # Try to re-add something better
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                for x in candidates[:25]:
                    if _valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        break
            continue
        elif S:
            # Swap
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i + 1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S_temp):
                    S = S_temp + [x]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)

    return best


def _local_search_aggressive(S, deadline):
    best = S[:]
    best_size = len(S)
    for restart in range(5):
        if time.time() > deadline:
            break
        S_perturbed = best[:]
        rng = random.Random(1000 + restart)
        n_remove = rng.randint(1, min(3, len(S_perturbed)))
        indices = rng.sample(range(len(S_perturbed)), n_remove)
        indices.sort(reverse=True)
        for idx in indices:
            S_perturbed.pop(idx)
        S_result = _local_search(S_perturbed, deadline, max_iters=120, seed=2000 + restart)
        if len(S_result) > best_size:
            best = S_result
            best_size = len(S_result)
    return best


def construct():
    start_time = time.time()
    # Contract: 60s wall-clock. Use a safety margin well under it.
    deadline = start_time + 55.0

    best = None
    best_size = 0

    def consider(c):
        nonlocal best, best_size
        if c and len(c) > best_size:
            best = c
            best_size = len(c)

    # Diverse greedy starting points
    consider(_greedy_largest_first(deadline))
    if time.time() < deadline - 8:
        consider(_greedy_smallest_first(deadline))
    if time.time() < deadline - 8:
        consider(_greedy_weighted(deadline, seed=42, alpha=0.5))
    for seed in [1, 7, 13, 99, 256]:
        if time.time() > deadline - 12:
            break
        consider(_greedy_randomized(deadline - 12, seed=seed))

    # Conflict-based removal heuristic (new)
    if time.time() < deadline - 15:
        for seed in [42, 123, 456]:
            if time.time() > deadline - 10:
                break
            consider(_conflict_based_removal(deadline - 10, seed=seed))

    # Refinement: local search on the best so far
    if best and time.time() < deadline - 10:
        consider(_local_search(best, deadline - 10, max_iters=300))

    # Simulated annealing (strongest refinement) from a couple of starts
    for seed in [55555, 12345]:
        if time.time() > deadline - 8:
            break
        consider(_simulated_annealing(best, deadline - 8, max_iters=2500, seed=seed))

    # 2-opt pair swap
    if best and time.time() < deadline - 6:
        consider(_two_opt_search(best, deadline - 6, max_iters=300))

    # Aggressive multi-restart local search from a few fresh greedy starts
    for seed in [2, 5, 11, 23, 42]:
        if time.time() > deadline - 4:
            break
        c = _greedy_randomized(deadline - 4, seed=seed)
        if c:
            consider(_local_search_aggressive(c, deadline - 4))

    if best is None:
        # Fallback: all primes (a valid set: for distinct primes p,q,r, p never divides q*r)
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        best = [x for x in range(2, NMAX + 1) if is_prime(x)]

    # Ensure we return a list of Python ints
    return [int(x) for x in best]
