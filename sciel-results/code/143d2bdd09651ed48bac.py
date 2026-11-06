"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _is_sidon(lst):
    """Fast Sidon check for sorted list (all unordered pair sums unique including diagonals)."""
    n = len(lst)
    sums = set()
    for i in range(n):
        a = lst[i]
        for j in range(i, n):
            s = a + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _has_sidon_of_size(ambient, k):
    """Return True if any k-subset of ambient is Sidon."""
    for combo in itertools.combinations(ambient, k):
        if _is_sidon(list(combo)):
            return True
    return False

def _max_sidon_size(ambient):
    """Compute max Sidon subset size for sorted 12-element list."""
    n = len(ambient)
    for k in range(min(n, 6), 2, -1):
        if _has_sidon_of_size(ambient, k):
            return k
    return 2

def _count_sidon_of_size(ambient, k):
    """Count k-subsets that are Sidon."""
    cnt = 0
    for combo in itertools.combinations(ambient, k):
        if _is_sidon(list(combo)):
            cnt += 1
    return cnt

def _structured_candidates():
    """Generate a diverse set of structured starting candidates."""
    cands = []
    # Consecutive integers
    cands.append(list(range(12)))
    # Arithmetic progressions with various steps
    for step in [2, 3, 4, 5, 6, 7, 8, 10, 11]:
        ap = list(range(0, 12 * step, step))
        if len(ap) == 12 and max(ap) <= 128:
            cands.append(ap)
    # Near-AP: consecutive with a few elements perturbed
    base = list(range(12))
    for shift in [1, 2, 3]:
        pert = base[:]
        pert[11] = 12 + shift
        cands.append(pert)
    # Gaps in the middle
    cands.append([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    cands.append([0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13])
    # Dense but spread
    cands.append([0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14])
    return cands

def _local_search(start, rng, deadline, current_best_k, current_best_c5):
    """
    Local search minimizing (max_sidon_size, count_of_5_sidon_subsets).
    The 5-Sidon count is a smooth surrogate: fewer 5-Sidon subsets
    makes it harder for the evaluator to find a large Sidon subset.
    """
    current = sorted(start)
    current_k = current_best_k
    current_c5 = current_best_c5
    best = current[:]
    best_k = current_k
    best_c5 = current_c5
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline and no_improve < max_no_improve:
        idx = rng.randint(0, 11)
        old_val = current[idx]
        new_val = rng.randint(0, 128)

        if new_val == old_val:
            continue
        if new_val in [x for i, x in enumerate(current) if i != idx]:
            continue

        current[idx] = new_val
        current.sort()

        # Evaluate: first check if there's a 5-Sidon
        has_5 = _has_sidon_of_size(current, 5)
        if has_5:
            k = _max_sidon_size(current)
            c5 = _count_sidon_of_size(current, 5)
        else:
            k = _max_sidon_size(current)
            c5 = 0

        # Accept if (k, c5) improves lexicographically
        if (k < current_k) or (k == current_k and c5 < current_c5):
            current_k = k
            current_c5 = c5
            no_improve = 0
            if (k < best_k) or (k == best_k and c5 < best_c5):
                best_k = k
                best_c5 = c5
                best = current[:]
        else:
            # Revert
            current[idx] = old_val
            current.sort()
            no_improve += 1

    return best, best_k, best_c5

def _construct_candidate():
    """
    Generates a candidate with a small max Sidon subset size.
    Strategy:
    1. Evaluate structured candidates to find a good starting point.
    2. Multi-restart local search minimizing (max_sidon_size, 5-Sidon count).
    """
    deadline = time.time() + 50  # Safety margin for 60s limit
    rng = random.Random(123)

    best_cand = list(range(12))
    best_k = _max_sidon_size(best_cand)
    best_c5 = _count_sidon_of_size(best_cand, 5)

    # Phase 1: Evaluate structured candidates
    for cand in _structured_candidates():
        if len(cand) != 12:
            continue
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        if time.time() > deadline - 10:
            break
        k = _max_sidon_size(cand)
        c5 = _count_sidon_of_size(cand, 5) if k >= 5 else 0
        if (k < best_k) or (k == best_k and c5 < best_c5):
            best_k = k
            best_c5 = c5
            best_cand = sorted(cand)
        if best_k <= 3:
            break

    # Phase 2: Local search from the best structured candidate
    if time.time() < deadline - 5:
        improved, ik, ic5 = _local_search(best_cand, rng, deadline - 2, best_k, best_c5)
        if (ik < best_k) or (ik == best_k and ic5 < best_c5):
            best_k = ik
            best_c5 = ic5
            best_cand = improved

    # Phase 3: Additional restarts from random dense sets if time remains
    for trial in range(50):
        if time.time() > deadline - 3:
            break
        if best_k <= 3:
            break
        # Random dense set: sample from a small range
        span = rng.choice([12, 14, 16, 18, 20])
        start = rng.randint(0, max(0, 128 - span))
        try:
            cand = sorted(rng.sample(range(start, start + span + 1), 12))
        except ValueError:
            continue
        if len(set(cand)) != 12:
            continue
        k = _max_sidon_size(cand)
        c5 = _count_sidon_of_size(cand, 5) if k >= 5 else 0
        if (k < best_k) or (k == best_k and c5 < best_c5):
            best_k = k
            best_c5 = c5
            best_cand = cand
            if best_k <= 3:
                break
        # Quick local search from this random start
        if time.time() < deadline - 3:
            improved, ik, ic5 = _local_search(cand, rng, deadline - 2, best_k, best_c5)
            if (ik < best_k) or (ik == best_k and ic5 < best_c5):
                best_k = ik
                best_c5 = ic5
                best_cand = improved
                if best_k <= 3:
                    break

    return best_cand

def construct():
    return _construct_candidate()
# EVOLVE-BLOCK-END
