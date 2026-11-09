import random
import time

N = 256

def _is_sidon(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def _is_maximal(S, sums):
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for s in S_list:
            if x + s in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_maximal(random_order=None):
    selected = []
    sums = set()
    if random_order is None:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
    
    for x in random_order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _reduce(S, deadline):
    S = sorted(S)
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for i in range(len(S)):
            if time.time() > deadline:
                break
            candidate = S[:i] + S[i + 1:]
            if not candidate:
                continue
            if not _is_sidon(candidate):
                continue
            sums = _get_sums(candidate)
            if _is_maximal(candidate, sums):
                S = candidate
                changed = True
                break
    return S

def _search_maximal_sidon(k, deadline):
    """
    Backtracking search for a maximal Sidon set of exactly size k.
    """
    if time.time() > deadline:
        return None

    def backtrack(current_set, current_sums, remaining, target_size):
        if time.time() > deadline:
            return None

        if len(current_set) == target_size:
            if _is_maximal(current_set, current_sums):
                return current_set[:]
            return None

        if not remaining:
            return None

        # Heuristic: Try candidates in random order
        candidates = list(remaining)
        random.shuffle(candidates)

        for x in candidates:
            new_sums_added = [x + a for a in current_set] + [2 * x]
            if len(set(new_sums_added)) != len(new_sums_added):
                continue
            if current_sums.intersection(set(new_sums_added)):
                continue

            new_set = current_set + [x]
            new_sums = current_sums.union(set(new_sums_added))

            res = backtrack(new_set, new_sums, [y for y in remaining if y > x], target_size)
            if res:
                return res

        return None

    res = backtrack([], set(), list(range(1, N + 1)), k)
    return res

def construct():
    deadline = time.time() + 28  # Safety margin
    random.seed(42)

    best = _greedy_maximal()
    best_size = len(best)
    
    # Initial reduction
    reduced = _reduce(best, deadline)
    if len(reduced) < best_size:
        best = reduced
        best_size = len(best)

    # Iterate random greedy + reduction
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Try different biases
        order = list(range(1, N + 1))
        if attempts % 3 == 0:
            # Low bias
            order.sort(key=lambda x: random.random() * x)
        elif attempts % 3 == 1:
            # High bias
            order.sort(key=lambda x: random.random() * (N - x))
        else:
            random.shuffle(order)
            
        S = _greedy_maximal(order)
        if len(S) >= best_size:
            # Still might reduce to something smaller? Unlikely if greedy is larger.
            # But reduction is powerful. Let's try if time permits.
            if time.time() < deadline - 5:
                reduced = _reduce(S, deadline)
                if len(reduced) < best_size:
                    best = reduced
                    best_size = len(best)
            continue

        reduced = _reduce(S, deadline)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(best)

    # Targeted search for smaller k
    # Lower bound is ~16. We try from 16 up to best_size - 1
    # We search in increasing k. If we find one for k, it's a valid maximal set of size k.
    # We want the MINIMUM, so we search k=16, 17, ...
    # However, backtracking is expensive. We only do it if we have time.
    
    min_k_to_try = 16
    max_k_to_try = best_size - 1
    
    # Sort k by likelihood? Smaller k is harder to find.
    # Let's try k from best_size - 1 down to 16? No, we want min.
    # If we find k=20, we know min <= 20.
    # If we find k=16, we are done.
    # Backtracking for small k is faster? No, it's more constrained, so it might fail fast,
    # but if it succeeds, it's great.
    
    # Let's try a few specific k values.
    for k in range(min_k_to_try, max_k_to_try + 1):
        if time.time() > deadline - 2:
            break
        # Give each k a small time budget
        sub_deadline = min(deadline, time.time() + 2)
        found = _search_maximal_sidon(k, sub_deadline)
        if found:
            best = found
            best_size = k
            # If we found a very small one, we might stop early?
            # The problem asks to minimize. If we find 16, it's likely optimal or near.
            # But we should check if we can go lower? 16 is the theoretical lower bound approx.
            # So if we find 16, we can probably stop.
            if k <= 17:
                break

    return best
