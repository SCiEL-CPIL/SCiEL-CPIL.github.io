"""Marginal-yield greedy packing for sumset/difference-set exponent lower bound."""
import time
import math
import random
import numpy as np

_DEADLINE = 120.0
_TIME_BUDGET = 113.0  # safety margin before the 120s kill


def _score_from_counts(diff_count, sum_count, max_u):
    if sum_count <= 0 or diff_count <= 0 or max_u <= 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count) / math.log(2 * max_u + 1)


def _golomb_ruler(n):
    """Small optimal Golomb rulers (distinct pairwise differences)."""
    tables = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 28],
        8: [0, 1, 4, 10, 18, 23, 29, 36],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68],
        13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76],
        14: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84],
        15: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84, 92],
        16: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84, 92, 100],
    }
    if n in tables:
        return list(tables[n])
    # Fallback: greedy construction
    elems = [0]
    diffs = set()
    while len(elems) < n:
        x = 1
        while True:
            ok = True
            for e in elems:
                if abs(x - e) in diffs:
                    ok = False
                    break
            if ok:
                for e in elems:
                    diffs.add(abs(x - e))
                elems.append(x)
                break
            x += 1
    return elems


def _counts_from_set(U):
    """Return (diff_count, sum_count, max_u) using FFT."""
    U = sorted(set(int(x) for x in U))
    if not U:
        return 0, 0, 0
    max_u = U[-1]
    ind = np.zeros(max_u + 1, dtype=np.float64)
    for x in U:
        ind[x] = 1.0

    # Sumset: convolve(ind, ind); support is [0, 2*max_u]
    n_conv = 2 * max_u + 1
    conv = np.fft.irfft(np.fft.rfft(ind, n=n_conv) ** 2, n=n_conv)
    sum_count = int(np.sum(conv > 1e-6))

    # Difference set: convolve(ind, reversed(ind)); support [-(max_u), max_u]
    ind_rev = ind[::-1]
    conv_d = np.fft.irfft(np.fft.rfft(ind, n=n_conv) * np.fft.rfft(ind_rev, n=n_conv), n=n_conv)
    diff_count = int(np.sum(conv_d > 1e-6))

    return diff_count, sum_count, max_u


def _score_of(U):
    diff_count, sum_count, max_u = _counts_from_set(U)
    return _score_from_counts(diff_count, sum_count, max_u)


def _marginal_yield(U_set, candidate):
    """
    Compute the marginal yield of adding `candidate` to U_set.
    Yield = (new_diffs_added / max(1, new_sums_added)) / log(2*new_max+1)
    Higher is better: more new differences per new sum, normalized by span growth.
    """
    if candidate in U_set:
        return -1.0
    
    new_U = U_set | {candidate}
    U_sorted = sorted(new_U)
    max_u = U_sorted[-1]
    
    if max_u <= 0:
        return -1.0
    
    ind = np.zeros(max_u + 1, dtype=np.float64)
    for x in U_sorted:
        ind[x] = 1.0
    
    n_conv = 2 * max_u + 1
    conv = np.fft.irfft(np.fft.rfft(ind, n=n_conv) ** 2, n=n_conv)
    new_sum_count = int(np.sum(conv > 1e-6))
    
    ind_rev = ind[::-1]
    conv_d = np.fft.irfft(np.fft.rfft(ind, n=n_conv) * np.fft.rfft(ind_rev, n=n_conv), n=n_conv)
    new_diff_count = int(np.sum(conv_d > 1e-6))
    
    # Compute old counts for comparison
    old_max = max(U_set) if U_set else 0
    if old_max <= 0:
        old_sum_count = 1  # {0+0=0}
        old_diff_count = 1  # {0-0=0}
    else:
        old_U = sorted(U_set)
        old_ind = np.zeros(old_max + 1, dtype=np.float64)
        for x in old_U:
            old_ind[x] = 1.0
        old_n_conv = 2 * old_max + 1
        old_conv = np.fft.irfft(np.fft.rfft(old_ind, n=old_n_conv) ** 2, n=old_n_conv)
        old_sum_count = int(np.sum(old_conv > 1e-6))
        old_ind_rev = old_ind[::-1]
        old_conv_d = np.fft.irfft(np.fft.rfft(old_ind, n=old_n_conv) * np.fft.rfft(old_ind_rev, n=old_n_conv), n=old_n_conv)
        old_diff_count = int(np.sum(old_conv_d > 1e-6))
    
    diffs_added = new_diff_count - old_diff_count
    sums_added = max(1, new_sum_count - old_sum_count)
    
    if diffs_added <= 0:
        return -1.0
    
    log_span = math.log(2 * max_u + 1)
    yield_val = (diffs_added / sums_added) / log_span
    
    return yield_val


def _marginal_yield_remove(U_set, element):
    """
    Compute the marginal yield of removing `element` from U_set.
    Yield = (diffs_lost / max(1, sums_lost)) / log(2*new_max+1)
    Lower is better (less loss). We want to remove elements with low yield (harmful).
    """
    if element == 0 or element not in U_set:
        return float('inf')
    
    new_U = U_set - {element}
    if not new_U:
        return float('inf')
    
    U_sorted = sorted(new_U)
    max_u = U_sorted[-1]
    
    if max_u <= 0:
        return 0.0
    
    ind = np.zeros(max_u + 1, dtype=np.float64)
    for x in U_sorted:
        ind[x] = 1.0
    
    n_conv = 2 * max_u + 1
    conv = np.fft.irfft(np.fft.rfft(ind, n=n_conv) ** 2, n=n_conv)
    new_sum_count = int(np.sum(conv > 1e-6))
    
    ind_rev = ind[::-1]
    conv_d = np.fft.irfft(np.fft.rfft(ind, n=n_conv) * np.fft.rfft(ind_rev, n=n_conv), n=n_conv)
    new_diff_count = int(np.sum(conv_d > 1e-6))
    
    # Old counts
    old_max = max(U_set)
    old_U = sorted(U_set)
    old_ind = np.zeros(old_max + 1, dtype=np.float64)
    for x in old_U:
        old_ind[x] = 1.0
    old_n_conv = 2 * old_max + 1
    old_conv = np.fft.irfft(np.fft.rfft(old_ind, n=old_n_conv) ** 2, n=old_n_conv)
    old_sum_count = int(np.sum(old_conv > 1e-6))
    old_ind_rev = old_ind[::-1]
    old_conv_d = np.fft.irfft(np.fft.rfft(old_ind, n=old_n_conv) * np.fft.rfft(old_ind_rev, n=old_n_conv), n=old_n_conv)
    old_diff_count = int(np.sum(old_conv_d > 1e-6))
    
    diffs_lost = old_diff_count - new_diff_count
    sums_lost = max(1, old_sum_count - new_sum_count)
    
    if diffs_lost <= 0:
        return -1.0  # Removing this element doesn't lose differences (good)
    
    log_span = math.log(2 * max_u + 1) if max_u > 0 else 1.0
    yield_val = (diffs_lost / sums_lost) / log_span
    
    return yield_val


def _greedy_add_pass(U_set, deadline, max_candidates=200):
    """
    Greedily add elements with highest marginal yield until no improvement or deadline.
    Returns the updated U_set.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        max_u = max(U_set) if U_set else 0
        
        # Generate candidate positions
        candidates = set()
        for x in U_set:
            for d in (1, 2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 30, 40, 50):
                candidates.add(x + d)
                candidates.add(x - d)
        for d in (1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 200):
            candidates.add(max_u + d)
        # Random candidates
        for _ in range(30):
            candidates.add(random.randint(1, max_u + 100))
        
        # Filter valid candidates
        valid_candidates = [c for c in candidates if c > 0 and c not in U_set]
        
        if not valid_candidates:
            break
        
        # Score candidates by marginal yield
        scored = []
        for c in valid_candidates[:max_candidates]:
            y = _marginal_yield(U_set, c)
            if y > 0:
                scored.append((y, c))
        
        if not scored:
            break
        
        # Sort by yield descending
        scored.sort(reverse=True)
        
        # Try the best candidates
        for yield_val, c in scored[:10]:
            new_U = U_set | {c}
            s = _score_of(new_U)
            old_s = _score_of(U_set)
            if s > old_s + 1e-10:
                U_set = new_U
                improved = True
                break
        
        # If we couldn't improve with top candidates, try a few more random ones
        if not improved:
            for c in random.sample(valid_candidates, min(5, len(valid_candidates))):
                new_U = U_set | {c}
                s = _score_of(new_U)
                old_s = _score_of(U_set)
                if s > old_s + 1e-10:
                    U_set = new_U
                    improved = True
                    break
    
    return U_set


def _greedy_remove_pass(U_set, deadline):
    """
    Greedily remove elements with lowest marginal yield (most harmful) until no improvement.
    Returns the updated U_set.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        removable = [x for x in U_set if x != 0]
        
        if not removable:
            break
        
        # Score removable elements by yield (lower is better to remove)
        scored = []
        for x in removable:
            y = _marginal_yield_remove(U_set, x)
            scored.append((y, x))
        
        # Sort by yield ascending (most harmful first)
        scored.sort()
        
        # Try removing the most harmful elements
        for yield_val, x in scored[:10]:
            if yield_val >= 0:  # Only consider if removing helps
                new_U = U_set - {x}
                s = _score_of(new_U)
                old_s = _score_of(U_set)
                if s > old_s + 1e-10:
                    U_set = new_U
                    improved = True
                    break
    
    return U_set


def _local_polish(U_set, deadline, max_iterations=5000):
    """
    Fine-tuning local search with small moves after greedy passes.
    """
    current_score = _score_of(U_set)
    best_score = current_score
    best_U = set(U_set)
    
    iterations = 0
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        U = sorted(U_set)
        max_u = U[-1] if U else 0
        
        move_type = random.random()
        if move_type < 0.4:
            # Add
            candidates = []
            for x in U:
                for d in (1, 2, 3, 5, 8):
                    candidates.append(x + d)
            candidates.append(max_u + 1)
            candidates.append(max_u + 2)
            candidates.append(max_u + 3)
            candidates.append(random.randint(1, max_u + 15))
            random.shuffle(candidates)
            for x in candidates[:8]:
                if x in U_set or x <= 0:
                    continue
                new_U = U_set | {x}
                s = _score_of(new_U)
                if s > current_score + 1e-10:
                    U_set = new_U
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_U = set(U_set)
                    break
        elif move_type < 0.6:
            # Remove
            removable = [x for x in U if x != 0]
            if not removable:
                continue
            random.shuffle(removable)
            for x in removable[:5]:
                new_U = U_set - {x}
                s = _score_of(new_U)
                if s > current_score + 1e-10:
                    U_set = new_U
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_U = set(U_set)
                    break
        else:
            # Move
            movable = [x for x in U if x != 0]
            if not movable:
                continue
            random.shuffle(movable)
            for x in movable[:4]:
                for d in (-3, -2, -1, 1, 2, 3, 5, 8):
                    y = x + d
                    if y < 0 or y in U_set:
                        continue
                    new_U = (U_set - {x}) | {y}
                    s = _score_of(new_U)
                    if s > current_score + 1e-10:
                        U_set = new_U
                        current_score = s
                        if current_score > best_score:
                            best_score = current_score
                            best_U = set(U_set)
                        break
                else:
                    continue
                break
    
    return best_U


def construct():
    start = time.time()
    deadline = start + _TIME_BUDGET

    random.seed(12345)

    # Multiple initial candidates
    best_U = None
    best_score = 0.0
    
    for n0 in (8, 10, 12, 14, 16, 18, 20):
        if time.time() > deadline - 10:
            break
        cand = _golomb_ruler(n0)
        s = _score_of(cand)
        if s > best_score:
            best_score = s
            best_U = set(cand)
    
    if best_U is None:
        best_U = set(_golomb_ruler(10))
        best_score = _score_of(best_U)

    # Phase 1: Greedy add pass
    U_set = _greedy_add_pass(set(best_U), deadline - 30)
    s = _score_of(U_set)
    if s > best_score:
        best_score = s
        best_U = set(U_set)

    # Phase 2: Greedy remove pass
    U_set = _greedy_remove_pass(set(best_U), deadline - 25)
    s = _score_of(U_set)
    if s > best_score:
        best_score = s
        best_U = set(U_set)

    # Phase 3: Another add pass
    U_set = _greedy_add_pass(set(best_U), deadline - 20)
    s = _score_of(U_set)
    if s > best_score:
        best_score = s
        best_U = set(U_set)

    # Phase 4: Another remove pass
    U_set = _greedy_remove_pass(set(best_U), deadline - 15)
    s = _score_of(U_set)
    if s > best_score:
        best_score = s
        best_U = set(U_set)

    # Phase 5: Local polish
    U_set = _local_polish(set(best_U), deadline - 3, max_iterations=3000)
    s = _score_of(U_set)
    if s > best_score:
        best_score = s
        best_U = set(U_set)

    # Final: try a few fresh starts with different seeds if time permits
    if time.time() < deadline - 10:
        for seed in (42, 7, 123):
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            for n0 in (10, 12, 14):
                if time.time() > deadline - 5:
                    break
                cand = _golomb_ruler(n0)
                U_temp = _greedy_add_pass(set(cand), deadline - 5, max_candidates=100)
                U_temp = _greedy_remove_pass(set(U_temp), deadline - 3)
                s = _score_of(U_temp)
                if s > best_score:
                    best_score = s
                    best_U = set(U_temp)

    return sorted(best_U)


if __name__ == "__main__":
    result = construct()
    print("Set size:", len(result))
    print("Max element:", max(result))
    print("Score:", _score_of(result))
    print("Set:", result)
