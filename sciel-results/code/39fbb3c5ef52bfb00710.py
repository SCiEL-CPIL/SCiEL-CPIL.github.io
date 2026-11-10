"""Covering code K_q(n, radius): greedy + iterated local search with constraint propagation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools
from collections import defaultdict

_DEADLINE = 145.0  # safety margin below the 150 s wall-clock kill


def _all_words(q, n):
    total = q ** n
    words = np.empty((total, n), dtype=np.uint8)
    for i in range(n):
        words[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    return words


def _build_neighbours(q, n, radius, words):
    """Precompute neighbour lists (indices within Hamming distance <= radius)."""
    total = words.shape[0]
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    if radius == 1:
        nb = np.empty((total, 1 + n * (q - 1)), dtype=np.int64)
        for idx in range(total):
            digits = words[idx].astype(np.int64)
            pos = 0
            nb[idx, pos] = idx
            pos += 1
            for i in range(n):
                v = digits[i]
                for vp in range(q):
                    if vp == v:
                        continue
                    nb[idx, pos] = idx + (vp - v) * stride[i]
                    pos += 1
        return nb
    else:
        patterns = []
        for k in range(radius + 1):
            for subset in itertools.combinations(range(n), k):
                if k == 0:
                    patterns.append([])
                else:
                    for assign in itertools.product(range(q), repeat=k):
                        patterns.append([(subset[j], assign[j]) for j in range(k)])
        nb_list = []
        for idx in range(total):
            digits = words[idx].astype(np.int64)
            nbs = set()
            for pat in patterns:
                new_idx = idx
                for (coord, new_sym) in pat:
                    new_idx += (new_sym - digits[coord]) * stride[coord]
                if 0 <= new_idx < total:
                    nbs.add(new_idx)
            nb_list.append(sorted(nbs))
        return nb_list


def _verify_coverage(codewords, q, n, radius, words, nb):
    """Check if codewords cover all words. Returns (covered_all, uncovered_count)."""
    total = words.shape[0]
    covered = np.zeros(total, dtype=bool)
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        nbs = nb[idx]
        for c in nbs:
            covered[c] = True
    return covered.all(), int((~covered).sum())


def _greedy_cover(q, n, radius, words, nb, seed, time_limit):
    """Greedy set cover with deadline guard using vectorized gain updates."""
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()

    covered = np.zeros(total, dtype=bool)
    codewords = []

    # gain[i] = number of uncovered words covered by position i
    gain = np.zeros(total, dtype=np.int32)
    for i in range(total):
        for c in nb[i]:
            gain[i] += 1

    # Precompute reverse neighbours for faster gain updates
    # rev_nb[w] = list of codeword indices that cover word w
    rev_nb = []
    for w in range(total):
        rev_nb.append([])
    for i in range(total):
        for c in nb[i]:
            rev_nb[c].append(i)

    while True:
        if time.time() - start > time_limit:
            break

        best_idx = int(np.argmax(gain))
        if gain[best_idx] == 0:
            break

        codewords.append(words[best_idx].tolist())
        newly_covered = []
        for c in nb[best_idx]:
            if not covered[c]:
                covered[c] = True
                newly_covered.append(c)
        for c in newly_covered:
            for w in rev_nb[c]:
                gain[w] -= 1

    return codewords, covered


def _remove_redundant(codewords, q, n, radius, words, nb, time_limit):
    """Remove redundant codewords efficiently."""
    start = time.time()
    total = words.shape[0]
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    codewords = list(codewords)

    coverage_count = np.zeros(total, dtype=np.int32)
    included = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        included[idx] = True
        for c in nb[idx]:
            coverage_count[c] += 1

    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        included_indices = np.where(included)[0]
        for idx in included_indices:
            if not included[idx]:
                continue
            redundant = True
            for c in nb[idx]:
                if coverage_count[c] == 1:
                    redundant = False
                    break
            if redundant:
                included[idx] = False
                for c in nb[idx]:
                    coverage_count[c] -= 1
                changed = True

    return [words[i].tolist() for i in np.where(included)[0]]


def _local_search(codewords, q, n, radius, words, nb, seed, time_limit):
    """
    Iterated Local Search: aggressively remove redundant codewords, then try
    structural swaps to enable further removals.
    """
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    included = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        included[idx] = True

    coverage_count = np.zeros(total, dtype=np.int32)
    for i in range(total):
        if included[i]:
            for c in nb[i]:
                coverage_count[c] += 1

    best_count = int(included.sum())
    best_included = included.copy()

    # Precompute reverse neighbours
    rev_nb = [[] for _ in range(total)]
    for i in range(total):
        for c in nb[i]:
            rev_nb[c].append(i)

    def try_remove_pass():
        """Repeatedly remove any codeword that is redundant."""
        nonlocal coverage_count, included, best_count, best_included
        progress = True
        while progress:
            progress = False
            included_indices = np.where(included)[0]
            for idx in included_indices:
                if not included[idx]:
                    continue
                redundant = True
                for c in nb[idx]:
                    if coverage_count[c] == 1:
                        redundant = False
                        break
                if redundant:
                    included[idx] = False
                    for c in nb[idx]:
                        coverage_count[c] -= 1
                    progress = True
        current_count = int(included.sum())
        if current_count < best_count:
            best_count = current_count
            best_included = included.copy()

    def try_swap(idx):
        """Try to swap codeword idx with another to enable future removals."""
        nonlocal coverage_count, included
        # Find words that would become uncovered if we remove idx
        would_be_uncovered = []
        for c in nb[idx]:
            if coverage_count[c] == 1:
                would_be_uncovered.append(c)
        
        if not would_be_uncovered:
            return True  # Can just remove
        
        # Find candidate that covers all of would_be_uncovered
        # Use reverse neighbours of the uncovered words
        candidate_set = set()
        for w in would_be_uncovered:
            for cand in rev_nb[w]:
                if not included[cand]:
                    candidate_set.add(cand)
        
        if not candidate_set:
            return False
        
        # Try candidates, prefer those covering more would_be_uncovered
        candidates = list(candidate_set)
        if len(candidates) > 50:
            candidates = rng.sample(candidates, 50)
        
        best_cand = -1
        best_gain = 0
        for cand in candidates:
            gain = 0
            for w in would_be_uncovered:
                if cand in rev_nb[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        
        if best_cand >= 0 and best_gain == len(would_be_uncovered):
            included[idx] = False
            for c in nb[idx]:
                coverage_count[c] -= 1
            included[best_cand] = True
            for c in nb[best_cand]:
                coverage_count[c] += 1
            return True
        return False

    # Main ILS loop
    while time.time() - start < time_limit - 0.5:
        # Phase 1: Remove all redundant codewords
        try_remove_pass()
        
        if int(included.sum()) <= best_count:
            # Try to make structural changes to enable more removals
            included_indices = np.where(included)[0]
            if len(included_indices) == 0:
                break
            
            # Pick a random codeword and try to swap it
            idx = included_indices[rng.randint(0, len(included_indices) - 1)]
            swapped = try_swap(idx)
            if swapped:
                # After swap, try to remove redundancies again
                try_remove_pass()
        
        # Random perturbation: remove k random non-essential codewords and re-greedy
        if rng.random() < 0.3:
            included_indices = np.where(included)[0]
            k = min(3, len(included_indices) // 10)
            if k > 0:
                to_remove = rng.sample(list(included_indices), k)
                for idx in to_remove:
                    if not included[idx]:
                        continue
                    # Check if safe to remove
                    safe = True
                    for c in nb[idx]:
                        if coverage_count[c] == 1:
                            safe = False
                            break
                    if safe:
                        included[idx] = False
                        for c in nb[idx]:
                            coverage_count[c] -= 1
                # Re-greedy to cover any gaps
                try_remove_pass()

    result = []
    for i in range(total):
        if best_included[i]:
            result.append(words[i].tolist())
    return result


def _backtracking_improve(codewords, q, n, radius, words, nb, seed, time_limit):
    """
    Systematic improvement: iteratively find and remove the least-essential
    codeword, using full reverse-neighbour lookups for swap candidates.
    """
    rng = random.Random(seed)
    total = words.shape[0]
    start = time.time()
    stride = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    included = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = int(np.dot(cw, stride))
        included[idx] = True

    coverage_count = np.zeros(total, dtype=np.int32)
    for i in range(total):
        if included[i]:
            for c in nb[i]:
                coverage_count[c] += 1

    best_count = int(included.sum())
    best_included = included.copy()

    # Precompute reverse neighbours
    rev_nb = [[] for _ in range(total)]
    for i in range(total):
        for c in nb[i]:
            rev_nb[c].append(i)

    for iteration in range(200):
        if time.time() - start > time_limit - 1:
            break
        
        # Find codewords with minimum essential coverage
        essential_counts = np.zeros(total, dtype=np.int32)
        for i in range(total):
            if included[i]:
                for c in nb[i]:
                    if coverage_count[c] == 1:
                        essential_counts[i] += 1
        
        included_indices = np.where(included)[0]
        if len(included_indices) == 0:
            break
        
        # Sort by essential count ascending (least essential first)
        sorted_indices = sorted(included_indices, key=lambda x: essential_counts[x])
        
        # Try removing from least essential
        removed = False
        for idx in sorted_indices:
            if not included[idx]:
                continue
            if essential_counts[idx] == 0:
                included[idx] = False
                for c in nb[idx]:
                    coverage_count[c] -= 1
                best_count = int(included.sum())
                best_included = included.copy()
                removed = True
                break
            elif essential_counts[idx] <= 2:
                # Try to swap: find a replacement that covers the essential words
                would_be_uncovered = [c for c in nb[idx] if coverage_count[c] == 1]
                if len(would_be_uncovered) <= 2:
                    candidate_set = set()
                    for w in would_be_uncovered:
                        for cand in rev_nb[w]:
                            if not included[cand]:
                                candidate_set.add(cand)
                    if candidate_set:
                        candidates = list(candidate_set)
                        if len(candidates) > 30:
                            candidates = rng.sample(candidates, 30)
                        best_cand = -1
                        best_gain = 0
                        for cand in candidates:
                            gain = sum(1 for w in would_be_uncovered if cand in rev_nb[w])
                            if gain > best_gain:
                                best_gain = gain
                                best_cand = cand
                        if best_cand >= 0 and best_gain == len(would_be_uncovered):
                            included[idx] = False
                            for c in nb[idx]:
                                coverage_count[c] -= 1
                            included[best_cand] = True
                            for c in nb[best_cand]:
                                coverage_count[c] += 1
                            removed = True
                            break
        if not removed:
            break
    
    result = []
    for i in range(total):
        if best_included[i]:
            result.append(words[i].tolist())
    return result


def construct(q, n, radius):
    start = time.time()

    # Precompute words and neighbours
    words = _all_words(q, n)
    nb = _build_neighbours(q, n, radius, words)
    total = words.shape[0]

    best_codewords = None
    best_count = float('inf')

    # Phase 1: Greedy constructions with multiple seeds
    greedy_seeds = [42, 123, 456, 789, 1011, 2022, 3141, 2718]
    time_for_greedy = min(12.0, (_DEADLINE - time.time()) * 0.2)
    time_per_greedy = time_for_greedy / len(greedy_seeds)

    for seed in greedy_seeds:
        if time.time() > _DEADLINE - 50:
            break
        try:
            codewords, covered = _greedy_cover(q, n, radius, words, nb, seed, time_per_greedy)
            cov_all, unc = _verify_coverage(codewords, q, n, radius, words, nb)
            if cov_all and len(codewords) < best_count:
                best_count = len(codewords)
                best_codewords = codewords[:]
        except Exception:
            continue

    if best_codewords is None:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Phase 2: Remove redundant codewords
    if time.time() < _DEADLINE - 35:
        try:
            improved = _remove_redundant(best_codewords, q, n, radius, words, nb, 5.0)
            cov_all, unc = _verify_coverage(improved, q, n, radius, words, nb)
            if cov_all and len(improved) < best_count:
                best_count = len(improved)
                best_codewords = improved[:]
        except Exception:
            pass

    # Phase 3: Local search polish (multiple restarts)
    remaining = _DEADLINE - time.time() - 5
    if remaining > 5 and best_codewords is not None:
        ls_seeds = [42, 123, 456, 789, 1011, 2022]
        time_per_ls = remaining / len(ls_seeds)
        for seed in ls_seeds:
            if time.time() > _DEADLINE - 5:
                break
            try:
                ls_codewords = _local_search(
                    best_codewords, q, n, radius, words, nb, seed, time_per_ls
                )
                cov_all, unc = _verify_coverage(ls_codewords, q, n, radius, words, nb)
                if cov_all and len(ls_codewords) < best_count:
                    best_count = len(ls_codewords)
                    best_codewords = ls_codewords[:]
            except Exception:
                continue

        # Phase 4: Backtracking improvement
        remaining2 = _DEADLINE - time.time() - 3
        if remaining2 > 3:
            try:
                bt_codewords = _backtracking_improve(
                    best_codewords, q, n, radius, words, nb, 42, remaining2
                )
                cov_all, unc = _verify_coverage(bt_codewords, q, n, radius, words, nb)
                if cov_all and len(bt_codewords) < best_count:
                    best_count = len(bt_codewords)
                    best_codewords = bt_codewords[:]
            except Exception:
                pass

        # Final redundancy removal
        if time.time() < _DEADLINE - 3:
            try:
                improved = _remove_redundant(best_codewords, q, n, radius, words, nb, 2.0)
                cov_all, unc = _verify_coverage(improved, q, n, radius, words, nb)
                if cov_all and len(improved) < best_count:
                    best_count = len(improved)
                    best_codewords = improved[:]
            except Exception:
                pass

    return best_codewords
# EVOLVE-BLOCK-END
