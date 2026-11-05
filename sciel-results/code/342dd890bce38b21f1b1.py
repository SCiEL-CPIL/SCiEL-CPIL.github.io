"""
Small inclusion-maximal Sidon sets in [1,256].

Strategy (structural change):
1. Model as a blocking problem: find smallest Sidon set S such that for every
   x not in S, adding x creates a sum collision with S's existing sums.
2. Use randomized greedy to build candidate maximal Sidon sets.
3. Use targeted removal with fast maximality checking to shrink sets.
4. Use bitmask-based operations for speed.
"""

import time
import random
import sys

N = 256
MAX_SUM = 2 * N  # 512


def _is_sidon(s):
    """Check if s is a Sidon set (all pair sums a+b with a<=b are distinct)."""
    seen = set()
    n = len(s)
    for i in range(n):
        val = 2 * s[i]
        if val in seen:
            return False
        seen.add(val)
        for j in range(i + 1, n):
            val = s[i] + s[j]
            if val in seen:
                return False
            seen.add(val)
    return True


def _get_sums(s):
    """Return the set of all pair sums a+b with a<=b."""
    sums = set()
    n = len(s)
    for i in range(n):
        sums.add(2 * s[i])
        for j in range(i + 1, n):
            sums.add(s[i] + s[j])
    return sums


def _is_maximal_fast(s, sums):
    """
    Check if s is inclusion-maximal using precomputed sums.
    For each x not in s, check if adding x creates a collision.
    """
    s_set = set(s)
    for x in range(1, N + 1):
        if x in s_set:
            continue
        # Check if x + a is in sums for any a in s, or 2*x is in sums
        blocked = False
        for a in s:
            if (a + x) in sums:
                blocked = True
                break
        if not blocked and (2 * x) in sums:
            blocked = True
        if not blocked:
            return False
    return True


def _greedy_construct(order):
    """
    Greedy Sidon set construction following a given order.
    Returns (selected_list, sums_set).
    """
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added: 2*x and x+a for all a in selected must be new
        if (2 * x) in sums:
            continue
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
        selected.append(x)
        sums.add(2 * x)
        for a in selected[:-1]:  # don't add x+x again
            sums.add(x + a)
    return selected, sums


def _try_shrink(s, sums, deadline):
    """
    Try to remove elements from s while maintaining maximality.
    Greedy removal: repeatedly try removing each element.
    """
    s_list = sorted(s)
    current_sums = _get_sums(s_list)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(s_list)):
            candidate = s_list[:i] + s_list[i+1:]
            if len(candidate) == 0:
                continue
            # Quick Sidon check
            if not _is_sidon(candidate):
                continue
            cand_sums = _get_sums(candidate)
            if _is_maximal_fast(candidate, cand_sums):
                s_list = candidate
                current_sums = cand_sums
                improved = True
                break
    
    return s_list


def _randomized_greedy(seed, deadline):
    """Generate a maximal Sidon set using randomized greedy with a given seed."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    
    selected, sums = _greedy_construct(order)
    
    if not selected:
        return None, None
    
    # Verify
    if not _is_sidon(selected):
        return None, None
    
    sel_sums = _get_sums(selected)
    if not _is_maximal_fast(selected, sel_sums):
        return None, None
    
    return selected, sel_sums


def _biased_greedy(seed, deadline, bias='low'):
    """
    Greedy with bias toward small or large numbers.
    This tends to produce different maximal structures.
    """
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    
    if bias == 'low':
        # Prefer smaller numbers: sort by (random_weight, value)
        weighted = [(rng.random() * 0.5 + x / N, x) for x in order]
    else:
        # Prefer larger numbers
        weighted = [(rng.random() * 0.5 + (N - x) / N, x) for x in order]
    weighted.sort(key=lambda t: t[0])
    order = [x for _, x in weighted]
    
    selected, sums = _greedy_construct(order)
    
    if not selected:
        return None, None
    
    if not _is_sidon(selected):
        return None, None
    
    sel_sums = _get_sums(selected)
    if not _is_maximal_fast(selected, sel_sums):
        return None, None
    
    return selected, sel_sums


def _local_search_shrink(s, deadline):
    """
    Local search: try single removals and 1-for-1 swaps to minimize size.
    """
    best = sorted(s)
    best_size = len(best)
    
    for _ in range(50):
        if time.time() >= deadline:
            break
        
        improved = False
        
        # Try single removals
        for i in range(len(best)):
            candidate = best[:i] + best[i+1:]
            if len(candidate) == 0:
                continue
            if not _is_sidon(candidate):
                continue
            cand_sums = _get_sums(candidate)
            if _is_maximal_fast(candidate, cand_sums):
                best = candidate
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add one missing
        best_set = set(best)
        found_swap = False
        for i in range(len(best)):
            if found_swap:
                break
            for x in range(1, N + 1):
                if x in best_set:
                    continue
                candidate = best[:i] + best[i+1:] + [x]
                candidate.sort()
                if not _is_sidon(candidate):
                    continue
                cand_sums = _get_sums(candidate)
                if _is_maximal_fast(candidate, cand_sums):
                    best = candidate
                    best_size = len(best)
                    best_set = set(best)
                    improved = True
                    found_swap = True
                    break
        
        if not improved:
            break
    
    return best


def construct():
    deadline = time.time() + 55  # Safety margin within 60s
    
    best_set = None
    best_size = N + 1
    
    # Phase 1: Generate diverse maximal Sidon sets
    # Use multiple strategies with many seeds
    
    # Strategy A: Pure random order
    for seed in range(0, 100):
        if time.time() >= deadline:
            break
        s, sums = _randomized_greedy(seed, deadline)
        if s is None:
            continue
        size = len(s)
        if size < best_size:
            best_size = size
            best_set = list(s)
    
    # Strategy B: Biased toward low numbers
    for seed in range(100, 200):
        if time.time() >= deadline:
            break
        s, sums = _biased_greedy(seed, deadline, bias='low')
        if s is None:
            continue
        size = len(s)
        if size < best_size:
            best_size = size
            best_set = list(s)
    
    # Strategy C: Biased toward high numbers
    for seed in range(200, 300):
        if time.time() >= deadline:
            break
        s, sums = _biased_greedy(seed, deadline, bias='high')
        if s is None:
            continue
        size = len(s)
        if size < best_size:
            best_size = size
            best_set = list(s)
    
    # Phase 2: Try to shrink the best set found
    if best_set is not None and time.time() < deadline:
        shrunken = _try_shrink(best_set, _get_sums(best_set), deadline)
        if len(shrunken) < best_size:
            best_size = len(shrunken)
            best_set = shrunken
    
    # Phase 3: Local search on the best set
    if best_set is not None and time.time() < deadline:
        improved = _local_search_shrink(best_set, deadline)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
    
    # Phase 4: If we have time, do more diverse searches and shrink
    if time.time() < deadline:
        # Try more seeds with different biases
        for seed in range(300, 500):
            if time.time() >= deadline:
                break
            bias = 'low' if seed % 2 == 0 else 'high'
            s, sums = _biased_greedy(seed, deadline, bias=bias)
            if s is None:
                continue
            size = len(s)
            if size < best_size:
                best_size = size
                best_set = list(s)
                # Try to shrink immediately
                if time.time() < deadline:
                    shrunken = _try_shrink(best_set, _get_sums(best_set), deadline)
                    if len(shrunken) < best_size:
                        best_size = len(shrunken)
                        best_set = shrunken
    
    # Fallback: deterministic ascending greedy
    if best_set is None:
        selected, sums = _greedy_construct(list(range(1, N + 1)))
        if _is_sidon(selected) and _is_maximal_fast(selected, _get_sums(selected)):
            best_set = selected
    
    if best_set is None:
        # Absolute fallback: return ascending greedy result
        selected, sums = _greedy_construct(list(range(1, N + 1)))
        best_set = selected
    
    return best_set


# Quick self-test if run directly
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Size: {len(result)}, Time: {elapsed:.2f}s")
    print(f"Set: {result}")
    assert _is_sidon(result), "Not a Sidon set!"
    assert _is_maximal_fast(result, _get_sums(result)), "Not maximal!"
    print("Valid maximal Sidon set.")
