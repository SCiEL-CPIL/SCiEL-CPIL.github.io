"""Ambient integer sets with small largest Sidon subset.
We want a 12-element set in [0,128] whose largest 'strong Sidon' subset
(all unordered pair sums including diagonals unique) is as small as possible."""
import random
import time
import math
from itertools import combinations
from collections import Counter
import numpy as np


def max_sidon_size(S):
    """Exact largest k such that some k-subset of S has all unordered pair sums
    (including diagonals) unique. Returns that k."""
    S = sorted(S)
    n = len(S)
    if n <= 1:
        return n

    # Precompute all pair sums (including diagonals)
    allsums = set()
    for i in range(n):
        for j in range(i, n):
            allsums.add(S[i] + S[j])
    allsums = sorted(allsums)
    sum_to_id = {s: k for k, s in enumerate(allsums)}
    m = len(allsums)

    # Precompute: for each pair (i,j) with i<=j, the sum id
    pair_sum_id = {}
    for i in range(n):
        for j in range(i, n):
            pair_sum_id[(i, j)] = sum_to_id[S[i] + S[j]]

    # For n=12, iterate all subsets and check validity
    best = 0
    for mask in range(1, 1 << n):
        k = bin(mask).count('1')
        if k <= best:
            continue
        # Check if this subset is a valid Sidon set
        elems = [i for i in range(n) if mask & (1 << i)]
        # Check all pair sums are unique
        seen_sums = set()
        valid = True
        for idx_a in range(len(elems)):
            i = elems[idx_a]
            for idx_b in range(idx_a, len(elems)):
                j = elems[idx_b]
                s = S[i] + S[j]
                if s in seen_sums:
                    valid = False
                    break
                seen_sums.add(s)
            if not valid:
                break
        if valid:
            best = k

    return best


def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)


def _count_duplicate_sums(S):
    """Count the number of duplicate unordered pair sums (including diagonals).
    A 'duplicate' means a sum value that appears more than once among all pairs.
    Higher is better for our purpose (more collisions = smaller max Sidon subset)."""
    S = sorted(S)
    n = len(S)
    sum_counts = Counter()
    for i in range(n):
        for j in range(i, n):
            sum_counts[S[i] + S[j]] += 1
    # Count excess occurrences (sums that appear k times contribute k-1 to duplicates)
    duplicates = sum(v - 1 for v in sum_counts.values() if v > 1)
    return duplicates


def _collision_score(S):
    """A smoother surrogate: sum of (count-1)^2 for each sum value.
    This penalizes high-multiplicity collisions more heavily, which should
    correlate better with reducing the max Sidon subset size."""
    S = sorted(S)
    n = len(S)
    sum_counts = Counter()
    for i in range(n):
        for j in range(i, n):
            sum_counts[S[i] + S[j]] += 1
    score = sum((v - 1) ** 2 for v in sum_counts.values() if v > 1)
    return score


def construct():
    start = time.time()
    deadline = start + 87.0  # safety margin under 90s

    best_set = None
    best_val = 99
    seen = set()

    def consider(c):
        nonlocal best_set, best_val
        if time.time() > deadline:
            return None
        key = tuple(sorted(c))
        if key in seen:
            return None
        seen.add(key)
        if not _valid(c):
            return None
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    rng = random.Random(42)

    # Phase 1: Try consecutive sets (known to have small Sidon subsets)
    for lo in range(0, 117):
        if time.time() > deadline:
            break
        c = list(range(lo, lo + 12))
        consider(c)

    # Phase 2: Structured sets with many sum collisions
    # Two interleaved APs
    for d in range(1, 8):
        if time.time() > deadline:
            break
        for a in range(0, 100):
            for gap in range(1, 15):
                b = a + gap
                if b + 5 * d > 128:
                    continue
                c1 = [a + i * d for i in range(6)]
                c2 = [b + i * d for i in range(6)]
                c = sorted(set(c1 + c2))
                if len(c) == 12 and _valid(c):
                    consider(c)

    # Phase 3: Random sets in small windows
    for trial in range(3000):
        if time.time() > deadline:
            break
        lo = rng.randint(0, 90)
        width = rng.randint(12, 25)
        hi = min(lo + width, 129)
        if hi - lo < 12:
            continue
        c = sorted(rng.sample(range(lo, hi), 12))
        consider(c)

    # Phase 4: Collision-maximization local search using the smoother surrogate
    if best_set is not None:
        cur = list(best_set)
        cur_val = best_val
        cur_score = _collision_score(cur)
        
        improved = True
        max_iter = 800
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False
            
            # Try all possible single-element swaps
            best_swap = None
            best_swap_score = cur_score
            best_swap_val = cur_val
            
            for pos in range(12):
                old_val = cur[pos]
                # Try replacing cur[pos] with a new value
                # Focus on values close to existing elements to create collisions
                candidates = set()
                # Add values near existing elements
                for other in cur:
                    for delta in range(-4, 5):
                        nv = other + delta
                        if 0 <= nv <= 128 and nv != old_val:
                            candidates.add(nv)
                # Also try some random values
                for _ in range(8):
                    nv = rng.randint(0, 128)
                    if nv != old_val:
                        candidates.add(nv)
                
                for nv in candidates:
                    if nv in cur:
                        continue
                    new_cur = cur.copy()
                    new_cur[pos] = nv
                    new_cur.sort()
                    
                    # Quick check: use the smoother collision score
                    new_score = _collision_score(new_cur)
                    
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap = (pos, nv)
                        # Compute the actual Sidon size for tracking
                        new_val = max_sidon_size(new_cur)
                        best_swap_val = new_val
                        if new_val < cur_val:
                            improved = True
                            break
                    elif new_score == best_swap_score:
                        # Tie-breaker: prefer lower Sidon size
                        new_val = max_sidon_size(new_cur)
                        if new_val < best_swap_val:
                            best_swap = (pos, nv)
                            best_swap_val = new_val
                            if new_val < cur_val:
                                improved = True
                                break
                if improved and best_swap_val < cur_val:
                    break
            
            if best_swap is not None:
                pos, nv = best_swap
                cur[pos] = nv
                cur.sort()
                cur_score = best_swap_score
                if best_swap_val < cur_val:
                    cur_val = best_swap_val
                    improved = True
                    consider(cur)
        
        # Final consideration
        if cur_val < best_val:
            consider(cur)

    # Phase 5: If we have time, try more structured patterns
    # Sets with repeated differences
    if time.time() < deadline - 10:
        for base in range(0, 50):
            if time.time() > deadline - 5:
                break
            for d1 in range(1, 10):
                for d2 in range(1, 10):
                    c = []
                    # 6 elements from AP with step d1, 6 from AP with step d2
                    c1 = [base + i * d1 for i in range(6)]
                    c2 = [base + d2 + i * d2 for i in range(6)]
                    c = sorted(set(c1 + c2))
                    if len(c) == 12 and _valid(c):
                        consider(c)

    # Phase 6: Additional collision-maximization search from multiple starting points
    if time.time() < deadline - 15:
        # Start from several different initial sets and do collision-maximization
        start_sets = []
        # Consecutive sets
        for lo in [0, 10, 20, 30, 50, 70]:
            start_sets.append(list(range(lo, lo + 12)))
        # Random sets
        for _ in range(10):
            lo = rng.randint(0, 90)
            width = rng.randint(12, 25)
            hi = min(lo + width, 129)
            if hi - lo >= 12:
                start_sets.append(sorted(rng.sample(range(lo, hi), 12)))
        
        for ss in start_sets:
            if time.time() > deadline - 5:
                break
            if not _valid(ss):
                continue
            cur = list(ss)
            cur_score = _collision_score(cur)
            cur_val = max_sidon_size(cur)
            
            # Quick collision-maximization using the smoother surrogate
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                best_swap = None
                best_swap_score = cur_score
                
                for pos in range(12):
                    old_val = cur[pos]
                    candidates = set()
                    for other in cur:
                        for delta in range(-3, 4):
                            nv = other + delta
                            if 0 <= nv <= 128 and nv != old_val:
                                candidates.add(nv)
                    
                    for nv in candidates:
                        if nv in cur:
                            continue
                        new_cur = cur.copy()
                        new_cur[pos] = nv
                        new_cur.sort()
                        new_score = _collision_score(new_cur)
                        if new_score > best_swap_score:
                            best_swap_score = new_score
                            best_swap = (pos, nv)
                
                if best_swap is None:
                    break
                
                pos, nv = best_swap
                cur[pos] = nv
                cur.sort()
                cur_score = best_swap_score
                cur_val = max_sidon_size(cur)
            
            consider(cur)

    # Phase 7: Final polish - try to improve the best set with a tighter search
    if best_set is not None and time.time() < deadline - 5:
        cur = list(best_set)
        cur_val = best_val
        cur_score = _collision_score(cur)
        
        # Try more aggressive local moves: swap two elements at once
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for pos1 in range(12):
                for pos2 in range(pos1 + 1, 12):
                    if time.time() > deadline - 5:
                        break
                    old1, old2 = cur[pos1], cur[pos2]
                    # Try swapping with nearby values
                    for d1 in range(-3, 4):
                        for d2 in range(-3, 4):
                            nv1 = old1 + d1
                            nv2 = old2 + d2
                            if not (0 <= nv1 <= 128 and 0 <= nv2 <= 128):
                                continue
                            if nv1 == nv2:
                                continue
                            if nv1 in cur or nv2 in cur:
                                continue
                            new_cur = cur.copy()
                            new_cur[pos1] = nv1
                            new_cur[pos2] = nv2
                            new_cur.sort()
                            new_score = _collision_score(new_cur)
                            if new_score > cur_score:
                                new_val = max_sidon_size(new_cur)
                                if new_val < cur_val:
                                    cur = new_cur
                                    cur_val = new_val
                                    cur_score = new_score
                                    improved = True
                                    consider(cur)
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    # If nothing found, return a default
    if best_set is None:
        best_set = list(range(12))

    return best_set
