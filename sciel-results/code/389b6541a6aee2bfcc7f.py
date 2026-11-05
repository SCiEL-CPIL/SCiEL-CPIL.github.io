"""Improved solver for odd covering systems with moduli dividing 51975."""
import time
import random
import sys
import math

BASE = 51975
# Precompute all odd divisors > 1 of BASE
# 51975 = 3^3 * 5^2 * 7 * 11
# Odd divisors > 1
_DIVS_CACHE = None

def _get_divisors():
    global _DIVS_CACHE
    if _DIVS_CACHE is None:
        divs = []
        for d in range(3, BASE + 1, 2):
            if BASE % d == 0:
                divs.append(d)
        _DIVS_CACHE = divs
    return _DIVS_CACHE

def _compute_uncovered_count(covered):
    """Count zeros in covered bytearray."""
    return covered.count(0)

def _greedy_construct(divs, order, deadline):
    """
    Greedy construction: for each modulus in given order, pick residue
    that covers most currently-uncovered residues.
    Returns (classes, covered_bytearray, uncovered_count).
    """
    covered = bytearray(BASE)
    classes = []
    for m in order:
        if time.time() > deadline:
            break
        # Find best residue r in [0, m) maximizing newly covered count
        best_r = 0
        best_gain = -1
        # Use slicing for speed
        for r in range(m):
            seg = covered[r::m]
            gain = seg.count(0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
                if gain == 0:
                    break
        classes.append([m, best_r])
        # Mark covered
        for i in range(best_r, BASE, m):
            covered[i] = 1
    return classes, covered, covered.count(0)

def _local_search(classes, covered, deadline, max_iter=200):
    """
    Local search: try changing residue for each modulus to reduce uncovered count.
    Also try swapping two moduli's residues.
    Returns (classes, covered, uncovered_count).
    """
    uncovered = covered.count(0)
    classes = [list(c) for c in classes]
    
    def _try_change(idx, new_r):
        """Try changing classes[idx] residue to new_r. Return new uncovered if better."""
        m = classes[idx][0]
        old_r = classes[idx][1]
        if new_r == old_r:
            return uncovered
        # Unmark old, mark new
        # Count how many positions are covered by other classes
        # Fast: just recompute the delta
        # Positions covered by old_r: old_r, old_r+m, old_r+2m, ...
        # Positions covered by new_r: new_r, new_r+m, new_r+2m, ...
        # A position is newly covered if it's in new_r's set but not in old_r's set
        # and was uncovered (i.e., only this class covered it)
        # A position becomes uncovered if it's in old_r's set but not in new_r's set
        # and was covered only by this class
        
        # Efficient approach: for positions in symmetric difference, check coverage
        delta = 0
        # Positions only in old_r
        for pos in range(old_r, BASE, m):
            if (pos - new_r) % m != 0:  # not in new_r's set
                # Check if covered by any other class
                covered_by_other = False
                for j, (mj, rj) in enumerate(classes):
                    if j == idx:
                        continue
                    if (pos - rj) % mj == 0:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    delta += 1  # becomes uncovered
        # Positions only in new_r
        for pos in range(new_r, BASE, m):
            if (pos - old_r) % m != 0:  # not in old_r's set
                if covered[pos] == 0:  # currently uncovered
                    delta -= 1  # becomes covered
        
        return uncovered + delta
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter and time.time() < deadline:
        improved = False
        iter_count += 1
        for idx in range(len(classes)):
            if time.time() > deadline:
                break
            m = classes[idx][0]
            current_r = classes[idx][1]
            best_r = current_r
            best_uncovered = uncovered
            # Try a subset of residues for speed
            # If m is large, sample; if small, try all
            if m <= 30:
                candidates = list(range(m))
            else:
                # Try current, 0, and some random
                candidates = [current_r, 0]
                for _ in range(min(20, m)):
                    candidates.append(random.randrange(m))
            
            for r in candidates:
                if r == current_r:
                    continue
                new_unc = _try_change(idx, r)
                if new_unc < best_uncovered:
                    best_uncovered = new_unc
                    best_r = r
            
            if best_r != current_r:
                # Apply change
                old_r = current_r
                classes[idx][1] = best_r
                # Update covered array
                for pos in range(old_r, BASE, m):
                    # Check if still covered by this or another class
                    still_covered = False
                    for j, (mj, rj) in enumerate(classes):
                        if (pos - rj) % mj == 0:
                            still_covered = True
                            break
                    if not still_covered:
                        covered[pos] = 0
                for pos in range(best_r, BASE, m):
                    covered[pos] = 1
                uncovered = best_uncovered
                improved = True
    
    return classes, covered, uncovered

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    divs = _get_divisors()
    
    # Sort divisors: smaller first tends to cover more densely
    divs_sorted = sorted(divs)
    
    # Initial greedy with sorted order
    classes, covered, unc = _greedy_construct(divs, divs_sorted, deadline)
    best_classes = [list(c) for c in classes]
    best_uncovered = unc
    
    if time.time() > deadline:
        return best_classes
    
    # Local search
    classes, covered, unc = _local_search(classes, covered, deadline, max_iter=50)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = [list(c) for c in classes]
    
    if time.time() > deadline:
        return best_classes
    
    # Multiple restarts with different orderings
    best_classes = [list(c) for c in best_classes]
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if restart == 0:
            order = sorted(divs, reverse=True)  # largest first
        elif restart == 1:
            order = list(divs)
            random.shuffle(order)
        elif restart == 2:
            # Primarily small, but shuffle among similar sizes
            order = sorted(divs, key=lambda d: (d // 10, random.random()))
        elif restart == 3:
            # Reverse of sorted
            order = sorted(divs, reverse=True)
        else:
            order = list(divs)
            random.shuffle(order)
        
        classes, covered, unc = _greedy_construct(divs, order, deadline)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [list(c) for c in classes]
        
        if time.time() > deadline:
            break
        
        # Quick local search on this result
        classes, covered, unc = _local_search(classes, covered, deadline, max_iter=30)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [list(c) for c in classes]
    
    # Final polish: try to improve best solution with more local search
    if time.time() < deadline - 5.0:
        # Reconstruct covered from best_classes
        covered = bytearray(BASE)
        for m, r in best_classes:
            for pos in range(r, BASE, m):
                covered[pos] = 1
        best_classes, covered, unc = _local_search(best_classes, covered, deadline, max_iter=100)
        if unc < best_uncovered:
            best_uncovered = unc
    
    return best_classes
