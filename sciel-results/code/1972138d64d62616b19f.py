"""Odd covering systems with moduli dividing 51975 (fewest uncovered residues)."""
import time
import random
from collections import defaultdict

_BASE = 51975
_DEADLINE = 140.0  # safety margin before 150s kill

# Precompute divisors of 51975 = 3^3 * 5^2 * 7 * 11, all > 1
_DIVISORS = None

def _get_divisors():
    global _DIVISORS
    if _DIVISORS is None:
        divs = []
        for d in range(3, _BASE + 1, 2):
            if _BASE % d == 0:
                divs.append(d)
        _DIVISORS = divs
    return _DIVISORS

def _score(covered):
    """Number of uncovered residues."""
    return covered.count(0)

def _greedy_build(order=None):
    """Greedy: for each modulus in given order, pick residue covering most uncovered."""
    if order is None:
        order = sorted(_get_divisors())
    covered = bytearray(_BASE)
    classes = []
    for m in order:
        # Find residue r that covers most uncovered positions
        best_r = 0
        best_count = -1
        for r in range(m):
            # Count uncovered in positions r, r+m, r+2m, ...
            count = 0
            for pos in range(r, _BASE, m):
                if covered[pos] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        for pos in range(best_r, _BASE, m):
            covered[pos] = 1
    return classes, covered

def _try_move(classes, covered, m_idx, new_r):
    """Try changing residue for modulus m_idx to new_r. Return delta in uncovered count."""
    m = classes[m_idx][0]
    old_r = classes[m_idx][1]
    if new_r == old_r:
        return 0
    
    # Count how many positions would change
    # Old class covers positions old_r, old_r+m, ...
    # New class covers positions new_r, new_r+m, ...
    
    # Positions only in old: old_r, old_r+m, ... not in new
    # Positions only in new: new_r, new_r+m, ... not in old
    
    # A position is affected if it's in old XOR new
    # For old positions: if covered becomes 0 (no other class covers it), uncovered increases
    # For new positions: if covered was 0, uncovered decreases
    
    delta = 0
    # Check old positions that won't be covered by new
    for pos in range(old_r, _BASE, m):
        # Is pos also in new class? pos % m == old_r, new class has pos where pos%m==new_r
        if pos % m != new_r:
            # This position loses coverage from this class
            # Check if any other class still covers it
            still_covered = False
            for i, (m2, r2) in enumerate(classes):
                if i == m_idx:
                    continue
                if pos % m2 == r2:
                    still_covered = True
                    break
            if not still_covered:
                delta += 1
    
    # Check new positions that weren't covered by old
    for pos in range(new_r, _BASE, m):
        if pos % m != old_r:
            # This position gains coverage from this class
            if covered[pos] == 0:
                delta -= 1
    
    return delta

def _local_search(classes, covered, time_limit):
    """Local search: try changing residues to reduce uncovered count."""
    best_score = _score(covered)
    best_classes = [c[:] for c in classes]
    best_covered = bytearray(covered)
    
    start = time.time()
    n = len(classes)
    improved = True
    
    while improved and (time.time() - start) < time_limit:
        improved = False
        for m_idx in range(n):
            m = classes[m_idx][0]
            current_r = classes[m_idx][1]
            for new_r in range(m):
                if new_r == current_r:
                    continue
                delta = _try_move(classes, covered, m_idx, new_r)
                if delta < 0:
                    # Apply the move
                    old_r = classes[m_idx][1]
                    classes[m_idx][1] = new_r
                    # Update covered
                    for pos in range(old_r, _BASE, m):
                        if pos % m != new_r:
                            # Check if still covered by others
                            still = False
                            for i, (m2, r2) in enumerate(classes):
                                if i == m_idx:
                                    continue
                                if pos % m2 == r2:
                                    still = True
                                    break
                            if not still:
                                covered[pos] = 0
                    for pos in range(new_r, _BASE, m):
                        if pos % m != old_r:
                            covered[pos] = 1
                    best_score += delta
                    best_classes = [c[:] for c in classes]
                    best_covered = bytearray(covered)
                    improved = True
                    if best_score == 0:
                        return best_classes, best_covered
        if (time.time() - start) > time_limit:
            break
    
    return best_classes, best_covered

def _random_restart(classes, covered, n_restarts, time_per_restart):
    """Try random perturbations and local search."""
    start = time.time()
    best_classes = [c[:] for c in classes]
    best_covered = bytearray(covered)
    best_score = _score(covered)
    
    for _ in range(n_restarts):
        if (time.time() - start) > time_per_restart:
            break
        # Perturb: randomly change a few residues
        perturbed = [c[:] for c in classes]
        perturbed_covered = bytearray(covered)
        n_changes = random.randint(1, 3)
        indices = random.sample(range(len(classes)), min(n_changes, len(classes)))
        for idx in indices:
            m = perturbed[idx][0]
            perturbed[idx][1] = random.randint(0, m - 1)
            # Rebuild covered for perturbed
        # Rebuild covered from scratch for perturbed
        perturbed_covered = bytearray(_BASE)
        for m, r in perturbed:
            for pos in range(r, _BASE, m):
                perturbed_covered[pos] = 1
        
        # Local search on perturbed
        ps_classes, ps_covered = _local_search(perturbed, perturbed_covered, time_per_restart / 2)
        ps_score = _score(ps_covered)
        if ps_score < best_score:
            best_score = ps_score
            best_classes = [c[:] for c in ps_classes]
            best_covered = bytearray(ps_covered)
            if best_score == 0:
                break
    
    return best_classes, best_covered

def construct():
    start_time = time.time()
    divisors = _get_divisors()
    
    # Try multiple greedy orders
    candidates = []
    
    # Ascending greedy
    classes_asc, covered_asc = _greedy_build(sorted(divisors))
    candidates.append((classes_asc, covered_asc))
    
    # Descending greedy
    classes_desc, covered_desc = _greedy_build(sorted(divisors, reverse=True))
    candidates.append((classes_desc, covered_desc))
    
    # Pick best initial
    best_classes = None
    best_covered = None
    best_score = float('inf')
    for c, cov in candidates:
        s = _score(cov)
        if s < best_score:
            best_score = s
            best_classes = [x[:] for x in c]
            best_covered = bytearray(cov)
    
    # Local search on best
    remaining = _DEADLINE - (time.time() - start_time)
    if remaining > 5:
        best_classes, best_covered = _local_search(best_classes, best_covered, remaining - 5)
        best_score = _score(best_covered)
    
    # Random restarts if time permits
    remaining = _DEADLINE - (time.time() - start_time)
    if remaining > 10:
        n_restarts = int(remaining / 5)
        best_classes, best_covered = _random_restart(best_classes, best_covered, n_restarts, remaining - 5)
        best_score = _score(best_covered)
    
    return best_classes
