"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_BASE = 51975
# All divisors of 51975 = 3^3 * 5^2 * 7 * 11 that are > 1
# 3^3=27, 5^2=25, 7, 11
# Divisors: product of 3^a * 5^b * 7^c * 11^d where a in 0..3, b in 0..2, c in 0..1, d in 0..1
# Filter out 1
_DIVS = []
for a in range(4):
    for b in range(3):
        for c in range(2):
            for d in range(2):
                m = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if m > 1:
                    _DIVS.append(m)
_DIVS.sort()
_NUM_DIVS = len(_DIVS)


def _count_uncovered(covered, m, r):
    """Count uncovered residues in class (m, r)."""
    # covered is a bytearray of 0/1
    total = 0
    for i in range(r, _BASE, m):
        total += (1 - covered[i])
    return total


def _update_covered(covered, m, r, add=True):
    """Update covered bytearray for class (m, r)."""
    for i in range(r, _BASE, m):
        if add:
            covered[i] = 1
        else:
            covered[i] = 0


def _score(covered):
    """Number of uncovered residues."""
    return sum(1 - c for c in covered)


def _greedy_descending():
    """Greedy: moduli in descending order, each takes the class covering most uncovered residues."""
    covered = bytearray(_BASE)
    classes = []
    for m in sorted(_DIVS, reverse=True):
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = _count_uncovered(covered, m, r)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            classes.append([m, best_r])
            _update_covered(covered, m, best_r, add=True)
    return classes, _score(covered)


def _local_search(classes, covered, max_iters=5000, time_limit=10.0):
    """Local search: try swapping residues of existing classes, or adding/removing classes."""
    start_time = time.time()
    current_score = _score(covered)
    best_classes = [c[:] for c in classes]
    best_score = current_score
    best_covered = covered[:]
    
    # Build a set of used moduli for quick lookup
    used_moduli = set(m for m, r in classes)
    available_moduli = [m for m in _DIVS if m not in used_moduli]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try 1: change residue of an existing class
        if classes:
            idx = random.randint(0, len(classes) - 1)
            m, r = classes[idx]
            new_r = random.randint(0, m - 1)
            if new_r != r:
                # Evaluate change
                old_gain = _count_uncovered(covered, m, r)
                # Temporarily remove
                _update_covered(covered, m, r, add=False)
                new_gain = _count_uncovered(covered, m, new_r)
                # Restore
                _update_covered(covered, m, r, add=True)
                
                delta = new_gain - old_gain
                if delta > 0:
                    # Apply
                    _update_covered(covered, m, r, add=False)
                    classes[idx][1] = new_r
                    _update_covered(covered, m, new_r, add=True)
                    current_score -= delta
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in classes]
                        best_covered = covered[:]
        
        # Try 2: add a new class
        if available_moduli:
            m = random.choice(available_moduli)
            r = random.randint(0, m - 1)
            gain = _count_uncovered(covered, m, r)
            if gain > 0:
                classes.append([m, r])
                _update_covered(covered, m, r, add=True)
                available_moduli.remove(m)
                current_score -= gain
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
                    best_covered = covered[:]
        
        # Try 3: remove a class with small contribution
        if len(classes) > 1:
            idx = random.randint(0, len(classes) - 1)
            m, r = classes[idx]
            gain = _count_uncovered(covered, m, r)
            if gain < 500:  # Only consider removing if contribution is small
                # Check if removal helps (it will by definition, but check if we want to)
                # Actually removing always increases uncovered by 'gain', so only do if gain is negative? No.
                # We only remove if it allows adding something better, but that's complex.
                # Skip for now, or only remove if gain == 0
                pass
    
    return best_classes, best_score


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Phase 1: Greedy descending
    classes, score = _greedy_descending()
    
    if time.time() > deadline - 20.0:
        return classes
    
    # Phase 2: Local search
    covered = bytearray(_BASE)
    for m, r in classes:
        _update_covered(covered, m, r, add=True)
    
    classes, score = _local_search(classes, covered, max_iters=10000, time_limit=100.0)
    
    if time.time() > deadline - 5.0:
        return classes
    
    # Phase 3: Another greedy pass on remaining gaps? Or just return.
    # Let's try a final greedy addition pass
    covered = bytearray(_BASE)
    for m, r in classes:
        _update_covered(covered, m, r, add=True)
    
    used_moduli = set(m for m, r in classes)
    for m in sorted(_DIVS, reverse=True):
        if time.time() > deadline - 2.0:
            break
        if m in used_moduli:
            continue
        best_r = 0
        best_gain = 0
        for r in range(m):
            gain = _count_uncovered(covered, m, r)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            classes.append([m, best_r])
            _update_covered(covered, m, best_r, add=True)
            used_moduli.add(m)
    
    return classes
# EVOLVE-BLOCK-END
