"""Optimized odd covering search for moduli dividing 51975."""
import time
import random
from collections import defaultdict

BASE = 51975
DEADLINE = 115  # safety margin before 150s limit

# Precompute all divisors of 51975 = 3^3 * 5^2 * 7 * 11
def _get_divisors():
    divs = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            divs.append(d)
    return divs

DIVISORS = _get_divisors()
DIVISOR_SET = set(DIVISORS)

# For each divisor m, precompute the list of residues r in [0, m)
# and the indices in [0, BASE) covered by class (m, r)
# indices[m][r] = list of positions i in [0,BASE) such that i % m == r
# But storing full lists is expensive; instead we use the fact that
# positions are r, r+m, r+2m, ...
# We'll use a coverage count array approach.

def _compute_coverage(classes):
    """Compute uncovered count given a list of [m, r] classes."""
    # Use a difference-array / counting approach
    # For each class (m, r), it covers positions r, r+m, r+2m, ...
    # We need to count how many positions are covered by at least one class.
    # Use inclusion via a boolean array is O(BASE * num_classes) which is too slow.
    # Instead, use a frequency array.
    freq = bytearray(BASE)  # 0 or 1 for each position
    for m, r in classes:
        for i in range(r, BASE, m):
            freq[i] = 1
    # Count zeros
    uncovered = 0
    for i in range(BASE):
        if freq[i] == 0:
            uncovered += 1
    return uncovered

def _compute_coverage_fast(classes):
    """Faster coverage computation using numpy if available, else manual."""
    try:
        import numpy as np
        covered = np.zeros(BASE, dtype=np.uint8)
        for m, r in classes:
            idx = range(r, BASE, m)
            if len(idx) > 0:
                covered[idx] = 1
        return int(BASE - covered.sum())
    except ImportError:
        return _compute_coverage(classes)

def _greedy_build(order='ascending'):
    """Greedy construction: process moduli in given order, each picks best residue."""
    if order == 'ascending':
        moduli_list = sorted(DIVISORS)
    else:
        moduli_list = sorted(DIVISORS, reverse=True)
    
    covered = bytearray(BASE)
    classes = []
    for m in moduli_list:
        # Find residue r that covers most uncovered positions
        best_r = 0
        best_count = -1
        for r in range(m):
            # Count uncovered in positions r, r+m, r+2m, ...
            cnt = 0
            for i in range(r, BASE, m):
                if covered[i] == 0:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        for i in range(best_r, BASE, m):
            covered[i] = 1
    return classes

def _local_search(classes, max_time=100):
    """Local search: try swapping residues or replacing moduli."""
    start = time.time()
    classes = [list(c) for c in classes]
    best_score = _compute_coverage_fast(classes)
    best_classes = [list(c) for c in classes]
    
    # Build a set of used moduli for quick lookup
    used_moduli = set(m for m, r in classes)
    
    # Precompute for each modulus m, the list of residues
    # For local search, we try:
    # 1. Changing the residue of an existing class
    # 2. Swapping a class with a different modulus not currently used
    
    # To speed up, maintain a coverage count array
    def update_coverage(freq, m, r, sign):
        """Add or remove class (m, r) from frequency array."""
        for i in range(r, BASE, m):
            if sign > 0:
                freq[i] += 1
            else:
                freq[i] -= 1
    
    # Build initial frequency array
    freq = bytearray(BASE)
    for m, r in classes:
        update_coverage(freq, m, r, 1)
    
    # Count uncovered
    def count_uncovered(freq):
        cnt = 0
        for i in range(BASE):
            if freq[i] == 0:
                cnt += 1
        return cnt
    
    current_uncovered = count_uncovered(freq)
    
    # Local search iterations
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        
        # Try changing residue of each class
        for idx, (m, r) in enumerate(classes):
            # Try all residues for this modulus
            for new_r in range(m):
                if new_r == r:
                    continue
                # Remove old, add new
                update_coverage(freq, m, r, -1)
                update_coverage(freq, m, new_r, 1)
                new_uncovered = count_uncovered(freq)
                update_coverage(freq, m, new_r, -1)
                update_coverage(freq, m, r, 1)
                
                if new_uncovered < current_uncovered:
                    classes[idx] = [m, new_r]
                    update_coverage(freq, m, r, -1)
                    update_coverage(freq, m, new_r, 1)
                    current_uncovered = new_uncovered
                    if current_uncovered < best_score:
                        best_score = current_uncovered
                        best_classes = [list(c) for c in classes]
                    improved = True
                    break  # restart scan from beginning
        
        # Try replacing a modulus with an unused one
        if (time.time() - start) > max_time:
            break
        
        unused_moduli = [m for m in DIVISORS if m not in used_moduli]
        if not unused_moduli:
            continue
            
        for idx, (m, r) in enumerate(classes):
            for new_m in unused_moduli:
                # Remove old class
                update_coverage(freq, m, r, -1)
                # Find best residue for new_m
                best_new_r = 0
                best_new_uncovered = current_uncovered + 1  # optimistic
                # We need to find the residue that minimizes uncovered after adding new_m
                # This is expensive; approximate by trying a few residues
                # Try the residue that covers most currently uncovered
                for new_r in range(new_m):
                    # Count how many uncovered positions are covered by (new_m, new_r)
                    cnt = 0
                    for i in range(new_r, BASE, new_m):
                        if freq[i] == 0:
                            cnt += 1
                    # After adding, uncovered decreases by cnt (but some may be double-counted)
                    # Actually, positions with freq==0 that are in new class become covered
                    # Positions with freq>=1 stay covered
                    # So new_uncovered = current_uncovered - cnt
                    candidate_uncovered = current_uncovered - cnt
                    if candidate_uncovered < best_new_uncovered:
                        best_new_uncovered = candidate_uncovered
                        best_new_r = new_r
                
                if best_new_uncovered < current_uncovered:
                    # Apply the change
                    classes[idx] = [new_m, best_new_r]
                    update_coverage(freq, new_m, best_new_r, 1)
                    used_moduli.discard(m)
                    used_moduli.add(new_m)
                    current_uncovered = best_new_uncovered
                    if current_uncovered < best_score:
                        best_score = current_uncovered
                        best_classes = [list(c) for c in classes]
                    improved = True
                    break  # restart scan
            
            if improved:
                break
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    
    # Generate multiple initial solutions using different greedy orders
    candidates = []
    
    # Ascending greedy
    if (time.time() - start_time) < DEADLINE - 30:
        c1 = _greedy_build('ascending')
        s1 = _compute_coverage_fast(c1)
        candidates.append((s1, c1))
    
    # Descending greedy
    if (time.time() - start_time) < DEADLINE - 30:
        c2 = _greedy_build('descending')
        s2 = _compute_coverage_fast(c2)
        candidates.append((s2, c2))
    
    # Select best initial
    candidates.sort(key=lambda x: x[0])
    best_score, best_classes = candidates[0]
    
    # Local search from best initial
    if (time.time() - start_time) < DEADLINE - 10:
        ls_classes, ls_score = _local_search(best_classes, max_time=DEADLINE - 15)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    # If time remains, try local search from other candidates
    for score, classes in candidates[1:]:
        if (time.time() - start_time) > DEADLINE - 15:
            break
        if score >= best_score:
            continue
        ls_classes, ls_score = _local_search(classes, max_time=DEADLINE - 15)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    # Final check
    if (time.time() - start_time) > DEADLINE:
        # Return best found so far
        pass
    
    return [[m, r] for m, r in best_classes]
