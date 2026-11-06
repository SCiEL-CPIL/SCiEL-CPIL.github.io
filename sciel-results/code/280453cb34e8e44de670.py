import time
import numpy as np
import random

BASE = 51975
# Precompute all odd divisors > 1 of 51975
def _get_divisors():
    divs = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors()

def _greedy_init(order, covered=None):
    """Greedy: given an order of moduli, pick best residue for each."""
    if covered is None:
        covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in order:
        best_r = 0
        best_gain = -1
        for r in range(m):
            # Count uncovered in this residue class
            gain = np.count_nonzero(covered[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes, covered

def _score(classes):
    """Count uncovered residues."""
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.count_nonzero(covered == 0))

def _local_search(classes, covered, time_limit=10.0):
    """Local search: try changing residues to reduce uncovered count."""
    start = time.time()
    best_score = int(np.count_nonzero(covered == 0))
    best_classes = [c[:] for c in classes]
    
    # Precompute for each modulus the list of residue positions
    # For efficiency, we'll try random swaps and single changes
    
    n = len(classes)
    # Map from modulus to index in classes
    m_to_idx = {}
    for i, (m, r) in enumerate(classes):
        m_to_idx[m] = i
    
    for iteration in range(2000):
        if time.time() - start > time_limit:
            break
        
        # Try changing residue of a random class
        idx = random.randint(0, n - 1)
        m, r_old = classes[idx]
        
        # Try a few random residues
        for _ in range(3):
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
            
            # Compute change in coverage
            # Old residue class covers positions r_old::m
            # New residue class covers positions r_new::m
            # Change = newly covered - newly uncovered
            
            old_mask = covered[r_old::m]
            new_mask = covered[r_new::m]
            
            # Positions that were covered by old but not by new
            lost = np.count_nonzero(old_mask == 1)
            # Positions that will be covered by new but weren't by old
            gained = np.count_nonzero(new_mask == 0)
            
            delta = gained - lost
            if delta > 0:
                # Apply
                covered[r_old::m] = 0
                covered[r_new::m] = 1
                classes[idx][1] = r_new
                new_score = best_score - delta
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in classes]
                break
    
    return best_classes, best_score, covered

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    best_classes = None
    best_score = BASE  # Worst case: nothing covered
    
    # Multiple greedy initializations with different orderings
    strategies = []
    
    # Ascending
    strategies.append(sorted(_DIVS))
    # Descending  
    strategies.append(sorted(_DIVS, reverse=True))
    # A few random orderings
    for _ in range(5):
        random_order = _DIVS[:]
        random.shuffle(random_order)
        strategies.append(random_order)
    
    for order in strategies:
        if time.time() > deadline - 20:
            break
        
        classes, covered = _greedy_init(order)
        score = int(np.count_nonzero(covered == 0))
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        # Local search if we have time
        if time.time() < deadline - 15:
            ls_classes, ls_score, ls_covered = _local_search(
                classes, covered, 
                time_limit=min(8.0, deadline - time.time() - 5)
            )
            if ls_score < best_score:
                best_score = ls_score
                best_classes = [c[:] for c in ls_classes]
        
        if best_score == 0:
            break
    
    if best_classes is None:
        # Fallback: simple ascending greedy
        best_classes, _ = _greedy_init(sorted(_DIVS))
    
    return best_classes
