import time
import random
import numpy as np
from math import isqrt

BASE = 51975
DEADLINE_FRAC = 0.85  # aim to finish well under the 150s kill

# Precompute divisors of 51975 = 3^3 * 5^2 * 7 * 11
def _get_divisors(n):
    divs = []
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

ALL_DIVS = [d for d in _get_divisors(BASE) if d > 1]
DIVS_ASC = sorted(ALL_DIVS)
DIVS_DESC = sorted(ALL_DIVS, reverse=True)

def _score_from_counts(counts):
    """Compute number of uncovered residues from a coverage count array."""
    return int(np.count_nonzero(counts == 0))

def _greedy_construct(order, time_limit=None):
    """
    Greedy set cover: at each step, pick the class (m, r) that covers the most
    uncovered residues. Moduli must be distinct.
    Returns classes, counts (int array)
    """
    start = time.time()
    covered = np.zeros(BASE, dtype=np.uint8)
    
    classes = []
    used_moduli = set()
    remaining_moduli = [m for m in order if m not in used_moduli]
    
    while remaining_moduli:
        if time_limit is not None and time.time() - start > time_limit:
            break
        
        best_gain = -1
        best_m = None
        best_r = None
        
        for m in remaining_moduli:
            # Find r that maximizes gain (uncovered elements)
            best_r_for_m = 0
            best_gain_for_m = -1
            
            for r in range(m):
                sl = covered[r::m]
                # Gain is number of 0s in sl
                gain = len(sl) - np.sum(sl)
                if gain > best_gain_for_m:
                    best_gain_for_m = gain
                    best_r_for_m = r
            
            if best_gain_for_m > best_gain:
                best_gain = best_gain_for_m
                best_m = m
                best_r = best_r_for_m
        
        if best_gain <= 0:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        remaining_moduli.remove(best_m)
        
        # Update covered
        covered[best_r::best_m] = 1
    
    # Rebuild counts for local search
    counts = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        counts[r::m] += 1
        
    return classes, counts

def _local_search(classes, counts, time_limit=None):
    """
    Local search using marginal gain calculation with counts array.
    Tries to change residue of existing classes to improve coverage.
    """
    start = time.time()
    if not classes:
        return classes, counts
    
    current_score = _score_from_counts(counts)
    
    improved = True
    while improved:
        if time_limit is not None and time.time() - start > time_limit:
            break
        improved = False
        
        # 1. Try changing residue of existing classes
        for i, (m, r) in enumerate(classes):
            if time_limit is not None and time.time() - start > time_limit:
                break
            
            # Calculate loss if we remove current class (m, r)
            # Loss = number of x in class (m, r) where counts[x] == 1
            sl_old = counts[r::m]
            loss = np.sum(sl_old == 1)
            
            # Try all other residues r_new
            best_r_new = r
            best_net_gain = 0 # Net decrease in uncovered count
            
            for r_new in range(m):
                if r_new == r:
                    continue
                
                sl_new = counts[r_new::m]
                # Gain is number of x in new class that are currently uncovered (counts==0)
                gain = np.sum(sl_new == 0)
                
                net_gain = gain - loss
                if net_gain > best_net_gain:
                    best_net_gain = net_gain
                    best_r_new = r_new
            
            if best_r_new != r and best_net_gain > 0:
                # Apply change
                counts[r::m] -= 1
                counts[best_r_new::m] += 1
                classes[i][1] = best_r_new
                
                current_score -= best_net_gain
                improved = True
        
    return classes, counts

def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRAC  # 127.5 seconds
    
    best_classes = None
    best_score = BASE  # worst case
    
    # Strategy: Multiple Greedy restarts with different orders, followed by Local Search.
    
    # Orders to try
    orders = [
        DIVS_ASC,
        DIVS_DESC,
        sorted(ALL_DIVS, key=lambda m: BASE // m),  # Most frequent first
    ]
    
    # Add random orders
    for _ in range(5):
        random_order = ALL_DIVS[:]
        random.shuffle(random_order)
        orders.append(random_order)
    
    for idx, order in enumerate(orders):
        if time.time() > deadline - 10:  # Leave 10s for final checks
            break
        
        # Greedy
        classes, counts = _greedy_construct(order)
        score = _score_from_counts(counts)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        # Local Search
        remaining = deadline - time.time()
        ls_time_limit = min(5, remaining / 2) # Don't spend more than 5s or half remaining
        
        if ls_time_limit > 0.1:
            classes_ls, counts_ls = _local_search([c[:] for c in classes], counts, time_limit=ls_time_limit)
            score_ls = _score_from_counts(counts_ls)
            
            if score_ls < best_score:
                best_score = score_ls
                best_classes = [c[:] for c in classes_ls]
    
    # If we have time, try a few more random restarts from the best solution found
    if time.time() < deadline - 5 and best_classes is not None:
        for _ in range(3):  # 3 random restarts
            if time.time() > deadline - 2:
                break
            
            # Perturb the best solution: randomly change a few residues
            classes_pert = [c[:] for c in best_classes]
            for _ in range(min(3, len(classes_pert))):
                idx = random.randint(0, len(classes_pert) - 1)
                m = classes_pert[idx][0]
                classes_pert[idx][1] = random.randint(0, m - 1)
            
            # Rebuild counts
            counts_pert = np.zeros(BASE, dtype=np.int16)
            for m, r in classes_pert:
                counts_pert[r::m] += 1
            
            classes_ls, counts_ls = _local_search(classes_pert, counts_pert, time_limit=2)
            score_ls = _score_from_counts(counts_ls)
            
            if score_ls < best_score:
                best_score = score_ls
                best_classes = [c[:] for c in classes_ls]
    
    if best_classes is None:
        # Fallback: simple greedy
        best_classes, _ = _greedy_construct(DIVS_ASC)
    
    return best_classes
