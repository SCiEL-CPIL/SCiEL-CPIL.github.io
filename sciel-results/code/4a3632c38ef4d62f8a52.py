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
# Sort by size descending for potential priority, but we'll use various orders
DIVS_ASC = sorted(ALL_DIVS)
DIVS_DESC = sorted(ALL_DIVS, reverse=True)

def _count_uncovered(covered, m, r):
    """Count how many uncovered residues are in class (m, r)."""
    # covered is a bytearray of length BASE, 1 if covered
    # We need count of i in [r, r+m, ...] < BASE where covered[i] == 0
    # Vectorized: covered[r::m] gives the slice
    return np.count_nonzero(covered[r::m] == 0)

def _greedy_construct(order, covered_init=None, time_limit=None):
    """
    Greedy set cover: at each step, pick the class (m, r) that covers the most
    uncovered residues. Moduli must be distinct.
    """
    start = time.time()
    if covered_init is not None:
        covered = np.array(covered_init, dtype=np.uint8)
    else:
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
            # Find the residue r that maximizes newly covered count
            # Try all r in 0..m-1
            # Optimization: only try r that are uncovered? No, r is the class offset.
            # We need to find r such that covered[r::m] has max zeros.
            
            # Vectorized approach: for each r, count zeros in covered[r::m]
            # This is m separate slices. For large m, this is expensive.
            # Alternative: use convolution or FFT? Or just loop for small m.
            
            # For speed, let's try a smarter approach.
            # Actually, for a fixed m, the classes partition the residues modulo m.
            # The number of uncovered residues covered by class (m, r) is:
            # sum over k where r + k*m < BASE of (1 - covered[r + k*m])
            
            # Let's just loop r for each m. m is at most 51975, but usually smaller.
            # Total work per step: sum over remaining m of O(m) = O(BASE * num_remaining) worst case?
            # No, sum of m for m in divisors is bounded.
            
            # Better: precompute for each m the counts for all r?
            # Let's just do it naively but efficiently with numpy.
            
            best_r_for_m = 0
            best_count_for_m = -1
            
            # Optimization: if m is large, the number of terms in the slice is small (BASE/m)
            # If m is small, the number of terms is large.
            
            for r in range(m):
                cnt = np.count_nonzero(covered[r::m] == 0)
                if cnt > best_count_for_m:
                    best_count_for_m = cnt
                    best_r_for_m = r
            
            if best_count_for_m > best_gain:
                best_gain = best_count_for_m
                best_m = m
                best_r = best_r_for_m
        
        if best_gain <= 0:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        remaining_moduli.remove(best_m)
        
        # Update covered
        covered[best_r::best_m] = 1
    
    return classes, covered

def _score(classes):
    """Compute number of uncovered residues."""
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.count_nonzero(covered == 0))

def _local_search(classes, time_limit=None):
    """
    Local search: try to swap out one class for another (unused modulus) to improve score.
    Also try to change the residue of an existing class.
    """
    start = time.time()
    if not classes:
        return classes
    
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    
    current_score = int(np.count_nonzero(covered == 0))
    used_moduli = set(m for m, r in classes)
    available_moduli = [m for m in ALL_DIVS if m not in used_moduli]
    
    improved = True
    while improved:
        if time_limit is not None and time.time() - start > time_limit:
            break
        improved = False
        
        # 1. Try changing residue of existing classes
        for i, (m, r) in enumerate(classes):
            if time_limit is not None and time.time() - start > time_limit:
                break
            
            # Remove current class
            # Rebuilding covered from scratch is expensive. Let's do incremental.
            # Actually, let's just try a few random residues or the best one.
            
            # Find best residue for this m
            # Temporarily "uncover" the current class? Hard to do incrementally with numpy easily.
            # Let's just rebuild covered without this class, then try all r.
            
            covered_no_i = np.zeros(BASE, dtype=np.uint8)
            for j, (mj, rj) in enumerate(classes):
                if j != i:
                    covered_no_i[rj::mj] = 1
            
            best_r_new = r
            best_gain = 0
            for r_try in range(m):
                # How many NEW residues would this cover?
                # It's the count of zeros in covered_no_i[r_try::m]
                cnt = np.count_nonzero(covered_no_i[r_try::m] == 0)
                if cnt > best_gain:
                    best_gain = cnt
                    best_r_new = r_try
            
            if best_r_new != r:
                classes[i][1] = best_r_new
                covered = np.zeros(BASE, dtype=np.uint8)
                for mj, rj in classes:
                    covered[rj::mj] = 1
                new_score = int(np.count_nonzero(covered == 0))
                if new_score < current_score:
                    current_score = new_score
                    improved = True
        
        # 2. Try swapping a class with an unused modulus
        if time_limit is not None and time.time() - start > time_limit:
            break
            
        for i, (m_old, r_old) in enumerate(classes):
            if time_limit is not None and time.time() - start > time_limit:
                break
            
            for m_new in available_moduli:
                if m_new == m_old:
                    continue
                
                # Rebuild covered without class i
                covered_no_i = np.zeros(BASE, dtype=np.uint8)
                for j, (mj, rj) in enumerate(classes):
                    if j != i:
                        covered_no_i[rj::mj] = 1
                
                # Find best r for m_new
                best_r_new = 0
                best_gain = -1
                for r_try in range(m_new):
                    cnt = np.count_nonzero(covered_no_i[r_try::m_new] == 0)
                    if cnt > best_gain:
                        best_gain = cnt
                        best_r_new = r_try
                
                # Calculate new score
                # new_score = score(covered_no_i with m_new, best_r_new added)
                # = current_score - (residues covered by old class not covered by others) + (residues covered by new class not covered by others)
                # This is complex. Let's just compute directly.
                
                covered_test = covered_no_i.copy()
                covered_test[best_r_new::m_new] = 1
                new_score = int(np.count_nonzero(covered_test == 0))
                
                if new_score < current_score:
                    classes[i] = [m_new, best_r_new]
                    used_moduli.remove(m_old)
                    used_moduli.add(m_new)
                    available_moduli.remove(m_new)
                    available_moduli.append(m_old)
                    current_score = new_score
                    covered = covered_test
                    improved = True
                    break  # Restart the inner loop for this i? Or just continue?
        
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRAC  # 127.5 seconds
    
    best_classes = None
    best_score = BASE  # worst case
    
    # Strategy 1: Greedy with different moduli orders
    orders = [
        DIVS_ASC,
        DIVS_DESC,
        sorted(ALL_DIVS, key=lambda m: BASE // m),  # Most frequent first
    ]
    
    for order in orders:
        if time.time() > deadline - 10:  # Leave 10s for local search
            break
        
        classes, covered = _greedy_construct(order)
        score = int(np.count_nonzero(covered == 0))
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        # Local search on this result
        if time.time() > deadline - 5:
            break
        
        classes_ls = _local_search([c[:] for c in classes], time_limit=5)
        score_ls = _score(classes_ls)
        
        if score_ls < best_score:
            best_score = score_ls
            best_classes = [c[:] for c in classes_ls]
    
    # If we have time, try a few random restarts with perturbation
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
            
            classes_ls = _local_search(classes_pert, time_limit=3)
            score_ls = _score(classes_ls)
            
            if score_ls < best_score:
                best_score = score_ls
                best_classes = [c[:] for c in classes_ls]
    
    if best_classes is None:
        # Fallback: simple greedy
        best_classes, _ = _greedy_construct(DIVS_ASC)
    
    return best_classes
