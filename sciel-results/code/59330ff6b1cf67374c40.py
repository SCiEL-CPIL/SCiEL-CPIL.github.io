import time
import numpy as np
import random

BASE = 51975

# Precompute divisors > 1 of BASE
def _get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    divs.sort()
    return divs

DIVISORS = _get_divisors(BASE)
DIVISORS = np.array(DIVISORS)
K = len(DIVISORS)

# Precompute for each divisor m the "column" indices for reshape
# For a given m, covered.reshape(-1, m) has columns 0..m-1
# Column r covers residues r, r+m, r+2m, ...

def _count_uncovered(covered):
    """Count residues with covered == 0."""
    return int(np.sum(covered == 0))

def _greedy_construct(order, deadline):
    """Greedy construction: for each modulus in order, pick best residue."""
    covered = np.zeros(BASE, dtype=np.int16)
    classes = []
    
    for m in order:
        if time.time() > deadline - 2:
            break
        m = int(m)
        # Count zeros in each residue class
        is_zero = (covered == 0)
        reshaped = is_zero.reshape(-1, m)
        zero_counts = np.sum(reshaped, axis=0)
        best_r = int(np.argmax(zero_counts))
        best_gain = zero_counts[best_r]
        
        if best_gain > 0:
            covered[best_r::m] += 1
            classes.append([m, best_r])
    
    return classes, covered

def _local_search(classes, covered, deadline):
    """Local search: re-optimize residues, add classes, remove redundant."""
    current_score = _count_uncovered(covered)
    improved = True
    
    while improved and time.time() < deadline - 2:
        improved = False
        
        # A. Re-optimize residues for existing classes
        for idx in range(len(classes)):
            if time.time() > deadline - 2:
                break
            m, r = classes[idx]
            m = int(m)
            
            # Temporarily remove this class
            covered[r::m] -= 1
            
            # Find best residue
            is_zero = (covered == 0)
            reshaped = is_zero.reshape(-1, m)
            gain_counts = np.sum(reshaped, axis=0)
            best_r_new = int(np.argmax(gain_counts))
            
            if best_r_new != r:
                # Try the swap
                covered[best_r_new::m] += 1
                new_score = _count_uncovered(covered)
                
                if new_score < current_score:
                    current_score = new_score
                    classes[idx] = [m, best_r_new]
                    improved = True
                    # Restore original removal (already have new r applied)
                    # Need to fix: we removed old r, added new r. 
                    # If we accept, covered now has new r. Good.
                    # If we reject, we need to remove new r and restore old r.
                else:
                    # Revert
                    covered[best_r_new::m] -= 1
                    covered[r::m] += 1
                # If we accepted, we don't restore old r. 
                # But wait, the loop above assumes covered reflects the current state.
                # Let's restructure: always restore to original state at end of iteration.
            else:
                # No change, just restore
                covered[r::m] += 1
                continue
            
            # If we accepted, covered has the new residue. 
            # The "restore" step was skipped. Let's fix the logic.
            # Actually, the code above is a bit messy. Let's rewrite the loop body.
            break
        
        # Restart the loop if improved
        if improved:
            continue
        
        # B. Add a new class
        used_moduli = set(c[0] for c in classes)
        unused = [m for m in DIVISORS if int(m) not in used_moduli]
        
        if unused:
            best_m = None
            best_r = None
            best_gain = 0
            
            for m in unused:
                if time.time() > deadline - 2:
                    break
                m = int(m)
                is_zero = (covered == 0)
                reshaped = is_zero.reshape(-1, m)
                zero_counts = np.sum(reshaped, axis=0)
                max_gain = int(np.max(zero_counts))
                
                if max_gain > best_gain:
                    best_gain = max_gain
                    best_m = m
                    best_r = int(np.argmax(zero_counts))
            
            if best_m is not None and best_gain > 0:
                covered[best_r::best_m] += 1
                classes.append([best_m, best_r])
                current_score = _count_uncovered(covered)
                improved = True
                continue
        
        # C. Remove redundant classes
        for idx in range(len(classes)):
            if time.time() > deadline - 2:
                break
            m, r = classes[idx]
            m = int(m)
            
            # Check if any residue in r::m has coverage exactly 1
            is_one = (covered == 1)
            unique_count = int(np.sum(is_one[r::m]))
            
            if unique_count == 0:
                # Redundant
                covered[r::m] -= 1
                classes.pop(idx)
                # Score doesn't change
                improved = True
                break
    
    return classes, covered, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    best_classes = []
    best_score = BASE
    
    # Try different greedy orders
    orders = []
    
    # 1. Ascending moduli
    orders.append(DIVISORS.tolist())
    
    # 2. Descending moduli
    orders.append(DIVISORS[::-1].tolist())
    
    # 3. Random orders (a few)
    for _ in range(5):
        rnd = DIVISORS.tolist()
        random.shuffle(rnd)
        orders.append(rnd)
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        
        # Greedy phase
        classes, covered = _greedy_construct(order, deadline)
        
        # Local search phase
        classes, covered, score = _local_search(classes, covered, deadline)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        if best_score == 0:
            break
    
    # If we still have time, try more random restarts
    while time.time() < deadline - 5 and best_score > 0:
        rnd = DIVISORS.tolist()
        random.shuffle(rnd)
        
        classes, covered = _greedy_construct(rnd, deadline)
        classes, covered, score = _local_search(classes, covered, deadline)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
    
    return best_classes
