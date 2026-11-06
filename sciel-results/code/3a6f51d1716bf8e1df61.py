"""Two Sidon sets with disjoint difference sets – ILS with Bitsets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin

def _can_add_to_set(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """
    Check if elem can be added to the set defined by elems, sum_bits, diff_bits.
    other_diff_bits is the diff mask of the OTHER set.
    """
    # 1. Check sum collisions within the set
    # elem + elem
    if (sum_bits >> (elem + elem)) & 1:
        return False
    # elem + a for each a in elems
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
            
    # 2. Check difference collisions within the set
    # abs(elem - a) must not be in diff_bits
    for a in elems:
        d = abs(elem - a)
        if (diff_bits >> d) & 1:
            return False
            
    # 3. Check difference disjointness with the OTHER set
    # The new differences {abs(elem - a) for a in elems} must not intersect with other_diff_bits
    for a in elems:
        d = abs(elem - a)
        if (other_diff_bits >> d) & 1:
            return False
            
    return True

def _update_masks_add(elems, sum_bits, diff_bits, elem):
    """Update masks after adding elem to elems. Returns new masks."""
    # Update sum_bits
    sum_bits |= (1 << (elem + elem))
    for a in elems:
        sum_bits |= (1 << (elem + a))
        
    # Update diff_bits
    for a in elems:
        d = abs(elem - a)
        diff_bits |= (1 << d)
        
    return sum_bits, diff_bits

def _update_masks_remove(elems, sum_bits, diff_bits, elem):
    """
    Rebuild masks after removing elem from elems.
    Rebuilding is O(k^2) which is small (k <= 64), so it's fast and safe.
    """
    new_elems = [x for x in elems if x != elem]
    new_sum_bits = 0
    new_diff_bits = 0
    
    for a, b in combinations_with_replacement(new_elems, 2):
        new_sum_bits |= (1 << (a + b))
    for a, b in combinations(new_elems, 2):
        new_diff_bits |= (1 << abs(a - b))
        
    return new_elems, new_sum_bits, new_diff_bits

def _greedy_build(other_diff_bits=0, max_size=64):
    """Greedily build a Sidon set."""
    elems = []
    sum_bits = 0
    diff_bits = 0
    
    # Sort candidates? Random order might help escape symmetries, 
    # but for initialization, let's try a few random orders or just 1..N
    # Actually, standard greedy from 1..N is a good baseline.
    
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if len(elems) >= max_size:
            break
        if _can_add_to_set(x, elems, sum_bits, diff_bits, other_diff_bits):
            elems.append(x)
            sum_bits, diff_bits = _update_masks_add(elems[:-1], sum_bits, diff_bits, x)
            
    return elems, sum_bits, diff_bits

def construct():
    start_time = time.time()
    best_A = []
    best_B = []
    best_size = 0
    
    # Initialize with a few random restarts of greedy to find a good starting point
    for _ in range(5):
        if time.time() - start_time > DEADLINE:
            break
        
        # Build A
        A, sum_bits_A, diff_bits_A = _greedy_build(0, MAX_EACH)
        # Build B constrained by A's diffs
        B, sum_bits_B, diff_bits_B = _greedy_build(diff_bits_A, MAX_EACH)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)
            
    # Local Search / ILS
    # We will use the best found so far as the starting point for the main search
    A = list(best_A)
    B = list(best_B)
    if not A:
        A, sum_bits_A, diff_bits_A = _greedy_build(0, MAX_EACH)
    else:
        # Rebuild masks for A
        sum_bits_A, diff_bits_A = _rebuild_bits(A)
        
    if not B:
        B, sum_bits_B, diff_bits_B = _greedy_build(diff_bits_A, MAX_EACH)
    else:
        sum_bits_B, diff_bits_B = _rebuild_bits(B)
        
    current_size = len(A) + len(B)
    
    # Main Loop: Random Local Search with "Shakes"
    # We try to add elements. If stuck, we remove one and try to add another.
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # 1. Try to add to A
        if len(A) < MAX_EACH:
            # Try a few random candidates
            candidates = [x for x in range(1, N + 1) if x not in A]
            random.shuffle(candidates)
            for x in candidates[:10]: # Limit checks per iteration
                if _can_add_to_set(x, A, sum_bits_A, diff_bits_A, diff_bits_B):
                    A.append(x)
                    sum_bits_A, diff_bits_A = _update_masks_add(A[:-1], sum_bits_A, diff_bits_A, x)
                    current_size += 1
                    improved = True
                    break
                    
        # 2. Try to add to B
        if not improved and len(B) < MAX_EACH:
            candidates = [x for x in range(1, N + 1) if x not in B]
            random.shuffle(candidates)
            for x in candidates[:10]:
                if _can_add_to_set(x, B, sum_bits_B, diff_bits_B, diff_bits_A):
                    B.append(x)
                    sum_bits_B, diff_bits_B = _update_masks_add(B[:-1], sum_bits_B, diff_bits_B, x)
                    current_size += 1
                    improved = True
                    break
                    
        if improved:
            if current_size > best_size:
                best_size = current_size
                best_A = list(A)
                best_B = list(B)
            continue
            
        # 3. Stuck: Perform a "Shake" (Remove and Re-add)
        # Remove a random element from A or B to free up differences
        # and try to fill the gap with a better element.
        
        if A:
            # Remove random from A
            idx = random.randint(0, len(A) - 1)
            removed_a = A.pop(idx)
            A, sum_bits_A, diff_bits_A = _update_masks_remove(A, sum_bits_A, diff_bits_A, removed_a)
            current_size -= 1
            
            # Try to add a DIFFERENT element to A
            added = False
            candidates = [x for x in range(1, N + 1) if x not in A and x != removed_a]
            random.shuffle(candidates)
            for x in candidates[:20]:
                if _can_add_to_set(x, A, sum_bits_A, diff_bits_A, diff_bits_B):
                    A.append(x)
                    sum_bits_A, diff_bits_A = _update_masks_add(A[:-1], sum_bits_A, diff_bits_A, x)
                    current_size += 1
                    added = True
                    break
            
            # If we couldn't add a new one, maybe put the old one back?
            # Or just leave it out if it was blocking?
            # For simplicity, if we didn't add a new one, we might try to add the old one back
            # to restore state, or keep it removed if it was "bad".
            # Let's try to add the old one back to ensure we don't lose size unnecessarily
            # unless we found a replacement.
            if not added:
                # Try to put removed_a back
                if _can_add_to_set(removed_a, A, sum_bits_A, diff_bits_A, diff_bits_B):
                    A.append(removed_a)
                    sum_bits_A, diff_bits_A = _update_masks_add(A[:-1], sum_bits_A, diff_bits_A, removed_a)
                    current_size += 1
                else:
                    # It's stuck, try to add ANY element to A to maximize size
                    candidates = [x for x in range(1, N + 1) if x not in A]
                    random.shuffle(candidates)
                    for x in candidates[:10]:
                        if _can_add_to_set(x, A, sum_bits_A, diff_bits_A, diff_bits_B):
                            A.append(x)
                            sum_bits_A, diff_bits_A = _update_masks_add(A[:-1], sum_bits_A, diff_bits_A, x)
                            current_size += 1
                            break
                            
        elif B:
            # Remove random from B
            idx = random.randint(0, len(B) - 1)
            removed_b = B.pop(idx)
            B, sum_bits_B, diff_bits_B = _update_masks_remove(B, sum_bits_B, diff_bits_B, removed_b)
            current_size -= 1
            
            # Try to add a DIFFERENT element to B
            added = False
            candidates = [x for x in range(1, N + 1) if x not in B and x != removed_b]
            random.shuffle(candidates)
            for x in candidates[:20]:
                if _can_add_to_set(x, B, sum_bits_B, diff_bits_B, diff_bits_A):
                    B.append(x)
                    sum_bits_B, diff_bits_B = _update_masks_add(B[:-1], sum_bits_B, diff_bits_B, x)
                    current_size += 1
                    added = True
                    break
            
            if not added:
                if _can_add_to_set(removed_b, B, sum_bits_B, diff_bits_B, diff_bits_A):
                    B.append(removed_b)
                    sum_bits_B, diff_bits_B = _update_masks_add(B[:-1], sum_bits_B, diff_bits_B, removed_b)
                    current_size += 1
                else:
                    candidates = [x for x in range(1, N + 1) if x not in B]
                    random.shuffle(candidates)
                    for x in candidates[:10]:
                        if _can_add_to_set(x, B, sum_bits_B, diff_bits_B, diff_bits_A):
                            B.append(x)
                            sum_bits_B, diff_bits_B = _update_masks_add(B[:-1], sum_bits_B, diff_bits_B, x)
                            current_size += 1
                            break
                            
        # Update best if we improved
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2] if 2 not in best_A else [3]
        
    return [best_A, best_B]

def _rebuild_bits(elems):
    """Rebuild sum_bits and diff_bits for a list of elements."""
    sum_bits = 0
    for a, b in combinations_with_replacement(elems, 2):
        sum_bits |= (1 << (a + b))
    diff_bits = 0
    for a, b in combinations(elems, 2):
        diff_bits |= (1 << abs(a - b))
    return sum_bits, diff_bits
# EVOLVE-BLOCK-END
