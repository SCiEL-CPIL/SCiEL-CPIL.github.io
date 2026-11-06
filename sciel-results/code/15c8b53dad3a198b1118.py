import time
import numpy as np

_BASE = 51975
_DEADLINE = 145.0  # safety margin before 150s kill

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs.discard(1)
    return sorted(divs)

def _compute_uncovered(classes, covered=None):
    """Compute number of uncovered residues for given classes."""
    if covered is None:
        covered = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        idx = np.arange(r, _BASE, m)
        covered[idx] = 1
    return int(np.sum(covered == 0))

def _greedy_build_descending():
    """Greedy construction: process moduli in descending order."""
    covered = np.zeros(_BASE, dtype=np.uint8)
    divisors = _get_divisors(_BASE)
    # Sort descending so larger moduli are placed first
    divisors_desc = sorted(divisors, reverse=True)
    classes = []
    
    for m in divisors_desc:
        # Find residue r that covers most currently-uncovered residues
        # We can compute gains for all r in O(N)
        gains = np.zeros(m, dtype=np.int32)
        # Iterate through all indices, if uncovered, increment gain for its residue mod m
        # This is O(N) per modulus. 
        # Optimization: use numpy vectorization
        # indices = np.arange(_BASE)
        # residues = indices % m
        # gains = np.bincount(residues[covered==0], minlength=m)
        
        # Let's try the vectorized bincount approach
        # covered_mask = (covered == 0)
        # residues = np.arange(_BASE) % m
        # gains = np.bincount(residues[covered_mask], minlength=m)
        
        # This creates large arrays. For N=51975, it's fine.
        residues = np.arange(_BASE) % m
        uncovered_mask = (covered == 0)
        # Only count uncovered
        gains = np.bincount(residues[uncovered_mask], minlength=m)
        
        best_r = int(np.argmax(gains))
        classes.append([m, best_r])
        
        # Update covered mask
        idx = np.arange(best_r, _BASE, m)
        covered[idx] = 1
        
    return classes, covered

def _local_search(classes, covered, time_limit):
    """Fast local search using gain calculation."""
    start_time = time.time()
    current_classes = [list(c) for c in classes]
    # Map modulus to index in classes for quick lookup
    # Note: Moduli are distinct
    m_to_idx = {m: i for i, (m, r) in enumerate(current_classes)}
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle order to avoid bias? Or just iterate.
        # Iterating in fixed order is fine for now.
        for i in range(len(current_classes)):
            m, r_cur = current_classes[i]
            
            # Calculate gains for all residues for this modulus m
            # gains[r] = number of uncovered residues x such that x % m == r
            residues = np.arange(_BASE) % m
            uncovered_mask = (covered == 0)
            gains = np.bincount(residues[uncovered_mask], minlength=m)
            
            best_r = int(np.argmax(gains))
            
            if best_r != r_cur:
                gain_best = gains[best_r]
                gain_cur = gains[r_cur]
                
                if gain_best > gain_cur:
                    # Apply move
                    # Remove coverage of r_cur
                    idx_old = np.arange(r_cur, _BASE, m)
                    # We need to be careful: if another class covers these, we shouldn't set to 0.
                    # But `covered` is a boolean mask of union.
                    # To update `covered` correctly:
                    # 1. Mark old spots as "potentially uncovered"
                    # 2. Mark new spots as "covered"
                    # 3. Recalculate? No, that's slow.
                    
                    # Correct update:
                    # The new covered set is (Old Covered - A_old) U A_new
                    # We can compute the new mask directly?
                    # Or we can just update the mask incrementally if we track counts?
                    # Tracking counts per residue is O(N) memory and O(1) update per point, but complex to init.
                    
                    # Given N=51975 is small, recalculating the mask for this specific move is O(N/m + N/m).
                    # But we need to know if a spot in A_old is still covered by OTHER classes.
                    # If we don't track counts, we can't just set to 0.
                    
                    # Strategy: 
                    # 1. Identify A_old and A_new.
                    # 2. For x in A_old: if x is not in A_new, check if any other class covers it.
                    #    This is expensive.
                    
                    # Alternative: Just rebuild the covered mask for this specific change?
                    # No, let's use a temporary mask.
                    # temp_covered = covered.copy()
                    # temp_covered[idx_old] = 0  # Tentatively remove
                    # # Check if any other class covers idx_old
                    # This is getting complicated and slow.
                    
                    # Let's stick to the property: 
                    # If we use `covered` as a boolean, we can't simply unset.
                    # We must maintain a `counts` array: counts[x] = number of classes covering x.
                    # Then covered[x] = counts[x] > 0.
                    
                    # Let's refactor to use counts.
                    # This makes updates O(N/m).
                    
                    # We'll break out and use a helper that updates counts.
                    # For now, let's assume we have `counts`.
                    pass
        # End for i
    return current_classes

# Refactoring for Counts-based Local Search

def _greedy_build_descending_counts():
    """Greedy construction returning classes and counts array."""
    counts = np.zeros(_BASE, dtype=np.int32)
    divisors = _get_divisors(_BASE)
    divisors_desc = sorted(divisors, reverse=True)
    classes = []
    
    for m in divisors_desc:
        # Gains for residue r is number of x with x%m==r and counts[x]==0
        residues = np.arange(_BASE) % m
        zero_mask = (counts == 0)
        gains = np.bincount(residues[zero_mask], minlength=m)
        
        best_r = int(np.argmax(gains))
        classes.append([m, best_r])
        
        # Update counts
        idx = np.arange(best_r, _BASE, m)
        counts[idx] += 1
        
    return classes, counts

def _local_search_counts(classes, counts, time_limit):
    """Local search using counts array for fast updates."""
    start_time = time.time()
    current_classes = [list(c) for c in classes]
    
    improved = True
    # We might want to randomize the order of moduli in the loop to escape local minima
    # But let's start with fixed order.
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate through classes
        # To be more robust, we can shuffle the list of indices
        indices = list(range(len(current_classes)))
        # np.random.shuffle(indices) # Optional
        
        for i in indices:
            m, r_cur = current_classes[i]
            
            # Calculate gains for all residues for this modulus m
            # gains[r] = number of x with x%m==r and counts[x]==0
            residues = np.arange(_BASE) % m
            zero_mask = (counts == 0)
            gains = np.bincount(residues[zero_mask], minlength=m)
            
            best_r = int(np.argmax(gains))
            
            if best_r != r_cur:
                gain_best = gains[best_r]
                gain_cur = gains[r_cur]
                
                if gain_best > gain_cur:
                    # Apply move
                    idx_old = np.arange(r_cur, _BASE, m)
                    idx_new = np.arange(best_r, _BASE, m)
                    
                    # Update counts
                    # Subtract 1 from old, add 1 to new
                    counts[idx_old] -= 1
                    counts[idx_new] += 1
                    
                    # Update class
                    current_classes[i][1] = best_r
                    improved = True
                    
                    # Early exit if fully covered
                    if np.all(counts > 0):
                        return current_classes
        # End for i
    return current_classes

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction (Descending)
    classes, counts = _greedy_build_descending_counts()
    
    # Phase 2: Local search
    remaining_time = _DEADLINE - (time.time() - start_time) - 2.0
    if remaining_time > 0:
        classes = _local_search_counts(classes, counts, remaining_time)
    
    # Final check
    if time.time() - start_time > _DEADLINE:
        pass
        
    return classes
