"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_BASE = 51975
_DEADLINE = 115.0  # safety margin before 150s kill

def _divisors(n):
    """Return all divisors > 1 of n, sorted."""
    divs = []
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.append(n)
    return sorted(divs)

def _coverage_mask(m, r):
    """Return a boolean numpy array of length _BASE where positions congruent to r mod m are True."""
    mask = np.zeros(_BASE, dtype=np.uint8)
    mask[r::_BASE // m + 1 if False else m] = 1
    # Actually simpler: mask[r::m] = 1
    mask = np.zeros(_BASE, dtype=np.uint8)
    mask[r::m] = 1
    return mask

def _uncovered_count(covered):
    """Count uncovered residues (0s) in the covered array."""
    return int(np.sum(covered == 0))

def _greedy_build(moduli_order):
    """Greedy: for each modulus in order, pick residue covering most uncovered residues."""
    covered = np.zeros(_BASE, dtype=np.uint8)
    classes = []
    for m in moduli_order:
        best_r = 0
        best_count = -1
        for r in range(m):
            count = int(np.sum(covered[r::m] == 0))
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes, covered

def _local_search(classes, covered, max_time):
    """Try to improve by changing residues of individual moduli."""
    start = time.time()
    classes = [list(c) for c in classes]
    covered = covered.copy()
    
    # Precompute masks for each class
    masks = []
    for m, r in classes:
        mask = np.zeros(_BASE, dtype=np.uint8)
        mask[r::m] = 1
        masks.append(mask)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start > max_time:
            break
        for i in range(len(classes)):
            if time.time() - start > max_time:
                break
            m, old_r = classes[i]
            old_mask = masks[i]
            # Remove old coverage
            covered = (covered & ~old_mask) | (covered & old_mask & 0)  # just zero out
            # Actually: covered = covered & ~old_mask doesn't work with uint8
            # Let's recompute: zero out old, then try each residue
            covered_zeroed = covered.copy()
            covered_zeroed[old_r::m] = 0
            # But this might zero out positions covered by other classes too!
            # We need to be careful. Let's use a different approach:
            # Count uncovered after removing class i and adding new residue
            # For each candidate residue r:
            #   new_covered = (covered & ~old_mask) | new_mask
            #   But positions covered by multiple classes would be wrong
            # Simpler: just recompute from scratch for each candidate? Too slow.
            # Alternative: track coverage counts per position
            pass
        
        # Let's use a simpler local search: for each class, try all residues and pick best
        # Recompute uncovered for each candidate
        for i in range(len(classes)):
            if time.time() - start > max_time:
                break
            m, old_r = classes[i]
            best_r = old_r
            best_uncovered = _uncovered_count(covered)
            
            # For each candidate residue, compute how many new positions get covered
            # that weren't covered before, minus how many positions lose coverage
            # that only this class covers
            # This is complex. Let's just try a few promising residues.
            # Try residues that cover the most currently-uncovered positions
            for r in range(m):
                if r == old_r:
                    continue
                # Simulate: remove old class, add new class
                # Positions that were only covered by class i and are not in new class -> become uncovered
                # Positions that were uncovered and are in new class -> become covered
                # Net change = (newly covered) - (newly uncovered)
                # newly covered = positions in new_mask that are uncovered in (covered without class i)
                # newly uncovered = positions in old_mask but not in new_mask that are covered only by class i
                # This requires knowing which positions are covered only by class i.
                # Too complex for a quick local search. Let's skip this.
                pass
            improved = False  # Skip complex local search for now
    
    return classes

def construct():
    start_time = time.time()
    moduli = _divisors(_BASE)
    
    # Greedy build with moduli in increasing order
    classes, covered = _greedy_build(moduli)
    best_classes = classes
    best_uncovered = _uncovered_count(covered)
    
    # Try different orderings: larger moduli first (they cover more positions each)
    if time.time() - start_time < _DEADLINE - 10:
        classes2, covered2 = _greedy_build(sorted(moduli, reverse=True))
        unc2 = _uncovered_count(covered2)
        if unc2 < best_uncovered:
            best_uncovered = unc2
            best_classes = classes2
    
    # Try a mixed ordering: interleave small and large
    if time.time() - start_time < _DEADLINE - 10:
        # Sort by number of residues per position (smaller m means more residues per position)
        # Actually, larger m means fewer positions per residue, so each class covers fewer positions
        # But larger m has more residue choices
        # Let's try: sort by m descending (largest first)
        classes3, covered3 = _greedy_build(sorted(moduli, key=lambda m: -m))
        unc3 = _uncovered_count(covered3)
        if unc3 < best_uncovered:
            best_uncovered = unc3
            best_classes = classes3
    
    # Simple local search: for each class, try to find a better residue
    if time.time() - start_time < _DEADLINE - 5:
        classes_ls = [list(c) for c in best_classes]
        covered_ls = np.zeros(_BASE, dtype=np.uint8)
        for m, r in classes_ls:
            covered_ls[r::m] = 1
        current_unc = _uncovered_count(covered_ls)
        
        for i in range(len(classes_ls)):
            if time.time() - start_time > _DEADLINE - 5:
                break
            m, old_r = classes_ls[i]
            # For each candidate residue, compute the uncovered count
            # We need to efficiently compute this
            # Temporarily remove class i, then add each candidate
            # Positions covered by class i: old_r::m
            # Positions covered by other classes: covered_ls & ~mask_i
            mask_i = np.zeros(_BASE, dtype=np.uint8)
            mask_i[old_r::m] = 1
            other_covered = covered_ls & (1 - mask_i)  # positions covered by others
            
            best_r = old_r
            best_unc = current_unc
            for r in range(m):
                if r == old_r:
                    continue
                mask_new = np.zeros(_BASE, dtype=np.uint8)
                mask_new[r::m] = 1
                new_covered = other_covered | mask_new
                unc = int(np.sum(new_covered == 0))
                if unc < best_unc:
                    best_unc = unc
                    best_r = r
            
            if best_r != old_r:
                classes_ls[i][1] = best_r
                # Update covered_ls
                covered_ls = np.zeros(_BASE, dtype=np.uint8)
                for m2, r2 in classes_ls:
                    covered_ls[r2::m2] = 1
                current_unc = best_unc
        
        if current_unc < best_uncovered:
            best_uncovered = current_unc
            best_classes = classes_ls
    
    return best_classes
# EVOLVE-BLOCK-END
