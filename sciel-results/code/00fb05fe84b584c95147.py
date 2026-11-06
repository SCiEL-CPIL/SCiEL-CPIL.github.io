"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s kill
    base = 51975
    
    # Get all divisors > 1 of 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by coverage descending (small m first covers more residues)
    divisors_by_coverage = sorted(divisors, key=lambda m: base // m, reverse=True)
    
    # Precompute position arrays for each (m, r) pair for fast access
    # For efficiency, we'll compute on the fly but use numpy
    
    # Greedy construction with multiple passes
    best_classes = []
    best_uncovered = base
    
    # Strategy 1: Greedy from smallest moduli up
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    used_mods = set()
    
    for m in divisors_by_coverage:
        if time.time() > deadline - 10.0:
            break
        # Find best r for this m that maximizes new coverage
        best_r = 0
        best_gain = -1
        for r in range(m):
            pos = np.arange(r, base, m)
            gain = int((covered[pos] == 0).sum())
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            classes.append([m, best_r])
            pos = np.arange(best_r, base, m)
            covered[pos] = 1
            used_mods.add(m)
    
    uncovered_count = int((covered == 0).sum())
    if uncovered_count < best_uncovered:
        best_uncovered = uncovered_count
        best_classes = classes[:]
    
    # Strategy 2: Greedy from largest moduli down (might complement small ones)
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    used_mods = set()
    
    for m in reversed(divisors_by_coverage):
        if time.time() > deadline - 10.0:
            break
        best_r = 0
        best_gain = -1
        for r in range(m):
            pos = np.arange(r, base, m)
            gain = int((covered[pos] == 0).sum())
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            classes.append([m, best_r])
            pos = np.arange(best_r, base, m)
            covered[pos] = 1
            used_mods.add(m)
    
    uncovered_count = int((covered == 0).sum())
    if uncovered_count < best_uncovered:
        best_uncovered = uncovered_count
        best_classes = classes[:]
    
    # Strategy 3: Local search improvement on best solution
    if time.time() < deadline - 15.0:
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            pos = np.arange(r, base, m)
            covered[pos] = 1
        
        current_uncovered = int((covered == 0).sum())
        improved = True
        max_iterations = 50
        
        while improved and time.time() < deadline - 5.0 and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Try changing residue for each existing class
            for i, (m, r) in enumerate(best_classes):
                if time.time() > deadline - 5.0:
                    break
                
                # Remove this class temporarily
                pos_old = np.arange(r, base, m)
                
                # Count coverage before change
                before_uncovered = int((covered == 0).sum())
                
                # Try all other residues
                best_r_new = r
                best_uncovered_after = before_uncovered
                
                for r_new in range(m):
                    if r_new == r:
                        continue
                    
                    # Compute what happens if we use r_new instead of r
                    # Positions that were covered by old r but not new r_new
                    # Positions that are covered by new r_new but not old r
                    
                    # Efficient approach: recompute coverage
                    # But that's expensive. Instead, use incremental update.
                    
                    # For now, just try a subset of residues for large m
                    if m > 50:
                        # Only try a few random residues
                        candidates = [0, m//2, (m-1)//2, min(1, m-1), max(0, m-2)]
                        candidates = [c for c in candidates if 0 <= c < m and c != r]
                    else:
                        candidates = [c for c in range(m) if c != r]
                    
                    for r_new in candidates:
                        pos_new = np.arange(r_new, base, m)
                        
                        # Compute uncovered if we replace r with r_new
                        # Temporarily mark old positions as potentially uncovered
                        # This is complex, so let's just recompute for small m
                        
                        if m <= 30:
                            # Recompute coverage without this class, then with new r
                            temp_covered = covered.copy()
                            # We need to know which positions are covered ONLY by this class
                            # This requires tracking coverage counts, which we don't have
                            # For simplicity, skip for now
                            pass
                
                # Skip complex local search for now, focus on adding new classes
            
            # Try adding unused moduli
            available_mods = [m for m in divisors_by_coverage if m not in used_mods]
            for m in available_mods:
                if time.time() > deadline - 5.0:
                    break
                if base // m < 2:
                    continue
                
                best_r = 0
                best_gain = -1
                for r in range(m):
                    pos = np.arange(r, base, m)
                    gain = int((covered[pos] == 0).sum())
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_gain > 0:
                    best_classes.append([m, best_r])
                    pos = np.arange(best_r, base, m)
                    covered[pos] = 1
                    used_mods.add(m)
                    improved = True
                    current_uncovered = int((covered == 0).sum())
    
    # Final strategy: Randomized restarts
    if time.time() < deadline - 20.0:
        import random
        random.seed(42)
        
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            
            covered = np.zeros(base, dtype=np.uint8)
            classes = []
            used_mods = set()
            
            # Random order of moduli
            mod_order = divisors_by_coverage[:]
            random.shuffle(mod_order)
            
            for m in mod_order:
                if time.time() > deadline - 10.0:
                    break
                best_r = 0
                best_gain = -1
                for r in range(m):
                    pos = np.arange(r, base, m)
                    gain = int((covered[pos] == 0).sum())
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_gain > 0:
                    classes.append([m, best_r])
                    pos = np.arange(best_r, base, m)
                    covered[pos] = 1
                    used_mods.add(m)
            
            uncovered_count = int((covered == 0).sum())
            if uncovered_count < best_uncovered:
                best_uncovered = uncovered_count
                best_classes = classes[:]
    
    return best_classes
# EVOLVE-BLOCK-END
