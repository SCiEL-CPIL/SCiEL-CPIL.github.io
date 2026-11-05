"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    # Precompute positions for each (modulus, residue) pair as numpy arrays for fast slicing
    # positions_cache[m][r] is an array of indices
    positions_cache = {}
    for m in divisors:
        positions_cache[m] = [np.arange(r, base, m) for r in range(m)]
        
    # Map divisor value to index in divisors list for quick lookup
    div_index = {d: i for i, d in enumerate(divisors)}

    best_uncovered = base
    best_residues = [0] * n_div
    best_cover_count = None

    def evaluate_residues(residues):
        """Calculate uncovered count for a given set of residues."""
        cover_count = np.zeros(base, dtype=np.uint8)
        for i, m in enumerate(divisors):
            r = residues[i]
            cover_count[positions_cache[m][r]] += 1
        return int(np.sum(cover_count == 0)), cover_count

    # Greedy initialization with adaptive look-ahead
    # We will try a few different strategies
    for attempt in range(3):
        if time.time() > deadline - 30:
            break
        
        if attempt == 0:
            order = list(range(n_div)) # Increasing
        elif attempt == 1:
            order = list(range(n_div-1, -1, -1)) # Decreasing
        else:
            order = list(range(n_div))
            random.shuffle(order)
            
        residues = [0] * n_div
        cover_count = np.zeros(base, dtype=np.uint8)
        covered_set = np.zeros(base, dtype=bool) # For fast gain calculation? 
        # Actually, just using cover_count is fine.
        
        # To speed up greedy, we can precompute the "gain" for each modulus/residue
        # But gain depends on current cover_count.
        
        for idx in order:
            m = divisors[idx]
            best_r = 0
            best_gain = -1
            
            # Heuristic: prioritize residues that cover the most currently uncovered spots
            # We can sample a few residues if m is large, but m <= 51975/3 = 17325.
            # Iterating all r is O(m * base/m) = O(base) which is ~52k ops.
            # With ~50 divisors, that's ~2.6M ops per greedy step. 50 steps -> 130M ops.
            # This might be too slow if done naively in Python loops, but we use numpy.
            # positions_cache[m][r] is an array. cover_count[pos] == 0 is a boolean array.
            # np.sum is fast.
            
            # Optimization: Only check a subset of residues if time is tight?
            # For now, full search per step.
            
            for r in range(m):
                pos = positions_cache[m][r]
                gain = np.sum(cover_count[pos] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[idx] = best_r
            cover_count[positions_cache[m][best_r]] += 1
            
        current_uncovered, current_cover = evaluate_residues(residues)
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues[:]
            best_cover_count = current_cover

    # Local Search with Incremental Update
    residues = best_residues[:]
    cover_count = best_cover_count.copy() if best_cover_count is not None else np.zeros(base, dtype=np.uint8)
    current_uncovered = best_uncovered
    
    # Simulated Annealing / Random Local Search
    # Use a faster neighborhood: swap residues? Or just single residue change.
    # Single residue change is O(base/m) per eval.
    
    temp = 1.0
    cooling = 0.995
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5:
        if temp < 0.01:
            break
        if no_improve_count > max_no_improve:
            break
            
        i = random.randint(0, n_div - 1)
        mi = divisors[i]
        old_ri = residues[i]
        
        # Choose new residue
        # Bias towards random to escape local minima
        new_ri = random.randint(0, mi - 1)
        if new_ri == old_ri:
            continue
            
        old_pos = positions_cache[mi][old_ri]
        new_pos = positions_cache[mi][new_ri]
        
        # Calculate delta
        # Remove old:
        #   if cover_count[p] == 1, it becomes 0 (uncovered) -> +1 to uncovered
        # Add new:
        #   if cover_count[p] == 0, it becomes 1 (covered) -> -1 to uncovered
        # Note: old_pos and new_pos are disjoint for same m.
        
        # We need to be careful with uint8 underflow/overflow, but counts are small.
        # Let's compute delta without modifying cover_count first to decide accept/reject.
        
        # Count how many in old_pos have count 1
        old_counts = cover_count[old_pos]
        # Count how many in new_pos have count 0
        new_counts = cover_count[new_pos]
        
        delta = np.sum(old_counts == 1) - np.sum(new_counts == 0)
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            # Accept move
            cover_count[old_pos] -= 1
            cover_count[new_pos] += 1
            residues[i] = new_ri
            current_uncovered += delta
            no_improve_count = 0
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues[:]
                best_cover_count = cover_count.copy() # Expensive, only do if improved significantly?
                # Actually, copying 52k bytes is fast.
        else:
            no_improve_count += 1
            
        temp *= cooling

    # Final Polish: Hill Climbing
    # Try to find any improvement by scanning all residues for each modulus
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n_div):
            m = divisors[i]
            old_r = residues[i]
            best_r = old_r
            best_local = current_uncovered
            
            old_pos = positions_cache[m][old_r]
            old_counts = cover_count[old_pos]
            
            # We can precompute the "loss" of removing old_r
            # loss = sum(old_counts == 1)
            # gain for new r = sum(cover_count[new_pos] == 0) - (overlap with old_pos? no, disjoint)
            # But cover_count changes if we hypothetically removed old_r.
            # So for each r:
            #   new_counts_hyp = cover_count[new_pos] (unchanged by removing old_r since disjoint)
            #   Wait, if we remove old_r, cover_count[new_pos] is the same.
            #   But if we add new_r, cover_count[new_pos] increases.
            #   The delta is:
            #   + sum(cover_count[old_pos] == 1)  (these become uncovered)
            #   - sum(cover_count[new_pos] == 0)  (these become covered)
            #   BUT, if old_r and new_r are the same, delta is 0.
            #   Also, if new_r != old_r, they are disjoint.
            
            # So we can calculate delta for each r efficiently.
            
            for r in range(m):
                if r == old_r:
                    continue
                
                new_pos = positions_cache[m][r]
                new_counts = cover_count[new_pos]
                
                delta = np.sum(old_counts == 1) - np.sum(new_counts == 0)
                
                local_uncovered = current_uncovered + delta
                
                if local_uncovered < best_local:
                    best_local = local_uncovered
                    best_r = r
            
            if best_r != old_r:
                # Apply move
                new_pos = positions_cache[m][best_r]
                cover_count[old_pos] -= 1
                cover_count[new_pos] += 1
                current_uncovered = best_local
                residues[i] = best_r
                improved = True
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = residues[:]
                    best_cover_count = cover_count.copy()

    classes = [[divisors[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
