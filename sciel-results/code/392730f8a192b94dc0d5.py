"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    # Precompute positions for each (modulus, residue) pair as numpy arrays for fast slicing
    # positions_cache[m][r] is an array of indices
    positions_cache = {}
    for m in divisors:
        positions_cache[m] = [np.arange(r, base, m) for r in range(m)]

    best_uncovered = base
    best_residues = [0] * n_div

    def evaluate_residues(residues):
        """Calculate uncovered count for a given set of residues."""
        cover_count = np.zeros(base, dtype=np.uint8)
        for i, m in enumerate(divisors):
            r = residues[i]
            cover_count[positions_cache[m][r]] += 1
        return int(np.sum(cover_count == 0))

    # Greedy initialization
    # We will try a few different greedy strategies to get a good starting point
    # Strategy 1: Increasing modulus
    # Strategy 2: Decreasing modulus
    # Strategy 3: Randomized with restarts
    
    for attempt in range(5):
        if time.time() > deadline - 60:
            break
        
        if attempt == 0:
            order = list(range(n_div))
        elif attempt == 1:
            order = list(range(n_div-1, -1, -1))
        else:
            order = list(range(n_div))
            random.shuffle(order)
            
        residues = [0] * n_div
        cover_count = np.zeros(base, dtype=np.uint8)
        
        for idx in order:
            m = divisors[idx]
            best_r = 0
            best_gain = -1
            
            # To speed up greedy, we can just pick the first one that gives a gain or random
            # But let's do a simple greedy: pick r that maximizes new coverage
            # This is expensive if we do it fully. 
            # Instead, let's just pick random r for speed in init, then polish.
            # Or, pick r that covers the most currently uncovered spots.
            
            # Efficient greedy step:
            # We want to maximize sum(cover_count[pos] == 0 for pos in positions_cache[m][r])
            # We can compute this vectorially for all r? No, m varies.
            # Let's just pick a random r to start, as local search will fix it.
            # Actually, a better init is to use the "density" heuristic:
            # Small moduli cover many spots.
            
            # Let's stick to a simple greedy: pick r that minimizes overlap with existing coverage
            # which is equivalent to maximizing new coverage.
            # We'll iterate r for this m.
            for r in range(m):
                pos = positions_cache[m][r]
                # Count how many are currently uncovered
                gain = np.sum(cover_count[pos] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[idx] = best_r
            cover_count[positions_cache[m][best_r]] += 1
            
        current_uncovered = int(np.sum(cover_count == 0))
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues[:]

    # Local Search with Incremental Update
    residues = best_residues[:]
    cover_count = np.zeros(base, dtype=np.uint8)
    for i, m in enumerate(divisors):
        cover_count[positions_cache[m][residues[i]]] += 1
    
    current_uncovered = int(np.sum(cover_count == 0))
    
    temp = 10.0
    cooling = 0.99
    
    # Precompute index lists for fast update
    # We already have positions_cache
    
    while time.time() < deadline - 10:
        if temp < 0.01:
            break
            
        i = random.randint(0, n_div - 1)
        mi = divisors[i]
        old_ri = residues[i]
        
        # Choose new residue
        if random.random() < 0.5:
            new_ri = random.randint(0, mi - 1)
        else:
            # Neighborhood: +/- 1 or random
            new_ri = (old_ri + random.randint(1, 3)) % mi
            
        if new_ri == old_ri:
            continue
            
        old_pos = positions_cache[mi][old_ri]
        new_pos = positions_cache[mi][new_ri]
        
        # Calculate delta
        # Remove old
        # For p in old_pos:
        #   if cover_count[p] == 1: uncovered += 1
        #   cover_count[p] -= 1
        # Add new
        # For p in new_pos:
        #   if cover_count[p] == 0: uncovered -= 1
        #   cover_count[p] += 1
        
        delta = 0
        
        # We need to be careful with the order if old_pos and new_pos overlap.
        # However, for a fixed modulus m, residues r1 and r2 are disjoint modulo m.
        # But wait, positions_cache[m][r] contains numbers congruent to r mod m.
        # If r1 != r2, the sets of positions are disjoint.
        # So we can just update.
        
        # Update cover_count and delta
        # Note: cover_count is uint8. 
        # We must ensure we don't underflow. 
        # Removing: cover_count[p] is at least 1 (since this class covers it).
        # Adding: cover_count[p] can be 0 or more.
        
        # Vectorized update is tricky for delta counting without branching.
        # Let's use numpy where possible.
        
        # 1. Identify which old positions become uncovered
        # old_covered_only = (cover_count[old_pos] == 1)
        # delta += sum(old_covered_only)
        
        # 2. Identify which new positions were uncovered
        # new_was_uncovered = (cover_count[new_pos] == 0)
        # delta -= sum(new_was_uncovered)
        
        # But we must update cover_count.
        # If we update cover_count first, the "was uncovered" check for new_pos might be wrong if there's overlap?
        # No overlap between old_pos and new_pos for same m.
        
        # So:
        # 1. Check old_pos: how many have count 1?
        # 2. Check new_pos: how many have count 0?
        # 3. Update counts.
        
        old_counts = cover_count[old_pos]
        new_counts = cover_count[new_pos]
        
        delta += np.sum(old_counts == 1)
        delta -= np.sum(new_counts == 0)
        
        # Update cover_count
        cover_count[old_pos] -= 1
        cover_count[new_pos] += 1
        
        new_uncovered = current_uncovered + delta
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            residues[i] = new_ri
            current_uncovered = new_uncovered
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues[:]
        else:
            # Revert
            cover_count[old_pos] += 1
            cover_count[new_pos] -= 1
            
        temp *= cooling

    # Final Polish: Hill Climbing
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n_div):
            if time.time() > deadline - 2:
                break
            m = divisors[i]
            old_r = residues[i]
            best_r = old_r
            best_local = current_uncovered
            
            old_pos = positions_cache[m][old_r]
            
            for r in range(m):
                if r == old_r:
                    continue
                
                new_pos = positions_cache[m][r]
                
                # Calculate delta for moving from old_r to r
                # Same logic as above
                old_counts = cover_count[old_pos]
                new_counts = cover_count[new_pos]
                
                delta = 0
                delta += np.sum(old_counts == 1)
                delta -= np.sum(new_counts == 0)
                
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

    classes = [[divisors[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
