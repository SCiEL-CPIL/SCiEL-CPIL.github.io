"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    base = 51975
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # Sort divisors to have a consistent order
    divisors.sort()
    
    num_divs = len(divisors)
    # Precompute indices for each divisor
    # indices[i] = list of residues covered by class (divisors[i], r) for r=0..divisors[i]-1
    # Actually, just using range(r, base, m) is fast enough in numpy?
    # Let's use numpy arrays for cover counts.
    
    def get_uncovered_count(cover_counts):
        return np.count_nonzero(cover_counts == 0)

    best_uncovered = base
    best_classes = None
    
    # Time limit check
    start_time = time.time()
    time_limit = 100.0 # Leave some buffer
    
    for restart in range(10): # 10 restarts
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random residues
        residues = np.array([random.randint(0, d-1) for d in divisors])
        cover_counts = np.zeros(base, dtype=np.int32)
        
        # Build initial cover counts
        for i, m in enumerate(divisors):
            r = residues[i]
            cover_counts[r::m] += 1
            
        current_uncovered = get_uncovered_count(cover_counts)
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_classes = [[divisors[i], residues[i]] for i in range(num_divs)]
            
        # Local Search Phase
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # Try to improve each class
            # We iterate through divisors. For each, find best residue.
            # To speed up, we can just check all residues for each divisor.
            # Total work per pass: sum(d_i) operations * cost of update.
            # Cost of update is O(base/d_i).
            # Total O(sum(base)) = O(num_divs * base) which is ~ 48 * 52k ~ 2.5M ops per pass.
            # This is fast in Python if vectorized, but loops are slow.
            # Let's vectorize the evaluation of best residue for a single divisor.
            
            for i in range(num_divs):
                m = divisors[i]
                r_old = residues[i]
                
                # Calculate change in uncovered count for each possible new residue r_new
                # Current contribution of this class to cover_counts:
                # old_mask = (np.arange(base) % m == r_old)
                
                # We want to find r_new that maximizes reduction in uncovered count.
                # Let U be boolean array where cover_counts == 0.
                # Let L be boolean array where x % m == r_old.
                # Let G_new be boolean array where x % m == r_new.
                
                # If we swap:
                # cover_counts_new = cover_counts - L + G_new
                # uncovered_new = (cover_counts_new == 0)
                
                # This is hard to vectorize over all r_new at once for the whole base.
                # Instead, let's just iterate r_new in range(m) and compute delta.
                # For m=51975, this is 51975 iterations. Too slow if done for every divisor every pass.
                
                # Optimization:
                # Only try a subset of residues? Or use a smarter heuristic?
                # Actually, for small m, it's fast. For large m, it's slow.
                # Let's try a simpler hill climbing: random perturbation.
                pass # We will replace this with a more efficient local search step below.
                
        # Better Local Search: Random Perturbation
        # Pick a random divisor, change its residue to a random other value, accept if better.
        # This is Simulated Annealing / Random Walk.
        
        # Let's implement a proper coordinate descent with random restarts on the coordinate choice.
        # But the inner loop "find best r" is the bottleneck.
        
        # Alternative: 
        # Since we have time, let's just do random restarts with a simple greedy initialization 
        # followed by random swaps.
        
        # Let's stick to the prompt's request: "design a new program ... from an approach family that none of the current programs use".
        # The current program is Greedy Construction.
        # My approach: Local Search (Hill Climbing) on Residues.
        
        # To make it fast, I will use a heuristic:
        # For each divisor, only test a few promising residues (e.g., current, 0, 1, random).
        # Or, simply perform random swaps and accept if better.
        
        # Let's implement Random Swap Hill Climbing.
        
        if time.time() - start_time > time_limit:
            continue
            
        for step in range(500): # 500 steps per restart
            if time.time() - start_time > time_limit:
                break
                
            # Pick random divisor
            i = random.randint(0, num_divs - 1)
            m = divisors[i]
            r_old = residues[i]
            
            # Pick random new residue
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
                
            # Calculate delta
            # Remove old coverage
            old_mask_indices = np.arange(r_old, base, m)
            new_mask_indices = np.arange(r_new, base, m)
            
            # We need to check if any x in old_mask becomes uncovered (cover_count goes 1->0)
            # and if any x in new_mask becomes covered (cover_count goes 0->1).
            
            # Count how many x in old_mask have cover_count == 1 (will become 0)
            lost_uncovered = np.count_nonzero(cover_counts[old_mask_indices] == 1)
            
            # Count how many x in new_mask have cover_count == 0 (will become 1)
            gained_covered = np.count_nonzero(cover_counts[new_mask_indices] == 0)
            
            # Net change in uncovered count:
            # Uncovered increases by lost_uncovered (these were covered by 1, now 0)
            # Uncovered decreases by gained_covered (these were 0, now 1)
            delta_uncovered = lost_uncovered - gained_covered
            
            if delta_uncovered < 0:
                # Apply change
                cover_counts[old_mask_indices] -= 1
                cover_counts[new_mask_indices] += 1
                residues[i] = r_new
                current_uncovered += delta_uncovered
                improved = True
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [[divisors[j], residues[j]] for j in range(num_divs)]
                    
            # If delta >= 0, reject (pure hill climbing)
            
    if best_classes is None:
        # Fallback to simple greedy if search failed
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        divs = sorted(divisors, reverse=True)
        for m in divs:
            best_r = 0
            best_count = -1
            for r in range(m):
                count = np.count_nonzero(covered[r::m] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes
        
    return best_classes
# EVOLVE-BLOCK-END
