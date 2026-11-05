"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    base = 51975
    
    # Get all odd divisors of 51975 greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by increasing modulus. Small moduli cover more residues, 
    # so fixing them first in greedy might be a good heuristic.
    divisors.sort()
    
    num_moduli = len(divisors)
    
    # Greedy Initialization with count array
    # counts[x] = number of classes covering residue x
    counts = np.zeros(base, dtype=np.int16)
    residues = [0] * num_moduli
    
    # We will assign residues to each divisor.
    # To make greedy effective, we process moduli in a specific order.
    # Let's try processing in increasing order of m.
    for i, m in enumerate(divisors):
        best_r = 0
        best_gain = -1
        # Try all residues for this modulus
        # Optimization: if m is large, N/m is small. 
        # But iterating m times is fine for small m.
        # For large m, m is large, but N/m is small.
        # Cost per r is O(N/m). Total cost O(N).
        # So total cost for this m is O(N).
        # Total cost for all m is O(N * num_moduli).
        # 51975 * 48 ~ 2.5M. This is fast.
        
        for r in range(m):
            idx = np.arange(r, base, m)
            # Gain is number of currently uncovered residues (counts == 0)
            gain = np.count_nonzero(counts[idx] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[i] = best_r
        counts[best_r::m] += 1
    
    current_score = int(np.count_nonzero(counts == 0))
    best_score = current_score
    best_residues = residues[:]

    # Local Search: Hill Climbing
    # We iterate through all moduli and try to change their residue to minimize score.
    # We use the delta calculation to avoid full recomputation.
    
    def compute_delta(m, r_old, r_new, counts):
        """
        Returns change in number of uncovered residues if we change residue of modulus m 
        from r_old to r_new.
        """
        if r_old == r_new:
            return 0
        
        old_idx = np.arange(r_old, base, m)
        new_idx = np.arange(r_new, base, m)
        
        # Find intersection to avoid double counting effects
        # Since they are APs, intersection is also an AP or empty
        # But np.intersect1d is O(N log N) which might be slow if called too often?
        # N/m is small for large m. For small m, N/m is large.
        # Let's check sizes.
        # If m=3, N/m=17325. intersect1d on 17k arrays is fast.
        # If m=17000, N/m=3. intersect1d on 3 arrays is instant.
        
        # Alternative: 
        # delta = (count of positions where count becomes 0) - (count of positions where count becomes 1)
        # Positions becoming 0: were 1, now 0. This happens in old_idx \ new_idx where count was 1.
        # Positions becoming 1: were 0, now 1. This happens in new_idx \ old_idx where count was 0.
        
        # We can compute this without explicit set difference if we are careful, 
        # but set difference is safer.
        
        # Optimization: If the sets are small, just do it.
        old_set = set(old_idx)
        new_set = set(new_idx)
        
        old_only = [x for x in old_idx if x not in new_set]
        new_only = [x for x in new_idx if x not in old_set]
        
        # This Python loop might be slow. 
        # Let's stick to numpy operations which are C-optimized.
        # np.setdiff1d is sorted, which adds overhead.
        # np.intersect1d is also sorted.
        
        # Let's try a vectorized approach without sorting if possible?
        # Or just accept the cost. 2.5M ops * 48 moduli * passes.
        # If one pass takes 1s, we get 100 passes.
        
        # Let's use a faster delta check:
        # 1. Temporarily remove old coverage
        # 2. Temporarily add new coverage
        # 3. Count changes
        # 4. Restore
        # This modifies the array, which is risky if not done carefully, but fast.
        
        # However, we can just use the formula:
        # delta = sum( (counts[x] == 1) for x in old_only ) - sum( (counts[x] == 0) for x in new_only )
        
        # To avoid Python loops for old_only/new_only, we can use boolean masks.
        # But creating masks of size N for each check is O(N).
        # We only need to look at old_idx and new_idx.
        
        # Let's use the intersection method with numpy.
        # It's robust enough for N=52k.
        
        inter = np.intersect1d(old_idx, new_idx)
        old_only = np.setdiff1d(old_idx, inter, assume_unique=False) # assume_unique=False is safe
        new_only = np.setdiff1d(new_idx, inter, assume_unique=False)
        
        # Note: setdiff1d returns sorted arrays. 
        # The values in counts are accessed by index, so order doesn't matter.
        
        newly_uncovered = np.count_nonzero(counts[old_only] == 1)
        newly_covered = np.count_nonzero(counts[new_only] == 0)
        
        return newly_uncovered - newly_covered

    improved = True
    iteration = 0
    max_iterations = 1000 # High limit, time will stop us
    
    while improved and time.time() < deadline - 10:
        improved = False
        iteration += 1
        
        for i in range(num_moduli):
            if time.time() > deadline - 10:
                break
                
            m = divisors[i]
            r_old = residues[i]
            
            best_r = r_old
            best_delta = 0
            
            # Try all residues
            for r_new in range(m):
                if r_new == r_old:
                    continue
                
                delta = compute_delta(m, r_old, r_new, counts)
                
                if delta < best_delta:
                    best_delta = delta
                    best_r = r_new
            
            if best_r != r_old:
                # Apply change
                counts[r_old::m] -= 1
                counts[best_r::m] += 1
                residues[i] = best_r
                current_score += best_delta
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_residues = residues[:]
                    
    # Restore best found
    if best_score < current_score:
        # We need to rebuild counts from best_residues to return valid classes
        # But we just need to return the classes list.
        # The score is calculated by the evaluator.
        # We just return the classes corresponding to best_residues.
        pass

    # Final Polish: Try to remove classes that don't help?
    # Since we must use distinct moduli, and adding a class doesn't hurt, 
    # we should keep all. But wait, does the problem allow *subset*?
    # "construct() -> classes [[m, r], ...] with distinct moduli m > 1 dividing 51975"
    # It doesn't say "all divisors". It says "with distinct moduli...".
    # If a class covers 0 *new* residues, it doesn't help.
    # But it doesn't *hurt* either.
    # However, if we can remove a class, we simplify the output.
    # But the score is just "number of residues covered by no class".
    # So removing a redundant class doesn't change the score.
    # We can skip removal to save time, or do it if we have time.
    
    # Let's do a quick removal pass if time permits
    if time.time() < deadline - 5:
        # Rebuild counts for current best
        counts_best = np.zeros(base, dtype=np.int16)
        for i, m in enumerate(divisors):
            r = best_residues[i]
            counts_best[r::m] += 1
            
        classes_to_keep = []
        for i in range(num_moduli):
            m = divisors[i]
            r = best_residues[i]
            # Check if this class covers any unique residue
            # i.e., is there any x in r::m such that counts_best[x] == 1?
            idx = np.arange(r, base, m)
            # If all are covered by others (count > 1), we can remove it?
            # No, if we remove it, count becomes 0 for those where it was the *only* cover.
            # So we can remove it if min(counts_best[idx]) > 1.
            if np.min(counts_best[idx]) > 1:
                # Redundant
                continue
            else:
                classes_to_keep.append([m, r])
                # We don't need to update counts_best if we just want to check redundancy sequentially?
                # Actually, if we remove one, others might become redundant.
                # But this is a heuristic. 
                # Let's just keep it simple: if it's redundant in the *current* full set, remove it.
                # This might not be optimal if removals cascade, but it's a good start.
                # To be safe, let's not remove to avoid bugs, as score is invariant.
                pass
        
        # Actually, let's just return the best_residues mapped to divisors.
        # The problem doesn't penalize extra classes.
        
    result_classes = []
    for i, m in enumerate(divisors):
        result_classes.append([m, best_residues[i]])
        
    return result_classes

# EVOLVE-BLOCK-END
