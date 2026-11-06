import numpy as np
import random
import time

def construct():
    base = 51975
    start_time = time.time()
    
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    divisors.sort()
    
    best_uncovered = base
    best_classes = [[divisors[0], 0]] # Fallback
    
    # We will try optimizing residues for sets of the smallest divisors.
    # The number of moduli K can vary.
    # Let's try K from 5 to 20.
    
    def run_sa(moduli, max_iters=5000, t_start=1.0, t_end=0.01, seed=None):
        """
        Run Simulated Annealing to optimize residues for given moduli.
        Returns (uncovered_count, residues_list)
        """
        if seed is not None:
            random.seed(seed)
            
        k = len(moduli)
        if k == 0:
            return base, []
            
        # Initialize residues randomly
        residues = [random.randint(0, m - 1) for m in moduli]
        
        # Initialize coverage counts
        # counts[x] = number of moduli covering x
        counts = np.zeros(base, dtype=np.int32)
        uncovered = base
        
        for i, m in enumerate(moduli):
            r = residues[i]
            # Add coverage
            covered_slice = np.arange(r, base, m)
            counts[covered_slice] += 1
            # Update uncovered
            # We need to know which were 0 before
            # It's easier to just recompute uncovered from counts at the end of init?
            # No, let's do it properly.
            # Actually, simpler:
            # Initialize counts to 0.
            # For each modulus, add 1 to its residues.
            # Then uncovered = sum(counts == 0).
            
        # Let's re-init properly
        counts = np.zeros(base, dtype=np.int32)
        for i, m in enumerate(moduli):
            r = residues[i]
            counts[r::m] += 1
        uncovered = np.sum(counts == 0)
        
        current_uncovered = uncovered
        best_uncovered_sa = current_uncovered
        best_residues_sa = residues.copy()
        
        t = t_start
        cooling = (t_start - t_end) / max_iters
        
        for it in range(max_iters):
            if time.time() - start_time > 110:
                break
                
            # Pick a random modulus to change
            i = random.randint(0, k - 1)
            m = moduli[i]
            old_r = residues[i]
            
            # Pick a new random residue
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
            
            # Calculate delta in uncovered count
            # Remove old coverage
            old_slice = np.arange(old_r, base, m)
            new_slice = np.arange(new_r, base, m)
            
            # We need to be careful with overlapping elements if we just subtract/add.
            # But since we are changing one modulus, the counts change by -1 on old_slice and +1 on new_slice.
            # Delta uncovered = (number of x in old_slice where counts[x] becomes 0) - (number of x in new_slice where counts[x] was 0)
            
            # Note: If x is in both old_slice and new_slice, net change in counts is 0, so no change in uncovered status for x.
            # We can optimize by only looking at symmetric difference, but for simplicity and speed with numpy:
            
            # 1. Decrement counts for old_slice
            counts[old_slice] -= 1
            # Count how many became 0
            newly_uncovered = np.sum((counts[old_slice] == 0))
            
            # 2. Increment counts for new_slice
            counts[new_slice] += 1
            # Count how many were 0 before increment (so became covered)
            # We need to check which ones were 0 *before* this increment.
            # Since we already decremented old_slice, if there is overlap, the counts might be affected.
            # Let's be precise.
            
            # Let's revert and do it in a way that tracks the state.
            # Actually, the logic above has a flaw if slices overlap.
            # If x is in both, counts[x] -= 1 then += 1. Net 0.
            # But `newly_uncovered` counts x if counts[x] becomes 0 after decrement.
            # And "newly covered" should count x if counts[x] was 0 before increment.
            
            # Let's restart the delta calculation logic carefully.
            
            # Revert changes
            counts[old_slice] += 1
            counts[new_slice] -= 1
            
            # Identify unique elements in old_slice and new_slice
            # This is complex with numpy. Let's just compute the new uncovered count if we make the change?
            # That would be O(N) per step, which is 51975 * 5000 = 2.5e8 ops. Might be too slow in Python/Numpy overhead.
            
            # Better:
            # 1. Save counts for old_slice and new_slice? No, too much memory.
            # 2. Use the property that only elements in old_slice U new_slice change status.
            
            # Let's try a simpler heuristic update:
            # We know the exact change in counts for each x in old_slice is -1.
            # We know the exact change in counts for each x in new_slice is +1.
            
            # Let's compute delta_uncovered:
            # Delta = (# x in old_slice with count[x]==1) - (# x in new_slice with count[x]==0)
            # Wait, if x is in both, count changes by 0.
            # So we should only consider x in (old_slice \ new_slice) for the -1 part, and x in (new_slice \ old_slice) for the +1 part.
            
            # Finding set difference in numpy is not trivial without creating boolean masks.
            # Mask approach:
            # mask_old = np.zeros(base, bool); mask_old[old_slice] = True
            # mask_new = np.zeros(base, bool); mask_new[new_slice] = True
            # This allocates 51975 booleans twice per step. 5000 steps -> 10000 allocations. Might be okay.
            
            # Let's try this.
            
            # Optimization: Precompute slices? No, residues change.
            
            # Let's just do the full re-evaluation of uncovered if the delta logic is too complex/slow.
            # But 51975 * 5000 is borderline.
            
            # Let's stick to the delta logic with masks, but optimize.
            
            # Actually, a faster way:
            # We only care about x where count[x] is 0 or 1 (for old) or 0 (for new).
            
            # Let's just use the mask approach, it's vectorized.
            
            mask_old = np.zeros(base, dtype=bool)
            mask_old[old_slice] = True
            mask_new = np.zeros(base, dtype=bool)
            mask_new[new_slice] = True
            
            # Unique old: mask_old & ~mask_new
            unique_old = mask_old & ~mask_new
            # Unique new: mask_new & ~mask_old
            unique_new = mask_new & ~mask_old
            
            # Elements in unique_old will have count decreased by 1.
            # They become uncovered if their current count is 1.
            delta_neg = np.sum((counts[unique_old] == 1))
            
            # Elements in unique_new will have count increased by 1.
            # They become covered if their current count is 0.
            delta_pos = np.sum((counts[unique_new] == 0))
            
            delta_uncovered = delta_neg - delta_pos
            
            # Acceptance
            if delta_uncovered <= 0 or random.random() < np.exp(-delta_uncovered / t):
                # Apply change
                residues[i] = new_r
                counts[old_slice] -= 1
                counts[new_slice] += 1
                current_uncovered += delta_uncovered
                
                if current_uncovered < best_uncovered_sa:
                    best_uncovered_sa = current_uncovered
                    best_residues_sa = residues.copy()
            
            t -= cooling
            if t < t_end:
                t = t_end
                
        return best_uncovered_sa, best_residues_sa

    # Try different sets of moduli
    # We'll try the first K divisors for K in range(5, 25)
    for K in range(5, 25):
        if time.time() - start_time > 100:
            break
            
        moduli_subset = divisors[:K]
        
        # Run SA with a few restarts
        for restart in range(3):
            if time.time() - start_time > 105:
                break
            
            uncovered, residues = run_sa(moduli_subset, max_iters=2000, t_start=1.0, t_end=0.01, seed=restart)
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_classes = [[moduli_subset[i], residues[i]] for i in range(K)]
                print(f"K={K}, restart={restart}, uncovered={uncovered}", flush=True)
                
    return best_classes
