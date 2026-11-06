import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin under 150s kill
    N = 1000

    def incremental_add(counts, mask, x):
        """Add x to the set; update counts in-place. Returns True if still valid (max<=2)."""
        # For each y in mask (including x itself for a<=b), add 1 to counts[x+y]
        # We need to handle a<=b: sum s = x+y where y >= x contributes 1
        # But also y < x: sum s = x+y where y < x, the pair (y,x) was already counted when y was added
        # Actually, let's think carefully.
        # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
        # When we add x, new pairs are (x, y) for y in S, y >= x, AND (y, x) for y in S, y < x
        # But (y,x) with y<x is the same pair as (x,y) with x>y. In our convention a<=b,
        # so (y,x) with y<x means a=y, b=x, which is valid.
        # So new pairs: (x,y) for all y in S with y >= x, and (y,x) for all y in S with y < x.
        # Wait, that's just all y in S. For each y in S, the pair (min(x,y), max(x,y)) is new.
        # The sum is x+y. So we add 1 to counts[x+y] for each y in S (including y=x).
        # But we must be careful: when we add x, the existing S doesn't include x yet.
        # So we iterate over current S (before adding x), and for each y, add 1 to counts[x+y].
        # Then set mask[x]=True.
        
        present = np.where(mask)[0]
        present = present[present >= 1]
        if len(present) == 0:
            mask[x] = True
            counts[2*x] += 1
            return counts[2*x] <= 2
        
        sums = x + present
        counts[sums] += 1
        mask[x] = True
        
        # Check if any count exceeds 2
        return np.max(counts) <= 2

    def incremental_remove(counts, mask, x):
        """Remove x from the set; update counts in-place."""
        present = np.where(mask)[0]
        present = present[(present >= 1) & (present != x)]
        if len(present) > 0:
            sums = x + present
            counts[sums] -= 1
        # Also remove the self-pair (x,x)
        counts[2*x] -= 1
        mask[x] = False

    def is_valid_counts(counts):
        return np.max(counts) <= 2

    def randomized_greedy(seed, time_limit=None):
        """Build a set using randomized greedy with a given seed."""
        rng = random.Random(seed)
        mask = np.zeros(N + 1, dtype=bool)
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        
        # Randomize the order of candidates
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        size = 0
        for x in candidates:
            # Try adding x
            if incremental_add(counts, mask, x):
                size += 1
            else:
                # Revert
                # We need to undo the add. But incremental_add already modified counts and mask.
                # Let's fix: check validity BEFORE committing.
                pass
        
        # The above is wrong because incremental_add modifies state. Let me redo.
        # Actually, let me restructure: check if adding would keep valid, then commit.
        
        return mask, size

    def randomized_greedy_v2(seed):
        """Proper randomized greedy."""
        rng = random.Random(seed)
        mask = np.zeros(N + 1, dtype=bool)
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        size = 0
        for x in candidates:
            # Check if adding x keeps all counts <= 2
            # Compute new counts: counts[x+y] for y in present
            present = np.where(mask)[0]
            present = present[present >= 1]
            if len(present) > 0:
                sums = x + present
                # Check if any of these sums would exceed 2
                would_exceed = np.any(counts[sums] >= 2)
                # Also check self-pair
                if counts[2*x] >= 2:
                    would_exceed = True
            else:
                would_exceed = counts[2*x] >= 2
            
            if not would_exceed:
                # Commit the add
                if len(present) > 0:
                    counts[x + present] += 1
                counts[2*x] += 1
                mask[x] = True
                size += 1
        
        return mask, size, counts

    def local_search(mask, counts, seed, time_budget):
        """Local search with swaps and adds to improve size."""
        rng = random.Random(seed)
        size = int(np.sum(mask))
        t0 = time.time()
        
        improved = True
        while improved and (time.time() - t0) < time_budget:
            improved = False
            
            # Try to add: find any absent x that can be added
            absent = np.where(~mask)[0]
            absent = absent[absent >= 1]
            if len(absent) == 0:
                break
            
            # Shuffle absent for randomness
            rng.shuffle(absent.tolist())
            
            for x in absent:
                present = np.where(mask)[0]
                present = present[present >= 1]
                would_exceed = False
                if len(present) > 0:
                    sums = x + present
                    if np.any(counts[sums] >= 2):
                        would_exceed = True
                if counts[2*x] >= 2:
                    would_exceed = True
                
                if not would_exceed:
                    # Add it
                    if len(present) > 0:
                        counts[x + present] += 1
                    counts[2*x] += 1
                    mask[x] = True
                    size += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove a, add b
            present_list = np.where(mask)[0]
            present_list = present_list[present_list >= 1]
            if len(present_list) == 0:
                break
            
            # Try a few random swaps
            for _ in range(50):
                a = present_list[rng.randrange(len(present_list))]
                b = absent[rng.randrange(len(absent))] if len(absent) > 0 else None
                if b is None:
                    break
                
                # Check if swap (remove a, add b) is valid
                # After removing a: counts decrease for sums a+y for y in present\{a}, and 2a
                # After adding b: counts increase for sums b+y for y in present\{a}, and 2b
                # Net effect: for each y in present\{a}:
                #   count[a+y] -= 1, count[b+y] += 1
                #   count[2a] -= 1, count[2b] += 1
                # We need all resulting counts <= 2
                
                present_no_a = np.delete(present_list, np.where(present_list == a)[0][0])
                
                # Compute net change
                valid = True
                # Check sums that increase: b+y for y in present_no_a, and 2b
                for y in present_no_a:
                    # count[b+y] increases by 1, count[a+y] decreases by 1
                    if counts[b + y] >= 2:
                        valid = False
                        break
                if valid and counts[2*b] >= 2:
                    valid = False
                # Also need to check that decreasing doesn't cause issues (it can't, counts only go down)
                # But we need to make sure the net effect doesn't make any count > 2
                # The only counts that increase are b+y and 2b. So checking those >= 2 is sufficient.
                
                if valid:
                    # Perform swap
                    # Remove a
                    if len(present_no_a) > 0:
                        counts[a + present_no_a] -= 1
                    counts[2*a] -= 1
                    mask[a] = False
                    
                    # Add b
                    if len(present_no_a) > 0:
                        counts[b + present_no_a] += 1
                    counts[2*b] += 1
                    mask[b] = True
                    
                    improved = True
                    break
        
        return mask, size, counts

    # Main: multi-restart
    best_size = 0
    best_mask = None
    best_counts = None
    
    # First, deterministic greedy as baseline
    mask_det = np.zeros(N + 1, dtype=bool)
    counts_det = np.zeros(2 * N + 1, dtype=np.int32)
    size_det = 0
    for x in range(1, N + 1):
        present = np.where(mask_det)[0]
        present = present[present >= 1]
        would_exceed = False
        if len(present) > 0:
            sums = x + present
            if np.any(counts_det[sums] >= 2):
                would_exceed = True
        if counts_det[2*x] >= 2:
            would_exceed = True
        if not would_exceed:
            if len(present) > 0:
                counts_det[x + present] += 1
            counts_det[2*x] += 1
            mask_det[x] = True
            size_det += 1
    
    best_size = size_det
    best_mask = mask_det.copy()
    best_counts = counts_det.copy()
    
    # Multi-restart with different seeds
    seed = 0
    while time.time() < deadline - 5:
        mask_r, size_r, counts_r = randomized_greedy_v2(seed)
        
        # Local search
        time_left = deadline - time.time() - 2
        if time_left > 1:
            mask_r, size_r, counts_r = local_search(mask_r, counts_r, seed, min(time_left, 30))
        
        if size_r > best_size:
            best_size = size_r
            best_mask = mask_r.copy()
            best_counts = counts_r.copy()
        
        seed += 1
    
    # Final local search on best
    time_left = deadline - time.time() - 2
    if time_left > 1:
        best_mask, best_size, best_counts = local_search(best_mask, best_counts, seed, time_left)
    
    result = [int(x) for x in np.where(best_mask)[0] if x >= 1]
    return result
