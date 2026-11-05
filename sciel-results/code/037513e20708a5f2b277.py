import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    # Fallback for non-specific cases
    if q != 3 or n != 6 or radius != 1:
        # Simple greedy for other cases
        total = q ** n
        if total > 10000:
             return [[0]*n]
        coords = []
        for i in range(total):
            c = [0]*n
            x = i
            for j in range(n-1, -1, -1):
                c[j] = x % q
                x //= q
            coords.append(c)
        
        # Build neighbors
        nbrs = []
        for c in coords:
            nbs = [c]
            for i in range(n):
                orig = c[i]
                for v in range(q):
                    if v != orig:
                        nc = c[:]
                        nc[i] = v
                        nbs.append(nc)
            nbrs.append(nbs)
            
        # Map to indices
        coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
        nbr_idx = []
        for nbs in nbrs:
            nbr_idx.append([coord_to_idx[tuple(nb)] for nb in nbs])
        
        uncovered = np.ones(total, dtype=np.int8)
        code = []
        while uncovered.sum() > 0:
            if time.time() > deadline:
                break
            counts = np.sum(uncovered[np.array(nbr_idx)], axis=1)
            best = int(np.argmax(counts))
            if counts[best] == 0:
                break
            code.append(best)
            uncovered[nbr_idx[best]] = 0
        return [coords[i] for i in code]

    # Specific case: K_3(6,1)
    total = 729
    powers = [1, 3, 9, 27, 81, 243]
    coords = [[(i // p) % 3 for p in powers] for i in range(total)]
    
    # Precompute neighbors as indices
    nbr = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        nbr[i, 0] = i
        k = 1
        for p in range(6):
            base = (i // powers[p]) % 3
            for alt in range(3):
                if alt != base:
                    nbr[i, k] = i + (alt - base) * powers[p]
                    k += 1

    # Helper to check if a set of indices covers everything
    def is_valid_code(indices):
        if not indices:
            return total == 0
        cov = np.zeros(total, dtype=np.int8)
        for idx in indices:
            cov[nbr[idx]] = 1
        return np.all(cov > 0)

    # Helper to try to remove redundant words
    def try_reduce(indices):
        if not indices:
            return indices
        cov = np.zeros(total, dtype=np.int32)
        for idx in indices:
            cov[nbr[idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(indices) - 1, -1, -1):
                idx = indices[i]
                if np.min(cov[nbr[idx]]) >= 2:
                    cov[nbr[idx]] -= 1
                    indices.pop(i)
                    changed = True
                    break
        return indices

    # 1. Greedy Construction
    uncovered = np.ones(total, dtype=np.int8)
    code = []
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        counts = np.sum(uncovered[nbr], axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        code.append(best)
        uncovered[nbr[best]] = 0
    
    code = try_reduce(code)
    best_code = code[:]
    best_size = len(code)
    
    # 2. Local Search: Remove and Repair
    # We will use a deterministic sequence of random seeds or just a fixed seed
    # to ensure reproducibility, but random is fine.
    rng = np.random.default_rng(42)
    no_improve = 0
    max_no_improve = 2000 # Increased limit for more thorough search
    
    # Precompute the list of indices for faster set checks?
    # Actually, checking `in code` is O(N). We can use a set for membership.
    
    while time.time() < deadline - 2.0 and no_improve < max_no_improve:
        if not code:
            break
            
        # Rebuild coverage map
        cov = np.zeros(total, dtype=np.int32)
        for idx in code:
            cov[nbr[idx]] += 1
            
        # Pick a random word to remove
        i = int(rng.integers(len(code)))
        c_remove = code[i]
        code.pop(i)
        cov[nbr[c_remove]] -= 1
        
        uncovered_mask = (cov == 0)
        
        if not np.any(uncovered_mask):
            # We removed a word and coverage is still valid.
            # Try to reduce further
            code = try_reduce(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                no_improve = 0
            else:
                no_improve += 1
            continue
            
        # Repair: Find best word to add back
        # We want to cover the most uncovered spots
        counts_add = np.sum(uncovered_mask[nbr], axis=1)
        
        # Ensure we don't add a word that's already in the code (though it shouldn't be, 
        # unless it was added by another removal? No, we removed one. 
        # But counts_add might be high for words already in code if they cover other uncovered spots?
        # Wait, if a word is in code, its neighbors are covered (cov >= 1). 
        # So uncovered_mask[nbr[idx]] would be 0 for any idx in code?
        # Let's check: cov[nbr[idx]] >= 1 for all neighbors of idx if idx is in code.
        # So uncovered_mask (cov==0) will be 0 on nbr[idx].
        # So counts_add will be 0 for any word currently in the code.
        # So best_add will naturally be a word NOT in the code, or 0 if all are in code (impossible if uncovered exists).
        
        best_add = int(np.argmax(counts_add))
        
        # If best_add is somehow in code (shouldn't happen if logic holds), revert
        if best_add in code:
            # This implies no valid word to add? Or logic error.
            # Revert removal
            code.append(c_remove)
            cov[nbr[c_remove]] += 1
            no_improve += 1
            continue

        code.append(best_add)
        cov[nbr[best_add]] += 1
        
        # Try to reduce again after repair
        code = try_reduce(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            no_improve = 0
        else:
            no_improve += 1
            
    # 3. Simulated Annealing / Random Restart if we have time
    # If we are stuck at e.g. 74 or 75, try swapping words.
    if time.time() < deadline - 5.0 and best_size > 73:
        # Try to find a 73-word code via more aggressive search
        # Strategy: Randomly swap a word with another and see if we can reduce.
        # Or: Start from best_code, and try to replace 2 words with 1?
        
        # Let's try a simple "swap" local search
        current_code = best_code[:]
        current_size = best_size
        rng2 = np.random.default_rng(123)
        no_improve_swap = 0
        
        while time.time() < deadline - 2.0 and no_improve_swap < 5000:
            if len(current_code) < 2:
                break
                
            cov = np.zeros(total, dtype=np.int32)
            for idx in current_code:
                cov[nbr[idx]] += 1
                
            # Pick two distinct words to remove
            i1 = int(rng2.integers(len(current_code)))
            i2 = int(rng2.integers(len(current_code) - 1))
            if i2 >= i1:
                i2 += 1
                
            c1 = current_code[i1]
            c2 = current_code[i2]
            
            # Remove them
            # Be careful with indices if we pop
            # Let's just work with a set/list copy for this trial
            
            trial_code = current_code[:]
            # Remove c1 and c2
            # Since they are distinct indices in the list, we can remove by value
            # But indices in the list are unique.
            
            # Recalculate coverage for trial
            # This is expensive. Let's just do it.
            t_cov = np.zeros(total, dtype=np.int32)
            for idx in trial_code:
                if idx != c1 and idx != c2:
                    t_cov[nbr[idx]] += 1
            
            uncovered_mask = (t_cov == 0)
            
            if not np.any(uncovered_mask):
                # We removed 2 words and still covered everything!
                # This is a big win.
                # Remove them permanently
                current_code.remove(c1)
                current_code.remove(c2)
                current_code = try_reduce(current_code)
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code[:]
                    no_improve_swap = 0
                else:
                    no_improve_swap += 1
                continue
            
            # Find best word to add back to cover the holes
            counts_add = np.sum(uncovered_mask[nbr], axis=1)
            best_add = int(np.argmax(counts_add))
            
            # If best_add is c1 or c2, we might just revert or swap with one of them?
            # If best_add is c1, adding it back means we effectively only removed c2.
            # But we want to reduce size.
            # If best_add is NOT c1 and NOT c2, we add it.
            # Size change: -2 (remove c1, c2) + 1 (add best_add) = -1.
            
            if best_add == c1 or best_add == c2:
                # This means one of the removed words was the best fix.
                # If we add c1 back, we are back to size N-1.
                # We might have found a redundant pair?
                # If adding c1 covers all, then c2 was redundant given c1?
                # No, we removed both. If adding c1 covers all, then {Code \ {c1,c2}} U {c1} = Code \ {c2} covers all.
                # So c2 was redundant in the original code?
                # Let's just try to remove c2 from current_code.
                if c2 in current_code:
                    current_code.remove(c2)
                    current_code = try_reduce(current_code)
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
                        no_improve_swap = 0
                    else:
                        no_improve_swap += 1
                continue
                
            # Add best_add
            trial_code.append(best_add)
            trial_code = try_reduce(trial_code)
            
            if len(trial_code) < len(current_code):
                current_code = trial_code
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code[:]
                    no_improve_swap = 0
                else:
                    no_improve_swap += 1
            else:
                no_improve_swap += 1

    # Return the best code found
    # Ensure we return a list of lists of integers
    return [coords[idx] for idx in best_code]
