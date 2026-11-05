"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill

def _precompute_spheres(q, n, radius):
    """Precompute the bitset representation of the sphere of radius r for each word."""
    space_size = q ** n
    # Map each word index to its bitset
    spheres = [0] * space_size
    
    # Helper to convert tuple to index
    def to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    # Precompute neighbors for each word
    # We can do this by iterating over all words and their neighbors
    for i in range(space_size):
        # Convert index i to word
        word = [0] * n
        temp = i
        for j in range(n - 1, -1, -1):
            word[j] = temp % q
            temp //= q
        
        mask = 0
        # Center
        mask |= (1 << i)
        
        # Neighbors
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    word[pos] = new_val
                    n_idx = to_idx(word)
                    mask |= (1 << n_idx)
                word[pos] = orig_val
        
        spheres[i] = mask
    return spheres

def _popcount(x):
    """Count set bits in integer x."""
    return bin(x).count('1')

def _greedy_cover_bitset(spheres, space_size, seed_idx=None, time_limit=120.0):
    """Greedy set cover using bitsets."""
    uncovered = (1 << space_size) - 1
    selected = []
    
    if seed_idx is not None:
        # If we have a seed, start with it? 
        # The prompt says "Keep the function names...". 
        # The current program uses tetracode as seed. 
        # Let's try starting from scratch first, or using a random subset.
        # Actually, starting from scratch with greedy is often good for covering codes.
        pass

    # To speed up, we can maintain a list of candidate indices.
    # However, checking all 6561 candidates each step is 6561 bitset ANDs + popcounts.
    # 500 steps * 6561 ops = 3.2M ops. This is very fast in Python if using ints.
    
    candidates = list(range(space_size))
    
    while uncovered:
        if time.time() - _start_time > time_limit - 5:
            break
            
        best_gain = -1
        best_idx = -1
        
        # Randomize tie-breaking to escape local minima
        # Shuffle candidates? No, too slow. Just pick first max.
        # We can add a small random perturbation to the gain.
        
        for idx in candidates:
            # If this candidate is already selected, skip? 
            # We don't remove from candidates, but we can check if it's in selected.
            # Better: maintain a boolean array or set for selected.
            # For now, let's just compute gain. If gain is 0, it's useless.
            
            # Optimization: if gain is 0, skip.
            # gain = popcount(spheres[idx] & uncovered)
            # We can approximate or just compute.
            
            gain = _popcount(spheres[idx] & uncovered)
            
            # Add tiny random noise to break ties
            if gain > 0:
                gain += random.random() * 0.001
                
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                
        if best_idx == -1 or best_gain <= 0:
            break
            
        selected.append(best_idx)
        uncovered &= ~spheres[best_idx]
        
        # Optional: Remove candidates that are now fully covered?
        # No, a candidate might cover new words later if others are removed? 
        # In greedy construction, we don't remove. So if a candidate's sphere is subset of union of selected, 
        # its future gain is 0. We could prune candidates list.
        # But checking subset is expensive. Let's just keep all candidates.
        
    return selected

def _greedy_prune_bitset(spheres, selected_indices, time_limit=120.0):
    """Prune redundant codewords from the selected list."""
    if not selected_indices:
        return []
    
    # Map index to position in selected list? 
    # We just work with the list.
    
    selected = list(selected_indices)
    
    # Precompute union of all
    # We will iterate and try to remove each.
    
    # To make pruning efficient, we can sort by "importance" (e.g., how many unique words they cover).
    # But let's just iterate.
    
    changed = True
    while changed:
        changed = False
        if time.time() - _start_time > time_limit - 5:
            break
            
        n_sel = len(selected)
        if n_sel == 0:
            break
            
        # Calculate union of all
        union_all = 0
        for idx in selected:
            union_all |= spheres[idx]
            
        # Try to remove each
        for i in range(n_sel):
            if time.time() - _start_time > time_limit - 5:
                break
                
            idx_to_remove = selected[i]
            
            # Calculate union of others
            union_others = 0
            for j in range(n_sel):
                if j == i:
                    continue
                union_others |= spheres[selected[j]]
                
            # Check if sphere[idx_to_remove] is subset of union_others
            # i.e., sphere[idx_to_remove] & ~union_others == 0
            if (spheres[idx_to_remove] & ~union_others) == 0:
                selected.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    return selected

def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    time_limit = _DEADLINE - 5  # 5s safety margin

    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (though task is specific)
        # Simple greedy for general case
        space_size = q ** n
        spheres = _precompute_spheres(q, n, radius)
        selected = _greedy_cover_bitset(spheres, space_size, time_limit=time_limit)
        selected = _greedy_prune_bitset(spheres, selected, time_limit=time_limit)
        # Convert indices back to words
        result = []
        for idx in selected:
            word = [0] * n
            temp = idx
            for j in range(n - 1, -1, -1):
                word[j] = temp % q
                temp //= q
            result.append(word)
        return result

    # Specific optimization for q=3, n=8, radius=1
    space_size = 6561
    spheres = _precompute_spheres(q, n, radius)
    
    best_selected = None
    best_size = float('inf')
    
    # Run multiple random greedy starts
    # Each run: Greedy Cover -> Prune
    # Budget: 120s total.
    
    # First run: Deterministic (or fixed seed)
    random.seed(42)
    start = time.time()
    sel = _greedy_cover_bitset(spheres, space_size, time_limit=time_limit)
    sel = _greedy_prune_bitset(spheres, sel, time_limit=time_limit)
    if len(sel) < best_size:
        best_size = len(sel)
        best_selected = sel
    
    # Subsequent runs with different random seeds
    # How many can we do?
    # Greedy cover takes ~0.5s? Prune takes ~0.5s?
    # Let's try to fit as many as possible.
    
    while time.time() - _start_time < time_limit - 10:
        # Randomize
        random.seed(int(time.time() * 1000) % 100000)
        
        # Maybe start with a random subset to break symmetry?
        # Or just rely on the random tie-breaking in greedy.
        
        sel = _greedy_cover_bitset(spheres, space_size, time_limit=time_limit)
        sel = _greedy_prune_bitset(spheres, sel, time_limit=time_limit)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = sel
            
        # Early exit if we hit the theoretical lower bound or known best?
        # Known best is 486. If we hit 486, we can stop.
        if best_size <= 486:
            break

    if best_selected is None:
        # Fallback to tetracode if something went wrong
        # Re-implementing tetracode quickly for fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        words = []
        for h in tetracode:
            for tail in np.ndindex(*(3,)*4):
                words.append(h + list(tail))
        return words

    # Convert indices back to words
    result = []
    for idx in best_selected:
        word = [0] * n
        temp = idx
        for j in range(n - 1, -1, -1):
            word[j] = temp % q
            temp //= q
        result.append(word)
    
    return result
# EVOLVE-BLOCK-END
