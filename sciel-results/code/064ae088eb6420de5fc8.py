import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 1..16, so bits 1..16
    candidates = []
    candidate_tuples = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        candidate_tuples.append(comb)
        
    N = len(candidates)
    
    # Precompute pairwise intersections for speed
    # pair_int[i][j] = candidates[i] & candidates[j]
    pair_int = [[0] * N for _ in range(N)]
    for i in range(N):
        ci = candidates[i]
        for j in range(i + 1, N):
            val = ci & candidates[j]
            pair_int[i][j] = val
            pair_int[j][i] = val

    def can_add(family_indices, k_idx):
        """
        Check if candidate k_idx can be added to family (list of indices)
        without forming a sunflower.
        """
        ck = candidates[k_idx]
        n_fam = len(family_indices)
        
        for i in range(n_fam):
            idx_a = family_indices[i]
            ca = candidates[idx_a]
            aik = ca & ck
            
            for j in range(i + 1, n_fam):
                idx_b = family_indices[j]
                cb = candidates[idx_b]
                
                ab = pair_int[idx_a][idx_b]
                if ab != aik:
                    continue
                
                bk = cb & ck
                if ab == bk:
                    return False
        return True

    def greedy_fill(family_indices, family_set, order):
        """
        Greedily add candidates from 'order' to 'family_indices' if valid.
        Returns the number of added elements.
        """
        added = 0
        for idx in order:
            if idx in family_set:
                continue
            if can_add(family_indices, idx):
                family_indices.append(idx)
                family_set.add(idx)
                added += 1
        return added

    best_family = []
    best_size = 0
    
    # List of all indices
    all_indices = list(range(N))
    
    while time.time() < deadline - 2.0:
        # 1. Randomized Greedy Initialization
        random.shuffle(all_indices)
        family = []
        fam_set = set()
        
        # Greedy phase
        for idx in all_indices:
            if can_add(family, idx):
                family.append(idx)
                fam_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
        # 2. Local Search / Polish Phase
        # We try to improve the current family by removing elements and refilling
        # or just adding more if possible.
        
        # First, try to add any remaining valid candidates
        remaining = [i for i in all_indices if i not in fam_set]
        random.shuffle(remaining)
        
        # Try to add from remaining
        for idx in remaining:
            if can_add(family, idx):
                family.append(idx)
                fam_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
        
        # If we are stuck (no more can be added), try swaps/kicks
        # We limit the number of kicks to avoid spending too much time on one bad seed
        kicks = 0
        max_kicks = 50
        while time.time() < deadline - 2.0 and kicks < max_kicks:
            kicks += 1
            
            # Try to add one more? (Check if we missed any in the shuffle)
            # Since we just did a full pass on 'remaining', we are likely maximal.
            # So we must kick.
            
            # Pick a random element to remove
            if not family:
                break
            
            victim_idx = random.choice(family)
            
            # Remove it
            family.remove(victim_idx)
            fam_set.remove(victim_idx)
            
            old_size = len(family)
            
            # Try to greedily fill with any candidates not in family
            # We prioritize candidates that were blocked by the victim, but 
            # simply shuffling the complement is a good heuristic.
            
            # To speed up, we can just iterate through all indices not in fam_set
            # But that's O(N). 
            # Let's create a list of candidates not in family.
            # Since N=1820, this is fast.
            
            candidates_to_try = [i for i in all_indices if i not in fam_set]
            random.shuffle(candidates_to_try)
            
            # Greedy add
            for idx in candidates_to_try:
                if can_add(family, idx):
                    family.append(idx)
                    fam_set.add(idx)
            
            new_size = len(family)
            
            if new_size > old_size:
                # Improvement!
                if new_size > best_size:
                    best_size = new_size
                    best_family = family.copy()
                # Reset kicks to allow more search on this good family
                kicks = 0 
                # Try to add more?
                # We just filled greedily, so likely maximal again.
                # Break out of kick loop to start a new greedy init? 
                # Or continue kicking?
                # Continuing kicking might find even better structures.
                # But let's restart the outer loop to get a fresh start with the new best.
                break 
            elif new_size == old_size:
                # Same size. Accept with low probability to randomize
                if random.random() < 0.1:
                    # Keep the new configuration
                    if new_size > best_size:
                        best_size = new_size
                        best_family = family.copy()
                    # Continue kicking
                    pass
                else:
                    # Restore victim
                    family.append(victim_idx)
                    fam_set.add(victim_idx)
            else:
                # Worse size (shouldn't happen if we greedily fill, but just in case)
                # Restore victim
                family.append(victim_idx)
                fam_set.add(victim_idx)

    # Convert best_family (list of indices) to a list of tuples
    result = []
    for idx in best_family:
        result.append(candidate_tuples[idx])
        
    return result
