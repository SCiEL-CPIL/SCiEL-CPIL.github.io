"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct():
    """
    Construct 40 distinct subsets of {0..11} maximizing the number of
    unordered pairs (A,B) where A is a strict subset of B.
    
    Strategy:
    1. Greedy chain initialization: start with a long chain of nested sets,
       then greedily add sets that maximize immediate comparability.
    2. Local search with 1-swap moves to improve the solution.
    3. Multiple restarts with different random seeds.
    4. Deadline guard to return best result before 90s limit.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 12
    m = 40
    all_masks = list(range(1 << n))  # 4096 masks
    
    # Precompute subset relations using bit operations
    # For each mask, precompute which masks are subsets/supersets
    # This is expensive to precompute fully, so we'll use a faster check
    
    def is_subset(a, b):
        """Check if set a is a strict subset of set b (using masks)."""
        return (a & b) == a and a != b
    
    def count_comparable(family):
        """Count unordered pairs where one is a strict subset of the other."""
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                a, b = family[i], family[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (a & b) == b and a != b:
                    count += 1
        return count
    
    def greedy_construct(seed):
        """Greedy construction starting from a chain."""
        rng = random.Random(seed)
        
        # Start with a chain of nested sets
        # Build a chain: {}, {v1}, {v1,v2}, ..., {v1,...,vk}
        vertices = list(range(n))
        rng.shuffle(vertices)
        
        family = []
        # Build initial chain
        current = 0
        for v in vertices:
            current |= (1 << v)
            family.append(current)
        
        # Now greedily add sets that maximize comparability
        remaining = [mask for mask in all_masks if mask not in family]
        rng.shuffle(remaining)
        
        while len(family) < m:
            best_mask = None
            best_gain = -1
            for mask in remaining:
                # Calculate gain: number of existing sets that are comparable with mask
                gain = 0
                for existing in family:
                    if (mask & existing) == mask and mask != existing:
                        gain += 1
                    elif (mask & existing) == existing and mask != existing:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
                if time.time() > deadline:
                    return family[:m], count_comparable(family[:m])
            
            if best_mask is None:
                break
            family.append(best_mask)
            remaining.remove(best_mask)
        
        return family[:m], count_comparable(family[:m])
    
    def local_search(family, max_iters=500):
        """Improve family via 1-swap moves."""
        family = list(family)
        current_score = count_comparable(family)
        
        in_family = set(family)
        candidates = [mask for mask in all_masks if mask not in in_family]
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            best_move = None
            best_new_score = current_score
            
            # Try swapping each family member with a candidate
            # Use a limited set of candidates for speed
            sample_size = min(len(candidates), 200)
            sampled = random.sample(candidates, sample_size) if len(candidates) > sample_size else candidates
            
            for i, old_mask in enumerate(family):
                for new_mask in sampled:
                    if new_mask in in_family:
                        continue
                    
                    # Calculate delta score
                    # Remove old_mask, add new_mask
                    # Gain from new_mask with other family members
                    gain_new = 0
                    for j, other in enumerate(family):
                        if j == i:
                            continue
                        if (new_mask & other) == new_mask and new_mask != other:
                            gain_new += 1
                        elif (new_mask & other) == other and new_mask != other:
                            gain_new += 1
                    
                    # Loss from old_mask with other family members
                    loss_old = 0
                    for j, other in enumerate(family):
                        if j == i:
                            continue
                        if (old_mask & other) == old_mask and old_mask != other:
                            loss_old += 1
                        elif (old_mask & other) == other and old_mask != other:
                            loss_old += 1
                    
                    delta = gain_new - loss_old
                    if delta > best_new_score - current_score:
                        best_new_score = current_score + delta
                        best_move = (i, old_mask, new_mask)
            
            if best_move is None:
                break
            
            i, old_mask, new_mask = best_move
            family[i] = new_mask
            in_family.discard(old_mask)
            in_family.add(new_mask)
            candidates.remove(old_mask)
            candidates.append(new_mask)
            current_score = best_new_score
        
        return family, current_score
    
    best_family = None
    best_score = -1
    
    # Try multiple restarts
    num_restarts = 10
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        family, score = greedy_construct(seed)
        if len(family) < m:
            # Pad with random sets if needed
            in_family = set(family)
            for mask in all_masks:
                if mask not in in_family:
                    family.append(mask)
                    in_family.add(mask)
                    if len(family) >= m:
                        break
            family = family[:m]
            score = count_comparable(family)
        
        # Local search
        family, score = local_search(family)
        
        if score > best_score:
            best_score = score
            best_family = list(family)
    
    # Final polish: try more local search iterations on the best family
    if best_family is not None and time.time() < deadline - 5:
        best_family, best_score = local_search(best_family, max_iters=2000)
    
    # Convert masks to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
