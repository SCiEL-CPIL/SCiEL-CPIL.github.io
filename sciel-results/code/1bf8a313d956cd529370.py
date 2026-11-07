"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _mask_to_list(mask):
    """Convert a bitmask to a list of vertex indices."""
    return [b for b in range(12) if (mask >> b) & 1]

def _count_comparable(sets_list):
    """Count unordered pairs where one set is a strict subset of the other."""
    n = len(sets_list)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if len(sets_list[i]) < len(sets_list[j]):
                if all(x in sets_list[j] for x in sets_list[i]):
                    count += 1
            elif len(sets_list[j]) < len(sets_list[i]):
                if all(x in sets_list[i] for x in sets_list[j]):
                    count += 1
    return count

def _count_comparable_masks(masks):
    """Count comparable pairs using bitmasks for speed."""
    n = len(masks)
    count = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            # Check if mi is subset of mj or mj is subset of mi
            if (mi & mj) == mi:
                if mi != mj:
                    count += 1
            elif (mi & mj) == mj:
                if mi != mj:
                    count += 1
    return count

def _greedy_chain_extension(initial_masks, target_size=40, deadline=None):
    """
    Extend a set of masks to target_size by greedily adding masks
    that maximize the number of new comparable pairs.
    """
    current = list(initial_masks)
    current_set = set(current)
    
    # Precompute all masks
    all_masks = list(range(1 << 12))
    
    # For efficiency, we can precompute subset relationships
    # subset[a][b] = True if a is subset of b
    # This is 4096x4096 = 16M entries, might be too large.
    # Instead, compute on the fly but cache if needed.
    
    while len(current) < target_size:
        best_mask = None
        best_gain = -1
        best_count = -1
        
        # Sample a subset of candidates to speed up
        candidates = all_masks
        if len(candidates) > 1000:
            candidates = random.sample(candidates, 1000)
        
        for m in candidates:
            if m in current_set:
                continue
            
            # Count how many existing sets are comparable with m
            gain = 0
            for cm in current:
                if (m & cm) == m or (m & cm) == cm:
                    if m != cm:
                        gain += 1
            
            if gain > best_gain or (gain == best_gain and m < (best_mask if best_mask is not None else 1 << 31)):
                best_gain = gain
                best_mask = m
                best_count = gain
        
        if best_mask is None:
            break
            
        current.append(best_mask)
        current_set.add(best_mask)
        
        if deadline and time.time() > deadline:
            break
    
    return current

def _local_search(masks, deadline):
    """
    Try to improve the set of masks by replacing one mask with another
    that increases the total comparable pair count.
    """
    current = list(masks)
    current_set = set(current)
    current_score = _count_comparable_masks(current)
    
    all_masks = list(range(1 << 12))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            # Try replacing current[i] with a random mask
            for _ in range(50):
                new_m = random.choice(all_masks)
                if new_m in current_set:
                    continue
                
                # Compute new score
                # Remove current[i], add new_m
                old_m = current[i]
                
                # Count pairs involving old_m
                old_gain = 0
                for j in range(len(current)):
                    if j == i:
                        continue
                    cm = current[j]
                    if (old_m & cm) == old_m or (old_m & cm) == cm:
                        if old_m != cm:
                            old_gain += 1
                
                # Count pairs involving new_m
                new_gain = 0
                for j in range(len(current)):
                    if j == i:
                        continue
                    cm = current[j]
                    if (new_m & cm) == new_m or (new_m & cm) == cm:
                        if new_m != cm:
                            new_gain += 1
                
                if new_gain > old_gain:
                    current[i] = new_m
                    current_set.remove(old_m)
                    current_set.add(new_m)
                    current_score += new_gain - old_gain
                    improved = True
                    break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for 90s limit
    
    # Start with a full chain: masks 0, 1, 3, 7, 15, ..., 4095
    # These are {0}, {0,1}, {0,1,2}, ..., {0,1,...,11}
    # Plus the empty set
    chain_masks = [0]
    for i in range(1, 13):
        chain_masks.append((1 << i) - 1)
    # That's 13 masks. We need 40.
    
    # Let's build a better initial solution
    # Use a chain of all prefixes of a random permutation
    random.seed(42)
    perm = list(range(12))
    random.shuffle(perm)
    
    chain = [0]
    mask = 0
    for v in perm:
        mask |= (1 << v)
        chain.append(mask)
    # 13 masks
    
    # Extend greedily
    current = _greedy_chain_extension(chain, 40, deadline)
    
    # Ensure we have exactly 40 distinct masks
    while len(current) < 40:
        # Add random masks
        m = random.randint(0, 4095)
        if m not in set(current):
            current.append(m)
    
    # Local search
    current, score = _local_search(current, deadline)
    
    # Convert to lists
    result = [_mask_to_list(m) for m in current]
    
    # Verify types
    for lst in result:
        for x in lst:
            assert isinstance(x, int) and not isinstance(x, bool)
    
    return result
# EVOLVE-BLOCK-END
