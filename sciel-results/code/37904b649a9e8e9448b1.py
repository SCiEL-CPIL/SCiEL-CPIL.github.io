import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill
    
    n = 12
    m = 40
    
    # Precompute all subsets as frozensets, grouped by size
    all_subsets_by_size = {k: [] for k in range(n + 1)}
    for k in range(n + 1):
        for combo in itertools.combinations(range(n), k):
            all_subsets_by_size[k].append(frozenset(combo))
    
    # Precompute subset relationships: for each pair, check if one is subset of other
    # We'll use bitmasks for efficiency
    # Map frozenset to bitmask
    def to_mask(fs):
        mask = 0
        for v in fs:
            mask |= (1 << v)
        return mask
    
    all_subsets = []
    for k in range(n + 1):
        all_subsets.extend(all_subsets_by_size[k])
    
    # Create mask representation for all subsets
    subset_masks = {fs: to_mask(fs) for fs in all_subsets}
    
    def count_comparable_pairs(sets):
        """Count unordered pairs where one is a strict subset of the other."""
        count = 0
        masks = [subset_masks[fs] for fs in sets]
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if masks[i] & masks[j] == masks[i] and masks[i] != masks[j]:
                    count += 1
                elif masks[i] & masks[j] == masks[j] and masks[i] != masks[j]:
                    count += 1
        return count
    
    # Strategy: Start with a chain (maximally comparable), then greedily add sets
    # that maximize additional comparable pairs
    
    # A full chain has 13 sets (sizes 0 through 12), all mutually comparable
    # We need 40 sets total. The chain gives us 13 sets with C(13,2)=78 comparable pairs.
    # We need 27 more sets.
    
    # Let's think about this differently. We want to maximize the number of comparable pairs.
    # A chain of length L has C(L,2) comparable pairs.
    # If we have multiple chains, pairs between chains may or may not be comparable.
    
    # Better approach: Use local search with bitmask acceleration.
    # Start with a good initial solution, then try swaps.
    
    # Initial solution: Start with chain of all 13 subsets, then add remaining 27
    # greedily by picking sets that maximize comparable pairs with existing sets.
    
    chain = [frozenset(range(k)) for k in range(n + 1)]  # sizes 0 to 12
    
    current = list(chain)
    current_set = set(current)
    
    # For greedy addition, compute for each candidate set how many comparable pairs it adds
    def comparable_with_existing(fs, existing):
        mask = subset_masks[fs]
        count = 0
        for ex in existing:
            em = subset_masks[ex]
            if mask & em == mask and mask != em:
                count += 1
            elif mask & em == em and mask != em:
                count += 1
        return count
    
    # Greedily add 27 more sets
    candidates = [fs for fs in all_subsets if fs not in current_set]
    
    for _ in range(m - len(current)):
        if time.time() > deadline:
            break
        
        best_fs = None
        best_score = -1
        for fs in candidates:
            if fs in current_set:
                continue
            score = comparable_with_existing(fs, current)
            if score > best_score:
                best_score = score
                best_fs = fs
        
        if best_fs is None:
            break
        
        current.append(best_fs)
        current_set.add(best_fs)
        candidates.remove(best_fs)
    
    # Now do local search: try swapping one set for another to improve the count
    def total_score(sets):
        return count_comparable_pairs(sets)
    
    current_score = total_score(current)
    
    # Precompute for faster evaluation
    # For local search, we'll try replacing one element with another
    # and see if the score improves.
    
    # To speed up, precompute comparable relationships
    # Let's use a different approach: maintain a matrix of comparability
    
    def build_comparability_matrix(sets):
        N = len(sets)
        masks = [subset_masks[fs] for fs in sets]
        comp = [[False] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if masks[i] & masks[j] == masks[i] and masks[i] != masks[j]:
                    comp[i][j] = True
                    comp[j][i] = True
                elif masks[i] & masks[j] == masks[j] and masks[i] != masks[j]:
                    comp[i][j] = True
                    comp[j][i] = True
        return comp
    
    def score_from_matrix(comp):
        count = 0
        N = len(comp)
        for i in range(N):
            for j in range(i + 1, N):
                if comp[i][j]:
                    count += 1
        return count
    
    # Local search with bitmask acceleration
    # We'll try swapping elements in the current set with unused elements
    
    unused = set(all_subsets) - set(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        comp = build_comparability_matrix(current)
        current_score = score_from_matrix(comp)
        
        # For each position i, try replacing current[i] with each unused set
        # To speed up, only try a subset of swaps
        for i in range(m):
            if time.time() > deadline:
                break
            
            # Compute how many comparable pairs involve position i
            i_comparable = sum(comp[i][j] for j in range(m) if j != i)
            
            best_swap = None
            best_gain = 0
            
            # Try swapping with unused sets (sample if too many)
            unused_list = list(unused)
            random.shuffle(unused_list)
            for new_fs in unused_list[:200]:  # limit to 200 candidates per position
                if new_fs in current:
                    continue
                
                # Compute new comparable count if we replace current[i] with new_fs
                # Remove i_comparable, add comparable pairs of new_fs with all other positions
                new_comparable = 0
                new_mask = subset_masks[new_fs]
                for j in range(m):
                    if j == i:
                        continue
                    other_mask = subset_masks[current[j]]
                    if new_mask & other_mask == new_mask and new_mask != other_mask:
                        new_comparable += 1
                    elif new_mask & other_mask == other_mask and new_mask != other_mask:
                        new_comparable += 1
                
                gain = new_comparable - i_comparable
                if gain > best_gain:
                    best_gain = gain
                    best_swap = new_fs
            
            if best_swap is not None and best_gain > 0:
                # Perform the swap
                old_fs = current[i]
                current[i] = best_swap
                unused.remove(best_swap)
                unused.add(old_fs)
                improved = True
    
    # Convert to list of lists
    result = [list(fs) for fs in current]
    
    # Ensure types are correct
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
    
    return result
