import random
import time
import numpy as np

# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))
_WEIGHTS = np.array([_INV[x] for x in _ALL], dtype=np.int64)
_ELEMENT_TO_IDX = {x: i for i, x in enumerate(_ALL)}

def _can_add(sums, w):
    """
    Check if weight w can be added to the set of subset sums 'sums'
    without causing any collisions.
    """
    for s in sums:
        if s + w in sums:
            return False
    return True

def _add_element(sums, w):
    """
    Add weight w to the set of subset sums.
    """
    return sums | {s + w for s in sums}

def _verify_dissociated_numpy(chosen):
    """
    Fast verification using NumPy: compute all 2^k subset sums and check uniqueness.
    Returns True if all subset sums are distinct.
    """
    k = len(chosen)
    if k == 0:
        return True
    if k > 20:  # 2^20 = 1M, still manageable but let's be safe
        # Fall back to set-based for very large k (though unlikely in practice)
        sums = {0}
        for x in chosen:
            w = _INV[x]
            sums = _add_element(sums, w)
        return len(sums) == (1 << k)
    
    idxs = np.array([_ELEMENT_TO_IDX[x] for x in chosen], dtype=np.int64)
    weights = _WEIGHTS[idxs]
    
    # Generate all 2^k subset sums using bit manipulation
    # For each subset mask from 0 to 2^k - 1, compute sum of weights where bit is set
    n_subsets = 1 << k
    masks = np.arange(n_subsets, dtype=np.int64)
    
    # Compute subset sums efficiently using iterative doubling
    # sums[mask] = sum of weights[i] for each bit i set in mask
    # Use the property: sums[2^i + m] = sums[m] + weights[i] for m < 2^i
    
    all_sums = np.zeros(n_subsets, dtype=np.int64)
    for i in range(k):
        bit = 1 << i
        # For all masks with bit i set, add weights[i]
        # Masks with bit i set: those where (mask & bit) != 0
        # We can compute this in blocks
        block_size = bit
        for start in range(bit, n_subsets, block_size * 2):
            end = min(start + block_size, n_subsets)
            all_sums[start:end] = all_sums[start - block_size:end - block_size] + weights[i]
    
    # Check if all sums are unique
    # Sort and check for duplicates
    sorted_sums = np.sort(all_sums)
    # Check if any adjacent elements are equal
    return np.all(np.diff(sorted_sums) > 0)

def _greedy(order, deadline):
    """
    Greedily add elements in the given order, maintaining dissociation.
    """
    sums = {0}
    chosen = set()
    for x in order:
        if x in chosen:
            continue
        w = _INV[x]
        if _can_add(sums, w):
            sums = _add_element(sums, w)
            chosen.add(x)
        if time.time() > deadline:
            break
    return chosen, sums

def _rebuild_sums(chosen):
    """
    Rebuild the set of subset sums from the chosen elements.
    """
    sums = {0}
    for x in chosen:
        w = _INV[x]
        sums = _add_element(sums, w)
    return sums

def _local_search(chosen, sums, deadline):
    """
    Local search to improve the set size with multi-swap capability.
    """
    remaining = [x for x in _ALL if x not in chosen]
    improved = True
    stall_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        
        # 1. Try to add any remaining element
        if remaining:
            random.shuffle(remaining)
            new_remaining = []
            for x in remaining:
                w = _INV[x]
                if _can_add(sums, w):
                    sums = _add_element(sums, w)
                    chosen.add(x)
                    improved = True
                    stall_count = 0
                else:
                    new_remaining.append(x)
            remaining = new_remaining
        
        # 2. If no simple add worked, try single swap
        if not improved and remaining and chosen and stall_count < 3:
            candidates = list(chosen)
            random.shuffle(candidates)
            for rem in candidates:
                if time.time() > deadline:
                    break
                
                new_chosen = set(chosen)
                new_chosen.discard(rem)
                new_sums = _rebuild_sums(new_chosen)
                
                found = False
                for add in remaining:
                    w = _INV[add]
                    if _can_add(new_sums, w):
                        new_sums = _add_element(new_sums, w)
                        new_chosen.add(add)
                        
                        if len(new_chosen) > len(chosen):
                            # Verify with NumPy before accepting
                            if _verify_dissociated_numpy(new_chosen):
                                chosen = new_chosen
                                sums = new_sums
                                improved = True
                                found = True
                                stall_count = 0
                                break
                            
                if found:
                    remaining = [x for x in _ALL if x not in chosen]
                    break
        
        # 3. If still stuck, try double swap (remove 2, add 2)
        if not improved and remaining and len(chosen) >= 2 and stall_count >= 2:
            candidates = list(chosen)
            random.shuffle(candidates)
            
            # Try a few random pairs to remove
            for _ in range(min(20, len(candidates) * (len(candidates)-1) // 2)):
                if time.time() > deadline:
                    break
                
                if len(candidates) < 2:
                    break
                    
                i, j = random.sample(range(len(candidates)), 2)
                rem1, rem2 = candidates[i], candidates[j]
                
                new_chosen = set(chosen)
                new_chosen.discard(rem1)
                new_chosen.discard(rem2)
                new_sums = _rebuild_sums(new_chosen)
                
                # Try to add two elements from remaining
                add_candidates = remaining[:]
                random.shuffle(add_candidates)
                
                added = 0
                temp_chosen = set(new_chosen)
                temp_sums = new_sums
                
                for add in add_candidates:
                    if added >= 2:
                        break
                    w = _INV[add]
                    if _can_add(temp_sums, w):
                        temp_sums = _add_element(temp_sums, w)
                        temp_chosen.add(add)
                        added += 1
                
                if added == 2 and len(temp_chosen) > len(chosen):
                    # Verify with NumPy
                    if _verify_dissociated_numpy(temp_chosen):
                        chosen = temp_chosen
                        sums = temp_sums
                        improved = True
                        stall_count = 0
                        break

    return chosen, sums

def construct():
    start = time.time()
    deadline = start + 82  # Safety margin for 90s limit
    best_size = 0
    best_set = set()

    # Generate a variety of orderings
    orderings = [
        sorted(_ALL),
        sorted(_ALL, reverse=True),
        sorted(_ALL, key=lambda x: _INV[x]),
        sorted(_ALL, key=lambda x: _INV[x], reverse=True),
    ]
    
    # Add random orderings - more of them since we have time
    num_random = 150
    for _ in range(num_random):
        order = _ALL[:]
        random.shuffle(order)
        orderings.append(order)

    for order in orderings:
        if time.time() > deadline:
            break
            
        chosen, sums = _greedy(order, deadline)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
        
        # Local search
        if time.time() < deadline - 3:
            chosen, sums = _local_search(chosen, sums, deadline - 3)
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)

    if not best_set:
        best_set = {1}

    return sorted(best_set)

# EVOLVE-BLOCK-END
