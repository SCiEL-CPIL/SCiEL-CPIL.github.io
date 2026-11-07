import time
import itertools
import random
import numpy as np

def _to_mask(lst):
    """Convert list of vertices to a 12-bit mask."""
    mask = 0
    for v in lst:
        mask |= (1 << v)
    return mask

def _from_mask(mask):
    """Convert 12-bit mask to list of vertices."""
    return [i for i in range(12) if (mask >> i) & 1]

def _is_comparable_mask(a_mask, b_mask):
    """Check if one mask is a strict subset of the other."""
    return (a_mask & b_mask) == a_mask and a_mask != b_mask or \
           (a_mask & b_mask) == b_mask and a_mask != b_mask

def _count_comparable_masks(masks):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(masks)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _is_comparable_mask(masks[i], masks[j]):
                count += 1
    return count

def _build_comparability_matrix(all_masks, selected_indices):
    """
    Build a numpy matrix where entry [i,j] = 1 if all_masks[i] is comparable with all_masks[j].
    Only for the selected subset of indices.
    Returns a 2D numpy array of shape (len(selected_indices), len(all_masks))
    where entry [k, j] = 1 if selected_indices[k] is comparable with all_masks[j].
    """
    n_sel = len(selected_indices)
    n_all = len(all_masks)
    # For each selected mask and each all mask, check comparability
    # Use vectorized bitwise ops
    sel_masks = np.array([all_masks[i] for i in selected_indices], dtype=np.uint32)
    all_arr = np.array(all_masks, dtype=np.uint32)
    
    # sel_masks shape: (n_sel, 1), all_arr shape: (1, n_all)
    sel_exp = sel_masks[:, np.newaxis]
    all_exp = all_arr[np.newaxis, :]
    
    # Comparable if (a & b) == a and a != b, OR (a & b) == b and a != b
    # i.e., (a & b) == min(a,b) in terms of subset, and a != b
    # More precisely: one is subset of other and they're not equal
    # (a & b) == a means a subset b; (a & b) == b means b subset a
    # Comparable iff ((a&b)==a XOR (a&b)==b) is false AND not (a==b)
    # Actually: comparable iff ((a&b)==a) OR ((a&b)==b), AND a!=b
    
    ab = sel_exp & all_exp
    a_subset_b = (ab == sel_exp)  # sel is subset of all
    b_subset_a = (ab == all_exp)  # all is subset of sel
    equal = (sel_exp == all_exp)
    
    comparable = ((a_subset_b | b_subset_a) & ~equal).astype(np.uint8)
    return comparable

def _local_search_masks(masks, all_masks, used_set, deadline, max_iter=200):
    """
    Local search using numpy-accelerated comparability checks.
    Tries to swap out one mask for a better one.
    """
    current_score = _count_comparable_masks(masks)
    n = len(masks)
    
    # Precompute all_masks as numpy array
    all_arr = np.array(all_masks, dtype=np.uint32)
    n_all = len(all_masks)
    
    # Create a boolean array of which masks are used
    used_arr = np.zeros(n_all, dtype=bool)
    for m in masks:
        for idx in range(n_all):
            if all_arr[idx] == m:
                used_arr[idx] = True
                break
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter and time.time() < deadline:
        improved = False
        iter_count += 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            saved_mask = masks[i]
            
            # Build comparability matrix for current selection (with mask i removed)
            # Actually, let's compute the gain of adding candidate c while removing mask i
            # Gain = (comparables of c with remaining 39) - (comparables of saved_mask with remaining 39)
            
            remaining = [masks[k] for k in range(n) if k != i]
            remaining_arr = np.array(remaining, dtype=np.uint32)
            n_rem = len(remaining_arr)
            
            # Compute comparability of saved_mask with remaining
            saved_arr = np.array([saved_mask], dtype=np.uint32)
            ab_saved = saved_arr & remaining_arr[np.newaxis, :]
            comp_saved = ((ab_saved == saved_arr) | (ab_saved == remaining_arr[np.newaxis, :])) & (saved_arr != remaining_arr[np.newaxis, :])
            saved_comp_count = comp_saved[0].sum()
            
            # For each candidate not in used_set, compute comparability with remaining
            # Candidates are those where used_arr is False
            candidate_indices = np.where(~used_arr)[0]
            if len(candidate_indices) == 0:
                continue
            
            cand_arr = all_arr[candidate_indices]
            ab_cand = cand_arr[:, np.newaxis] & remaining_arr[np.newaxis, :]
            comp_cand = ((ab_cand == cand_arr[:, np.newaxis]) | (ab_cand == remaining_arr[np.newaxis, :])) & (cand_arr[:, np.newaxis] != remaining_arr[np.newaxis, :])
            cand_comp_counts = comp_cand.sum(axis=1)
            
            best_local_idx = np.argmax(cand_comp_counts)
            best_gain = cand_comp_counts[best_local_idx] - saved_comp_count
            
            if best_gain > 0:
                best_global_idx = candidate_indices[best_local_idx]
                best_mask = all_arr[best_global_idx]
                
                # Perform the swap
                used_arr[used_arr == 0]  # just to keep numpy happy
                # Mark saved_mask as unused
                for idx in range(n_all):
                    if all_arr[idx] == saved_mask:
                        used_arr[idx] = False
                        break
                used_arr[best_global_idx] = True
                
                masks[i] = int(best_mask)
                current_score += int(best_gain)
                improved = True
                break  # Restart the outer loop
    
    return current_score

def _construct_base():
    """Construct a strong initial solution using chains and numpy-accelerated local search."""
    deadline = time.time() + 82  # 82 second deadline for safety margin
    
    # Generate all subsets as masks
    all_masks = []
    for r in range(13):
        for combo in itertools.combinations(range(12), r):
            all_masks.append(_to_mask(list(combo)))
    
    n_all = len(all_masks)
    
    random.seed(42)
    
    best_masks = []
    used_set = set()
    
    # Chain 1: Standard chain {0}, {0,1}, {0,1,2}, ..., {0..11}
    chain1_masks = [_to_mask(list(range(i))) for i in range(13)]
    for m in chain1_masks:
        best_masks.append(m)
        used_set.add(m)
    
    # Chain 2: Another chain using different elements
    # {1}, {1,2}, {1,2,3}, ...
    chain2_masks = []
    for i in range(1, 12):
        s = list(range(1, 1 + i))
        m = _to_mask(s)
        if m not in used_set:
            chain2_masks.append(m)
    
    # Add up to 13 more sets from chain2
    for m in chain2_masks[:13]:
        if len(best_masks) >= 40:
            break
        best_masks.append(m)
        used_set.add(m)
    
    # If we still need more, add greedily
    remaining = 40 - len(best_masks)
    if remaining > 0:
        candidates = [m for m in all_masks if m not in used_set]
        random.shuffle(candidates)
        
        for _ in range(remaining):
            if time.time() > deadline:
                break
            best_gain = -1
            best_mask = None
            best_idx = None
            
            for idx, m in enumerate(candidates):
                if time.time() > deadline:
                    break
                cnt = 0
                for existing in best_masks:
                    if _is_comparable_mask(m, existing):
                        cnt += 1
                if cnt > best_gain:
                    best_gain = cnt
                    best_mask = m
                    best_idx = idx
            
            if best_mask is not None and best_idx is not None:
                best_masks.append(best_mask)
                used_set.add(best_mask)
                candidates.pop(best_idx)
    
    # Local search to improve
    if time.time() < deadline:
        candidates = [m for m in all_masks if m not in used_set]
        random.shuffle(candidates)
        _local_search_masks(best_masks, candidates, used_set, deadline, max_iter=200)
    
    # Verify we have exactly 40 distinct sets
    if len(best_masks) != 40:
        # Fallback: fill with random sets
        candidates = [m for m in all_masks if m not in used_set]
        random.shuffle(candidates)
        for m in candidates:
            if len(best_masks) >= 40:
                break
            best_masks.append(m)
            used_set.add(m)
    
    assert len(best_masks) == 40
    assert len(set(best_masks)) == 40
    
    return [_from_mask(m) for m in best_masks]

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82 second deadline
    
    try:
        result = _construct_base()
        # Verify validity
        assert len(result) == 40
        seen = set()
        for s in result:
            assert isinstance(s, (list, tuple))
            for x in s:
                assert isinstance(x, int) and not isinstance(x, bool)
                assert 0 <= x <= 11
            fs = frozenset(s)
            assert fs not in seen
            seen.add(fs)
        return result
    except Exception:
        # Fallback: simple chain-based construction
        chain = [list(range(i)) for i in range(13)]
        # Add remaining sets from a fixed pattern
        extra = []
        for r in range(1, 12):
            for start in range(12 - r + 1):
                s = list(range(start, start + r))
                if frozenset(s) not in set(frozenset(c) for c in chain) and frozenset(s) not in set(frozenset(e) for e in extra):
                    extra.append(s)
                if len(chain) + len(extra) >= 40:
                    break
            if len(chain) + len(extra) >= 40:
                break
        result = chain + extra[:40 - len(chain)]
        while len(result) < 40:
            # Add more
            for r in range(13):
                for combo in itertools.combinations(range(12), r):
                    s = list(combo)
                    if frozenset(s) not in set(frozenset(x) for x in result):
                        result.append(s)
                        break
                if len(result) >= 40:
                    break
        result = result[:40]
        return [list(s) for s in result]
