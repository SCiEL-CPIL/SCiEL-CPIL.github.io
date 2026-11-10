"""Program to find a large pairwise non-coprime subset of [1,255] containing 255."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    Strategy: Use bitmask-based compatibility checking for speed,
    then greedy construction + local search (1-swap and 2-swap).
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for 90s limit
    
    n = 255
    # Candidates must share a factor with 255 (3, 5, or 17)
    candidates = [x for x in range(1, n+1) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    num_cands = len(candidates)
    
    # Map candidate value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build compatibility bitmask: compat[i] is a bitmask where bit j is set if gcd(candidates[i], candidates[j]) > 1
    compat = [0] * num_cands
    for i in range(num_cands):
        for j in range(i+1, num_cands):
            if math.gcd(candidates[i], candidates[j]) > 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Index of 255 in candidates
    idx_255 = val_to_idx[255]
    
    # Precompute degree for each candidate
    degrees = [bin(compat[i]).count('1') for i in range(num_cands)]
    
    def is_compatible_with_set(v_idx, S_bitmask):
        """Check if candidate v_idx is compatible with all elements in S (bitmask)."""
        # v_idx must be compatible with every element in S
        # S contains v_idx? No, we check before adding.
        # For each bit set in S, check if compat[v_idx] has that bit set.
        # This is equivalent to: (S & ~compat[v_idx]) == 0, but only for bits in S
        # Actually: all bits in S must be set in compat[v_idx]
        return (S_bitmask & ~compat[v_idx]) == 0
    
    def set_size(S_bitmask):
        return bin(S_bitmask).count('1')
    
    def set_to_list(S_bitmask):
        return [candidates[i] for i in range(num_cands) if (S_bitmask >> i) & 1]
    
    def list_to_set(lst):
        return set(lst)
    
    best_size = 0
    best_bitmask = 1 << idx_255
    
    # Try multiple greedy strategies
    strategies = []
    
    # Strategy 1: Greedy by degree (descending)
    # Strategy 2: Greedy by degree (ascending) 
    # Strategy 3: Random restarts with greedy
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if trial == 0:
            # Pure degree descending
            order = sorted(range(num_cands), key=lambda i: (-degrees[i], i))
        elif trial == 1:
            # Degree ascending
            order = sorted(range(num_cands), key=lambda i: (degrees[i], i))
        else:
            # Random perturbation of degree ordering
            import random
            random.seed(trial)
            order = list(range(num_cands))
            # Sort by degree with random tie-breaking
            order.sort(key=lambda i: (-degrees[i], random.random()))
        
        S_bitmask = 1 << idx_255
        S_set = {idx_255}
        
        for v_idx in order:
            if time.time() > deadline:
                break
            if v_idx == idx_255:
                continue
            if (S_bitmask >> v_idx) & 1:
                continue
            if is_compatible_with_set(v_idx, S_bitmask):
                S_bitmask |= (1 << v_idx)
                S_set.add(v_idx)
        
        if set_size(S_bitmask) > best_size:
            best_size = set_size(S_bitmask)
            best_bitmask = S_bitmask
        
        # If we've found a good set, do local search
        if best_size >= 85 and trial % 5 == 0:
            # Local search: 1-swap improvement
            improved = True
            while improved and time.time() < deadline:
                improved = False
                S_list = [candidates[i] for i in range(num_cands) if (best_bitmask >> i) & 1]
                S_indices = [i for i in range(num_cands) if (best_bitmask >> i) & 1]
                
                for u_idx in S_indices:
                    if time.time() > deadline:
                        break
                    # Try to remove u_idx and add w_idx
                    for w_idx in range(num_cands):
                        if time.time() > deadline:
                            break
                        if w_idx == u_idx:
                            continue
                        if (best_bitmask >> w_idx) & 1:
                            continue
                        # Check if w is compatible with all in S \ {u}
                        S_without_u = best_bitmask & ~(1 << u_idx)
                        if is_compatible_with_set(w_idx, S_without_u):
                            # Also need w to be compatible with 255 (already ensured by candidate filter)
                            new_bitmask = best_bitmask | (1 << w_idx) & ~(1 << u_idx)
                            # Wait, that's not right. We want to replace u with w.
                            new_bitmask = (best_bitmask & ~(1 << u_idx)) | (1 << w_idx)
                            # Verify 255 is still in the set
                            if (new_bitmask >> idx_255) & 1:
                                best_bitmask = new_bitmask
                                improved = True
                                break
                    if improved:
                        break
    
    # Final local search with 2-swap
    if time.time() < deadline:
        S_indices = [i for i in range(num_cands) if (best_bitmask >> i) & 1]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_indices = [i for i in range(num_cands) if (best_bitmask >> i) & 1]
            for i in range(len(S_indices)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(S_indices)):
                    if time.time() > deadline:
                        break
                    u_idx = S_indices[i]
                    v_idx = S_indices[j]
                    # Try to remove u and v, add w and x
                    S_without_uv = best_bitmask & ~(1 << u_idx) & ~(1 << v_idx)
                    for w_idx in range(num_cands):
                        if time.time() > deadline:
                            break
                        if w_idx == u_idx or w_idx == v_idx:
                            continue
                        if (best_bitmask >> w_idx) & 1:
                            continue
                        if not is_compatible_with_set(w_idx, S_without_uv):
                            continue
                        # Now find x compatible with S_without_uv ∪ {w}
                        S_with_w = S_without_uv | (1 << w_idx)
                        for x_idx in range(num_cands):
                            if time.time() > deadline:
                                break
                            if x_idx == u_idx or x_idx == v_idx or x_idx == w_idx:
                                continue
                            if (best_bitmask >> x_idx) & 1:
                                continue
                            if is_compatible_with_set(x_idx, S_with_w):
                                new_bitmask = S_with_w | (1 << x_idx)
                                if (new_bitmask >> idx_255) & 1:
                                    best_bitmask = new_bitmask
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Convert best_bitmask to list of values
    result = [candidates[i] for i in range(num_cands) if (best_bitmask >> i) & 1]
    
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
    
    return result
# EVOLVE-BLOCK-END
