"""Optimized program: fast paths for small k, backtracking for larger k."""
# EVOLVE-BLOCK-START
import time

def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)

# Build global smooth list up to 128 once
_SMOOTH = _gen_smooth(128)
_VAL_INDEX = {v: i for i, v in enumerate(_SMOOTH)}
_SMOOTH_SET = set(_SMOOTH)

def construct():
    start = time.time()
    DEADLINE = 85.0
    
    answers = [None] * 129
    smooth_vals = _SMOOTH
    n_smooth = len(smooth_vals)
    
    # Precompute incompatibility masks
    inc_mask = [0] * n_smooth
    for i in range(n_smooth):
        vi = smooth_vals[i]
        for j in range(i + 1, n_smooth):
            vj = smooth_vals[j]
            if vi % vj == 0 or vj % vi == 0:
                inc_mask[i] |= (1 << j)
                inc_mask[j] |= (1 << i)
    
    # Precompute index list for fast access
    # We will iterate over values, so we need quick index lookup
    
    for n in range(1, 129):
        if time.time() - start > DEADLINE:
            break
            
        found = None
        
        # k=1
        if n in _VAL_INDEX:
            found = [n]
        else:
            # k=2
            # a + b = n, a < b
            # a must be smooth, b = n - a must be smooth
            # and a, b must be compatible
            found = _find_k2(n, smooth_vals, inc_mask, _VAL_INDEX)
            if not found:
                # k=3
                # a + b + c = n, a < b < c
                found = _find_k3(n, smooth_vals, inc_mask, _VAL_INDEX)
                if not found:
                    # k >= 4: use backtracking
                    for k in range(4, 31):
                        if k > n: break
                        found = _find_sum_fast(smooth_vals, n, k, inc_mask, _VAL_INDEX)
                        if found: break
                        
        answers[n] = found

    # Fallback for any missing (should be rare with optimized search)
    for n in range(1, 129):
        if answers[n] is None:
            answers[n] = _greedy_fill(n, smooth_vals, inc_mask, _VAL_INDEX)

    return answers[1:129]

def _find_k2(n, smooth_vals, inc_mask, val_index):
    """Find 2 distinct smooth numbers summing to n, compatible."""
    # To avoid a == b, we can just check all a, if b = n-a is smooth and b != a and compatible.
    # Since we want minimal k, any valid pair works.
    # We can iterate a in smooth_vals.
    for a in smooth_vals:
        if a >= n: break
        b = n - a
        if b <= a: continue # Ensure a < b to avoid duplicates and self-check
        if b in _SMOOTH_SET:
            idx_a = val_index[a]
            idx_b = val_index[b]
            if not (inc_mask[idx_a] & (1 << idx_b)):
                return [a, b]
    return None

def _find_k3(n, smooth_vals, inc_mask, val_index):
    """Find 3 distinct smooth numbers summing to n, compatible."""
    # a < b < c
    # a + b + c = n
    # c = n - a - b
    # c must be smooth, c > b, and compatible with a, b.
    
    # To speed up, we can limit a and b.
    # a < n/3
    # b < (n - a) / 2
    
    for i_a, a in enumerate(smooth_vals):
        if a * 3 >= n: break # Since a < b < c, a must be < n/3
        
        # b must be > a
        for i_b in range(i_a + 1, len(smooth_vals)):
            b = smooth_vals[i_b]
            if a + b >= n: break # c must be > b > 0
            c = n - a - b
            if c <= b: continue # Ensure c > b
            
            if c in _SMOOTH_SET:
                idx_c = val_index[c]
                # Check compatibility
                # a and b are compatible?
                if inc_mask[i_a] & (1 << i_b): continue
                # a and c are compatible?
                if inc_mask[i_a] & (1 << idx_c): continue
                # b and c are compatible?
                if inc_mask[i_b] & (1 << idx_c): continue
                
                return [a, b, c]
    return None

def _find_sum_fast(smooth_vals, n, k, inc_mask, val_index):
    """Find k distinct smooth values summing to n, pairwise div-free."""
    # Filter candidates <= n
    candidates = [v for v in smooth_vals if v <= n]
    if len(candidates) < k:
        return None
    
    # Map to indices
    cand_indices = [val_index[v] for v in candidates]
    # Sort descending for pruning
    cand_indices.sort(key=lambda i: smooth_vals[i], reverse=True)
    
    best = None
    chosen = []
    chosen_mask = 0
    
    def backtrack(start, remaining_k, remaining_sum):
        nonlocal best, chosen, chosen_mask
        if best is not None:
            return True
        if remaining_k == 0:
            if remaining_sum == 0:
                best = [smooth_vals[i] for i in chosen]
                return True
            return False
        if remaining_k > remaining_sum:
            return False
        if start + remaining_k > len(cand_indices):
            return False
        
        # Pruning: Max possible sum with remaining_k items from start
        # We can precompute suffix max sums? Or just use the first item * k as a loose bound?
        # Loose bound: remaining_sum <= remaining_k * smooth_vals[cand_indices[start]]
        if remaining_sum > remaining_k * smooth_vals[cand_indices[start]]:
            return False
            
        for pos in range(start, len(cand_indices)):
            idx = cand_indices[pos]
            v = smooth_vals[idx]
            if v > remaining_sum:
                continue
            
            if chosen_mask & inc_mask[idx]:
                continue
            
            chosen.append(idx)
            chosen_mask |= (1 << idx)
            if backtrack(pos + 1, remaining_k - 1, remaining_sum - v):
                return True
            chosen.pop()
            chosen_mask ^= (1 << idx)
        
        return False
    
    backtrack(0, k, n)
    return best

def _greedy_fill(n, smooth_vals, inc_mask, val_index):
    """Fallback fill."""
    if n in val_index:
        return [n]
    for k in range(2, 10):
        found = _find_sum_fast(smooth_vals, n, k, inc_mask, val_index)
        if found:
            return found
    return [1] if n >= 1 else [1] # Should not happen
# EVOLVE-BLOCK-END
