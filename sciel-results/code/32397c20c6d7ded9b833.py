import itertools
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # But we should handle general inputs if possible, or at least not crash.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies q=3, n=7, radius=1.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple construction for other parameters
        # Direct product of q-ary codes?
        # For now, just return a trivial valid code (all words) if not the specific case
        # to ensure validity, though score will be bad.
        # Or use the tetracode logic if n>=4 and q==3.
        if q == 3 and n >= 4 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(7,1)
    total_words = 3 ** n
    words = [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Precompute neighbor masks using Python integers as bitsets
    # Bit i is set if word i is in the set
    neighbor_masks = [0] * total_words
    for i, w in enumerate(words):
        mask = 1 << i # Distance 0
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = int(''.join(map(str, new_w)), 3)
                    mask |= (1 << idx)
        neighbor_masks[i] = mask

    # Greedy Set Cover with Bitsets
    full_mask = (1 << total_words) - 1
    uncovered = full_mask
    code_indices = []
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        # Iterate through all words to find the one covering most uncovered
        for i in range(total_words):
            gain = bin(neighbor_masks[i] & uncovered).count('1')
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break # Should not happen
            
        code_indices.append(best_idx)
        uncovered &= ~neighbor_masks[best_idx]
        
    # Pruning Phase
    # Aggressive pruning: restart from 0 after any removal to catch new opportunities
    final_code = code_indices[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(final_code)):
            # Try removing final_code[i]
            candidate_code = final_code[:i] + final_code[i+1:]
            # Check if candidate_code covers everything
            union_mask = 0
            for idx in candidate_code:
                union_mask |= neighbor_masks[idx]
            if union_mask == full_mask:
                final_code = candidate_code
                changed = True
                break
                
    return [words[i] for i in final_code]
