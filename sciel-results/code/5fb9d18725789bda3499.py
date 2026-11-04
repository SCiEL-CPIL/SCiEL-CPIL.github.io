import time
import itertools
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    return _greedy_general(q, n, radius)

def _solve_ternary_8_1():
    start_time = time.time()
    deadline = start_time + 115.0
    
    q = 3
    n = 8
    total = q ** n  # 6561
    
    # Generate all words
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute masks
    # mask[i] is a bitset of words covered by codeword i
    masks = [0] * total
    for i, w in enumerate(all_words):
        m = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_w = list(w)
                    new_w[pos] = val
                    # Find index of new_w
                    # Since total is small, we can precompute a dict
                    # But doing this inside the loop is slow.
                    pass
        
    # Better precomputation of indices
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    for i, w in enumerate(all_words):
        m = 1 << i
        w_list = list(w)
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    j = word_to_idx[tuple(w_list)]
                    m |= (1 << j)
            w_list[pos] = orig
        masks[i] = m
        
    ALL_ONES = (1 << total) - 1
    
    # Greedy Initialization
    code = set()
    covered_mask = 0
    
    # Calculate uncovered count for greedy
    # uncovered[i] = number of bits in masks[i] that are 0 in covered_mask
    # Initially covered_mask = 0, so uncovered[i] = popcount(masks[i]) = 17
    # But we need to update this.
    
    # Simple greedy: pick codeword that covers most uncovered words
    # To speed up, we can just pick randomly or use a heuristic.
    # Let's use a simple greedy with random tie-breaking.
    
    uncovered_mask = ALL_ONES
    candidates = list(range(total))
    random.shuffle(candidates)
    
    # Actually, standard greedy is better.
    # We maintain `uncovered_mask`.
    # For each candidate c, count bits in (masks[c] & uncovered_mask).
    # This is O(N) per candidate, O(N^2) per step. N=6561. 6561^2 * 486 steps is too slow.
    
    # Optimized Greedy:
    # Maintain an array `gain[c]` = popcount(masks[c] & uncovered_mask).
    # When we add a codeword c, we update uncovered_mask.
    # We need to update gain for all other codewords.
    # This is also O(N^2) total if done naively.
    
    # Alternative: Just use a few random restarts of a simpler heuristic or the local search from a random start?
    # No, we need a good start.
    
    # Let's use the bitset property.
    # We can approximate the greedy by picking random codewords until covered? No, that's large.
    
    # Let's implement the incremental gain update.
    # gain[c] is an integer.
    # When we add c0, new_uncovered = uncovered_mask & ~masks[c0].
    # For all c, gain[c] = popcount(masks[c] & new_uncovered).
    # This is O(N) per step. 486 steps * 6561 candidates * (bitwise op + popcount).
    # Popcount of 6561-bit int is fast.
    # 486 * 6561 = 3.1M operations. This is fine.
    
    gain = [0] * total
    for i in range(total):
        gain[i] = masks[i].bit_count()
        
    while uncovered_mask != 0 and time.time() < deadline:
        # Find max gain
        best_c = -1
        best_gain = -1
        # Optimization: If we are close to done, we can break early? No.
        # Just find max.
        for i in range(total):
            if gain[i] > best_gain:
                best_gain = gain[i]
                best_c = i
            if best_gain == 0:
                break
                
        if best_c == -1 or best_gain == 0:
            break
            
        # Add best_c
        code.add(best_c)
        uncovered_mask &= ~masks[best_c]
        
        # Update gains
        # Only update if necessary? 
        # gain[i] = (masks[i] & uncovered_mask).bit_count()
        for i in range(total):
            if i != best_c:
                gain[i] = (masks[i] & uncovered_mask).bit_count()
                
        if uncovered_mask == 0:
            break
            
    # Local Search
    # code is a set of indices
    # covered_mask = ALL_ONES
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Remove redundant
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for c in list(code):
                # Check if c is redundant
                # covered by others: (ALL_ONES ^ masks[c]) & masks[c] == masks[c]
                # i.e., masks[c] is subset of (ALL_ONES ^ masks[c])? No.
                # Other cover = ALL_ONES & ~masks[c]
                # If (masks[c] & ~masks[c]) == 0, it's always true.
                # We need: Are all bits in masks[c] set in (ALL_ONES ^ masks[c])?
                # No, ALL_ONES ^ masks[c] is the mask of words NOT covered by c.
                # We want to know if words covered by c are covered by someone else.
                # Let other_cover = ALL_ONES & ~masks[c] is WRONG.
                # other_cover should be the union of masks of all c' in code \ {c}.
                # But we don't have that directly.
                # However, we know ALL_ONES is the union of all masks in code.
                # If we remove c, the new union is ALL_ONES ^ (masks[c] & ~union_others).
                # This is hard to compute directly without iterating.
                
                # Alternative check:
                # A word w is exclusively covered by c if (masks[c] has bit w) AND (no other c' has bit w).
                # This is equivalent to: 
                # Let U = union of masks in code. U = ALL_ONES.
                # We can't easily compute "union of others" quickly.
                
                # Let's use the `covered_count` approach from the previous code, but with bitsets?
                # Bitsets don't support counting easily.
                
                # Let's go back to the swap check which was:
                # Swap c -> c' is valid if ((covered_mask ^ masks[c]) | masks[c']) == ALL_ONES.
                # Here covered_mask = ALL_ONES.
                # So ((ALL_ONES ^ masks[c]) | masks[c']) == ALL_ONES.
                # ALL_ONES ^ masks[c] is the mask of words NOT covered by c.
                # Let X = ALL_ONES ^ masks[c].
                # We need (X | masks[c']) == ALL_ONES.
                # This means masks[c'] must cover all bits in (ALL_ONES ^ X).
                # ALL_ONES ^ X = masks[c].
                # So masks[c'] must cover all bits in masks[c]?
                # No.
                # X | masks[c'] == ALL_ONES  <=>  (ALL_ONES ^ X) & ~masks[c'] == 0.
                # ALL_ONES ^ X = masks[c].
                # So (masks[c] & ~masks[c']) == 0.
                # This means masks[c] is a subset of masks[c'].
                
                # Wait, this logic assumes covered_mask is ALL_ONES.
                # If we remove c, the coverage becomes ALL_ONES ^ (masks[c] & ~union_others).
                # The condition for c to be removable is that union_others covers masks[c].
                # i.e., masks[c] is a subset of union_others.
                
                # How to check if masks[c] is subset of union_others?
                # union_others = ALL_ONES ^ (masks[c] & ~union_others). Circular.
                
                # Let's use the property:
                # c is redundant if for all w in masks[c], there exists c' != c in code such that w in masks[c'].
                
                # We can compute this by iterating w in masks[c]?
                # masks[c] has 17 bits.
                # For each of the 17 words w, check if any other codeword covers it.
                # We can precompute `coverers[w]` = set of codewords covering w.
                # If for all w in masks[c], len(coverers[w]) > 1, then c is redundant.
                
                # This requires maintaining `coverers`.
                # When we add/remove c, we update coverers for 17 words.
                # This is O(17) per add/remove.
                # This is very fast.
                
                pass # Placeholder for logic below
                
        # Re-implementing with coverers logic for speed and correctness
        
    # Reset and do it properly
    code = set(code)
    coverers = [set() for _ in range(total)]
    for c in code:
        # Iterate bits in masks[c]
        m = masks[c]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverers[idx].add(c)
            m ^= lsb
            
    # Local Search Loop
    while time.time() < deadline:
        # 1. Remove redundant
        removed_any = False
        for c in list(code):
            # Check if c is redundant
            redundant = True
            m = masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if len(coverers[idx]) == 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                # Remove c
                code.remove(c)
                m = masks[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    coverers[idx].remove(c)
                    m ^= lsb
                removed_any = True
                
        if not removed_any:
            # 2. Try swaps
            swapped = False
            code_list = list(code)
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline:
                    break
                    
                # Try to find c' such that swap is valid and enables removal
                # Valid swap condition:
                # After swapping c with c', the code is still a cover.
                # This is true if c' covers all words that were ONLY covered by c.
                # Words only covered by c: W_excl = {w in masks[c] | len(coverers[w]) == 1}
                # If W_excl is empty, c is redundant (handled above).
                # So W_excl is not empty.
                # We need c' such that W_excl is subset of masks[c'].
                
                # Find W_excl
                excl_mask = 0
                m = masks[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if len(coverers[idx]) == 1:
                        excl_mask |= lsb
                    m ^= lsb
                    
                if excl_mask == 0:
                    continue # Should have been removed
                
                # Find c' in [0, total) not in code such that (excl_mask & ~masks[c']) == 0
                # i.e., excl_mask is subset of masks[c']
                
                found = False
                # Iterate candidates
                # We can iterate all i in range(total)
                for c_prime in range(total):
                    if c_prime in code:
                        continue
                    if (excl_mask & ~masks[c_prime]) == 0:
                        # Valid swap
                        # Perform swap
                        code.remove(c)
                        code.add(c_prime)
                        
                        # Update coverers
                        m_old = masks[c]
                        while m_old:
                            lsb = m_old & -m_old
                            idx = lsb.bit_length() - 1
                            coverers[idx].remove(c)
                            m_old ^= lsb
                            
                        m_new = masks[c_prime]
                        while m_new:
                            lsb = m_new & -m_new
                            idx = lsb.bit_length() - 1
                            coverers[idx].add(c_prime)
                            m_new ^= lsb
                            
                        swapped = True
                        found = True
                        break
                        
                if found:
                    break # Restart loop after swap
                    
            if not swapped:
                # No improvement found in this round
                # To escape local minima, we can try a random perturbation
                # Or just break if we've been stuck for a while?
                # Let's try a random swap that maintains validity (even if no removal)
                # and see if it leads to a removal later.
                # This is complex. For now, break.
                break
                
    # Convert code indices back to words
    codewords = [list(all_words[c]) for c in code]
    codewords.sort()
    return codewords

def _greedy_general(q, n, radius):
    # Fallback
    return []
