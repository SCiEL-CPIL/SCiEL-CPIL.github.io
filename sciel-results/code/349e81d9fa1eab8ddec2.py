import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Precompute neighbor bitmasks
    # mask[i] is an integer where bit j is 1 if word j is within radius of word i
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                j = i + diff * powers[pos]
                m |= (1 << j)
        masks[i] = m

    full_mask = (1 << total_space) - 1
    
    def greedy_cover(remaining_mask, current_set):
        """
        Greedily adds words to current_set to cover remaining_mask.
        Returns updated current_set and remaining_mask.
        """
        covered = 0
        for w in current_set:
            covered |= masks[w]
        
        rem = remaining_mask & ~covered
        
        # If nothing to cover, return
        if rem == 0:
            return current_set, 0
        
        # Find words that cover the most of 'rem'
        # Optimization: Only consider words that intersect 'rem'
        # Simple loop over all words might be slow if done many times, 
        # but total_space is 2187, which is small.
        
        while rem and time.time() < deadline - 5:
            best_gain = 0
            best_word = -1
            # We can optimize by checking only words not in current_set?
            # Or just all words.
            for w in range(total_space):
                if w in current_set:
                    continue
                gain = (masks[w] & rem).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            
            if best_word == -1:
                break
                
            current_set.add(best_word)
            rem &= ~masks[best_word]
            
        return current_set, rem

    # 1. Initial Greedy Solution
    S = set()
    rem = full_mask
    while rem and time.time() < deadline - 20:
        best_gain = 0
        best_word = -1
        for w in range(total_space):
            if w in S:
                continue
            gain = (masks[w] & rem).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_word = w
        
        if best_word == -1:
            break
            
        S.add(best_word)
        rem &= ~masks[best_word]
        
    best_size = len(S)
    best_S = set(S)
    
    # 2. Local Search / ILS
    # Helper: Check if S is a valid cover
    def is_valid_cover(S_set):
        covered = 0
        for w in S_set:
            covered |= masks[w]
        return (covered == full_mask)
        
    # Helper: Reduce S by removing redundant words
    def reduce_set(S_set):
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Iterate in random order to avoid bias
            indices = list(S_set)
            random.shuffle(indices)
            for w in indices:
                # Check if w is redundant
                # A word w is redundant if (Union of others) covers (mask[w])
                # i.e., mask[w] & ~(Union of others) == 0
                # Union of others = Full_Cover - mask[w] + (mask[w] & Union of others) ? No.
                # Let U = Union of masks of all w' in S_set \ {w}
                # w is redundant if mask[w] & ~U == 0
                
                U = 0
                for w2 in S_set:
                    if w2 == w:
                        continue
                    U |= masks[w2]
                
                if (masks[w] & ~U) == 0:
                    S_set.remove(w)
                    changed = True
                    break # Restart loop as set changed
        return S_set

    # Helper: Try to swap a word
    def try_swap(S_set):
        # Try to find a swap that allows further reduction
        # Or just a swap that maintains validity and changes structure
        if len(S_set) == 0:
            return False
            
        w_to_remove = random.choice(list(S_set))
        
        # Calculate what is lost if we remove w_to_remove
        U_others = 0
        for w in S_set:
            if w == w_to_remove:
                continue
            U_others |= masks[w]
            
        lost = masks[w_to_remove] & ~U_others
        
        if lost == 0:
            # Just remove
            S_set.remove(w_to_remove)
            return True
            
        # We need a word v such that mask[v] covers 'lost'
        # And v is not in S_set
        # And ideally, v is "efficient" (covers lots of others too, or just covers lost)
        
        # Find candidate v
        best_v = -1
        # Heuristic: pick v that covers 'lost' and has high overlap with existing coverage?
        # Or just any v that covers lost.
        # To make it a "swap", we add v and remove w_to_remove.
        # New set S' = S \ {w_to_remove} U {v}
        # S' is valid if mask[v] covers lost.
        
        # We want to diversify. Pick a random valid v.
        valid_vs = []
        for v in range(total_space):
            if v in S_set:
                continue
            if (masks[v] & lost) == lost:
                valid_vs.append(v)
                
        if valid_vs:
            v = random.choice(valid_vs)
            S_set.remove(w_to_remove)
            S_set.add(v)
            return True
            
        return False

    # Main ILS Loop
    current_S = set(best_S)
    
    while time.time() < deadline - 10:
        # Reduce
        current_S = reduce_set(current_S)
        
        if len(current_S) < best_size:
            best_size = len(current_S)
            best_S = set(current_S)
            
        # If we reached a very good size, we can try more aggressive swaps
        # Perturbation: Randomly break and re-greedy
        if random.random() < 0.3:
            # Remove random 5% of words
            num_remove = max(1, len(current_S) // 20)
            to_remove = random.sample(list(current_S), min(num_remove, len(current_S)))
            for w in to_remove:
                current_S.remove(w)
            
            # Re-greedy
            current_S, rem = greedy_cover(full_mask, current_S)
            
            if len(current_S) < best_size:
                best_size = len(current_S)
                best_S = set(current_S)
        else:
            # Local Swap Move
            # Try a few swaps
            for _ in range(10):
                if not try_swap(current_S):
                    break
            
            # Check if swap enabled a reduction
            current_S = reduce_set(current_S)
            
            if len(current_S) < best_size:
                best_size = len(current_S)
                best_S = set(current_S)

    # Final Polish
    best_S = reduce_set(best_S)
    
    if not is_valid_cover(best_S):
        # Fallback to greedy if something went wrong
        S = set()
        rem = full_mask
        while rem and time.time() < deadline - 1:
            best_gain = 0
            best_word = -1
            for w in range(total_space):
                if w in S:
                    continue
                gain = (masks[w] & rem).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word == -1: break
            S.add(best_word)
            rem &= ~masks[best_word]
        best_S = S

    return [int_to_tuple(v) for v in best_S]
