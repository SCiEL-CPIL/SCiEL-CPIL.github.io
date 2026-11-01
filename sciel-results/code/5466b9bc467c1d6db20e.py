import itertools
import random
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1.
    # We assume the evaluator passes these specific parameters as per the task description.
    # If other parameters are passed, we fall back to a simple construction.
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Direct sum construction (same as seed)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial case for small n
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(7,1)
    # Total words = 3^7 = 2187
    total_words = 3 ** n
    # Map tuple to index
    def to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    def to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices within Hamming distance <= radius
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = to_tuple(idx)
        # Self
        neighbors[idx].append(idx)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    neighbors[idx].append(to_index(tuple(new_w)))
    
    # Greedy Set Cover
    uncovered = set(range(total_words))
    code_indices = []
    
    while uncovered:
        best_cover = 0
        best_word = -1
        # Iterate through all possible words to find the one covering most uncovered
        for w_idx in range(total_words):
            count = 0
            for v in neighbors[w_idx]:
                if v in uncovered:
                    count += 1
            if count > best_cover:
                best_cover = count
                best_word = w_idx
        
        if best_word == -1:
            break
            
        code_indices.append(best_word)
        for v in neighbors[best_word]:
            uncovered.discard(v)
            
    # Local Search: Try to remove redundant codewords
    coverage_count = [0] * total_words
    for c in code_indices:
        for v in neighbors[c]:
            coverage_count[v] += 1
            
    improved = True
    while improved:
        improved = False
        for i in range(len(code_indices) - 1, -1, -1):
            c = code_indices[i]
            valid_removal = True
            for v in neighbors[c]:
                if coverage_count[v] == 1:
                    valid_removal = False
                    break
            
            if valid_removal:
                for v in neighbors[c]:
                    coverage_count[v] -= 1
                code_indices.pop(i)
                improved = True
                
    # Bit-parallel Local Search: Try to swap codewords to reduce size
    # We will try to find a pair (remove c1, add c2) such that size decreases or stays same but we accept if size decreases.
    # Actually, we want to minimize size. So we look for (remove c1, add c2) where c2 is NOT in code_indices
    # and the set of words uniquely covered by c1 (among current code) is a subset of words covered by c2.
    # If we remove c1, words uniquely covered by c1 become uncovered.
    # If we add c2, it covers its neighbors.
    # So we need: Unique(c1) subset of Neighbors(c2).
    # Also, c2 must not be in the current code.
    
    # Precompute bitmasks for neighbors?
    # 2187 bits is ~34 64-bit integers. Python ints are arbitrary precision, so we can use a single int per word.
    # This is efficient enough.
    
    neighbor_masks = [0] * total_words
    for idx in range(total_words):
        mask = 0
        for v in neighbors[idx]:
            mask |= (1 << v)
        neighbor_masks[idx] = mask
    
    code_set = set(code_indices)
    
    # Calculate unique coverage mask for each codeword in the current code
    # unique_mask[c] = bits that are covered ONLY by c in the current code
    # We can compute this by:
    # 1. For each c in code, start with neighbor_mask[c]
    # 2. Subtract intersection with all other c' in code? No.
    # 3. Better: For each word v, count how many c in code cover it. If count == 1, let c be that one, set bit in unique_mask[c].
    
    # This is O(N * |Code| * |Neighbors|) which might be slow if done naively.
    # Let's use the coverage_count array we already have logic for, but we need to identify WHICH code covers it if count==1.
    
    # Let's recompute coverage counts and unique owners.
    coverage_count = [0] * total_words
    unique_owner = [-1] * total_words
    
    for c in code_indices:
        for v in neighbors[c]:
            coverage_count[v] += 1
    
    for c in code_indices:
        for v in neighbors[c]:
            if coverage_count[v] == 1:
                unique_owner[v] = c
    
    # Now, for each c in code, let's see if we can swap it with some c2 not in code.
    # Condition: unique_mask[c] is a subset of neighbor_mask[c2].
    # Also, we want to ensure that adding c2 doesn't break anything else?
    # No, adding c2 only adds coverage. Removing c1 only removes coverage.
    # The only risk is if c1 covers some word that NO OTHER code covers (unique to c1), and c2 doesn't cover it.
    # That is exactly the condition: unique_mask[c1] subset of neighbor_mask[c2].
    
    # We can precompute unique_mask for each c in code as an integer.
    unique_masks = [0] * total_words
    for v in range(total_words):
        if unique_owner[v] != -1:
            c = unique_owner[v]
            unique_masks[c] |= (1 << v)
    
    # Now search for swaps
    # To speed up, we can iterate through c in code, and for each c, check potential c2.
    # How many c2 to check? All 2187 words not in code.
    # 186 * 2187 * (bit operations)
    # Bit operations on 2187-bit ints are fast in Python.
    
    # We can optimize by noting that c2 must cover unique_mask[c].
    # So we can filter c2 candidates.
    
    # Let's do a few passes of this swap.
    # Time limit check: 120s.
    start_time = time.time()
    
    swapped = True
    while swapped and (time.time() - start_time) < 110:
        swapped = False
        # Iterate through current code
        # We need to be careful with indices if we modify the list.
        # Let's collect valid swaps first, then apply them? Or apply one at a time and restart?
        # Applying one at a time is safer.
        
        for i in range(len(code_indices)):
            c1 = code_indices[i]
            umask = unique_masks[c1]
            if umask == 0:
                # c1 is redundant? No, if umask is 0, it means c1 has no unique words.
                # Then we can just remove c1?
                # Wait, the previous local search already removed all redundant codewords (umask == 0).
                # So umask should not be 0 for any c1 in code_indices after the first pass.
                # But if we add a new codeword c2, it might make c1 redundant?
                # Yes. So we should check for redundancy again.
                # Let's handle redundancy first in the inner loop?
                # Or just let the swap logic handle it?
                # If umask == 0, we can remove c1. That reduces size by 1.
                # Let's do that immediately.
                code_indices.pop(i)
                code_set.remove(c1)
                swapped = True
                # Recalculate unique masks?
                # Removing c1 decreases coverage counts.
                # Words that were covered by c1 and no one else? None (umask was 0).
                # Words covered by c1 and others? Their count decreases.
                # Could a word become unique to another code?
                # Yes. E.g. word w covered by c1 and c2. Count 2.
                # Remove c1. Count 1. Unique to c2.
                # So unique masks change.
                # We need to recalculate unique masks after any modification.
                # This is expensive.
                # Let's break and recalculate?
                # Or just break out of the loop and recalculate in the next while iteration.
                break
        
        if swapped:
            # Recalculate coverage and unique masks
            coverage_count = [0] * total_words
            for c in code_indices:
                for v in neighbors[c]:
                    coverage_count[v] += 1
            
            unique_owner = [-1] * total_words
            for c in code_indices:
                for v in neighbors[c]:
                    if coverage_count[v] == 1:
                        unique_owner[v] = c
            
            unique_masks = [0] * total_words
            for v in range(total_words):
                if unique_owner[v] != -1:
                    c = unique_owner[v]
                    unique_masks[c] |= (1 << v)
            continue
        
        # No redundancy found. Try swaps.
        for i in range(len(code_indices)):
            c1 = code_indices[i]
            umask = unique_masks[c1]
            if umask == 0:
                # Shouldn't happen if redundancy check passed, but just in case
                code_indices.pop(i)
                code_set.remove(c1)
                swapped = True
                break
            
            # Find c2 not in code_set such that umask is subset of neighbor_mask[c2]
            # And we want to pick the best c2?
            # Ideally, c2 covers many other words that are currently "tightly" covered?
            # Or just any c2 that satisfies the condition.
            # Let's iterate through all c2.
            # Optimization: c2 must cover all bits in umask.
            # We can precompute for each bit position, which c2 cover it?
            # That's essentially the reverse neighbor map.
            
            # Simple approach: iterate c2 from 0 to total_words-1
            # Skip if c2 in code_set
            # Check if (umask & ~neighbor_mask[c2]) == 0
            
            # To speed up, we can check if c2 covers the "hardest" bit of umask first?
            # Or just iterate. 2187 iterations is small.
            
            for c2 in range(total_words):
                if c2 in code_set:
                    continue
                # Check if c2 covers all unique words of c1
                if (umask & ~neighbor_masks[c2]) == 0:
                    # Valid swap
                    # Replace c1 with c2
                    code_indices[i] = c2
                    code_set.remove(c1)
                    code_set.add(c2)
                    swapped = True
                    break
            
            if swapped:
                # Recalculate
                coverage_count = [0] * total_words
                for c in code_indices:
                    for v in neighbors[c]:
                        coverage_count[v] += 1
                
                unique_owner = [-1] * total_words
                for c in code_indices:
                    for v in neighbors[c]:
                        if coverage_count[v] == 1:
                            unique_owner[v] = c
                
                unique_masks = [0] * total_words
                for v in range(total_words):
                    if unique_owner[v] != -1:
                        c = unique_owner[v]
                        unique_masks[c] |= (1 << v)
                break
    
    # Convert indices back to tuples/lists
    result = [to_tuple(idx) for idx in code_indices]
    return result
