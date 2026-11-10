import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 148  # Safety margin before 150s kill
    
    total_words = q ** n
    
    # For the specific task (q=3, n=6, radius=1), total_words = 729
    # This is small enough for efficient bitmask or numpy operations.
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute Hamming distances or use a fast method to determine coverage
    # For radius 1, a word covers itself and words differing in exactly 1 position.
    # Number of neighbors for radius 1: 1 + n*(q-1)
    # For q=3, n=6: 1 + 6*2 = 13 words per codeword.
    
    # Strategy: Greedy Set Cover
    # 1. Create a boolean array `covered` of size total_words, initially False.
    # 2. Repeat until all covered:
    #    a. For each candidate word i not yet in code, calculate how many NEW words it covers.
    #    b. Pick the candidate with max gain.
    #    c. Mark those words as covered.
    
    # To speed up gain calculation, we can precompute the list of words covered by each word i.
    # Let `neighbors[i]` be a list of indices of words within radius of word i.
    
    # Optimization: Instead of checking distance for every pair (O(N^2 * n)),
    # we can generate neighbors directly.
    # For radius r, the number of neighbors is sum_{k=0}^r C(n,k) * (q-1)^k.
    # For q=3, n=6, r=1: 1 + 6*2 = 13. Very small.
    
    # Let's build the neighbor lists.
    # Map word tuple to index.
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i, w in enumerate(all_words):
        # Center
        neigh_set = {i}
        
        if radius >= 1:
            # Change 1 position
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    # Create new word
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
        
        if radius >= 2:
            # Change 2 positions
            # This gets larger. For n=6, C(6,2)=15 pairs. (q-1)^2 = 4. Total 60.
            # 1 + 12 + 60 = 73 neighbors. Still manageable.
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = w[pos1]
                    v2 = w[pos2]
                    for nv1 in range(q):
                        if nv1 == v1: continue
                        for nv2 in range(q):
                            if nv2 == v2: continue
                            new_w_list = list(w)
                            new_w_list[pos1] = nv1
                            new_w_list[pos2] = nv2
                            new_w = tuple(new_w_list)
                            neigh_set.add(word_to_idx[new_w])
        
        if radius >= 3:
             # For n=6, r=3: C(6,3)*8 = 20*8 = 160. Total 1+12+60+160 = 233.
             # Still fine for 729 words.
             for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for pos3 in range(pos2 + 1, n):
                        v1 = w[pos1]; v2 = w[pos2]; v3 = w[pos3]
                        for nv1 in range(q):
                            if nv1 == v1: continue
                            for nv2 in range(q):
                                if nv2 == v2: continue
                                for nv3 in range(q):
                                    if nv3 == v3: continue
                                    new_w_list = list(w)
                                    new_w_list[pos1] = nv1
                                    new_w_list[pos2] = nv2
                                    new_w_list[pos3] = nv3
                                    new_w = tuple(new_w_list)
                                    neigh_set.add(word_to_idx[new_w])
        
        neighbors.append(neigh_set)
        
    # Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    # Precompute neighbor lists as numpy arrays for faster intersection?
    # Or just use sets. Sets are fast for small sizes.
    # Let's use a list of sets.
    
    # To speed up the "max gain" selection:
    # Gain of i = len(neighbors[i] & ~covered)
    # We can maintain a count of uncovered neighbors for each i.
    # When we add a codeword c, we mark its neighbors as covered.
    # For each i, if any of its neighbors becomes covered, its gain decreases.
    # This is the "incremental" greedy.
    
    # Let's initialize `uncovered_count[i]` = len(neighbors[i])
    # When we pick c, for every i, if neighbors[i] intersects with newly covered words, update count.
    # This is O(N * |neighbors|) per step.
    # N=729, |neighbors|~13-233. Steps ~73.
    # 729 * 233 * 73 ~ 12 million operations. Very fast in Python if optimized, or numpy.
    
    # Let's stick to a simpler approach first: recompute gain each step.
    # Steps ~ 73.
    # Each step: iterate 729 candidates.
    # For each candidate, compute gain.
    # If we use bitmasks, gain calculation is very fast.
    
    # Let's convert neighbors to bitmasks.
    masks = [0] * total_words
    for i in range(total_words):
        m = 0
        for idx in neighbors[i]:
            m |= (1 << idx)
        masks[i] = m
            
    uncovered_mask = (1 << total_words) - 1
    
    while uncovered_mask:
        best_candidate = -1
        max_gain = -1
        
        # Optimization: Only consider candidates that have at least 1 uncovered neighbor.
        # We can check if (masks[i] & uncovered_mask) != 0
        
        for i in range(total_words):
            # If i is already in code, skip? 
            # Actually, we can re-pick? No, set cover doesn't allow re-picking to add more coverage 
            # because the union is monotonic. Once picked, its contribution is fixed.
            # But we haven't marked "used" yet. We just need to pick distinct words.
            # If we pick i, it covers some. Next time, picking i again adds 0.
            # So we can just pick max gain. If gain is 0, we stop (or break if uncovered_mask != 0 implies error).
            
            # Calculate gain
            new_covered = masks[i] & uncovered_mask
            gain = new_covered.bit_count()
            
            if gain > max_gain:
                max_gain = gain
                best_candidate = i
                if max_gain == total_words.bit_count() - (uncovered_mask.bit_count()): # Max possible
                    # Actually max gain is just len(uncovered).
                    # If we cover all remaining, we can break early?
                    # But there might be another candidate with same gain?
                    # Random tie-breaking is better.
                    pass
        
        if best_candidate == -1 or max_gain == 0:
            # Should not happen if code is valid covering
            # But if we run out of time or logic error, break
            break
            
        code_indices.append(best_candidate)
        uncovered_mask &= ~masks[best_candidate]
        
        if time.time() > deadline:
            break
            
    # Post-processing: Remove redundant codewords
    # Greedy set cover often includes redundant words.
    # We can try to remove each word and see if coverage is still complete.
    # This is O(K^2 * cost_of_check).
    # K ~ 73. K^2 ~ 5000.
    # Check: union of masks of remaining words == target_mask.
    
    if time.time() < deadline - 5:
        # Try to remove redundancies
        changed = True
        while changed:
            changed = False
            for i in range(len(code_indices)):
                # Temporarily remove i
                # Check if union of others covers everything
                # Efficient way: 
                # Let current_union = target_mask (since we know it covers all)
                # We want to know if union(others) == target_mask.
                # This is equivalent to: Is there any word covered ONLY by code_indices[i]?
                # Let unique_to_i = masks[code_indices[i]] & ~union(others)
                # If unique_to_i == 0, then i is redundant.
                
                # Calculate union of others
                # We can maintain a running union? 
                # Simpler: Just compute union of all masks except i.
                # With bitmasks, this is fast.
                
                union_others = 0
                for j in range(len(code_indices)):
                    if j == i:
                        continue
                    union_others |= masks[code_indices[j]]
                    
                if (masks[code_indices[i]] & ~union_others) == 0:
                    # Redundant
                    code_indices.pop(i)
                    changed = True
                    break # Restart loop from beginning as indices shifted
            
            if time.time() > deadline:
                break

    # Convert indices back to words
    result = [all_words[i] for i in code_indices]
    
    # Final validation check (optional, but good for safety)
    # If result is empty or doesn't cover, return a known valid construction?
    # For q=3, n=6, radius=1, the direct product of Tetracode (size 9, length 4, radius 1) 
    # and something? No, Tetracode is perfect.
    # A simple valid code: All words. Size 729.
    # Or the 81-word code from inspiration 3.
    # Let's ensure we return something valid if the greedy failed or timed out badly.
    
    # Check validity
    if len(result) == 0:
        return all_words # Worst case
        
    # Quick check if result covers all
    final_union = 0
    for i in code_indices:
        final_union |= masks[i]
        
    if final_union != (1 << total_words) - 1:
        # Fallback to a known valid code if greedy failed
        # For q=3, n=6, radius=1:
        # Use the 81-word code: Tetracode x {0,1,2}^2 ?
        # Tetracode is length 4, size 9, radius 1.
        # Product with length 2, size 3, radius 0 (all 9 words)?
        # Distance in product is sum.
        # If we use Tetracode (radius 1) and Full Code (radius 0), total radius 1.
        # Size 9 * 9 = 81.
        if q == 3 and n == 6 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            tail_space = list(itertools.product(range(3), repeat=2))
            fallback = [h + list(tail) for h in tetracode for tail in tail_space]
            return fallback
        else:
            return all_words
            
    return result
