"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct(q, n, radius):
    deadline = time.time() + 110.0
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all-zero word covers only itself (radius 0 case) or brute force if small
        # For robustness, return a simple valid code if possible, else empty (invalid)
        # Since task is specific to q=3, n=7, r=1, we focus optimization there.
        return []

    # Specific optimization for q=3, n=7, r=1
    # 1. Generate all possible "neighbors" for a codeword.
    # A codeword c covers c itself and all words at distance 1.
    # Total words covered by one codeword: 1 + n*(q-1) = 1 + 7*2 = 15.
    
    # Represent codewords as integers 0..3^7-1 for fast hashing? 
    # Or just use numpy arrays. 
    # Let's stick to lists of integers for the final output, but use numpy for computation.
    
    # Precompute the "cover sets" for every possible word in the space?
    # Space size is 3^7 = 2187. This is small enough to precompute coverage masks?
    # A bitset of size 2187 fits in 2187 bits ~ 274 bytes. 
    # We can represent the coverage of each of the 2187 potential codewords as a boolean array or integer bitmask.
    # Python integers are arbitrary precision, so we can use a single integer as a bitset for the 2187 words.
    
    # Map word (tuple of 7 ints) to index 0..2186
    # index = w[0]*3^6 + w[1]*3^5 + ... + w[6]
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute cover_mask[idx] for each idx in 0..2186
    # cover_mask[idx] is an integer where bit j is 1 if word j is within distance 'radius' of word idx.
    # Since radius=1, word j is covered if dist(j, idx) <= 1.
    
    all_words = [idx_to_word(i) for i in range(q**n)]
    
    cover_masks = [0] * (q**n)
    
    # Optimization: compute distances efficiently?
    # With 2187 words, pairwise distance calculation is 2187^2 * 7 ~ 33 million ops. 
    # In Python, this might be slow. Let's use numpy to compute distances for all pairs at once?
    # Or just loop? 2187 * 2187 = ~4.7M pairs. 
    # Let's try a vectorized approach for precomputation.
    
    words_arr = np.array(all_words) # shape (2187, 7)
    
    # Compute hamming distance matrix? 2187x2187 matrix of ints. 
    # dist[i,j] = sum(words_arr[i] != words_arr[j])
    # This creates a 2187x2187 array. 4.7M entries. 
    # diff = np.abs(words_arr[:, np.newaxis, :] - words_arr[np.newaxis, :, :])
    # dist = (diff != 0).sum(axis=2)
    # This is memory heavy: 2187*2187*7 booleans ~ 33MB. Acceptable.
    
    diff = np.abs(words_arr[:, np.newaxis, :] - words_arr[np.newaxis, :, :])
    dist_matrix = (diff != 0).sum(axis=2)
    
    # Now, for each i, cover_mask[i] has bits set for all j where dist_matrix[i,j] <= radius
    for i in range(q**n):
        mask = 0
        # We need to set bits for j where dist_matrix[i,j] <= radius
        # This loop is 2187 * 2187 ~ 4.7M iterations total. 
        # We can do it faster with numpy?
        # np.where(dist_matrix[i] <= radius)[0] gives indices.
        # Then set bits.
        covered_indices = np.where(dist_matrix[i] <= radius)[0]
        for j in covered_indices:
            mask |= (1 << j)
        cover_masks[i] = mask

    # Greedy Set Cover
    # Universe U = {0..2186}
    # Sets S_i = cover_masks[i]
    # Goal: min |C| such that union_{i in C} S_i = U
    
    uncovered = (1 << (q**n)) - 1
    selected = []
    
    # To speed up greedy, we can pre-sort or just pick max intersection each time.
    # 2187 sets. 
    # Iteration 1: pick set with max |S_i & uncovered|
    # Iteration 2: update uncovered, pick next max.
    # Number of sets needed ~ 180.
    # Each step requires checking 2187 masks against current uncovered.
    # 180 * 2187 ~ 400k operations. Very fast.
    
    # However, we want to beat 189. Pure greedy often gets stuck.
    # We will run greedy, then local search.
    
    # Initial Greedy
    current_uncovered = (1 << (q**n)) - 1
    greedy_selected = []
    
    # For efficiency, we can maintain a list of available sets.
    # Since 2187 is small, we can just loop all every time.
    
    while current_uncovered != 0 and time.time() < deadline - 10:
        best_set = -1
        best_count = -1
        # Randomize tie-breaking to escape symmetries?
        # Let's iterate all
        for i in range(q**n):
            # Skip if already selected? 
            # In set cover, we don't need to skip if we allow re-adding, but greedy typically doesn't re-add.
            # Actually, standard greedy: pick best, add to C, remove covered.
            # We don't need to exclude selected sets from consideration in subsequent steps? 
            # No, once selected, its contribution is fixed. We only care about *new* coverage.
            # But wait, if we pick i again, it adds nothing new. So we can skip selected sets.
            if i in greedy_selected:
                continue
            
            inter = current_uncovered & cover_masks[i]
            count = inter.bit_count()
            if count > best_count:
                best_count = count
                best_set = i
                if count == current_uncovered.bit_count():
                    break # Perfect cover found in one step? Unlikely.
        
        if best_set == -1:
            break
            
        greedy_selected.append(best_set)
        current_uncovered &= ~cover_masks[best_set]
        
    # Initial solution size
    current_solution = list(greedy_selected)
    
    # Local Search: 1-opt removal and swap
    # Try to remove any codeword if coverage remains intact.
    # Try to swap one codeword with another to allow removal of another?
    # Simpler: Just try removals repeatedly until no more can be removed.
    # Then try "swap": remove i, add j, then try to remove k?
    # Let's start with just removals.
    
    def is_valid_solution(solution):
        union_mask = 0
        for idx in solution:
            union_mask |= cover_masks[idx]
        return union_mask == ((1 << (q**n)) - 1)
    
    # Optimize: maintain union mask
    union_mask = 0
    for idx in current_solution:
        union_mask |= cover_masks[idx]
    
    full_mask = (1 << (q**n)) - 1
    
    # Iterative Removal
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try to remove each element
        # To make it faster, we can check if removing i leaves a gap.
        # Gap exists if (union_mask ^ cover_masks[i]) != full_mask? 
        # No. (union_mask & ~cover_masks[i]) == full_mask is false means some bit in full_mask was ONLY covered by i.
        # So condition to remove i: (union_mask & ~cover_masks[i]) == full_mask is FALSE?
        # Wait. If we remove i, new_union = union_mask & ~cover_masks[i] is wrong. 
        # new_union = union_mask. We need to know if any word w is covered ONLY by i.
        # This is hard to check directly from union_mask.
        # We need to know, for each word w, how many codewords cover it?
        # Or, simply: 
        # If (union_mask & ~cover_masks[i]) != full_mask, then removing i breaks coverage.
        # Let's verify.
        # Let U be full set. 
        # Let S be current union. S == U.
        # Let C_i be set covered by i.
        # If we remove i, new union S' = S \ C_i? No. S' = Union of all C_j for j != i.
        # S' = S & ~ (C_i \ (Union of others)).
        # It is easier to check: Does C_i contain any element that is not in (Union of others)?
        # Union of others = S & ~ (C_i \ (Union of others))... circular.
        
        # Alternative: 
        # Check if (S & ~C_i) == U is false?
        # S & ~C_i removes from S all elements in C_i.
        # If an element w is in U and in C_i, and NOT in any other C_j, then w is in S and in C_i.
        # Is w in S & ~C_i? No, because w is in C_i, so it's removed.
        # So if w is uniquely covered by i, then w is in S, w is in C_i, so w is NOT in S & ~C_i.
        # Thus S & ~C_i != U.
        # If all elements in C_i are also covered by some other C_j, then for any w in C_i, w is in some C_j (j!=i).
        # So w is in S (via C_j) and w is in C_i.
        # Wait. S & ~C_i contains elements in S that are NOT in C_i.
        # Elements in C_i that are covered by others are in S. Are they in S & ~C_i? No, they are in C_i.
        # So S & ~C_i only contains elements NOT covered by i.
        # This logic is flawed.
        
        # Correct Logic:
        # We want to know if U \ (Union_{j!=i} C_j) is empty.
        # Union_{j!=i} C_j = S \ (C_i \ Union_{j!=i} C_j).
        # This is getting complicated.
        
        # Simplest correct check for removal:
        # If we remove i, the new union is S'.
        # S' = S. But we lose the "backing" of i.
        # Actually, we can just recompute the union if we remove i? 
        # No, that's O(N).
        
        # Better check:
        # An element w is lost if w in C_i and w not in Union_{j!=i} C_j.
        # This is equivalent to: w in C_i and w in (S \ Union_{j!=i} C_j).
        # Note that S \ Union_{j!=i} C_j is the set of elements covered ONLY by i (among the current set).
        # Let's call this Unique_i.
        # If Unique_i is empty, we can remove i.
        
        # How to compute Unique_i efficiently?
        # We can maintain a "coverage count" array? 
        # Given 2187 words and ~180 codewords, we can maintain an array `counts` of size 2187.
        # counts[w] = number of codewords in current_solution that cover w.
        # Then i can be removed if for all w in C_i, counts[w] > 1.
        
        # Let's implement this.
        counts = np.zeros(q**n, dtype=np.int32)
        for idx in current_solution:
            # Add cover_mask[idx] to counts
            # This is slow if done bit by bit.
            # But we can do it once at the start of local search.
            pass
        
        # Re-initialize counts properly
        counts = np.zeros(q**n, dtype=np.int32)
        # We need to add 1 to all positions where cover_mask[idx] has a 1.
        # This is tricky with numpy on integers.
        # Alternative: Just iterate through the set bits of cover_mask[idx]?
        # Or, since n is small, we can just use the distance matrix?
        # dist_matrix[i, j] <= 1.
        # So for each selected codeword i, we increment counts[j] for all j where dist_matrix[i,j] <= 1.
        
        # Let's precompute the list of covered indices for each codeword.
        covered_indices_list = [np.where(dist_matrix[i] <= radius)[0] for i in range(q**n)]
        
        for idx in current_solution:
            counts[covered_indices_list[idx]] += 1
        
        # Now try removals
        while time.time() < deadline - 5:
            removed_any = False
            # Try to remove in random order to break symmetry?
            # Or just iterate.
            for pos in range(len(current_solution)):
                if time.time() > deadline - 5:
                    break
                idx = current_solution[pos]
                # Check if any word in covered_indices_list[idx] has count == 1
                # If yes, cannot remove.
                # If no, can remove.
                # Check: min(counts[covered_indices_list[idx]]) > 1
                if np.min(counts[covered_indices_list[idx]]) > 1:
                    # Remove idx
                    # Update counts
                    counts[covered_indices_list[idx]] -= 1
                    current_solution.pop(pos)
                    removed_any = True
                    # Restart loop as indices changed? Or continue?
                    # Continuing might be unsafe if we pop from middle. 
                    # Let's break and restart the while loop.
                    break
            if not removed_any:
                break
                
    # Now try swaps?
    # "Swap" usually means: Remove i, Add j, then try to remove k?
    # Or simply: Try to replace i with j if it helps?
    # Given time constraints, let's try a simple "Add then Remove" heuristic.
    # Pick a random word j not in solution. Add it. Then try to remove any redundant codeword (including j? No, j is new).
    # If we can remove some i (i != j), we have effectively swapped i for j (size same) or just added j (size +1).
    # We want to reduce size. So we want to add j and remove TWO codewords? Or add j and remove i where i was redundant?
    # If i was redundant, we would have removed it in the previous phase.
    # So we need to add j, which might make some i redundant, allowing removal of i. Net change 0.
    # To reduce size, we need to add j and remove i AND k?
    # Or, more likely, the greedy solution has "clumps" where 3 codewords cover an area that 2 could cover.
    # A 2-opt local search might be needed.
    
    # Let's try a simple random restart of the greedy phase if we have time?
    # Or just return the current best.
    
    # Convert indices back to words
    best_solution = current_solution
    best_size = len(best_solution)
    
    # If we have time, try a few random restarts of greedy with different tie-breaking
    # But 189 is the seed. We want < 189.
    # Greedy usually gets 183-186.
    
    # Let's do one more pass: try to swap any pair (i, j) in solution with (k, l) outside?
    # Too complex.
    
    # Let's just ensure we return a valid solution.
    # Verify validity
    final_union = 0
    for idx in best_solution:
        final_union |= cover_masks[idx]
    if final_union != full_mask:
        # Fallback to seed if invalid (should not happen)
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [idx_to_word(idx) for idx in best_solution]

# EVOLVE-BLOCK-END
