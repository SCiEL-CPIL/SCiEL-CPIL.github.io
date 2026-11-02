import itertools
import time
import random

def construct(q, n, radius):
    # Parameters are fixed by the task, but we handle them generally if possible.
    # For q=3, n=7, radius=1, the space is 2187.
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Only optimized for the specific task parameters
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed behavior for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # 1. Precompute all words and their spheres as bitmasks
    # Total words: 3^7 = 2187
    total_words = q ** n
    # We need bitmasks of size 2187. Python ints can handle this.
    
    # Helper to convert list of digits to index
    def to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    # Helper to convert index to list of digits
    def from_idx(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Generate all words
    all_words = [from_idx(i) for i in range(total_words)]
    
    # Precompute sphere masks for each word
    # sphere_mask[c] is an integer bitmask where bit j is 1 if word j is within radius 1 of word c
    sphere_masks = [0] * total_words
    for c in range(total_words):
        cw = all_words[c]
        mask = 1 << c # The word itself
        for i in range(n):
            for delta in [-1, 1]:
                new_d = (cw[i] + delta) % q
                if new_d != cw[i]:
                    new_word = cw[:]
                    new_word[i] = new_d
                    idx = to_idx(new_word)
                    mask |= (1 << idx)
        sphere_masks[c] = mask

    # 2. Greedy Set Cover
    # Uncovered mask starts with all bits set
    full_mask = (1 << total_words) - 1
    uncovered = full_mask
    
    selected_codewords = []
    # We will store indices of selected codewords
    selected_indices = []
    
    # To speed up, we can precompute the count of uncovered bits for each candidate?
    # That's expensive to update. Instead, we just iterate.
    # Since 2187 is small, iterating 2187 times per step is fine.
    # Number of steps is roughly 2187 / 15 ~ 150.
    # 150 * 2187 ~ 328k operations. Very fast.
    
    while uncovered != 0:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = -1
        
        # Iterate through all potential codewords
        # Optimization: We can skip codewords that are already selected?
        # Yes, adding a duplicate doesn't help.
        # We can also skip if the intersection with uncovered is 0.
        
        for c in range(total_words):
            if c in selected_indices: # O(N) check, might be slow if N is large. 
                # Better to maintain a boolean array or set
                pass
            
            # Calculate intersection
            inter = uncovered & sphere_masks[c]
            if inter == 0:
                continue
                
            # Count bits
            # Python int.bit_count() is fast in 3.10+
            # If not available, use bin(x).count('1') or a lookup table for bytes.
            # Assuming modern Python or standard library support.
            cnt = inter.bit_count() if hasattr(int, 'bit_count') else bin(inter).count('1')
            
            if cnt > best_count:
                best_count = cnt
                best_idx = c
                
        if best_idx == -1:
            # Should not happen if uncovered != 0 and spheres cover space
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~sphere_masks[best_idx]
        
        # Check deadline again
        if time.time() > deadline:
            break

    # 3. Local Search / Removal Phase
    # Try to remove codewords that are redundant.
    # A codeword is redundant if its sphere is fully covered by the union of other selected spheres.
    
    # We can do a few passes of removal.
    # Let's create a set of selected indices for fast lookup
    selected_set = set(selected_indices)
    
    # To check if a codeword i is redundant:
    # Union of spheres of all j in selected_set \ {i} must cover sphere_masks[i]
    # This is expensive to compute from scratch each time.
    
    # Alternative: Random restarts of the greedy algorithm with different orderings?
    # Or just a simple local search: 
    # For each selected codeword, try removing it. If valid, keep it removed.
    # Then try adding a new codeword to see if we can reduce size? No, adding increases size.
    # The goal is to minimize size. So we only try to remove.
    
    # Efficient redundancy check:
    # Maintain a count of how many selected codewords cover each word in the space.
    # coverage_count[word_idx]
    
    # Initialize coverage counts
    # This requires iterating over all selected codewords and their spheres.
    # Size ~ 200. Sphere size 15. 3000 ops. Fast.
    
    def compute_coverage(indices):
        counts = [0] * total_words
        for idx in indices:
            mask = sphere_masks[idx]
            # Iterate through set bits in mask
            m = mask
            while m:
                # Get least significant set bit
                lsb = m & -m
                pos = lsb.bit_length() - 1
                counts[pos] += 1
                m ^= lsb
        return counts

    # Current indices
    current_indices = list(selected_indices)
    
    # Try to remove redundant codewords
    # We do this in a loop, removing one at a time if possible.
    # Order of removal can matter. We can try random orders or just iterate.
    
    # To make it more robust, we can try multiple random permutations of the removal attempt.
    # But let's stick to a simple pass first.
    
    # Let's implement a function to check if a specific index is redundant given the set.
    # This is O(N * SphereSize) if we recompute coverage, or O(SphereSize) if we maintain it.
    
    # Let's maintain coverage counts.
    coverage = compute_coverage(current_indices)
    
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        # Try to remove any codeword where all words in its sphere have coverage > 1
        # If we remove c, coverage of words in sphere(c) decreases by 1.
        # So c is removable if for all w in sphere(c), coverage[w] > 1.
        
        for idx in current_indices:
            mask = sphere_masks[idx]
            m = mask
            can_remove = True
            while m:
                lsb = m & -m
                pos = lsb.bit_length() - 1
                if coverage[pos] <= 1:
                    can_remove = False
                    break
                m ^= lsb
            
            if can_remove:
                # Remove idx
                current_indices.remove(idx)
                # Update coverage
                m = mask
                while m:
                    lsb = m & -m
                    pos = lsb.bit_length() - 1
                    coverage[pos] -= 1
                    m ^= lsb
                removed_any = True
                # Restart the loop to check others again, as removals can enable further removals
                break # Break to restart the for loop with updated state

    # If we have time, we can try a "swap" move? 
    # Replace 1 codeword with 2? No, that increases size.
    # Replace 2 codewords with 1? 
    # If we remove 2, we might uncover words. If those words can be covered by 1 new codeword, 
    # we reduced size by 1.
    # This is a 2-exchange local search.
    
    # Given time constraints, the greedy + removal is a solid improvement over the fixed 243.
    # Greedy usually gets < 220. Removal might get < 210.
    
    # Convert indices back to lists
    result = [all_words[i] for i in current_indices]
    
    # Ensure distinctness (should be by construction)
    # Ensure validity (should be by construction)
    
    return result

# Keep the function name and signature
# The prompt asks to keep function names and signatures.
# The current program has `construct(q, n, radius)`.
