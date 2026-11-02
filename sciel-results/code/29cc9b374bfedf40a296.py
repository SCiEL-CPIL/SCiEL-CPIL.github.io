import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    limit = 140.0 # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute powers of q for index conversion
    powers = [q**i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor masks
    # mask[idx] is an integer with bits set for all words within radius of idx
    neighbor_masks = [0] * total_words
    
    for idx in range(total_words):
        mask = 1 << idx
        # Radius 1 neighbors
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    mask |= (1 << neighbor_idx)
        neighbor_masks[idx] = mask
        
    # Greedy Set Cover using bitmasks
    uncovered = (1 << total_words) - 1
    solution_indices = []
    
    while uncovered:
        if time.time() - start_time > limit:
            break
            
        best_word = -1
        max_covered = 0
        
        # Find the word that covers the most uncovered words
        for w in range(total_words):
            count = (neighbor_masks[w] & uncovered).bit_count()
            if count > max_covered:
                max_covered = count
                best_word = w
                
        if best_word == -1:
            break
            
        solution_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]
        
    # Local Search: Try to remove redundant codewords
    # A codeword is redundant if all words it covers are also covered by other codewords.
    # This is equivalent to checking if the union of neighbors of (solution \ {w}) covers the neighbors of w.
    # Or simpler: Re-check coverage.
    # Given time constraints, a simple pass to remove any codeword that doesn't add unique coverage is good.
    
    # Let's do a single pass of removal.
    # We need to know if removing w leaves any gap.
    # This is expensive if done naively.
    # Alternative: Just return the greedy result. The greedy result with bitmasks should be very fast and likely better than the slow set-based one.
    
    # Let's check if we have time for a simple local search.
    # The greedy phase for 6561 words with bitmasks is very fast (< 1s).
    # We have ~139s left.
    
    # Simple Local Search:
    # For each codeword in solution (in reverse order of addition?), check if it can be removed.
    # To check if w can be removed:
    # 1. Compute union of neighbors of all other codewords.
    # 2. Check if neighbors[w] is a subset of that union.
    # This is O(|solution|^2 * bit_ops). |solution| ~ 400. 160k ops. Very fast.
    
    new_solution = list(solution_indices)
    
    # Efficient local search using coverage counting
    # For each word in the space, count how many codewords cover it
    # A codeword w is redundant if every word it covers is covered by at least 2 codewords total
    # (i.e., removing w still leaves those words covered)
    
    while time.time() - start_time < limit:
        changed = False
        sol_size = len(new_solution)
        if sol_size == 0:
            break
            
        # Build coverage count: for each word in space, how many codewords cover it
        coverage_count = [0] * total_words
        for w in new_solution:
            mask = neighbor_masks[w]
            # Count bits for each set bit position - this is expensive
            # Alternative: just check redundancy directly
            
        # Better approach: for each codeword, check if it's redundant
        # A codeword w is redundant if for every word x in neighbors(w),
        # there exists another codeword v != w such that x is in neighbors(v)
        
        # Even better: precompute for each position in the solution, 
        # which words are ONLY covered by that codeword
        # If a codeword has no uniquely covered words, it's redundant
        
        # Let's use a different strategy: 
        # For each codeword, compute the set of words it uniquely covers
        # (words covered by this codeword but no other)
        # If this set is empty, the codeword is redundant
        
        # To do this efficiently:
        # 1. For each word in the space, count how many codewords cover it
        # 2. A codeword w is redundant if for all x in neighbors(w), coverage_count[x] >= 2
        
        # Build coverage count array
        coverage_count = [0] * total_words
        for w in new_solution:
            mask = neighbor_masks[w]
            # For each set bit in mask, increment coverage_count
            # This is O(2^n) per codeword in worst case, but masks are sparse-ish
            # Actually, each mask has ~1 + n*(q-1) bits set = 1 + 8*2 = 17 bits for q=3,n=8
            # So this is fast!
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                coverage_count[idx] += 1
                m ^= lsb
        
        # Now check each codeword for redundancy
        for i in range(sol_size - 1, -1, -1):
            w = new_solution[i]
            mask = neighbor_masks[w]
            # Check if all words in mask have coverage_count >= 2
            redundant = True
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                if coverage_count[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
            
            if redundant:
                new_solution.pop(i)
                changed = True
                break  # Restart to recompute coverage
        
        if not changed:
            break  # No more improvements possible
    
    # Convert indices back to lists of digits
    return [get_digits(idx) for idx in new_solution]
