import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses Iterated Local Search (Greedy + Perturbation + Repair) to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbor bitmasks directly (skip intermediate list)
    neighbor_masks = [0] * total_words
    for i in range(total_words):
        mask = 1 << i  # self
        if radius >= 1:
            w = int_to_word(i, n, q)
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        mask |= (1 << neighbor_idx)
                w[pos] = original_val
        neighbor_masks[i] = mask
    
    # Precompute: for each word w, which codeword indices cover w
    # Actually, we'll compute coverage on the fly using masks

    # Initialize with a known good construction: Direct sum of tetracode
    initial_codewords = set()
    if q == 3 and n == 6 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    initial_codewords.add(idx)
    else:
        # Fallback: Greedy construction from scratch
        uncovered_mask = (1 << total_words) - 1
        while uncovered_mask:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                count = (uncovered_mask & neighbor_masks[candidate]).bit_count()
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            initial_codewords.add(best_center)
            uncovered_mask &= ~neighbor_masks[best_center]

    best_codewords = set(initial_codewords)
    best_size = len(best_codewords)
    
    def get_coverage_mask(codewords):
        """Calculate the coverage bitmask given a set of codewords."""
        mask = 0
        for c in codewords:
            mask |= neighbor_masks[c]
        return mask

    def reduce_codewords_bitmask(codewords, max_iterations=100):
        """Remove redundant codewords using bitmask coverage. Returns the reduced set."""
        changed = True
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            # Compute total coverage once
            total_cover = get_coverage_mask(codewords)
            # Try to remove multiple codewords in one pass
            to_remove = []
            for c in list(codewords):
                # Coverage without c: total_cover minus only what c uniquely covers
                # c is redundant if every neighbor of c is covered by at least one other codeword
                # i.e., (total_cover & ~neighbor_masks[c]) & neighbor_masks[c] == neighbor_masks[c]
                # which simplifies to: neighbor_masks[c] & ~(total_cover & ~neighbor_masks[c]) == 0
                # i.e., neighbor_masks[c] | total_cover == total_cover is NOT the right test
                # Correct: all bits in neighbor_masks[c] must be in total_cover when we remove c
                # That means: neighbor_masks[c] must be subset of (total_cover & ~neighbor_masks[c]) OR
                # more precisely: for each bit b in neighbor_masks[c], b must be in total_cover
                # AND b must be covered by some codeword other than c.
                # Since total_cover includes c's contribution, we need:
                # For each position p in neighbor_masks[c], there exists c' != c with p in neighbor_masks[c']
                # Equivalent: (total_cover & ~neighbor_masks[c]) must cover neighbor_masks[c]
                # i.e., (total_cover & ~neighbor_masks[c]) & neighbor_masks[c] == neighbor_masks[c]
                cover_without_c = total_cover & ~neighbor_masks[c]
                if (cover_without_c & neighbor_masks[c]) == neighbor_masks[c]:
                    to_remove.append(c)
            for c in to_remove:
                codewords.remove(c)
                changed = True
            if to_remove:
                # Continue loop to check again with updated set
                continue
        return codewords

    def greedy_repair(codewords):
        """Add codewords greedily to cover all uncovered words. Returns modified set."""
        covered = get_coverage_mask(codewords)
        uncovered = ((1 << total_words) - 1) & ~covered
        if not uncovered:
            return codewords
        # Precompute coverage counts for all candidates against current uncovered set
        # Use a list of (count, candidate) sorted by count descending for efficient selection
        while uncovered:
            if time.time() > deadline:
                break
            # Find best candidate by computing overlap with uncovered
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                # Use bit_count() which is much faster than bin().count('1')
                count = (uncovered & neighbor_masks[candidate]).bit_count()
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            codewords.add(best_center)
            uncovered &= ~neighbor_masks[best_center]
        return codewords

    # Main ILS Loop
    current_codewords = set(best_codewords)
    codeword_list = list(best_codewords)
    all_words = list(range(total_words))
    
    while time.time() < deadline:
        # 1. Reduce
        current_codewords = reduce_codewords_bitmask(current_codewords)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            
        if not current_codewords:
            break
            
        # 2. Perturbation (Kick)
        # Randomly choose between removal and swap
        perturbation_type = random.random()
        if perturbation_type < 0.4:
            # Remove 1-3 random codewords
            k = random.randint(1, 3)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
        elif perturbation_type < 0.7:
            # Swap: remove one, add one random
            if len(current_codewords) > 1:
                to_remove = random.choice(list(current_codewords))
                current_codewords.remove(to_remove)
                # Add a random word not in codewords - use rejection sampling
                to_add = random.randint(0, total_words - 1)
                while to_add in current_codewords:
                    to_add = random.randint(0, total_words - 1)
                current_codewords.add(to_add)
        else:
            # Remove 2-4 random codewords (larger kick)
            k = random.randint(2, 4)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
        
        # 3. Repair (Greedy Add)
        current_codewords = greedy_repair(current_codewords)
        
        # After repair, the set is valid again.
        # The next iteration of the main loop will reduce it.

    # Final polish: try harder to reduce the best solution
    # Try multiple reduction passes with different orderings
    final_codewords = set(best_codewords)
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        # Shuffle the order of checking for redundancy
        shuffled = list(final_codewords)
        random.shuffle(shuffled)
        # Compute total coverage once per pass
        total_cover = get_coverage_mask(final_codewords)
        to_remove = []
        for c in shuffled:
            if c not in final_codewords:
                continue
            cover_without_c = total_cover & ~neighbor_masks[c]
            if (cover_without_c & neighbor_masks[c]) == neighbor_masks[c]:
                to_remove.append(c)
        for c in to_remove:
            final_codewords.remove(c)
        if not to_remove:
            break
        if len(final_codewords) < best_size:
            best_size = len(final_codewords)
            best_codewords = set(final_codewords)

    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
