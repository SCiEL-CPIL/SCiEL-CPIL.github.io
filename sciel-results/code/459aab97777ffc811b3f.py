import time
import random

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses Bitset-based Iterated Local Search for q=3, n=6, r=1.
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

    # Precompute neighbors as bitmasks for speed
    # neighbor_mask[i] is a bitmask where bit j is 1 if word j is within radius of word i
    neighbor_masks = [0] * total_words
    for i in range(total_words):
        w = int_to_word(i, n, q)
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        mask |= (1 << neighbor_idx)
                w[pos] = original_val
        neighbor_masks[i] = mask

    # For the specific case q=3, n=6, r=1, use a specialized bitset ILS
    if q == 3 and n == 6 and radius == 1:
        # Initial construction: Direct sum of tetracode (81 words)
        # Tetracode: 9 words of length 4
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = set()
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    initial_codewords.add(idx)
        
        best_codewords = set(initial_codewords)
        best_size = len(best_codewords)
        
        # Helper to get the union of neighbor masks for a set of codewords
        def get_covered_mask(codewords):
            mask = 0
            for c in codewords:
                mask |= neighbor_masks[c]
            return mask
        
        # Helper to reduce a set of codewords by removing redundant ones
        def reduce_codewords(codewords):
            changed = True
            while changed:
                changed = False
                covered_mask = get_covered_mask(codewords)
                for c in list(codewords):
                    # Check if c is redundant
                    # c is redundant if all words covered by c are covered by codewords - {c}
                    # Let mask_others be the union of neighbors of all other codewords.
                    # If (mask_others & neighbor_masks[c]) == neighbor_masks[c], then c is redundant.
                    # To avoid recomputing mask_others from scratch for every c, we can compute
                    # total_mask = get_covered_mask(codewords)
                    # But we need to know if the *others* cover c's neighbors.
                    # This is equivalent to: Is there any word w in neighbors[c] such that w is NOT in neighbors[x] for any x != c?
                    # A faster check:
                    # Let M = OR of neighbor_masks[x] for x in codewords.
                    # This doesn't help directly because M includes c.
                    
                    # Efficient check:
                    # Compute mask_others = OR of neighbor_masks[x] for x in codewords if x != c.
                    # This is O(|C|) per c, so O(|C|^2) total. With |C|~80, 6400 ops, very fast.
                    mask_others = 0
                    for x in codewords:
                        if x != c:
                            mask_others |= neighbor_masks[x]
                    
                    if (mask_others & neighbor_masks[c]) == neighbor_masks[c]:
                        codewords.remove(c)
                        changed = True
                        break # Restart loop as set changed
            return codewords

        # Main ILS Loop
        current_codewords = set(best_codewords)
        
        while time.time() < deadline:
            # 1. Reduce
            current_codewords = reduce_codewords(current_codewords)
            
            if len(current_codewords) < best_size:
                best_size = len(current_codewords)
                best_codewords = set(current_codewords)
                # If we reached the known optimum (73), we can stop or continue to try for 72 (impossible)
                if best_size == 73:
                    # We found the best known. We can break early to save time, or try a bit more.
                    # Given the strict lower bound is 71, 73 is very hard to beat.
                    # Let's break to ensure we return a valid optimal-ish result quickly.
                    break

            if not current_codewords:
                break
            
            # 2. Perturbation
            # Strategy: Remove 1-3 random codewords
            k = random.randint(1, 3)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
            
            # 3. Repair (Greedy Add)
            # Find uncovered words
            covered_mask = get_covered_mask(current_codewords)
            # Uncovered words are those with bit 0 in covered_mask
            # We need to add codewords to cover them.
            
            # Instead of iterating all 729 candidates every time, we can be smarter,
            # but 729 is small.
            # We need to find a center that covers the most uncovered words.
            
            # Create a list of uncovered indices for faster checking?
            # Or just use bitmask operations.
            
            # Let's find the best center greedily.
            # We need to repeatedly pick the best center until all are covered.
            
            # To speed up, we can maintain a list of uncovered words?
            # For now, simple greedy:
            while True:
                if time.time() > deadline:
                    break
                
                # Check if all covered
                # Total mask is (1 << total_words) - 1
                total_mask = (1 << total_words) - 1
                if (covered_mask & total_mask) == total_mask:
                    break
                
                # Find best candidate
                best_center = -1
                best_count = -1
                
                # Iterate through all possible centers
                # Optimization: Only consider centers that cover at least one uncovered word?
                # All centers cover themselves. If a center is already covered, it might still cover others.
                
                # We can speed this up by precomputing for each word, the list of neighbors?
                # We have neighbor_masks.
                
                # Let's just loop all 729.
                for candidate in range(total_words):
                    # If candidate is already in code, adding it does nothing (we use a set)
                    # But we removed some, so it might be available.
                    # If candidate is in current_codewords, skip.
                    if candidate in current_codewords:
                        continue
                    
                    # Count how many *uncovered* words this candidate covers.
                    # Uncovered words are ~covered_mask & total_mask
                    # New coverage = neighbor_masks[candidate] & (~covered_mask & total_mask)
                    # Count bits in that intersection.
                    
                    # Python int.bit_count() is fast.
                    new_cover = neighbor_masks[candidate] & (~covered_mask & total_mask)
                    count = new_cover.bit_count()
                    
                    if count > best_count:
                        best_count = count
                        best_center = candidate
                
                if best_center == -1:
                    break
                
                current_codewords.add(best_center)
                covered_mask |= neighbor_masks[best_center]
                
        result = [int_to_word(c, n, q) for c in best_codewords]
        return result
    else:
        # Fallback for other parameters: Simple Greedy
        # Precompute neighbors as lists for generic case if needed, but we already have masks.
        # We can use the same bitset logic but without the specific initialization.
        
        # Greedy Construction
        codewords = set()
        covered_mask = 0
        total_mask = (1 << total_words) - 1
        
        while covered_mask != total_mask:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                new_cover = neighbor_masks[candidate] & (~covered_mask & total_mask)
                count = new_cover.bit_count()
                if count > best_count:
                    best_count = count
                    best_center = candidate
            
            if best_center == -1:
                break
            
            codewords.add(best_center)
            covered_mask |= neighbor_masks[best_center]
            
        # Local search reduction
        def reduce_codewords_generic(codewords):
            changed = True
            while changed:
                changed = False
                for c in list(codewords):
                    mask_others = 0
                    for x in codewords:
                        if x != c:
                            mask_others |= neighbor_masks[x]
                    if (mask_others & neighbor_masks[c]) == neighbor_masks[c]:
                        codewords.remove(c)
                        changed = True
                        break
            return codewords
        
        codewords = reduce_codewords_generic(codewords)
        
        result = [int_to_word(c, n, q) for c in codewords]
        return result
