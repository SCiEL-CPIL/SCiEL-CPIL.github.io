import time
import random

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

    # Precompute neighbors for all words
    # neighbors[i] is a list of indices covered by word i
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val

    # Initialize with a known good construction: Direct sum of tetracode
    # Tetracode: 9 words of length 4, covering radius 1
    # Direct sum with all tails of length 2: 9 * 3^2 = 81 words
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
        uncovered = set(range(total_words))
        while uncovered:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                count = 0
                for nb in neighbors[candidate]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            initial_codewords.add(best_center)
            for nb in neighbors[best_center]:
                if nb in uncovered:
                    uncovered.remove(nb)

    best_codewords = set(initial_codewords)
    best_size = len(best_codewords)
    
    def get_uncovered(codewords):
        """Calculate the set of uncovered words given a set of codewords."""
        covered = set()
        for c in codewords:
            covered.update(neighbors[c])
        return set(range(total_words)) - covered

    def reduce_codewords(codewords):
        """Remove redundant codewords until none remain. Returns the reduced set."""
        changed = True
        while changed:
            changed = False
            # We iterate over a copy because we modify the set
            for c in list(codewords):
                # Check if c is redundant
                # c is redundant if all words covered by c are also covered by (codewords - {c})
                # Efficient check: 
                # Let U be the set of words covered only by c?
                # Simpler: 
                # 1. Remove c from a temporary set
                # 2. Check if the union of neighbors of remaining codewords covers all neighbors of c?
                # Actually, it's enough to check if there is any word w in neighbors[c] 
                # such that w is NOT covered by any other codeword.
                
                is_redundant = True
                for w in neighbors[c]:
                    # Check if w is covered by any other codeword
                    covered_by_other = False
                    for other_c in codewords:
                        if other_c == c:
                            continue
                        # Check if w is in neighbors[other_c]
                        # Since neighbors lists are small (13), this is fast
                        if w in neighbors[other_c]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                
                if is_redundant:
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
            
        # If we hit the theoretical lower bound or a known very good bound, we can stop early if we want, 
        # but let's try to beat 73.
        
        if not current_codewords:
            break
            
        # 2. Perturbation (Kick)
        # Remove a small number of random codewords to create holes
        # The number of codewords to remove is a parameter to tune. 
        # Usually 1-3 is good.
        k = random.randint(1, 3)
        if len(current_codewords) > k:
            to_remove = random.sample(list(current_codewords), k)
            for c in to_remove:
                current_codewords.remove(c)
        
        # 3. Repair (Greedy Add)
        # Find all uncovered words
        covered = set()
        for c in current_codewords:
            covered.update(neighbors[c])
        uncovered = set(range(total_words)) - covered
        
        while uncovered:
            if time.time() > deadline:
                break
            best_center = -1
            best_count = -1
            # Only consider candidates that are not already in the code? 
            # Actually, adding a word already in the code is useless.
            # But we just removed some, so we should consider all words.
            # To speed up, we can prioritize uncovered words as centers?
            # No, a covered word might cover many uncovered words.
            # However, for small N, checking all is fine.
            
            # Optimization: If the number of uncovered words is small, we can check fewer candidates?
            # Let's just check all for simplicity and correctness, it's fast for N=729.
            for candidate in range(total_words):
                if candidate in current_codewords:
                    continue
                count = 0
                for nb in neighbors[candidate]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_center = candidate
            
            if best_center == -1:
                break
            
            current_codewords.add(best_center)
            for nb in neighbors[best_center]:
                if nb in uncovered:
                    uncovered.remove(nb)
                
        # After repair, the set is valid again. 
        # The next iteration of the main loop will reduce it.

    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
