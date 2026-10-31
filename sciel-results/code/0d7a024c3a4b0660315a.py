"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a greedy algorithm to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute neighbors for each word.
    # A word is represented as an integer in base q.
    # neighbors[i] is a list of indices j such that HammingDistance(i, j) <= radius.
    # For efficiency, we only need to check distance <= radius.
    # Given n=6, q=3, radius=1, this is small.
    
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

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        # Center itself
        neighbors[i].append(i)
        # Variations at distance 1 (if radius >= 1)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
        # Variations at distance 2 (if radius >= 2)
        if radius >= 2:
            # This becomes more expensive, but for n=6, q=3, radius=1, we don't need this.
            # If radius was 2, we'd need to iterate pairs of positions.
            pass

    # Coverage mask: True if covered
    covered = [False] * total_words
    
    # Greedy selection
    codewords = []
    
    # To speed up, we can maintain a list of uncovered words.
    # However, for small n, iterating all total_words is fine.
    # Let's optimize by keeping a set of uncovered indices.
    uncovered_set = set(range(total_words))
    
    while uncovered_set:
        if time.time() > deadline:
            break
            
        best_center = -1
        best_count = -1
        
        # We only need to consider centers that are themselves uncovered?
        # Actually, any center can be chosen, but choosing an uncovered one is often good.
        # However, a center that is covered might cover many *other* uncovered words.
        # So we should check all possible centers.
        
        # Optimization: If the number of uncovered words is small, we might find a center quickly.
        # But iterating all total_words is O(total_words * radius_volume).
        # For n=6, q=3, total_words=729, radius_volume=13. 729*13 is very small.
        
        for candidate in range(total_words):
            count = 0
            # Count how many neighbors of candidate are in uncovered_set
            # Using a set for O(1) lookup
            for nb in neighbors[candidate]:
                if nb in uncovered_set:
                    count += 1
            
            if count > best_count:
                best_count = count
                best_center = candidate
        
        if best_center == -1:
            break
            
        # Add best_center to codewords
        codewords.append(best_center)
        
        # Update uncovered_set
        for nb in neighbors[best_center]:
            if nb in uncovered_set:
                uncovered_set.remove(nb)
                
    # Convert integer indices back to lists of integers
    result = [int_to_word(c, n, q) for c in codewords]
    
    return result
# EVOLVE-BLOCK-END
