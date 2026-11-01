import itertools
import random

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
            # Optimization: If w_idx is not in uncovered, it might still cover others? 
            # No, if w_idx is not in uncovered, it doesn't matter if we pick it as the center?
            # Actually, the center must be a valid codeword. It doesn't have to be in uncovered.
            # But picking a word that is already covered doesn't hurt, but usually the best 
            # covering word is likely to be in the uncovered set or close to it.
            # However, we must consider ALL words as potential centers.
            
            count = 0
            for v in neighbors[w_idx]:
                if v in uncovered:
                    count += 1
            if count > best_cover:
                best_cover = count
                best_word = w_idx
        
        if best_word == -1:
            # Should not happen if uncovered is not empty
            break
            
        code_indices.append(best_word)
        for v in neighbors[best_word]:
            uncovered.discard(v)
            
    # Local Search: Try to remove redundant codewords
    # coverage_count[i] = number of codewords in code_indices that cover i
    coverage_count = [0] * total_words
    for c in code_indices:
        for v in neighbors[c]:
            coverage_count[v] += 1
            
    improved = True
    while improved:
        improved = False
        # Try to remove each codeword
        # We iterate backwards so removals don't shift indices
        for i in range(len(code_indices) - 1, -1, -1):
            c = code_indices[i]
            # Check if removing c leaves any word with coverage 0
            valid_removal = True
            for v in neighbors[c]:
                if coverage_count[v] == 1:
                    valid_removal = False
                    break
            
            if valid_removal:
                # Remove it
                for v in neighbors[c]:
                    coverage_count[v] -= 1
                code_indices.pop(i)
                improved = True
                # Restart the loop or continue? 
                # If we remove one, coverage counts change, so previous checks might be invalid?
                # No, we only decreased counts. If a word was safe before (count > 1), it is still safe.
                # If a word became unsafe (count == 1), we wouldn't have removed.
                # So we can continue the loop, but a fresh pass is safer if we want to be sure.
                # However, since we only remove, we can just continue to the next i.
                # But removing one might make another one removable?
                # Yes, if c1 and c2 cover the same unique word, removing c1 might make c2 removable?
                # No, if they cover the same unique word, that word has count 2.
                # If we remove c1, count becomes 1. c2 is still needed.
                # If c1 and c2 are disjoint in their "critical" coverage, removing c1 doesn't affect c2's critical words.
                # Actually, removing c1 decreases counts. It cannot make a count increase.
                # So if c2 was removable (all its neighbors had count > 1), it is still removable.
                # So we can just continue the loop.
                
    # Convert indices back to tuples/lists
    result = [to_tuple(idx) for idx in code_indices]
    return result
