import itertools
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # But we should handle general inputs if possible, or at least not crash.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies q=3, n=7, radius=1.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple construction for other parameters
        # Direct product of q-ary codes?
        # For now, just return a trivial valid code (all words) if not the specific case
        # to ensure validity, though score will be bad.
        # Or use the tetracode logic if n>=4 and q==3.
        if q == 3 and n >= 4 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(7,1)
    total_words = 3 ** n
    words = [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Precompute neighbors for each word
    # neighbor_indices[i] is a list of indices of words within distance <= radius
    neighbor_indices = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        # Find all words within distance 1
        # Distance 0: itself
        neighbor_indices[i].append(i)
        # Distance 1: change one position
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    # Create new word
                    new_w = w.copy()
                    new_w[pos] = new_val
                    # Find index of new_w
                    # We can map tuple to index
                    idx = int(''.join(map(str, new_w)), 3)
                    neighbor_indices[i].append(idx)
                    
    # Greedy Set Cover
    uncovered = set(range(total_words))
    code = []
    
    # To speed up, we can maintain a count of how many times each word is covered by the current code?
    # No, standard greedy picks the set that covers the most *uncovered* elements.
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(total_words):
            # Calculate gain: number of neighbors of i that are in uncovered
            gain = 0
            for nb in neighbor_indices[i]:
                if nb in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                # Optimization: if gain == len(uncovered), we can't do better
                # But gain is max 15. len(uncovered) starts at 2187.
        
        if best_idx == -1:
            break # Should not happen
            
        code.append(best_idx)
        # Remove covered words from uncovered
        for nb in neighbor_indices[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
                
    # Convert indices back to words
    result = [words[i] for i in code]
    
    # Pruning Phase
    # Try to remove words from the code if coverage is maintained
    # We need to check if removing a word leaves any word uncovered.
    # Efficient check: 
    # 1. Count coverage of each word by the current code.
    # 2. If any word has count == 1, it depends on the specific codeword covering it.
    # 3. If we remove a codeword c, all words that were ONLY covered by c become uncovered.
    
    # Let's implement a simpler pruning:
    # Iterate through the code. For each candidate for removal, check if the remaining code covers the space.
    # This is O(|code| * |code| * n) which is small.
    
    def is_valid_cover(codeword_indices):
        covered = [False] * total_words
        for idx in codeword_indices:
            for nb in neighbor_indices[idx]:
                covered[nb] = True
        return all(covered)
        
    # Greedy pruning
    i = 0
    while i < len(code):
        # Try to remove code[i]
        temp_code = code[:i] + code[i+1:]
        if is_valid_cover(temp_code):
            code = temp_code
            # Do not increment i, check the next element at the same position
        else:
            i += 1
            
    return [words[i] for i in code]
