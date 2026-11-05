import itertools
import numpy as np

def construct(q, n, radius):
    # We assume q=3, n=7, radius=1 based on the task description.
    # However, the function signature is general. 
    # The task says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # But the specific problem is K_3(7,1). 
    # If q, n, radius are different, the greedy approach still works, but performance might vary.
    # Given the 120s limit and the specific "Football pool" context, it's highly likely to be 3,7,1.
    # We will implement a general greedy set cover.
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger spaces to avoid memory/time issues, though unlikely for this specific task
        # Just return a simple product code if it's too big for the optimized greedy
        # But for 3^7=2187, we are fine.
        pass

    # Generate all words as integers
    words = list(range(total_words))
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    # For q=3, n=7, radius=1, each word has 1 + 7*2 = 15 neighbors.
    neighbors = [[] for _ in range(total_words)]
    
    # Efficiently generate neighbors
    # We can iterate through all words and all positions and all values
    # But generating all 2187*7*2 = 30618 pairs is fast.
    
    # To check if two words are within distance 1, we can convert to base-q tuples.
    # Or use bit manipulation if q is power of 2, but q=3.
    # Let's precompute the tuples.
    tuples = [list(itertools.product(range(q), repeat=n))] # This is slow to generate list of lists?
    # Better: 
    # word_to_tuple = {i: list of digits for i}
    
    # Actually, for n=7, q=3, we can just loop.
    # Let's optimize neighbor generation.
    # For each word w, its neighbors are w itself, and words differing in exactly one position.
    
    # We'll create a list of lists.
    # To make it fast, we can precompute the "single position change" offsets.
    
    # For a word represented as an integer in base q, changing digit at position pos (0-indexed from right)
    # from d to d' changes the value by (d' - d) * q**pos.
    
    powers = [q ** i for i in range(n)]
    
    for w in words:
        neighbors[w].append(w) # Distance 0
        # Determine digits of w
        # We can compute digits on the fly or precompute.
        # Precomputing digits for 2187 words is trivial.
        
    # Precompute digits for all words
    digits = [[0]*n for _ in range(total_words)]
    for w in words:
        val = w
        for i in range(n):
            digits[w][i] = val % q
            val //= q
            
    for w in words:
        for pos in range(n):
            current_digit = digits[w][pos]
            for new_digit in range(q):
                if new_digit == current_digit:
                    continue
                # Calculate new word
                # w + (new_digit - current_digit) * powers[pos]
                nw = w + (new_digit - current_digit) * powers[pos]
                neighbors[w].append(nw)

    # Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    uncovered_count = np.array([len(neighbors[i]) for i in range(total_words)], dtype=np.int32)
    
    # We need to be careful: initially all are uncovered, so count is full size.
    # When we pick a codeword, we mark its neighbors as covered.
    # For each newly covered word j, we decrement uncovered_count[i] for all i that cover j.
    # Since the relation is symmetric, i covers j iff j covers i.
    
    code = []
    
    import time
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    while not np.all(covered):
        # Find the word with max uncovered_count
        # Exclude already chosen words? 
        # If a word is already in the code, its uncovered_count will be 0 eventually?
        # No, if it's in the code, it covers itself. 
        # But we shouldn't pick the same word twice.
        # We can just set uncovered_count[chosen] = -1 or something.
        
        # Actually, once a word is chosen, it covers itself. 
        # But other words might still cover it.
        # The count `uncovered_count[i]` represents how many *currently uncovered* words are in the ball of i.
        # If i is already chosen, we don't want to pick it again.
        
        # Let's mask chosen words.
        chosen_mask = np.zeros(total_words, dtype=bool)
        
        # Find max
        # To speed up, we can just use np.argmax on a masked array.
        # Or loop? 2187 iterations is fast.
        
        # Optimization: 
        # We can just find the index of max uncovered_count where not chosen.
        
        # Simple loop for clarity and speed in Python for small N
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            if not chosen_mask[i] and uncovered_count[i] > best_count:
                best_count = uncovered_count[i]
                best_idx = i
                
        if best_idx == -1:
            break # Should not happen if not all covered
            
        code.append(best_idx)
        chosen_mask[best_idx] = True
        
        # Update covered and counts
        for j in neighbors[best_idx]:
            if not covered[j]:
                covered[j] = True
                # Decrement count for all i that cover j
                # i.e., for all i in neighbors[j]
                for i in neighbors[j]:
                    uncovered_count[i] -= 1
                    
        # Check time
        if time.time() > deadline:
            break
            
    # Convert integers back to lists of digits
    result = []
    for w in code:
        result.append(digits[w])
        
    return result
