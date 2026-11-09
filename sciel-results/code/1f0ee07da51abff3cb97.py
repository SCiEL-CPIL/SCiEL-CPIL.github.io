import itertools
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task
    # q=3, n=6, radius=1
    
    total_words = q ** n
    
    # 1. Generate all words and map them to indices
    # Word is a tuple of n integers
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # 2. Precompute neighbors (sphere of radius 1) for each word
    # neighbors[i] is a list of indices
    neighbors = []
    for w in all_words:
        w_list = list(w)
        nbrs = [word_to_idx[w]] # include self
        for i in range(n):
            orig_val = w_list[i]
            for v in range(q):
                if v != orig_val:
                    w_list[i] = v
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(nbrs)
    
    # 3. Greedy Set Cover
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    uncovered = set(range(total_words))
    code_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = 0
        
        # Find the word that covers the most uncovered words
        # Optimization: We can skip words that are already in the code?
        # No, we just pick the best among all. Since we remove covered words,
        # words in the code will have 0 uncovered neighbors eventually, but 
        # checking all is fine for 729.
        
        for i in range(total_words):
            # Count how many neighbors of i are in uncovered
            count = 0
            for nbr in neighbors[i]:
                if nbr in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            break
            
        code_indices.append(best_idx)
        
        # Remove covered words from uncovered
        for nbr in neighbors[best_idx]:
            uncovered.discard(nbr)
            
    # 4. Local Search: Remove redundant codewords
    # A codeword c is redundant if every word in neighbors[c] is covered by 
    # some other codeword in the code.
    
    # To check this efficiently, we can maintain a coverage count for each word.
    # coverage_count[w] = number of codewords in `code_indices` that cover w.
    
    coverage_count = [0] * total_words
    for c in code_indices:
        for nbr in neighbors[c]:
            coverage_count[nbr] += 1
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate through code_indices
        # We need to be careful with indices if we remove elements.
        # Better to iterate by index and remove by value, or rebuild list.
        
        for i in range(len(code_indices) - 1, -1, -1):
            c = code_indices[i]
            # Check if c is redundant
            # c is redundant if for all nbr in neighbors[c], coverage_count[nbr] > 1
            is_redundant = True
            for nbr in neighbors[c]:
                if coverage_count[nbr] <= 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # Remove c
                code_indices.pop(i)
                # Update coverage counts
                for nbr in neighbors[c]:
                    coverage_count[nbr] -= 1
                changed = True
                break # Restart loop as indices/counts changed
                
    # 5. Convert indices back to word lists
    result = [list(all_words[i]) for i in code_indices]
    
    # Fallback if something went wrong or time ran out before completion
    # The seed program returns 81 words. If our result is larger, use seed.
    # But greedy should be <= 81.
    
    # Verify validity just in case (optional, but good for safety)
    # If invalid, return seed.
    covered_final = set()
    for cw in result:
        idx = word_to_idx[tuple(cw)]
        covered_final.update(neighbors[idx])
        
    if len(covered_final) != total_words:
        # Fallback to direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return result
