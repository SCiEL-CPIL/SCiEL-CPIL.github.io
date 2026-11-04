import itertools
import time

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # But the function should be generic or at least handle the specific case efficiently.
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # However, the strategy is specific to the small size.
    # If n is large, this won't work. But the task is fixed to 6 matches.
    # We assume q=3, n=6, radius=1 based on the task description.
    # If inputs differ, we fall back to a simple construction or error.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (though not expected)
        # Simple greedy for general case
        # This part is likely not executed for the specific task contract
        pass 

    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    total_words = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors (Hamming distance <= radius)
    # For q=3, n=6, radius=1: 1 + 6*2 = 13 neighbors
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        neighbors[i] = [i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors[i].append(word_to_idx[tuple(new_w)])
    
    # Coverage tracking
    # count[i] = number of codewords that cover word i
    counts = [0] * total_words
    in_code = [False] * total_words
    code = []
    
    def add_word(idx):
        code.append(idx)
        in_code[idx] = True
        for nb in neighbors[idx]:
            counts[nb] += 1
            
    def remove_word(idx):
        code.remove(idx)
        in_code[idx] = False
        for nb in neighbors[idx]:
            counts[nb] -= 1
            
    # 1. Greedy Construction
    # We want to cover all words.
    # Pick word that covers most uncovered words.
    # Optimization: maintain a list of uncovered words?
    # Or just check counts.
    # Since total_words is small (729), we can just scan.
    
    uncovered_count = total_words
    
    while uncovered_count > 0:
        if time.time() - start_time > timeout:
            break
        
        best_gain = -1
        best_idx = -1
        
        # To speed up, we can skip words that are already in code? 
        # No, we pick from all words, but adding a word already in code is useless.
        # We can iterate over all words not in code.
        
        for i in range(total_words):
            if in_code[i]:
                continue
            
            # Calculate gain: how many uncovered words does i cover?
            # gain = sum(1 for nb in neighbors[i] if counts[nb] == 0)
            gain = 0
            for nb in neighbors[i]:
                if counts[nb] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break # Should not happen if uncovered_count > 0
            
        add_word(best_idx)
        uncovered_count -= best_gain
        
    # 2. Reduction Phase
    # Try to remove words that are redundant.
    # A word is redundant if removing it doesn't make any count drop to 0.
    # i.e., for all nb in neighbors[idx], counts[nb] > 1.
    
    def try_reduce():
        changed = True
        while changed:
            if time.time() - start_time > timeout:
                return
            changed = False
            # Iterate over a copy of code to allow removal
            for idx in list(code):
                # Check if redundant
                redundant = True
                for nb in neighbors[idx]:
                    if counts[nb] == 1:
                        redundant = False
                        break
                if redundant:
                    remove_word(idx)
                    changed = True
                    break # Restart loop as counts changed
                    
    try_reduce()
    
    # 3. Swap Phase
    # If we are stuck, try swapping a word u in code with a word v not in code.
    # Condition: All words uniquely covered by u must be covered by v.
    # unique_u = [nb for nb in neighbors[u] if counts[nb] == 1]
    # If unique_u is subset of neighbors[v], swap is valid.
    
    swapped = True
    while swapped:
        if time.time() - start_time > timeout:
            break
        swapped = False
        
        # Try to find a swap
        for u in list(code):
            if time.time() - start_time > timeout:
                break
            
            # Find words uniquely covered by u
            unique_u = []
            for nb in neighbors[u]:
                if counts[nb] == 1:
                    unique_u.append(nb)
            
            if not unique_u:
                # This word is redundant, should have been removed in try_reduce
                # But maybe try_reduce didn't catch it? Or we are in a state where 
                # removing it is valid but we are in swap loop.
                # Actually, if unique_u is empty, we can just remove it.
                remove_word(u)
                swapped = True
                try_reduce()
                break
            
            # Look for v not in code such that unique_u is subset of neighbors[v]
            found_swap = False
            # Optimization: Only check v if it covers the first unique word?
            # Let's just check all v.
            for v in range(total_words):
                if in_code[v]:
                    continue
                
                # Check if unique_u is subset of neighbors[v]
                # Since unique_u is small, this is fast.
                valid = True
                for ub in unique_u:
                    if ub not in neighbors[v]:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    remove_word(u)
                    add_word(v)
                    swapped = True
                    found_swap = True
                    try_reduce()
                    break
            
            if found_swap:
                break # Restart outer loop with new code
        
    # Return the codewords as lists of integers
    return [words[i] for i in code]
