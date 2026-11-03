import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    powers = [q**i for i in range(n)]
    
    # Precompute neighbors
    neighbors_list = []
    for idx in range(total_words):
        digits = []
        temp = idx
        for i in range(n):
            digits.append(temp % q)
            temp //= q
        
        neigh = [idx]
        for i in range(n):
            orig_digit = digits[i]
            for new_digit in range(q):
                if new_digit == orig_digit:
                    continue
                new_idx = idx - orig_digit * powers[i] + new_digit * powers[i]
                neigh.append(new_idx)
        neighbors_list.append(neigh)
        
    neighbors = np.array(neighbors_list)
    
    # Greedy Phase
    uncovered = np.ones(total_words, dtype=bool)
    used = np.zeros(total_words, dtype=bool)
    code_indices = []
    cover_count = np.zeros(total_words, dtype=int)
    
    while True:
        if time.time() > deadline - 5.0: # Leave 5s for reduction
            break
        if not np.any(uncovered):
            break
            
        counts = np.sum(uncovered[neighbors], axis=1)
        counts[used] = 0
        
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break
            
        code_indices.append(best_idx)
        used[best_idx] = True
        cover_count[neighbors[best_idx]] += 1
        uncovered[neighbors[best_idx]] = False
        
    # Reduction Phase
    # We have code_indices. Let's try to remove them.
    # To make it faster, we can shuffle the order of removal attempts.
    # But simple sequential is okay.
    
    # We need to be careful: removing one might make another removable.
    # So we loop until no removals happen in a pass.
    
    # However, checking "is it still covered" requires updating cover_count.
    
    # Let's use a list for current code
    current_code = list(code_indices)
    
    # Initialize cover_count based on current_code
    cover_count = np.zeros(total_words, dtype=int)
    for c in current_code:
        cover_count[neighbors[c]] += 1
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through a copy of the list to allow removal
        # We can iterate in reverse to handle indices, or use a set
        # Let's iterate through the list
        for i in range(len(current_code)):
            c = current_code[i]
            # Try removing c
            cover_count[neighbors[c]] -= 1
            
            # Check if any word has count 0
            # np.any(cover_count == 0)
            if not np.any(cover_count == 0):
                # Safe to remove
                current_code.pop(i)
                improved = True
                # Since we popped, we skip to next i (which is now the same index)
                # But wait, if we pop, the list shrinks.
                # We should restart the loop or be careful.
                # Restarting is safer but slower.
                # Given the small size, restart is fine.
                break
            else:
                # Not safe, put it back
                cover_count[neighbors[c]] += 1
                
    # Convert to output format
    code = []
    for idx in current_code:
        digits = []
        temp = idx
        for i in range(n):
            digits.append(temp % q)
            temp //= q
        code.append(digits)
        
    return code
