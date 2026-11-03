import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    timeout = 110.0
    
    total_words = q ** n
    
    # Fallback for large spaces
    if total_words > 5000:
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Return a simple greedy result or all words if too large
            # For safety, return a small valid set if possible, else all words
            # But the task implies small n.
            pass

    # Precompute neighbors
    neighbors = []
    for i in range(total_words):
        # Convert i to base-q digits
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % q
            temp //= q
        
        neigh = [i]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                digits[pos] = new_val
                # Convert back to index
                idx = 0
                for k in range(n):
                    idx = idx * q + digits[k]
                neigh.append(idx)
            digits[pos] = orig
        neighbors.append(neigh)
        
    # Convert neighbors to numpy arrays for faster indexing?
    # neighbors_np = [np.array(neigh) for neigh in neighbors]
    # This might be faster for uncovered[neighbors_np[c]]
    neighbors_np = [np.array(neigh, dtype=int) for neigh in neighbors]

    best_solution = None
    best_size = total_words

    # Number of restarts
    # Estimate time per restart
    # Let's try 100 restarts initially, adjust if too slow
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        
        # Greedy Phase
        while np.any(uncovered):
            best_gain = -1.0
            best_candidate = -1
            
            # Iterate candidates
            # To improve diversity, we can iterate in a random order or add noise
            # Adding noise is cheaper
            for c in range(total_words):
                gain = np.sum(uncovered[neighbors_np[c]])
                # Add small noise to break ties randomly
                gain += random.uniform(0, 0.01)
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
                    
            if best_candidate == -1:
                break
                
            selected.append(best_candidate)
            uncovered[neighbors_np[best_candidate]] = False
            
            if time.time() - start_time > timeout:
                break
        
        # Redundancy Removal
        cover_count = np.zeros(total_words, dtype=int)
        for c in selected:
            cover_count[neighbors_np[c]] += 1
            
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow popping
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                # Check if redundant
                # If all neighbors have count > 1
                if np.all(cover_count[neighbors_np[c]] > 1):
                    cover_count[neighbors_np[c]] -= 1
                    selected.pop(i)
                    changed = True
                    
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected[:]
            
    # Format output
    if best_solution is None:
        # Fallback: return all words (valid but large)
        result = []
        for i in range(total_words):
            digits = [0] * n
            temp = i
            for j in range(n):
                digits[j] = temp % q
                temp //= q
            result.append(digits)
        return result

    result = []
    for idx in best_solution:
        digits = [0] * n
        temp = idx
        for j in range(n):
            digits[j] = temp % q
            temp //= q
        result.append(digits)
        
    return result
