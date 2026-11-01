import time
import numpy as np

def construct(q, n, radius):
    # The task is specifically for q=3, n=7, radius=1.
    # We assume the evaluator calls this with these parameters.
    # If parameters differ, we fall back to a simple heuristic or seed logic.
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed logic for other parameters to ensure validity
        import itertools
        # Simple greedy for general case if needed, but seed is better for specific case
        # For non-target case, just return a valid set (even if not optimal)
        # We'll stick to the seed's logic for other cases to be safe, 
        # but the prompt implies the specific task.
        # Let's just implement a basic greedy for general case.
        total_words = q ** n
        all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=int)
        covered = np.zeros(total_words, dtype=bool)
        code = []
        # Precompute neighbors for general case is expensive if n is large, 
        # but for small n it's fine.
        # However, the specific task is 3,7,1. We focus on that.
        
        # If not the specific case, return a trivial covering code (all words)
        # to ensure validity, though score will be bad.
        # Actually, the seed logic is decent. Let's just return seed logic for non-target.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=7, radius=1
    total_words = 3 ** 7
    # Map word (list of ints) to index
    # Index = w[0] + w[1]*3 + ... + w[6]*3^6
    
    # Precompute neighbors for each word index
    # neighbor_list[i] is a list of indices within Hamming distance 1
    neighbor_lists = []
    for i in range(total_words):
        # Convert i to base-3 digits
        digits = [0] * n
        temp = i
        for k in range(n):
            digits[k] = temp % 3
            temp //= 3
        
        neighbors = [i]
        for pos in range(n):
            original_val = digits[pos]
            for new_val in range(3):
                if new_val != original_val:
                    # Calculate new index
                    delta = (new_val - original_val) * (3 ** pos)
                    neighbors.append(i + delta)
        neighbor_lists.append(neighbors)
    
    neighbor_lists = np.array(neighbor_lists, dtype=int)
    
    best_code_size = float('inf')
    best_code = None
    
    # Greedy Set Cover with random restarts
    # We can try different random seeds or just run the greedy once with a random start?
    # The standard greedy for covering codes usually picks the word covering the most *uncovered* words.
    # The order of selection matters. Randomizing the tie-breaking or the initial state can help.
    # Actually, the "uncovered" set changes.
    
    # To diversify, we can randomly permute the candidate order in case of ties, 
    # or start with a random word.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # Random start? No, greedy usually starts by picking the best word.
        # The "best" word is the one that covers the most uncovered words.
        # Initially, all words cover 15 words.
        # So the first choice is arbitrary (or random).
        # Let's pick a random starting word to break symmetry and explore different basins.
        start_word = np.random.randint(total_words)
        
        # Add start word
        code.append(start_word)
        covered[neighbor_lists[start_word]] = True
        uncovered_count = total_words - np.sum(covered)
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            # Find the word that covers the most currently uncovered words
            # For each word i, count how many of its neighbors are uncovered.
            # We can vectorize this.
            # For each candidate i, we want sum(1 - covered[neighbors[i]])
            
            # Efficient way:
            # We have a matrix M where M[i, j] = 1 if j is neighbor of i.
            # We want to compute for each i: sum_j M[i, j] * (1 - covered[j])
            
            # Let's precompute the adjacency matrix A (2187x2187)
            # This is 4.7 million entries. 16MB. Acceptable.
            
            # To avoid recomputing A every restart, we define it once.
            # But we are inside the loop. Let's move it out.
            
            # Actually, let's just compute it once outside the restart loop.
            # I'll refactor the code structure.
            pass
        
        # Since I need to be careful with time, I'll implement the loop properly below.
        # The pass above was a placeholder.
        
        # Let's restart the logic with A precomputed.
        # This block is effectively skipped in the real execution flow if I structure it right.
        
        # Correct structure:
        # 1. Precompute A
        # 2. Loop restarts
        # 3. Inner loop greedy steps
        
    # Re-doing the structure cleanly
    
    # Precompute adjacency matrix
    # A[i, j] = 1 if dist(i, j) <= 1
    # This is symmetric.
    A = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        A[i, neighbor_lists[i]] = 1
        
    # Now run greedy with restarts
    best_size = float('inf')
    best_code_list = None
    
    for restart in range(200): # Try many restarts
        if time.time() > deadline:
            break
            
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        # Random start
        start_idx = np.random.randint(total_words)
        code_indices.append(start_idx)
        covered[A[start_idx] == 1] = True
        uncovered_count = total_words - np.sum(covered)
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            # Calculate coverage gain for each candidate
            # gain[i] = number of uncovered neighbors of i
            # gain = A @ (1 - covered)
            # A is int8, (1-covered) is bool->int.
            # Result is int.
            gains = A @ (1 - covered).astype(np.int8)
            
            # Find max gain
            max_gain = np.max(gains)
            
            # If max_gain is 0, we have a problem (should not happen if space is connected)
            if max_gain == 0:
                # Fallback: pick any uncovered word? 
                # Actually if max_gain is 0, it means no word covers any uncovered word?
                # That implies uncovered words are isolated? No, radius 1 means every word is neighbor to itself.
                # So gain[i] is at least 1 if i is uncovered.
                # So max_gain should be >= 1.
                pass
            
            # Get all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to break ties (adds diversity)
            chosen_idx = candidates[np.random.randint(len(candidates))]
            
            code_indices.append(chosen_idx)
            covered[A[chosen_idx] == 1] = True
            new_uncovered = np.sum(1 - covered)
            
            # Optimization: only update uncovered_count if necessary?
            # We can just recalculate or track delta.
            # Delta = max_gain (since we added chosen_idx, it covers max_gain new words)
            # Wait, max_gain is the number of *currently* uncovered words it covers.
            # So uncovered_count -= max_gain
            uncovered_count -= max_gain
            
            # Safety check
            if uncovered_count < 0:
                uncovered_count = 0
                
        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code_list = code_indices.copy()
            # Optional: Local search / improvement?
            # For now, greedy is the baseline.
            
    # Convert indices back to lists of ints
    if best_code_list is None:
        # Fallback to seed if something went wrong
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_code_list:
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        result.append(digits)
        
    return result
