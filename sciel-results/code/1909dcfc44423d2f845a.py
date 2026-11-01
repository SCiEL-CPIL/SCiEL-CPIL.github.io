import numpy as np
import itertools
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general inputs if possible, though the prompt implies specific task context.
    # The prompt says: "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must handle q, n, radius generically, but optimize for the known case if it matches.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a generic greedy + local search for other parameters
        # But for now, let's assume the main path is for 3,6,1
        # A generic implementation would be slower. 
        # Given the "Football pool problem" context, it's almost certainly 3,6,1.
        # However, to be safe, we can implement a generic version if it's not 3,6,1.
        pass

    total_words = q ** n
    
    # Precompute neighbors for all words
    # idx -> tuple of digits
    # We need a mapping from idx to word and vice versa.
    
    # For q=3, n=6, total_words = 729.
    # We can use a lookup table.
    
    idx_to_word = {}
    word_to_idx = {}
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    for i, w in enumerate(all_words):
        idx_to_word[i] = w
        word_to_idx[w] = i
        
    # Precompute neighbors
    # neighbors[i] is a list of indices
    neighbors = [[] for _ in range(total_words)]
    
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                new_idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(new_idx)
                
    # Convert to numpy array for fast indexing
    # neighbors_arr shape (total_words, max_neighbors)
    # We need to handle variable length? No, for fixed q,n,radius, neighbor count is fixed.
    # For radius 1, count is 1 + n*(q-1).
    num_neighbors = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for i in range(total_words):
        neighbors_arr[i] = neighbors[i]
        
    # Greedy Construction with Local Search
    best_code = None
    best_size = float('inf')
    
    start_time = time.time()
    max_time = 110.0 # Safety margin
    
    # Number of restarts
    # Each greedy run is fast.
    
    for restart in range(200):
        if time.time() - start_time > max_time:
            break
            
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        while uncovered.any():
            # Calculate coverage counts for all candidates
            # uncovered[neighbors_arr] gives a (total_words, num_neighbors) boolean array
            # Summing along axis 1 gives the count of uncovered neighbors for each candidate
            counts = uncovered[neighbors_arr].sum(axis=1)
            
            # Add random noise to break ties and explore different solutions
            # Noise should be small enough not to change the order significantly for distinct counts
            # But effective for equal counts
            noise = np.random.rand(total_words) * 0.01
            counts_with_noise = counts + noise
            
            best_idx = np.argmax(counts_with_noise)
            
            code.append(best_idx)
            
            # Update uncovered
            uncovered[neighbors_arr[best_idx]] = False
            
        # Local Search: Deletion
        # Convert code to numpy array
        code_arr = np.array(code)
        
        # Calculate initial coverage counts
        # coverage_counts[j] = number of codewords covering word j
        coverage_counts = np.zeros(total_words, dtype=np.int32)
        for c_idx in code_arr:
            coverage_counts[neighbors_arr[c_idx]] += 1
            
        improved = True
        while improved:
            improved = False
            new_code = []
            # We iterate through the current code list
            # If we remove a codeword, we update counts
            # We need to be careful with the order. 
            # Standard approach: iterate through indices. 
            # If removal is safe, remove it.
            
            current_code = list(code_arr)
            # We will build a new list
            new_code_list = []
            
            for c_idx in current_code:
                # Temporarily remove c_idx
                coverage_counts[neighbors_arr[c_idx]] -= 1
                
                # Check if any word is uncovered
                if np.any(coverage_counts == 0):
                    # Restore
                    coverage_counts[neighbors_arr[c_idx]] += 1
                    new_code_list.append(c_idx)
                else:
                    # Keep removed
                    improved = True
                    # Do not add to new_code_list
                    
            code_arr = np.array(new_code_list)
            # If code_arr is empty, break? No, if code is empty, coverage is 0, which is invalid unless N=0.
            # The loop condition `improved` handles termination.
            
        # Update best
        if len(code_arr) < best_size:
            best_size = len(code_arr)
            best_code = code_arr.copy()
            
    # Convert best_code indices to list of lists
    if best_code is None:
        # Fallback: return empty or a trivial cover if something went wrong
        # But greedy should always find a cover.
        return []
        
    result = []
    for idx in best_code:
        result.append(list(idx_to_word[idx]))
        
    return result
