import time
import itertools
import random

def construct(q, n, radius):
    # The task specifies q=3, n=7, radius=1.
    # But we should handle general inputs if possible, though the contract implies this specific instance.
    # "The evaluator passes this task's alphabet size q, length n and covering radius."
    # We assume q=3, n=7, radius=1 based on the prompt description.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # If parameters are large, greedy might be too slow. 
    # For n=7, q=3, total words = 2187. This is small.
    # If n is larger, we need a heuristic.
    # For now, optimize for the specific case n=7, q=3.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the original direct sum construction or similar for other params
        # Just to be safe, though the task is specific.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Very small n, just return all words
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimized solver for q=3, n=7, R=1
    # Total words = 3^7 = 2187
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(3), repeat=n))
    num_words = len(all_words)
    
    # Map word tuple to index for fast lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (index)
    # Neighbor: same word + all words at distance 1
    neighbors = [[] for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # distance 0
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    w_list[pos] = new_val
                    n_idx = word_to_idx[tuple(w_list)]
                    neighbors[i].append(n_idx)
            w_list[pos] = orig_val
            
    # Greedy Set Cover
    uncovered = set(range(num_words))
    code_indices = []
    
    # Optimization: 
    # In each step, we need to find the word i that maximizes |neighbors[i] ∩ uncovered|.
    # Since num_words is small (2187), we can iterate through all candidates.
    # However, we can restrict candidates to those that are in `uncovered` or have many uncovered neighbors.
    # Actually, just iterating all 2187 words is fine.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = -1
        
        # We can iterate over all words, or just uncovered ones?
        # A word that is already covered can still be a good center.
        # But usually, the best center is in the uncovered set or adjacent to it.
        # Let's iterate all words to be safe and optimal for greedy.
        
        # To speed up, we can maintain a count of how many uncovered neighbors each word has.
        # But updating counts for all words after removing a set of neighbors is expensive.
        # Given 2187 words, a full scan is O(2187 * 15) ~ 32k ops.
        # With ~200 iterations, that's ~6M ops. Very fast.
        
        for i in range(num_words):
            # Count uncovered neighbors
            count = 0
            for n_idx in neighbors[i]:
                if n_idx in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
                if best_count == len(uncovered): # Perfect cover found? Unlikely but check
                    break
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        
        # Remove covered neighbors from uncovered
        for n_idx in neighbors[best_idx]:
            if n_idx in uncovered:
                uncovered.remove(n_idx)
                
    # Convert indices back to lists of integers
    code_words = [list(all_words[i]) for i in code_indices]
    
    return code_words
