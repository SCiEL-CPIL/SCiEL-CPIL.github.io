import time
import random
import itertools
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    # neighbors[i] is a list of indices of words within Hamming distance <= radius of all_words[i]
    neighbors = []
    for w in all_words:
        nbrs = [w]
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val != orig:
                    new_w = list(w)
                    new_w[i] = val
                    nbrs.append(tuple(new_w))
        neighbors.append([word_to_idx[t] for t in nbrs])
        
    # Greedy Construction with Gain Array
    # gain[i] = number of uncovered words in the neighborhood of i
    # Initially, all words are uncovered, so gain[i] = len(neighbors[i])
    
    gains = [len(nbrs) for nbrs in neighbors]
    uncovered = [True] * total_words
    code_indices = []
    current_max_gain = max(gains) if gains else 0
    
    # Optimization: Instead of scanning all N to find max every time,
    # we can just scan. N=6561 is small.
    # But we can be smarter.
    
    while current_max_gain > 0 and time.time() < deadline:
        # Find the word with max gain
        best_word = -1
        best_gain = -1
        for i in range(total_words):
            if gains[i] > best_gain:
                best_gain = gains[i]
                best_word = i
        
        if best_word == -1:
            break
            
        code_indices.append(best_word)
        
        # Update gains for all words that are neighbors of best_word
        # If a word 'u' is covered by best_word, then for any candidate 'c',
        # if 'u' is in neighbors[c], the gain of 'c' decreases by 1.
        # So we need to iterate over all candidates 'c' that have 'u' as a neighbor?
        # No, that's expensive.
        
        # Alternative view:
        # gain[c] = sum(1 for u in neighbors[c] if uncovered[u])
        # When we select best_word, we mark all u in neighbors[best_word] as covered.
        # For each such u, we need to decrement gain[c] for all c such that u in neighbors[c].
        
        # Let's precompute reverse neighbors?
        # reverse_neighbors[u] = list of c such that u in neighbors[c]
        # Since the relation is symmetric (Hamming distance), neighbors[c] contains u iff neighbors[u] contains c.
        # So reverse_neighbors[u] is just neighbors[u].
        
        for u in neighbors[best_word]:
            if uncovered[u]:
                uncovered[u] = False
                # Decrement gain for all c in neighbors[u]
                for c in neighbors[u]:
                    gains[c] -= 1
        
        # Update current_max_gain
        # We can just recompute max(gains) if we want to be safe, but it's O(N).
        # Or we can check if best_gain was the max, and if so, recompute.
        # Given N=6561, recomputing max is fast enough?
        # 6561 * 486 iterations ~ 3M ops. Yes.
        current_max_gain = max(gains)
        
    # Local Search: Removal
    cover_count = [0] * total_words
    
    def update_cover_count(idx, delta):
        for nbr_idx in neighbors[idx]:
            cover_count[nbr_idx] += delta
            
    for c_idx in code_indices:
        update_cover_count(c_idx, 1)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_code_indices = []
        for c_idx in code_indices:
            redundant = True
            for nbr_idx in neighbors[c_idx]:
                if cover_count[nbr_idx] <= 1:
                    redundant = False
                    break
                    
            if redundant:
                update_cover_count(c_idx, -1)
                improved = True
            else:
                new_code_indices.append(c_idx)
                
        code_indices = new_code_indices
        
    return [list(all_words[i]) for i in code_indices]
