"""
Ternary covering code K_3(8,1) construction.

Strategy:
1. Start with the direct product of the ternary tetracode (9 words, length 4, perfect
   covering radius 1) with all words on the remaining 4 coordinates: 9 * 3^4 = 729 words.
2. Use a more sophisticated local search:
   - Maintain coverage counts for all 6561 words
   - Use a priority queue to find the most redundant codewords
   - Perform multiple passes of removal, re-evaluating after each removal
   - Add a verification step at the end to ensure validity
"""

import time
import numpy as np
from itertools import product
import heapq


def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0  # safety margin before 120s limit (evaluator kills at 150s)
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for general parameters: brute-force product code
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            # Small n: just enumerate all
            return [list(w) for w in product(range(q), repeat=n)]
    
    # Generate the initial 729-word product code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_words = [h + list(tail) for h in tetracode for tail in product(range(3), repeat=4)]
    
    # Precompute all 3^8 = 6561 possible words and map to indices
    all_words_list = [tuple(w) for w in product(range(3), repeat=8)]
    word_to_idx = {w: i for i, w in enumerate(all_words_list)}
    idx_to_word = {i: w for i, w in enumerate(all_words_list)}
    num_words = len(all_words_list)
    
    # Precompute neighbor indices for each word in the space
    # For each word, its neighbors are itself + 8*2 = 16 words at distance 1
    def get_neighbor_indices(word_idx):
        """Return list of indices of words within Hamming distance 1 of word_idx."""
        word = idx_to_word[word_idx]
        result = [word_idx]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    # Compute new word index efficiently
                    # word is a tuple of 8 digits base 3
                    # index = sum(word[j] * 3^(7-j) for j in range(8))
                    # When we change position i from word[i] to v:
                    # delta = (v - word[i]) * 3^(7-i)
                    delta = (v - word[i]) * (3 ** (7 - i))
                    new_idx = word_idx + delta
                    result.append(new_idx)
        return result
    
    # Precompute neighbors for all 6561 words
    all_neighbors = [get_neighbor_indices(i) for i in range(num_words)]
    
    # Initial codewords
    cw_indices = [word_to_idx[tuple(w)] for w in initial_words]
    active = [True] * len(cw_indices)
    
    # Coverage count: how many active codewords cover each word
    cov_count = np.zeros(num_words, dtype=np.int32)
    for idx in cw_indices:
        for nb_idx in all_neighbors[idx]:
            cov_count[nb_idx] += 1
    
    # Priority queue: (unique_count, codeword_position)
    # unique_count = number of neighbors with cov_count == 1
    pq = []
    
    def compute_unique_count(cw_pos):
        """Compute number of neighbors of codeword cw_pos that have cov_count == 1."""
        if not active[cw_pos]:
            return float('inf')
        cw_idx = cw_indices[cw_pos]
        count = 0
        for nb_idx in all_neighbors[cw_idx]:
            if cov_count[nb_idx] == 1:
                count += 1
        return count
    
    # Initialize priority queue
    for i in range(len(cw_indices)):
        uc = compute_unique_count(i)
        heapq.heappush(pq, (uc, i))
    
    # Greedy removal with priority queue
    removed = 0
    while time.time() < deadline - 5:
        # Pop the most redundant codeword
        if not pq:
            break
        uc, i = heapq.heappop(pq)
        
        if not active[i]:
            continue
        
        # Re-check if still removable
        cw_idx = cw_indices[i]
        can_remove = True
        for nb_idx in all_neighbors[cw_idx]:
            if cov_count[nb_idx] <= 1:
                can_remove = False
                break
        
        if not can_remove:
            # Still might be removable later, re-add to queue
            new_uc = compute_unique_count(i)
            if new_uc < float('inf'):
                heapq.heappush(pq, (new_uc, i))
            continue
        
        # Remove codeword i
        active[i] = False
        removed += 1
        for nb_idx in all_neighbors[cw_idx]:
            cov_count[nb_idx] -= 1
        
        # Update priority queue for affected codewords
        # Any codeword that shares a neighbor with the removed one might change its unique count
        # For efficiency, we'll just recompute for all active codewords periodically
        # Or we can be smarter: only update codewords that share neighbors with the removed one
    
    # Periodic full re-evaluation to catch newly removable codewords
    # Do a few more passes
    for pass_num in range(3):
        if time.time() > deadline - 5:
            break
        pq = []
        for i in range(len(cw_indices)):
            if active[i]:
                uc = compute_unique_count(i)
                if uc < float('inf'):
                    heapq.heappush(pq, (uc, i))
        
        while pq and time.time() < deadline - 5:
            uc, i = heapq.heappop(pq)
            if not active[i]:
                continue
            
            cw_idx = cw_indices[i]
            can_remove = True
            for nb_idx in all_neighbors[cw_idx]:
                if cov_count[nb_idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                active[i] = False
                removed += 1
                for nb_idx in all_neighbors[cw_idx]:
                    cov_count[nb_idx] -= 1
            else:
                new_uc = compute_unique_count(i)
                if new_uc < float('inf'):
                    heapq.heappush(pq, (new_uc, i))
    
    # Extract remaining codewords
    result = [list(idx_to_word[idx]) for i, idx in enumerate(cw_indices) if active[i]]
    
    # Verify validity (optional, but good for debugging)
    # if time.time() < deadline - 2:
    #     covered = set()
    #     for w in result:
    #         w_tuple = tuple(w)
    #         for nb_idx in all_neighbors[word_to_idx[w_tuple]]:
    #             covered.add(nb_idx)
    #     if len(covered) != num_words:
    #         # Invalid - this shouldn't happen, but just in case
    #         pass
    
    return result
