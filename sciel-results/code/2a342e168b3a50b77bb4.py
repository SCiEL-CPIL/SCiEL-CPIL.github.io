import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: The limit is 150s. We stop at 140s to be safe.
    limit = 140.0
    
    total_words = q ** n
    
    # 1. Generate all words and convert to/from indices
    # We need a way to map index <-> word efficiently for neighbor generation
    # For small q, n, we can precompute neighbors for all words
    
    # Precompute neighbors for each word
    # neighbor_indices[i] is a list of indices of words within 'radius' of word i
    # Including i itself
    
    # To do this, we generate all words
    # Since N is small (6561), we can afford to do this in a reasonable time
    
    # Helper to convert index to list of digits
    def idx_to_digits(idx, q, n):
        digits = []
        for _ in range(n):
            digits.append(idx % q)
            idx //= q
        return digits
    
    # Helper to convert list of digits to index
    def digits_to_idx(digits, q, n):
        idx = 0
        for i in range(n):
            idx += digits[i] * (q ** i)
        return idx
    
    # Precompute neighbors
    # We will create a NumPy boolean matrix M where M[i, j] is 1 if dist(i, j) <= radius
    # Size N x N. For N=6561, this is ~43M booleans. Memory is fine.
    
    # However, generating M directly by iterating all pairs is O(N^2 * n) = 6561^2 * 8 ~ 344M ops. 
    # This might be slow in pure Python.
    
    # Better strategy for neighbor generation:
    # For each word, generate its neighbors by perturbing positions.
    # Radius 1: For each position, change to other values.
    # Number of neighbors for radius 1: 1 + n*(q-1).
    # For q=3, n=8: 1 + 8*2 = 17 neighbors per word.
    # Total edges: 6561 * 17 ~ 111k. This is very small.
    
    neighbor_lists = [None] * total_words
    for i in range(total_words):
        digits = idx_to_digits(i, q, n)
        neighbors = {i} # Include self
        
        if radius >= 1:
            for pos in range(n):
                original_val = digits[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        new_digits = digits.copy()
                        new_digits[pos] = new_val
                        new_idx = digits_to_idx(new_digits, q, n)
                        neighbors.add(new_idx)
        
        neighbor_lists[i] = list(neighbors)
        
    # Now, we have neighbor_lists.
    # We want to perform greedy set cover.
    # State: uncovered_mask (bool array of size N)
    # At each step, find i that maximizes sum(uncovered_mask[j] for j in neighbor_lists[i])
    
    # To speed this up, we can use a dense matrix M where M[i] is a bool array of neighbors.
    # But constructing M as a dense matrix takes time.
    # Alternative: Use sparse representation or just iterate.
    # Given N=6561 and ~17 neighbors, iterating over neighbors for all N candidates is 6561 * 17 ~ 111k operations per step.
    # If we need ~400 steps, that's ~44M operations. In Python, this might take a few seconds.
    # In NumPy, we can do it faster if we vectorize the "count overlap" step.
    
    # Let's build the dense boolean matrix M.
    # M[i, j] = 1 if j in neighbor_lists[i]
    M = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        M[i, neighbor_lists[i]] = True
        
    uncovered = np.ones(total_words, dtype=bool)
    solution_indices = []
    
    # Greedy Set Cover
    while True:
        if time.time() - start_time > limit:
            break
            
        # Find the word that covers the most uncovered words
        # count[i] = sum(M[i] & uncovered)
        # This is a matrix-vector product where M is bool and uncovered is bool.
        # np.dot or @ might be slow for bool? Let's try.
        # Actually, M @ uncovered (where uncovered is treated as 0/1) gives the counts.
        
        counts = M @ uncovered.astype(np.int8)
        
        if np.max(counts) == 0:
            break
            
        best_idx = np.argmax(counts)
        solution_indices.append(best_idx)
        uncovered[M[best_idx]] = False
        
        # Optional: Early termination if all covered
        if not np.any(uncovered):
            break
            
    # Local Search: Remove redundant codewords
    # A codeword i is redundant if all its neighbors are covered by the union of neighbors of other codewords.
    # i.e., union_neighbors_other & ~M[i] == 0 ? No.
    # Redundant if: for all j in M[i], there exists k in solution (k != i) such that j in M[k].
    # Equivalent: M[i] is a subset of (union of M[k] for k in solution, k!=i).
    
    # We can check this by maintaining a coverage count array.
    # cover_count[j] = number of codewords in solution that cover j.
    
    cover_count = np.zeros(total_words, dtype=np.int32)
    for idx in solution_indices:
        cover_count[M[idx]] += 1
        
    new_solution = list(solution_indices)
    changed = True
    while changed:
        if time.time() - start_time > limit:
            break
        changed = False
        
        # Try to remove from end to front
        for i in range(len(new_solution) - 1, -1, -1):
            idx = new_solution[i]
            # Check if any neighbor of idx has cover_count == 1
            # If yes, idx is essential.
            # If no, idx is redundant.
            # neighbors_of_idx = M[idx]
            # if np.min(cover_count[neighbors_of_idx]) > 1:
            #     Remove idx
            
            min_cover = np.min(cover_count[M[idx]])
            if min_cover > 1:
                # Redundant
                cover_count[M[idx]] -= 1
                new_solution.pop(i)
                changed = True
                break # Restart loop as state changed
                
    # Targeted swap search: try to replace a codeword with another that maintains coverage
    # A swap (remove a, add b) is valid if:
    #   1. b is not already in the solution
    #   2. For every word covered only by a (cover_count == 1 and in M[a]), b also covers it
    #      i.e., M[a] & (cover_count == 1) is a subset of M[b]
    
    best_solution = list(new_solution)
    best_count = len(new_solution)
    
    # Precompute for each codeword the set of words it uniquely covers
    # unique_cover[i] = M[i] & (cover_count == 1)
    
    # We'll iterate through rounds of swaps
    # In each round, try to find a valid swap
    # Targeted approach: for each codeword a, find candidates b that cover unique_cover[a]
    
    # Build a mapping: for each word w, which codewords cover it (from cover_count > 0)
    # Actually, we can precompute for each codeword a, the set of candidate b's
    # that would cover all of a's uniquely-covered words.
    
    # More efficient: for each codeword a, compute the set of words it uniquely covers.
    # Then a candidate b is valid if M[b] contains all of those words.
    # We can check this by: np.all(M[b][unique_words_of_a])
    
    # Let's do targeted swaps with a time budget
    swap_start = time.time()
    swap_budget = limit - (swap_start - start_time)
    
    # Limit swap search to avoid exceeding time
    max_swap_rounds = 200
    for round_num in range(max_swap_rounds):
        if time.time() - start_time > limit - 5.0:
            break
        
        # Find a valid swap
        found_swap = False
        
        # Shuffle codewords for diversity
        if round_num % 10 == 0:
            random.shuffle(new_solution)
        
        for a_idx in range(len(new_solution)):
            if time.time() - start_time > limit - 5.0:
                break
            
            a = new_solution[a_idx]
            
            # Get words uniquely covered by a
            unique_mask = M[a] & (cover_count == 1)
            unique_count = np.count_nonzero(unique_mask)
            
            if unique_count == 0:
                # a is redundant, should have been caught earlier
                continue
            
            # Find candidates b that cover all uniquely-covered words of a
            # b must not be in new_solution
            # And M[b] must contain all words in unique_mask
            
            # Efficient check: for each candidate b, check if M[b] covers unique_mask
            # We can precompute which b's cover each unique word
            
            # For speed, we'll check candidates in random order
            # Candidate b must cover all words in unique_mask
            # Let's get the list of uniquely covered words
            unique_words = np.where(unique_mask)[0]
            
            # For each candidate b (not in solution), check if M[b] covers all unique_words
            # This is expensive if done naively. Let's use a smarter approach.
            
            # Precompute: for each word w, the set of codewords that cover it
            # Actually, we can use the fact that M is a boolean matrix.
            # M[b][unique_words] should all be True.
            
            # Let's try a limited number of candidates per a
            # Candidates: all words not in new_solution
            
            # For efficiency, we'll sample candidates
            # But targeted search means we should check candidates that are likely to work.
            
            # A good candidate b should cover many of the uniquely-covered words of a.
            # We can score candidates by how many unique words they cover.
            
            # But computing this for all candidates is expensive.
            # Alternative: pick a random unique word w of a, and look for b's that cover w.
            
            if len(unique_words) > 0:
                # Pick a random unique word to use as a filter
                w = unique_words[random.randint(0, len(unique_words) - 1)]
                
                # Find all b's that cover w (from M[:, w])
                # b must cover w and all other unique words of a
                # And b must not be in new_solution
                
                # Get candidates that cover w
                # M[:, w] gives which codewords cover w
                # But we need b not in new_solution
                
                # Let's get the set of codewords that cover w
                # This is M[:, w].T or M.T[w]
                
                # Actually, M[i, j] = 1 if j is a neighbor of i.
                # So M[:, w] gives which i's have w as a neighbor, i.e., which codewords cover w.
                
                candidates_covering_w = np.where(M[:, w])[0]
                
                # Filter out codewords already in solution
                solution_set = set(new_solution)
                candidates = [b for b in candidates_covering_w if b not in solution_set]
                
                # Now check each candidate to see if it covers ALL unique words of a
                for b in candidates:
                    if time.time() - start_time > limit - 5.0:
                        break
                    
                    # Check if M[b] covers all unique_words
                    if np.all(M[b][unique_words]):
                        # Valid swap!
                        # Remove a, add b
                        new_solution[a_idx] = b
                        cover_count[M[a]] -= 1
                        cover_count[M[b]] += 1
                        found_swap = True
                        break
                
                if found_swap:
                    break
        
        if not found_swap:
            # No more swaps found
            break
    
    # Final redundancy removal after swaps
    changed = True
    while changed:
        if time.time() - start_time > limit - 2.0:
            break
        changed = False
        
        for i in range(len(new_solution) - 1, -1, -1):
            idx = new_solution[i]
            min_cover = np.min(cover_count[M[idx]])
            if min_cover > 1:
                cover_count[M[idx]] -= 1
                new_solution.pop(i)
                changed = True
                break
    
    # Convert indices back to lists of digits
    return [idx_to_digits(idx, q, n) for idx in new_solution]
