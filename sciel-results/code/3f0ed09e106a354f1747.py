import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # For q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback to the direct sum construction for other parameters
        # This specific construction is tuned for 3,8,1
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: trivial covering (all words)
            return [list(w) for w in product(range(q), repeat=n)]

    total_words = q ** n
    # Precompute neighbors for each word in the space
    # Word is represented as an integer from 0 to total_words-1
    # Convert int to base-q vector and back to int for neighbors
    
    def to_base_q(num, n, q):
        vec = [0] * n
        for i in range(n):
            vec[i] = num % q
            num //= q
        return vec

    def from_base_q(vec, q):
        num = 0
        for i in range(n):
            num = num * q + vec[i]
        return num

    neighbors = [set() for _ in range(total_words)]
    for w in range(total_words):
        vec = to_base_q(w, n, q)
        neighbors[w].add(w)
        for i in range(n):
            for delta in range(1, q):
                new_val = (vec[i] + delta) % q
                new_vec = vec.copy()
                new_vec[i] = new_val
                neighbors[w].add(from_base_q(new_vec, q))
        
    # Greedy algorithm with restarts
    best_code = None
    best_size = total_words + 1
    
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # Randomly select the first codeword to break symmetry and explore different basins
        first_choice = random.randint(0, total_words - 1)
        code.append(first_choice)
        covered[list(neighbors[first_choice])] = True
        
        remaining_uncovered = total_words - len(neighbors[first_choice])
        
        while remaining_uncovered > 0:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_word = -1
            
            # Iterate through all possible candidates
            # Optimization: We could maintain a list of candidates that might still cover something,
            # but with 6561 words, a full scan is fast enough in Python if vectorized or simple loops are used.
            # However, pure Python loops over 6561 items * ~400 iterations might be slow.
            # Let's try to speed up the "max uncovered" calculation.
            
            # We need to find w that maximizes |neighbors[w] \ covered|
            # This is equivalent to minimizing |neighbors[w] \cap covered|
            
            # For each w, count how many of its neighbors are already covered.
            # If a word w is already covered, it doesn't mean we can't pick it, but picking a covered word 
            # is never better than picking an uncovered word that covers new ones? 
            # Actually, we should only consider words that cover at least 1 new word.
            
            # Vectorized approach:
            # Create a matrix M where M[i, j] = 1 if j is in neighbors[i].
            # This matrix is 6561x6561, which is ~43M entries. Too large for memory? 
            # 43M booleans is ~43MB. Acceptable.
            
            # Let's precompute the neighbor matrix once.
            if not hasattr(construct, '_neighbor_matrix'):
                mat = np.zeros((total_words, total_words), dtype=np.uint8)
                for w in range(total_words):
                    for nb in neighbors[w]:
                        mat[w, nb] = 1
                construct._neighbor_matrix = mat
                construct._neighbor_sets = neighbors # Keep sets for fast list conversion if needed
                
            mat = construct._neighbor_matrix
            
            # Count covered neighbors for each candidate w
            # covered is a boolean vector of length total_words
            # counts[w] = sum(mat[w, :] * covered)
            # Using matrix multiplication might be faster
            counts = mat @ covered  # Shape (total_words,)
            
            # We want to maximize (size_of_neighbors[w] - counts[w])
            # size_of_neighbors[w] is constant for all w (radius 1, q=3 -> 1 + 8*2 = 17)
            # So we just want to minimize counts[w]
            
            # But we must ensure the word actually covers something new.
            # If counts[w] == size_of_neighbors[w], it covers nothing new.
            # We should pick w with minimum counts[w] among those where counts[w] < size_of_neighbors[w].
            
            min_count = np.min(counts)
            
            # If min_count is 17, we are stuck (shouldn't happen if covered is not full)
            if min_count == 17:
                 # This implies all remaining uncovered words are isolated from all candidates? 
                 # No, it implies every candidate covers only already covered words.
                 # This can happen if we are in a bad state, but greedy usually doesn't fail this way unless covered is full.
                 # If covered is not full, there must be some uncovered word u. 
                 # u is in neighbors[u]. So u covers itself.
                 # If u is not covered, counts[u] < 17.
                 # So min_count < 17 should hold.
                 pass
            
            # Find all words with minimum count
            candidates = np.where(counts == min_count)[0]
            
            # Randomly select from candidates to add diversity
            chosen = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(chosen)
            covered[list(neighbors[chosen])] = True
            remaining_uncovered -= (17 - min_count) # Approximation, exact update below is safer
            
            # Exact update of remaining_uncovered is tricky with vectorization in loop.
            # Just check if covered is all True.
            if np.all(covered):
                break
        
        if np.all(covered):
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                
        # Early stopping if we hit a very good solution? 
        # Theoretical lower bound is 403. If we get 403, we can stop.
        if best_size <= 403:
            break

    if best_code is None:
        # Fallback to deterministic direct sum if greedy failed (unlikely)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=4)]
        
    # Convert integer indices back to vectors
    result = []
    for w in best_code:
        result.append(to_base_q(w, n, q))
        
    return result
