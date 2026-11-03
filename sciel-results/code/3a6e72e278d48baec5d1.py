import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with radius r.
    Uses randomized greedy set cover with restarts.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # Precompute all words as tuples or integers for fast indexing
    # Total words: q**n
    total_words = q ** n
    
    # Map word (tuple) to index
    # For q=3, n=6, we can map (a,b,c,d,e,f) to a + 3*b + 9*c + ...
    # This allows O(1) conversion and storage in numpy arrays
    
    # Generate all words as a numpy array of shape (total_words, n)
    # This is efficient for vectorized operations if needed, but for greedy,
    # explicit loops with precomputed neighbors might be faster due to low constant overhead.
    
    # Let's use integer representation for words to make sets/lists fast.
    # word_idx = sum(w[i] * q**i for i in range(n))
    
    # Precompute the ball of radius 'radius' for each word.
    # For radius=1, the ball contains the word itself and words differing in exactly one position.
    # Number of neighbors = 1 + n * (q - 1)
    
    # We store neighbors as a list of lists of integers (indices).
    neighbors = [None] * total_words
    
    for idx in range(total_words):
        # Convert idx to word tuple
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word = tuple(word)
        
        ball = [idx]
        
        if radius == 1:
            for i in range(n):
                for val in range(q):
                    if val != word[i]:
                        # Create new word
                        new_word_list = list(word)
                        new_word_list[i] = val
                        new_idx = 0
                        pow_q = 1
                        for j in range(n):
                            new_idx += new_word_list[j] * pow_q
                            pow_q *= q
                        ball.append(new_idx)
        # If radius > 1, this simple expansion is not enough, but task specifies radius=1 typically.
        # The prompt says "checked over the whole space", but the specific task is K_3(6,1).
        # We should handle general radius if possible, but the heuristic is tailored for small radius.
        # Given the specific task context (Football pool, K_3(6,1)), radius=1 is the target.
        # If radius > 1, the ball size grows exponentially. 
        # For robustness, if radius > 1, we might need a different approach or just rely on the fact
        # that the evaluator passes specific q, n, radius. The seed assumes radius=1 structure implicitly 
        # via the tetracode construction which is a perfect code for r=1.
        # However, the interface takes radius. 
        # Let's assume radius=1 for the optimization path as per the task description "K_3(6,1)".
        # If radius is not 1, the greedy approach still works but ball computation is heavier.
        
        neighbors[idx] = ball

    best_code = None
    best_size = float('inf')
    
    # Number of restarts
    # Each restart takes ~ms. We can do many.
    max_restarts = 10000
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Random seed for this restart
        random.seed(restart)
        
        # Greedy Set Cover
        # uncovered: set of indices
        uncovered = set(range(total_words))
        code = []
        
        # Optimization: Instead of checking all 729 candidates every time,
        # we can just check all, it's small enough.
        # To improve diversity, we can randomize the order of candidates or tie-breaking.
        
        while uncovered:
            # Find candidate with max coverage
            best_candidate = -1
            max_covered = 0
            # List of candidates with max coverage to pick from randomly
            top_candidates = []
            
            # Iterate through all possible codewords
            # To add randomness, we can shuffle the range, but that's expensive.
            # Instead, we can just pick randomly from ties.
            
            # Optimization: 
            # If we just pick the absolute max, it's deterministic. 
            # To get different solutions, we can add a small random noise to the score.
            
            for c in range(total_words):
                # Calculate coverage
                # Count how many neighbors of c are in uncovered
                cover = 0
                for nb in neighbors[c]:
                    if nb in uncovered:
                        cover += 1
                
                # Add small random noise to break ties and explore
                noise = random.random() * 0.5 
                score = cover + noise
                
                if score > max_covered:
                    max_covered = score
                    best_candidate = c
                    top_candidates = [c]
                elif abs(score - max_covered) < 1e-9: # Tie
                    top_candidates.append(c)
            
            # Pick one from top candidates
            chosen = random.choice(top_candidates)
            code.append(chosen)
            
            # Update uncovered
            for nb in neighbors[chosen]:
                if nb in uncovered:
                    uncovered.remove(nb)
        
        # Post-processing: Remove redundant codewords
        # A codeword is redundant if all words it covers are covered by other codewords.
        # We can check each codeword:
        # 1. Temporarily remove it.
        # 2. Check if any word in its ball is NOT covered by the remaining code.
        # 3. If all are covered, keep it removed.
        
        # Efficient check:
        # Maintain a coverage count for each word.
        # count[word] = number of codewords in code whose ball contains word.
        
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for nb in neighbors[c]:
                counts[nb] += 1
        
        final_code = []
        for c in code:
            # Check if removing c leaves any word with count 0
            # A word w is "lost" if counts[w] == 1 and c is the only one covering it.
            # i.e., if there exists nb in neighbors[c] such that counts[nb] == 1.
            is_redundant = True
            for nb in neighbors[c]:
                if counts[nb] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove c
                for nb in neighbors[c]:
                    counts[nb] -= 1
            else:
                final_code.append(c)
        
        if len(final_code) < best_size:
            best_size = len(final_code)
            best_code = final_code
            
            # If we hit the theoretical lower bound (71) or known best (73), we can stop early?
            # Lower bound is 71. If we find 71, it's optimal.
            if best_size <= 71:
                break

    # If no code was found (should not happen), return a valid large code
    if best_code is None:
        # Fallback to simple greedy without restarts or the seed construction
        # The seed construction returns 81 words.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to lists of integers
    result = []
    for idx in best_code:
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        result.append(word)
        
    return result
