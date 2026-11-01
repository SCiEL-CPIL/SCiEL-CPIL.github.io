import time
import random

def construct(q, n, radius):
    # This implementation is specialized for q=3, n=7, radius=1
    # but should handle general cases by falling back to a simple method if needed.
    # However, the task contract says the evaluator passes q, n, radius.
    # The specific problem is K_3(7,1).
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases: simple greedy
        # But the task is specifically about K_3(7,1)
        # We will assume the evaluator calls with these parameters.
        # If not, we can return a trivial code or raise an error.
        # For safety, let's implement a generic greedy if parameters differ.
        pass

    total_words = q ** n
    # Precompute balls
    # Map word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    balls = [0] * total_words
    for i in range(total_words):
        # Convert index to word
        word = []
        temp = i
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        
        mask = 1 << i
        # Add neighbors
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                new_word = word[:]
                new_word[pos] = val
                idx = word_to_idx(new_word)
                mask |= (1 << idx)
        balls[i] = mask

    start_time = time.time()
    best_code = None
    best_len = float('inf')
    
    # Greedy with randomization
    # We can also try to improve the initial state or use a different heuristic.
    # For now, randomized greedy.
    
    while time.time() - start_time < 110:  # Safety margin
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            # Find word with max coverage
            max_cov = -1
            best_words = []
            
            # Iterate through all words
            # Optimization: Only consider words that are still uncovered?
            # No, a word can cover others even if it's already covered.
            # But if a word is already covered, its ball is partially covered.
            # We want to maximize NEW coverage.
            
            for i in range(total_words):
                cov = (uncovered & balls[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_words = [i]
                elif cov == max_cov:
                    best_words.append(i)
            
            if max_cov == 0:
                break # Should not happen if uncovered is not 0
                
            # Pick randomly from best_words
            chosen = random.choice(best_words)
            code.append([ (chosen // (q ** (n - 1 - i))) % q for i in range(n) ])
            uncovered &= ~balls[chosen]
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
    return best_code
