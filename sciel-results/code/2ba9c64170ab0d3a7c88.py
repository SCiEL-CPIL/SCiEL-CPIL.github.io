import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Construct a covering code using a randomized greedy heuristic followed by local search (reduction).
    """
    start_time = time.time()
    timeout = 115  # Safety margin from 120s
    
    total_words = q ** n
    
    # Precompute balls (neighbors within Hamming distance 'radius')
    # For radius 1, ball size is 1 + n*(q-1)
    # We represent words as integers 0..total_words-1
    
    # Helper to get neighbors
    def get_neighbors(word_int):
        digits = [0] * n
        temp = word_int
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        
        neighbors = [word_int]
        
        if radius >= 1:
            for i in range(n):
                orig = digits[i]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    diff = (new_val - orig) * (q ** i)
                    neighbors.append(word_int + diff)
        
        return neighbors

    # Precompute balls for all words
    balls = []
    for i in range(total_words):
        balls.append(get_neighbors(i))
        
    # balls is a list of lists of ints.
    # Note: For Hamming distance 1, the relation is symmetric.
    
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = np.ones(total_words, dtype=bool)
        # Initial coverage count is just the size of the ball, since all words are uncovered
        coverage_count = np.array([len(balls[i]) for i in range(total_words)], dtype=np.int32)
        
        code = []
        
        while True:
            if time.time() - start_time > timeout:
                break
                
            # Find max coverage
            # To allow randomization, we can find all indices with max value and pick one
            max_val = np.max(coverage_count)
            if max_val == 0:
                break
                
            candidates = np.where(coverage_count == max_val)[0]
            
            # Randomly pick one candidate to break ties / introduce randomness
            # Or pick the first one. Random picking is better for diversity.
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(best_idx)
            
            # Update coverage
            # For each word w in balls[best_idx]:
            #   if uncovered[w]:
            #     uncovered[w] = False
            #     for each d in balls[w]:
            #       coverage_count[d] -= 1
             
            for w in balls[best_idx]:
                if uncovered[w]:
                    uncovered[w] = False
                    # Decrement coverage for all d that cover w
                    # Since relation is symmetric, d covers w iff w covers d
                    # So d is in balls[w]
                    for d in balls[w]:
                        coverage_count[d] -= 1
                        
        return code

    def reduce_code(code, seed):
        random.seed(seed)
        # Compute cover_count: how many codewords cover each word
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        new_code = []
        # Shuffle to randomize removal order
        shuffled_code = code[:]
        random.shuffle(shuffled_code)
        
        for c in shuffled_code:
            # Try to remove c
            # Check if any word in balls[c] has count 1 (becoming 0)
            is_essential = False
            for w in balls[c]:
                cover_count[w] -= 1
                if cover_count[w] == 0:
                    is_essential = True
                    break
            
            if is_essential:
                # Put c back
                for w in balls[c]:
                    cover_count[w] += 1
                new_code.append(c)
            else:
                # c is redundant, keep it removed
                pass
                
        return new_code

    best_code = None
    best_size = total_words + 1
    
    # Run multiple iterations with different seeds
    # We can run many fast iterations
    max_iterations = 50
    current_seed = 42
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout - 5:
            break
            
        # 1. Greedy
        code = greedy_cover(current_seed)
        
        # 2. Reduce
        code = reduce_code(code, current_seed + 1)
        
        # 3. Reduce again (sometimes allows more removals)
        code = reduce_code(code, current_seed + 2)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # Optional: Try a few swaps if time permits? 
        # For now, greedy+reduce is a strong baseline.
        
        current_seed += 100

    if best_code is None:
        # Fallback: just return greedy without reduction if it failed?
        # Or a simple greedy
        code = greedy_cover(0)
        best_code = code

    # Convert integer codes to list of lists of digits
    final_code = []
    for c_int in best_code:
        digits = []
        temp = c_int
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        final_code.append(digits)
        
    return final_code
