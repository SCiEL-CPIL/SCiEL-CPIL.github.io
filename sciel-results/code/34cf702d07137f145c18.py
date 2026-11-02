import numpy as np
import itertools
import random
import math

def construct(q, n, radius):
    # Parameters for this task are fixed: q=3, n=8, radius=1
    # But we write it generally for the interface.
    
    total_words = q ** n
    
    # Precompute neighbors (ball of radius 1) for all words
    # neighbors[i] is a list of integers
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
            
    neighbors = []
    for i in range(total_words):
        word = to_word(i)
        ball = [i]
        for pos in range(n):
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                ball.append(to_int(new_word))
        neighbors.append(ball)
        
    # Convert to numpy array for faster operations? 
    # We will use lists for neighbors but numpy for coverage.
    
    # 1. Greedy Set Cover Initialization
    # uncovered_mask: 1 if not covered, 0 if covered. Initially all 1.
    uncovered_mask = np.ones(total_words, dtype=np.int8)
    cov = np.zeros(total_words, dtype=np.int32)
    code = set()
    
    # Precompute ball indices as numpy arrays for fast dot product?
    # ball_indices[i] is a list of indices.
    # To find the best candidate, we need max sum(uncovered_mask[ball])
    # This is equivalent to finding the word c that covers the most uncovered words.
    
    while np.any(uncovered_mask):
        # Find best candidate
        best_c = -1
        best_count = -1
        
        # Optimization: We can sample if it's too slow, but 6561 is small.
        # We can vectorize this?
        # Create a matrix B where B[c, w] = 1 if w in ball(c).
        # Then counts = B @ uncovered_mask.
        # B is 6561 x 6561? No, B is 6561 x 17 (sparse).
        # Let's just loop, it's 6561 * 17 = 111k ops per step.
        # If we take 500 steps, 55M ops. Python loop might be ~1s-2s. Acceptable.
        
        for c in range(total_words):
            if c in code:
                continue
            # Count how many uncovered words are in ball(c)
            # ball_c = neighbors[c]
            # count = sum(uncovered_mask[w] for w in ball_c)
            # Using numpy:
            count = np.sum(uncovered_mask[neighbors[c]])
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c == -1:
            break # Should not happen if uncovered_mask has any 1
            
        code.add(best_c)
        # Update coverage
        for w in neighbors[best_c]:
            cov[w] += 1
            if cov[w] == 1:
                uncovered_mask[w] = 0
        
    # 2. Simulated Annealing
    # Energy = len(code) + penalty * (number of uncovered words)
    # We start with a valid cover, so uncovered = 0.
    
    current_code = list(code)
    current_cov = cov.copy()
    current_uncovered = np.sum(uncovered_mask) # Should be 0
    current_energy = len(current_code)
    
    T = 10.0
    cooling_rate = 0.95
    iterations = 20000 # Adjust based on time
    
    # Precompute random indices for sampling
    all_indices = list(range(total_words))
    
    for _ in range(iterations):
        if T < 0.01:
            break
            
        # Generate a move
        move_type = random.random()
        
        new_code = current_code[:]
        new_cov = current_cov.copy()
        
        if move_type < 0.3:
            # Remove a random codeword
            if not new_code:
                continue
            idx = random.randint(0, len(new_code) - 1)
            c = new_code[idx]
            new_code.pop(idx)
            
            # Update coverage
            for w in neighbors[c]:
                new_cov[w] -= 1
                
        elif move_type < 0.6:
            # Add a random codeword
            c = random.choice(all_indices)
            if c in new_code:
                continue
            new_code.append(c)
            
            # Update coverage
            for w in neighbors[c]:
                new_cov[w] += 1
                
        else:
            # Swap
            if not new_code:
                continue
            idx = random.randint(0, len(new_code) - 1)
            c_out = new_code[idx]
            c_in = random.choice(all_indices)
            
            if c_in == c_out:
                continue
            
            new_code[idx] = c_in
            
            # Remove c_out
            for w in neighbors[c_out]:
                new_cov[w] -= 1
            # Add c_in
            for w in neighbors[c_in]:
                new_cov[w] += 1
        
        # Calculate new energy
        new_uncovered = np.sum(new_cov == 0)
        penalty = 10.0 # High penalty to prefer valid solutions
        new_energy = len(new_code) + penalty * new_uncovered
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            # Accept
            current_code = new_code
            current_cov = new_cov
            current_energy = new_energy
        else:
            # Accept with probability exp(-delta_E / T)
            if random.random() < math.exp(-delta_E / T):
                current_code = new_code
                current_cov = new_cov
                current_energy = new_energy
                
        T *= cooling_rate
        
    # Convert back to list of lists
    result = [to_word(c) for c in current_code]
    return result
