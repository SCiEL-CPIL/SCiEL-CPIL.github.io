import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # The evaluator passes q, n, radius. We assume the task is fixed to this instance 
    # but the function signature is generic. 
    # However, the problem statement says "Football pool problem for 6 matches... K_3(6,1)".
    # The seed code assumes n=4 logic for tetracode but generalizes. 
    # We will implement a generic solver that works well for q=3, n=6, r=1.
    
    total_words = q ** n
    if total_words > 10000:
        # Fallback to seed logic for larger instances if needed, 
        # but the task specifies n=6.
        # For safety, if n is large, use the seed's direct sum if applicable.
        # The seed uses tetracode for n>=4.
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Simple greedy for other cases
             pass

    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        start_time = time.time()
        deadline = start_time + 140.0  # 10s margin
        
        # Generate all words
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        # Precompute neighbors (ball of radius 1)
        # A word w is covered by c if dist(w, c) <= 1
        # dist <= 1 means w == c OR w differs in exactly 1 position.
        neighbors = []
        for i, w in enumerate(words):
            ball = [i]
            w_list = list(w)
            for pos in range(n):
                old_val = w_list[pos]
                for new_val in range(q):
                    if new_val != old_val:
                        w_list[pos] = new_val
                        idx = word_to_idx[tuple(w_list)]
                        ball.append(idx)
                        w_list[pos] = old_val
            neighbors.append(ball)
            
        # Convert to numpy arrays for faster summing? 
        # neighbors is a list of lists of ints.
        
        best_code = None
        best_size = total_words # Upper bound
        
        # Number of restarts
        max_restarts = 500
        restart_count = 0
        
        while True:
            if time.time() > deadline:
                break
            restart_count += 1
            
            # Randomized Greedy
            uncovered = np.ones(total_words, dtype=bool)
            code_indices = []
            
            while np.any(uncovered):
                # Find candidate with max coverage
                max_cov = -1
                best_cand = -1
                candidates = []
                
                # Optimization: Only consider candidates that cover at least one uncovered word?
                # Actually, every candidate covers itself. If it's uncovered, cov >= 1.
                # We can just iterate all.
                
                for c in range(total_words):
                    # Coverage is number of uncovered words in ball(c)
                    # neighbors[c] is a list of indices
                    cov = np.sum(uncovered[neighbors[c]])
                    if cov > max_cov:
                        max_cov = cov
                        best_cand = c
                        candidates = [c]
                    elif cov == max_cov:
                        candidates.append(c)
                        
                # Pick randomly from best candidates
                chosen = random.choice(candidates)
                code_indices.append(chosen)
                
                # Update uncovered
                uncovered[neighbors[chosen]] = False
                
            current_size = len(code_indices)
            if current_size < best_size:
                best_size = current_size
                best_code = [words[i] for i in code_indices]
                
                # Early exit if we hit the known lower bound or best known
                if best_size <= 73:
                    break

        if best_code is not None:
            return best_code
        else:
            # Fallback if no restarts finished (should not happen)
            return [words[i] for i in range(total_words)]

    # Generic fallback for other parameters (not expected in this task)
    # Simple greedy
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    neighbors = []
    for i, w in enumerate(words):
        ball = []
        for u in words:
            dist = sum(1 for a, b in zip(w, u) if a != b)
            if dist <= radius:
                ball.append(word_to_idx[u])
        neighbors.append(ball)
        
    uncovered = np.ones(len(words), dtype=bool)
    code_indices = []
    while np.any(uncovered):
        max_cov = -1
        best_cand = -1
        for c in range(len(words)):
            cov = np.sum(uncovered[neighbors[c]])
            if cov > max_cov:
                max_cov = cov
                best_cand = c
        code_indices.append(best_cand)
        uncovered[neighbors[best_cand]] = False
        
    return [words[i] for i in code_indices]
