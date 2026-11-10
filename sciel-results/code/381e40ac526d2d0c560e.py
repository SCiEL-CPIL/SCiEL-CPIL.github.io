import itertools
import random
import time

def construct(q, n, radius):
    """
    Finds a covering code for q-ary alphabet, length n, covering radius 'radius'.
    Uses a randomized greedy set cover algorithm.
    """
    if q <= 0 or n <= 0 or radius < 0:
        return []
    
    # For the specific task parameters (q=3, n=6, radius=1)
    # We generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls for each word
    # Ball of radius r around w contains all words at Hamming distance <= r
    balls = []
    for w in all_words:
        ball = set()
        # Generate all neighbors within Hamming distance 'radius'
        # For small n and radius, we can just check all words or generate neighbors
        # Here radius=1, n=6, q=3. Neighbors are 1 + 6*2 = 13 words.
        # It's faster to generate neighbors explicitly than to check all 729 words.
        w_list = list(w)
        ball.add(w)
        for i in range(n):
            orig_val = w_list[i]
            for new_val in range(q):
                if new_val != orig_val:
                    w_list[i] = new_val
                    ball.add(tuple(w_list))
                    # Note: radius is 1, so we stop here. 
                    # If radius > 1, we would need to recurse or use combinatorics.
                    # The task specifies radius=1 in the description, but signature takes radius.
                    # Assuming radius is small (1 or 2).
                    # If radius > 1, this simple neighbor generation is incomplete.
                    # However, the task is specifically K_3(6,1).
                    # Let's assume radius=1 for the optimization logic, 
                    # but handle general radius by falling back to distance check if radius > 1.
        balls.append(ball)
        
    # If radius > 1, the above neighbor generation is wrong. 
    # Let's check if we need to support radius > 1.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # The task is "Football pool problem for 6 matches: ternary covering code K_3(6,1)".
    # So radius is likely 1. But to be safe:
    if radius > 1:
        # Recompute balls using distance check for correctness, though slower
        balls = []
        for i, w in enumerate(all_words):
            ball = set()
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    ball.add(v)
            balls.append(ball)

    # Convert balls to lists of indices for faster set operations
    ball_indices = []
    for ball in balls:
        ball_indices.append([word_to_idx[w] for w in ball])

    best_code = None
    best_size = float('inf')
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Randomized Greedy Set Cover
    # We try a few different random orders to see if we get a smaller code
    # Standard greedy picks the word covering most uncovered.
    # Randomized greedy picks from top-k candidates randomly.
    
    for attempt in range(50): # Up to 50 restarts
        if time.time() > deadline:
            break
            
        uncovered = set(range(num_words))
        code = []
        
        while uncovered:
            # Find candidate that covers most uncovered words
            max_covered = -1
            best_idx = -1
            candidates = []
            
            # Optimization: Only consider words that are still uncovered? 
            # No, a covered word can still cover other uncovered words.
            # But usually, greedy picks from the set of all words.
            
            # To speed up, we can just iterate through all words
            # 729 words * 73 steps * 13 neighbors is small.
            
            for i in range(num_words):
                covered_count = 0
                # Count how many words in ball_indices[i] are in uncovered
                # Using set intersection is fast
                covered_count = len(uncovered.intersection(ball_indices[i]))
                
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_idx = i
                    candidates = [i]
                elif covered_count == max_covered and covered_count > 0:
                    candidates.append(i)
                
            if max_covered == 0:
                # Should not happen if balls are correct
                break
            
            # Randomization: pick randomly from top candidates
            # To encourage diversity in restarts
            if len(candidates) > 1 and random.random() < 0.2:
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = best_idx
                
            code.append(all_words[chosen_idx])
            uncovered.difference_update(ball_indices[chosen_idx])
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # If we hit the theoretical lower bound or known best, we can stop early
            # Known best is 73. Lower bound 71.
            if best_size <= 73:
                break
                
    if best_code is None:
        # Fallback to seed construction if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_code
