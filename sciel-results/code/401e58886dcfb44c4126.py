import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases if needed, though task implies specific
        # But we should handle generally or at least the specific case
        pass

    total_words = 3**n
    # Precompute balls
    # Ball of radius 1 in ternary code of length 8
    # Center: c
    # Neighbors: change one digit to one of the other 2 values.
    # Total 1 + 8*2 = 17 words.
    
    # Map int to word and word to int
    # We work with integers 0..total_words-1
    
    # Precompute neighbors for all words
    # This takes some time but is done once.
    neighbors_list = []
    for i in range(total_words):
        # Decode
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        # Generate neighbors
        balls = [i]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(3):
                if new_val != orig:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    new_val_int = 0
                    for j in range(n):
                        new_val_int += new_digits[j] * (3**j)
                    balls.append(new_val_int)
        neighbors_list.append(balls)
        
    balls_np = np.array(neighbors_list, dtype=np.int32)
    
    covered_counts = np.zeros(total_words, dtype=np.int32)
    code = []
    
    # Greedy Phase
    while time.time() < deadline - 5:
        # Calculate coverage sum for each candidate
        # We want to minimize the number of already covered words in the ball
        # sums[c] = sum(covered_counts[balls[c]])
        sums = covered_counts[balls_np].sum(axis=1)
        best_idx = np.argmin(sums)
        
        # If the best candidate covers 0 new words, we are done (or stuck)
        # But we need to cover everything.
        # Actually, if min sum is 17, it means the ball is fully covered.
        # We should stop if no uncovered words remain?
        # Check if all covered?
        # It's faster to just check if any count is 0.
        if np.any(covered_counts == 0):
             code.append(best_idx)
             covered_counts[balls_np[best_idx]] += 1
        else:
             break

    # Pruning Phase
    # Sort code? No, order doesn't matter for counts.
    # Iterate through code, try to remove.
    # We need to be careful: if we remove one, counts change.
    # A single pass might miss some. Multiple passes better.
    
    max_prune_passes = 10
    for p in range(max_prune_passes):
        if time.time() > deadline - 2:
            break
        removed_any = False
        new_code = []
        for c in code:
            # Check if c is essential
            essential = False
            for v in balls_np[c]:
                if covered_counts[v] == 1:
                    essential = True
                    break
            if essential:
                new_code.append(c)
            else:
                # Remove c
                covered_counts[balls_np[c]] -= 1
                removed_any = True
        code = new_code
        if not removed_any:
            break

    # Convert back to lists of ints
    result = []
    for c in code:
        digits = []
        temp = c
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        result.append(digits)
        
    return result
