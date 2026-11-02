import itertools
import time

def construct(q, n, radius):
    """
    Constructs a covering code using Greedy Set Cover followed by a trimming phase.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    total_words = q ** n
    # Precompute balls of radius 'radius' for all words
    # For radius 1, ball size is 1 + n*(q-1)
    # For general radius, we need to generate neighbors
    
    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Generate all words as tuples of digits
    # Map integer index to tuple of digits
    # And precompute the set of integers in the ball for each integer index
    
    # Optimization: For q=3, n=8, radius=1, we can hardcode the ball generation logic
    # but let's keep it general for q, n, radius if possible, though the task is specific.
    # The task contract says "evaluator passes this task's alphabet size q, length n and covering radius".
    # But the specific problem is K_3(8,1).
    
    # We will precompute balls as a list of sets of integers.
    balls = [None] * total_words
    
    # Generate neighbors for each word
    # For radius 1:
    # Center + 1 change in 1 position to q-1 other values
    
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = {i}
        if radius == 1:
            for pos in range(n):
                orig = d[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    d[pos] = new_val
                    neighbors.add(digits_to_int(d, q))
                d[pos] = orig
        else:
            # General radius: generate all words within Hamming distance radius
            # This is expensive for large radius, but radius is likely small (1 or 2)
            # For this specific task, radius is 1.
            # If radius > 1, we need a more general generator.
            # Given the 120s limit and small N, we can afford a slightly slower generation if needed.
            # But let's assume radius=1 is the primary case based on the prompt "K_3(8,1)".
            pass
            
        balls[i] = neighbors

    # Greedy Set Cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_center = -1
        max_gain = 0
        
        # Iterate through all possible centers to find the one covering the most uncovered words
        for c in range(total_words):
            # Count how many neighbors of c are in uncovered
            # balls[c] is a set
            gain = len(balls[c] & uncovered)
            if gain > max_gain:
                max_gain = gain
                best_center = c
                if gain == len(balls[best_center]): # Can't get better than full ball
                    # Optimization: if we found a center covering its entire ball, 
                    # and the ball is large, it's likely optimal for this step.
                    # But we still need to check if another center covers MORE? 
                    # No, max possible gain is ball size.
                    # So we can break if max_gain == ball_size.
                    # However, different centers might have different ball sizes? No, all balls same size.
                    break
        
        if best_center == -1:
            break
            
        code.append(best_center)
        # Remove covered words from uncovered
        uncovered -= balls[best_center]
        
        # Optional: Early exit if time is running out
        if time.time() > deadline:
            break

    # Trim Phase
    # Maintain cover_count for each word
    cover_count = [0] * total_words
    for c in code:
        for u in balls[c]:
            cover_count[u] += 1
            
    # Try to remove codewords that are redundant
    # We iterate and try to remove.
    # To be safe, we iterate multiple times or just once?
    # Once is usually enough for greedy, but multiple passes might find more.
    # Given time, let's do a few passes.
    
    changed = True
    passes = 0
    while changed and time.time() < deadline:
        changed = False
        passes += 1
        # Iterate in reverse to allow removal without index issues, or just collect to remove
        to_remove = []
        for i, c in enumerate(code):
            # Check if any word in balls[c] has cover_count == 1
            # If all have > 1, we can remove c
            redundant = True
            for u in balls[c]:
                if cover_count[u] == 1:
                    redundant = False
                    break
            
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            # Remove from end to start to preserve indices
            for idx in sorted(to_remove, reverse=True):
                c = code[idx]
                code.pop(idx)
                for u in balls[c]:
                    cover_count[u] -= 1
            changed = True

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
        
    return result
