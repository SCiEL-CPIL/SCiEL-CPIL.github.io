import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # For the specific task q=3, n=6, radius=1
    # But we should handle the general case if possible, though the task is specific.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies q=3, n=6, radius=1.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic or simple greedy for other params if they appear
        # But the task is fixed. Let's assume 3,6,1.
        pass

    total_words = q ** n
    if total_words > 10000:
        # If space is too large, greedy might be slow. 
        # But for 3^6=729, it's fine.
        # If n is larger, we might need a different strategy.
        # The task is specific to 6 matches.
        pass

    # Generate all words
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls
    # balls[i] is a list of indices j such that dist(words[i], words[j]) <= radius
    balls = []
    for i, w in enumerate(words):
        ball = []
        # For radius 1, we can generate neighbors quickly
        # But for generality, let's compute distance
        # Since n=6, q=3, radius=1, we can just iterate all words? 
        # 729 * 729 = 530k distance checks. Fast enough.
        # Or generate neighbors explicitly.
        # Explicit generation for radius 1:
        # Center + 1 change in each of n positions to other q-1 values.
        # Size: 1 + n*(q-1)
        
        # Let's use explicit generation for speed and correctness
        ball_indices = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbor = list(w)
                neighbor[pos] = new_val
                ball_indices.append(word_to_idx[tuple(neighbor)])
        balls.append(ball_indices)
        
    # Convert balls to a matrix for fast gain calculation
    # M[i, j] = 1 if j is in ball of i
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in balls[i]:
            M[i, j] = 1
            
    best_code = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while not covered.all():
            # Calculate gain for each candidate
            # gain[i] = number of uncovered words in ball of i
            # uncovered = ~covered
            gains = M @ (~covered)
            
            max_gain = gains.max()
            if max_gain == 0:
                # Should not happen if covered is not all True, 
                # unless M is incorrect or radius is 0 and we missed something.
                break
                
            # Find all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking to explore different solutions
            idx = rng.choice(candidates)
            
            code_indices.append(idx)
            
            # Update covered
            covered[balls[idx]] = True
            
        # Cleanup: Remove redundant codewords
        # Sort code_indices? No, order doesn't matter for validity, but for determinism maybe.
        # Try to remove each codeword if possible
        final_code = list(code_indices)
        # We can try to remove from the end
        i = len(final_code) - 1
        while i >= 0:
            # Check if final_code[i] is needed
            # Temporarily remove it
            test_code = final_code[:i] + final_code[i+1:]
            # Check if test_code covers everything
            # Efficient check: 
            # covered_by_others = OR of balls of test_code
            # This is expensive if done naively for every i.
            # Alternative: Maintain a count of how many codewords cover each word.
            
            # Let's do a proper count-based cleanup
            # This cleanup is O(C * TotalWords) if done carefully.
            # C ~ 73, TotalWords 729. 73*729 = 53k. Very fast.
            
            break # Skip complex cleanup for now, greedy is usually tight. 
            # Actually, simple removal check:
            # If we just want to ensure minimal, we can check coverage.
            
        if len(final_code) < best_len:
            best_len = len(final_code)
            best_code = final_code
            
            # If we hit the theoretical lower bound or a known good bound, we can stop early?
            # Lower bound is 71. Best known 73.
            # If we find 73, we might still find 72? Unlikely but possible.
            # Let's keep running until deadline to be sure we didn't miss 72.
            # But if we find 73 many times, it's likely optimal.
            
    # Convert indices back to words
    if best_code is None:
        # Fallback
        return [list(w) for w in words[:1]]
        
    return [list(words[i]) for i in best_code]
