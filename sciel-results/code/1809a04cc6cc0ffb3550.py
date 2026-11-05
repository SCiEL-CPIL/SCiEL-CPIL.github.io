import itertools
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # But we should write it generally enough or at least robustly for the given parameters.
    # The evaluator passes q, n, radius.
    
    # If parameters are large, this might be too slow. 
    # But the task specifies n=7, q=3. 
    # 3**7 = 2187. This is small enough for brute-force greedy.
    
    if q**n > 100000:
        # Fallback to the algebraic construction if the space is too large for this heuristic
        # to finish in time, though for 7 matches it's fine.
        # We'll just stick to the heuristic for now as the prompt implies n=7.
        pass

    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Precompute balls for all words
    # ball[w] = set of words within Hamming distance <= radius from w
    balls = {}
    for w in all_words:
        w_tuple = tuple(w)
        ball = {w_tuple}
        # Generate all words at distance 1
        # For radius > 1, we'd need more, but task is radius 1.
        # Let's handle radius=1 specifically for speed, or general radius if small.
        if radius == 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w_tuple)
                    new_w[i] = val
                    ball.add(tuple(new_w))
        balls[w_tuple] = ball

    best_code = None
    best_size = float('inf')
    
    # Number of restarts. 
    # Each restart takes ~ 200 steps * 2187 candidates * 16 neighbors checks.
    # ~ 7 million ops. In Python, maybe 0.1 - 0.5 seconds?
    # Let's aim for 10-20 restarts to be safe within 120s.
    # We can monitor time.
    
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        uncovered = set(all_words)
        code = []
        
        while uncovered:
            # Find candidate with max coverage
            max_cov = -1
            best_candidates = []
            
            for w in all_words:
                cov = 0
                for u in balls[w]:
                    if u in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [w]
                elif cov == max_cov:
                    best_candidates.append(w)
            
            # Pick randomly from best candidates to allow exploration
            selected = random.choice(best_candidates)
            code.append(selected)
            
            # Update uncovered
            for u in balls[selected]:
                if u in uncovered:
                    uncovered.remove(u)
        
        # Deletion Heuristic
        # Try to remove each codeword if possible
        # We do this iteratively until no more can be removed
        changed = True
        while changed:
            changed = False
            # Shuffle order to avoid bias? Or just iterate.
            # Iterating is fine.
            i = 0
            while i < len(code):
                c = code[i]
                # Temporarily remove
                code.remove(c)
                
                # Check if all words are covered by remaining code
                # Efficient check: 
                # We can maintain a coverage count array? 
                # Or just check if union of balls of remaining code covers all_words.
                # Union of balls size is at most 2187.
                # Check: create a set of covered words.
                covered = set()
                for cw in code:
                    covered.update(balls[cw])
                
                if len(covered) == num_words:
                    # Valid removal
                    changed = True
                    # Do not increment i, because list shifted
                else:
                    # Put it back
                    code.insert(i, c)
                    i += 1
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # Optional: Local search / Swap? 
        # Given the time limit, simple greedy+delete is strong.
        # If we have time, we can try swapping one codeword for another 
        # to see if it enables a deletion.
        
    return [list(c) for c in best_code]
