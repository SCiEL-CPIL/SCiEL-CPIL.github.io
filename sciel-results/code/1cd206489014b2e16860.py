import itertools
import random
import time

def construct(q, n, radius):
    # Specific case for the task: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_3_6_1()
    
    # Fallback for other cases: Greedy Set Cover
    # This is a generic implementation, might be slow for large n
    total_words = q ** n
    if total_words > 10000:
        # Too large for brute force greedy in reasonable time without optimization
        # Return a trivial cover (all words) to ensure validity, though score will be bad
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    # Precompute balls
    # Map word tuple to index
    word_to_idx = {}
    idx_to_word = {}
    idx = 0
    for w in itertools.product(range(q), repeat=n):
        word_to_idx[w] = idx
        idx_to_word[idx] = list(w)
        idx += 1
        
    def get_ball(word_idx):
        word = tuple(idx_to_word[word_idx])
        ball = {word_idx}
        for i in range(n):
            for val in range(q):
                if val == word[i]:
                    continue
                new_word = list(word)
                new_word[i] = val
                ball.add(word_to_idx[tuple(new_word)])
        return ball
        
    all_balls = [get_ball(i) for i in range(total_words)]
    
    # Greedy Set Cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        best_word = -1
        max_covered = 0
        for w_idx in range(total_words):
            if w_idx in code: # Optimization: skip if already in code? 
                # Actually, adding the same word again is useless.
                continue
            covered = len(all_balls[w_idx] & uncovered)
            if covered > max_covered:
                max_covered = covered
                best_word = w_idx
        if best_word == -1:
            break
        code.append(best_word)
        uncovered -= all_balls[best_word]
        
    # Local Search: Try to remove redundant codewords
    cov_count = [0] * total_words
    for c in code:
        for p in all_balls[c]:
            cov_count[p] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            can_remove = True
            for p in all_balls[c]:
                if cov_count[p] <= 1:
                    can_remove = False
                    break
            if can_remove:
                for p in all_balls[c]:
                    cov_count[p] -= 1
                code.pop(i)
                changed = True
                
    return [idx_to_word[c] for c in code]

def _solve_3_6_1():
    q = 3
    n = 6
    radius = 1
    total_words = q ** n # 729
    
    # Precompute index mapping
    word_to_idx = {}
    idx_to_word = {}
    idx = 0
    for w in itertools.product(range(q), repeat=n):
        word_to_idx[w] = idx
        idx_to_word[idx] = list(w)
        idx += 1
        
    # Precompute balls for all 729 words
    # Ball size is 1 + 6*2 = 13
    all_balls = []
    for i in range(total_words):
        word = tuple(idx_to_word[i])
        ball = {i}
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                new_word = list(word)
                new_word[pos] = val
                ball.add(word_to_idx[tuple(new_word)])
        all_balls.append(ball)
        
    # Convert balls to lists for faster iteration in some contexts, but sets are good for intersection
    # For coverage counting, we just need the elements.
    
    start_time = time.time()
    timeout = 140 # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    # Strategy: Multiple randomized greedy runs + local search
    # Randomization: In each step of greedy, pick randomly from top K candidates
    
    num_restarts = 200 # We can afford many restarts because each is fast
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Randomized Greedy
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            # Find top candidates
            # Optimization: Instead of checking all 729, we can just check all. 729 is small.
            candidates = []
            for w_idx in range(total_words):
                if w_idx in code:
                    continue
                covered = len(all_balls[w_idx] & uncovered)
                if covered > 0:
                    candidates.append((covered, w_idx))
            
            if not candidates:
                break
                
            # Sort by coverage descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from top 10 (or all if fewer)
            top_k = min(10, len(candidates))
            # Weighted random or uniform from top? Uniform is simple.
            chosen = random.choice(candidates[:top_k])
            w_idx = chosen[1]
            
            code.append(w_idx)
            uncovered -= all_balls[w_idx]
            
        # Local Search: Remove redundant
        # Build coverage counts
        cov_count = [0] * total_words
        for c in code:
            for p in all_balls[c]:
                cov_count[p] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                can_remove = True
                for p in all_balls[c]:
                    if cov_count[p] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for p in all_balls[c]:
                        cov_count[p] -= 1
                    code.pop(i)
                    changed = True
                    
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
            # If we hit the known optimum (73), we can stop early?
            # Yes, 73 is the best known.
            if best_size == 73:
                break
                
    # Fallback if no code found (should not happen)
    if best_code is None:
        # Return a valid cover, e.g., all words
        return [idx_to_word[i] for i in range(total_words)]
        
    return [idx_to_word[c] for c in best_code]
