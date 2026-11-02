import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Generate initial code (81 words from tetracode)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # 2. Precompute balls
    # Map word to index
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Ball for each word
    balls = []
    for w in all_words:
        ball = set([tuple(w)])
        for i in range(n):
            for d in range(1, q):
                new_w = w.copy()
                new_w[i] = (w[i] + d) % q
                ball.add(tuple(new_w))
        balls.append(ball)
        
    # 3. Initialize coverage counts
    # code_set is a set of tuples for fast lookup
    code_set = set(tuple(w) for w in initial_code)
    coverage = [0] * (q ** n)
    for c in code_set:
        idx = word_to_idx[c]
        for w in balls[idx]:
            coverage[word_to_idx[w]] += 1
            
    # Helper to check if coverage is valid (all > 0)
    def is_valid():
        return min(coverage) > 0
        
    # 4. Local Search: Remove redundant, then Swap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to remove redundant codewords
        removed = True
        while removed and time.time() < deadline:
            removed = False
            for c in list(code_set):
                idx = word_to_idx[c]
                # Tentatively remove
                for w in balls[idx]:
                    coverage[word_to_idx[w]] -= 1
                if is_valid():
                    code_set.remove(c)
                    improved = True
                    removed = True
                    break
                else:
                    # Revert
                    for w in balls[idx]:
                        coverage[word_to_idx[w]] += 1
        
        # Try swaps
        if time.time() > deadline:
            break
            
        # To make swaps efficient, we might want to prioritize swaps that "uncover" fewer words
        # But for now, random or sequential swap attempts.
        # Sequential might get stuck in local minima. Random is better.
        import random
        random.seed(42) # Fixed seed for reproducibility? Or random? 
        # Let's use random to escape local minima.
        
        for _ in range(1000): # Limit swap attempts per iteration
            if time.time() > deadline:
                break
            if len(code_set) == 0:
                break
                
            c = random.choice(list(code_set))
            # Pick a random word w not in code_set
            # To avoid picking from huge space, we can pick a random index
            w_idx = random.randint(0, q**n - 1)
            w = all_words[w_idx]
            if tuple(w) in code_set:
                continue
                
            c_idx = word_to_idx[c]
            # Tentatively remove c
            for w_ in balls[c_idx]:
                coverage[word_to_idx[w_]] -= 1
                
            if is_valid():
                # Add w
                code_set.remove(c)
                code_set.add(tuple(w))
                w_idx_new = word_to_idx[tuple(w)]
                for w_ in balls[w_idx_new]:
                    coverage[word_to_idx[w_]] += 1
                improved = True
                break # Restart removal phase
            else:
                # Revert
                for w_ in balls[c_idx]:
                    coverage[word_to_idx[w_]] += 1

    return [list(w) for w in code_set]
