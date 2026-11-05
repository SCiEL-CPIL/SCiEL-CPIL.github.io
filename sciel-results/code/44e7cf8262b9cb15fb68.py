import numpy as np
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # The evaluator passes these, but we optimize for this specific case if possible, 
    # or write generic code. The prompt implies generic interface but specific task.
    # "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must handle q, n, radius.
    # However, the known hard case is 3, 7, 1.
    # For other cases, a simple greedy might be sufficient or we fallback to simple methods.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Greedy construction
        return _greedy_cover(q, n, radius)
        
    # Specific optimization for K_3(7,1)
    total_words = 3**7
    # Precompute balls for all 2187 words
    # Ball of radius 1 in ternary code of length 7
    # Center + 7 positions * 2 other values = 15 words
    
    # Convert word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0]*n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Generate all balls
    balls = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        ball = [idx]
        for i in range(n):
            for val in range(q):
                if val == w[i]: continue
                w[i] = val
                ball.append(word_to_idx(w))
                w[i] = w[i] # restore? No, we modified w. 
                # Better to generate from scratch or track changes
        # Let's do it cleaner
        ball = [idx]
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val == orig: continue
                w[i] = val
                ball.append(word_to_idx(w))
            w[i] = orig
        balls.append(ball)
        
    balls = np.array(balls, dtype=np.int32)
    
    # Initial code: The 186 word code from the seed
    # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    import itertools
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    code = [word_to_idx(w) for w in initial_code_words]
    code_set = set(code)
    
    # Coverage array
    coverage = np.zeros(total_words, dtype=np.int32)
    for c in code:
        coverage[balls[c]] += 1
        
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Local Search
    while time.time() < deadline:
        # 1. Try to delete
        deleted = False
        # Randomize order of checking to avoid bias
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for i in indices:
            c = code[i]
            # Check if redundant: all neighbors have coverage > 1
            # coverage[balls[c]] contains the count including c
            # So we need min(coverage[balls[c]]) >= 2
            if np.min(coverage[balls[c]]) >= 2:
                # Remove c
                coverage[balls[c]] -= 1
                code.pop(i)
                code_set.remove(c)
                deleted = True
                # Restart deletion loop immediately
                break
                
        if deleted:
            continue
            
        # 2. Try to swap
        # If we get here, no single deletion is possible.
        # Pick a codeword to move.
        # Heuristic: Pick one that covers many critical words (count=1)
        # Or just random.
        
        # Let's try to find a promising swap
        # Identify critical words (count == 1)
        critical_mask = (coverage == 1)
        num_critical = np.sum(critical_mask)
        
        if num_critical == 0:
            # Should have been deletable?
            # If no critical words, min coverage >= 2 for all words?
            # Not necessarily. Some words might have count 2, others 3.
            # But if min coverage >= 2 everywhere, then any c is redundant.
            # So if we are here, there must be critical words.
            pass
        
        # Pick a random codeword c to remove
        # Bias towards codewords covering critical words?
        # For now, random
        c_idx = random.randint(0, len(code)-1)
        c = code[c_idx]
        
        # Find a candidate c'
        # Strategy: Pick a random critical word w, pick a random c' in B(w)
        # This ensures c' covers a word that is at risk.
        critical_words = np.where(critical_mask)[0]
        
        swap_accepted = False
        # Try a few swaps
        for _ in range(10):
            if len(critical_words) == 0:
                break
            w = critical_words[random.randint(0, len(critical_words)-1)]
            # Pick random neighbor of w
            c_prime = balls[w, random.randint(0, 14)]
            
            if c_prime in code_set:
                continue
                
            # Test swap
            # Remove c
            coverage[balls[c]] -= 1
            # Add c'
            coverage[balls[c_prime]] += 1
            
            if np.min(coverage) >= 0:
                # Valid swap
                code[c_idx] = c_prime
                code_set.remove(c)
                code_set.add(c_prime)
                swap_accepted = True
                break
            else:
                # Revert
                coverage[balls[c_prime]] -= 1
                coverage[balls[c]] += 1
                
        if not swap_accepted:
            # If we can't find a swap with this c, try another c or restart?
            # Just continue loop, next iteration will pick different c
            pass

    # Convert back to list of lists
    result = [idx_to_word(c) for c in code]
    return result

def _greedy_cover(q, n, radius):
    """Fallback for other parameters."""
    total = q**n
    # This is a placeholder. For the specific task, the main logic is used.
    # A simple greedy:
    code = []
    covered = [0]*total
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0]*n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls
    balls = []
    for idx in range(total):
        w = idx_to_word(idx)
        ball = [idx]
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val == orig: continue
                w[i] = val
                ball.append(word_to_idx(w))
            w[i] = orig
        balls.append(ball)

    while True:
        # Find uncovered word
        uncovered = [i for i in range(total) if covered[i] == 0]
        if not uncovered:
            break
        
        # Pick random uncovered word
        w = uncovered[random.randint(0, len(uncovered)-1)]
        
        # Find best c in B(w)
        best_c = -1
        best_count = -1
        for c in balls[w]:
            count = sum(1 for x in balls[c] if covered[x] == 0)
            if count > best_count:
                best_count = count
                best_c = c
                
        # Add best_c
        code.append(best_c)
        for x in balls[best_c]:
            covered[x] += 1
            
    return [idx_to_word(c) for c in code]
