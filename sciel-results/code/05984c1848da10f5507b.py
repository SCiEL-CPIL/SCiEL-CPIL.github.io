import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    alphabet = range(q)
    total_words = q ** n
    all_words = list(itertools.product(alphabet, repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = {}
    for w in all_words:
        ball = [w]
        for i in range(n):
            for val in alphabet:
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                ball.append(tuple(new_w))
        balls[w] = ball
        
    # Phase 1: Greedy
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        if time.time() > deadline:
            break
        u = random.choice(tuple(uncovered))
        candidates = balls[u]
        best_c = None
        best_gain = -1
        for c in candidates:
            gain = sum(1 for w in balls[c] if w in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c is not None:
            code.append(best_c)
            for w in balls[best_c]:
                uncovered.discard(w)
                
    # Phase 2: Local Search
    cover_count = [0] * total_words
    for c in code:
        for w in balls[c]:
            cover_count[word_to_idx[w]] += 1
            
    while time.time() < deadline:
        removed = False
        swapped = False
        
        # Try Removal
        random.shuffle(code)
        for i in range(len(code)):
            c = code[i]
            safe = True
            for w in balls[c]:
                if cover_count[word_to_idx[w]] == 1:
                    safe = False
                    break
            if safe:
                code.pop(i)
                for w in balls[c]:
                    cover_count[word_to_idx[w]] -= 1
                removed = True
                break
        
        if removed:
            continue
            
        # Try Swap
        # Limit swap attempts to avoid infinite loops on bad swaps
        swap_attempts = 0
        while swap_attempts < 100:
            swap_attempts += 1
            i = random.randint(0, len(code) - 1)
            c = code[i]
            
            # Find words only covered by c
            U_c = [w for w in balls[c] if cover_count[word_to_idx[w]] == 1]
            
            if not U_c:
                # Should have been removed, but maybe logic gap?
                # Just remove it
                code.pop(i)
                for w in balls[c]:
                    cover_count[word_to_idx[w]] -= 1
                removed = True
                break
                
            # Compute intersection of balls of U_c
            S = set(balls[U_c[0]])
            for w in U_c[1:]:
                S.intersection_update(balls[w])
                if not S:
                    break
            
            if S:
                # Pick a candidate from S
                # Prefer candidates not already in code? Or just random.
                # If we pick a word already in code, it's a duplicate, which is bad for distinctness?
                # The problem says "codewords must be distinct".
                # So c' must not be in code.
                # Also c' != c.
                valid_candidates = [x for x in S if x not in code]
                if valid_candidates:
                    c_prime = random.choice(valid_candidates)
                    # Perform swap
                    for w in balls[c]:
                        cover_count[word_to_idx[w]] -= 1
                    for w in balls[c_prime]:
                        cover_count[word_to_idx[w]] += 1
                    code[i] = c_prime
                    swapped = True
                    break
                    
        if swapped:
            continue
            
        if not removed and not swapped:
            # Stuck
            break
            
    return code
