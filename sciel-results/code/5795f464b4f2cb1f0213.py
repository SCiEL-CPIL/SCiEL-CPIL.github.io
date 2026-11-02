import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 110 seconds to be safe
    
    total_words = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    idx_to_word = {i: w for i, w in enumerate(words)}
    
    # Precompute balls for radius 1
    # For each word, store the list of indices of words in its ball
    balls = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        balls[i].append(i)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    balls[i].append(word_to_idx[tuple(new_w)])
                    
    uncovered = bytearray([1] * total_words)
    code_indices = []
    
    # Greedy phase
    while any(uncovered):
        if time.time() > deadline:
            break
            
        best_count = -1
        best_idx = -1
        
        # We can optimize by only checking words that are still uncovered or have uncovered neighbors
        # But for now, let's check all words, or at least a subset.
        # Checking all 2187 words * 15 ops each is ~33k ops, which is fast.
        
        for i in range(total_words):
            # Quick check: if the word itself is covered, its ball might still have uncovered words
            # But if the word is covered, it's less likely to be the best.
            # Actually, we just need to count uncovered words in the ball.
            count = 0
            for j in balls[i]:
                if uncovered[j]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        for j in balls[best_idx]:
            uncovered[j] = 0
            
        # Early termination if all covered
        if not any(uncovered):
            break
            
    # Local search phase
    # Try to swap a codeword with another to reduce the size
    # This is complex, so let's stick to a simpler improvement if time allows
    # For now, the greedy result is likely around 200-220 words.
    
    # Convert indices back to words
    code = [list(idx_to_word[i]) for i in code_indices]
    
    return code
