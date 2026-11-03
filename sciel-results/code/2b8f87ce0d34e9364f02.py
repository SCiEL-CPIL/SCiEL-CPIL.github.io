import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Initial Code: Direct sum of tetracode (len 4) and full space (len n-4)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    codewords = []
    for h in tetracode:
        for tail in tails:
            codewords.append(tuple(h + list(tail)))
            
    space_size = q ** n
    word_to_idx = {}
    idx_to_word = {}
    for w in itertools.product(range(q), repeat=n):
        idx = 0
        for val in w:
            idx = idx * q + val
        word_to_idx[w] = idx
        idx_to_word[idx] = w
        
    balls = [[] for _ in range(space_size)]
    for w in itertools.product(range(q), repeat=n):
        idx = word_to_idx[w]
        balls[idx].append(idx)
        for i in range(n):
            for val in range(q):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                new_idx = word_to_idx[tuple(new_w)]
                balls[idx].append(new_idx)
                
    # We will try multiple random permutations of the removal order
    best_size = len(codewords)
    best_code = set(codewords)
    
    # Try a few times with different random seeds/orders
    for attempt in range(5):
        if time.time() > deadline - 10: # Leave 10s for finalization
            break
            
        current_code = set(codewords)
        cover_count = [0] * space_size
        
        for c in current_code:
            c_idx = word_to_idx[c]
            for w_idx in balls[c_idx]:
                cover_count[w_idx] += 1
                
        # Shuffle the list of codewords to vary removal order
        code_list = list(current_code)
        random.shuffle(code_list)
        
        while True:
            removed = False
            for c in code_list:
                if c not in current_code:
                    continue
                if time.time() > deadline - 10:
                    break
                    
                c_idx = word_to_idx[c]
                can_remove = True
                for w_idx in balls[c_idx]:
                    if cover_count[w_idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    current_code.remove(c)
                    for w_idx in balls[c_idx]:
                        cover_count[w_idx] -= 1
                    removed = True
                    # Re-shuffle remaining? Or just continue. 
                    # Continuing is faster.
                    
            if not removed:
                break
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code
            
    return [list(c) for c in best_code]
