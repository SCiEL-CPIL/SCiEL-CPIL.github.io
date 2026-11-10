import itertools
import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Ensure inputs are integers
    q = int(q)
    n = int(n)
    radius = int(radius)
    
    space_size = q ** n
    
    # If the space is too large for this specific heuristic to finish in time, 
    # we might need a fallback. But for q=3, n=6, it's small.
    
    def decode(val, length, base):
        word = [0] * length
        for i in range(length - 1, -1, -1):
            word[i] = int(val % base)
            val //= base
        return word

    def encode(word, base):
        val = 0
        for sym in word:
            val = val * base + int(sym)
        return val

    if space_size > 10000:
        # Fallback for larger spaces
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            return code
        else:
            return [list(x) for x in itertools.product(range(q), repeat=n)]

    # Fast path
    balls = [None] * space_size
    balls_np = [None] * space_size
    
    for i in range(space_size):
        word = decode(i, n, q)
        neighbors = {i}
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for sym in range(q):
                    if sym == orig: continue
                    new_word = word[:]
                    new_word[pos] = sym
                    neighbors.add(encode(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1+1, n):
                    o1, o2 = word[pos1], word[pos2]
                    for s1 in range(q):
                        if s1 == o1: continue
                        for s2 in range(q):
                            if s2 == o2: continue
                            new_word = word[:]
                            new_word[pos1] = s1
                            new_word[pos2] = s2
                            neighbors.add(encode(new_word, q))
        balls[i] = list(neighbors)
        # FIX: Convert set to list before np.array to avoid potential type errors
        balls_np[i] = np.array(list(neighbors), dtype=np.int32)
    
    best_code = None
    best_len = float('inf')
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        cover_count = np.zeros(space_size, dtype=np.int32)
        code = []
        uncovered_count = space_size
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            best_w = -1
            best_score = -1
            
            random_offset = rng.random()
            
            for w in range(space_size):
                cnt = np.sum(cover_count[balls_np[w]] == 0)
                score = cnt + (w * 0.001 * random_offset) % 0.01
                if score > best_score:
                    best_score = score
                    best_w = w
            
            if best_w == -1:
                break
                
            code.append(best_w)
            for u in balls[best_w]:
                if cover_count[u] == 0:
                    uncovered_count -= 1
                cover_count[u] += 1
        
        if time.time() > deadline:
            break
            
        current_len = len(code)
        
        # Pruning Phase
        indices = list(range(len(code)))
        rng.shuffle(indices)
        
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            for idx in indices:
                if time.time() > deadline:
                    break
                
                c = code[idx]
                if c is None:
                    continue
                
                redundant = True
                for u in balls[c]:
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for u in balls[c]:
                        cover_count[u] -= 1
                    code[idx] = None
                    current_len -= 1
                    removed_any = True
            
            indices = [i for i in indices if code[i] is not None]
            rng.shuffle(indices)
        
        final_code = [c for c in code if c is not None]
        
        if len(final_code) < best_len:
            best_len = len(final_code)
            best_code = final_code
            
    if best_code is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert encoded integers back to lists of Python integers
    result = [decode(w, n, q) for w in best_code]
    return result
