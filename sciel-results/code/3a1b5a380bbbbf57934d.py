import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    total_words = q ** n
    words = list(range(total_words))
    
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return t
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls
    balls = []
    for i in range(total_words):
        t_i = idx_to_tuple(i)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == t_i[pos]:
                        continue
                    t_new = list(t_i)
                    t_new[pos] = val
                    ball.append(tuple_to_idx(t_new))
        balls.append(ball)
        
    def greedy_construct():
        cover_count = np.zeros(total_words, dtype=int)
        uncovered = set(words)
        code = []
        
        while uncovered:
            if time.time() > deadline:
                break
            u = random.choice(list(uncovered))
            
            best_c = -1
            max_gain = -1
            
            for c in balls[u]:
                gain = 0
                for w in balls[c]:
                    if w in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            
            if best_c == -1:
                break
                
            code.append(best_c)
            
            for w in balls[best_c]:
                cover_count[w] += 1
                if cover_count[w] == 1:
                    uncovered.discard(w)
                    
        return code, cover_count

    def local_search(code, cover_count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                c = code[i]
                
                unique_words = []
                for w in balls[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                        
                if not unique_words:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    improved = True
                    break
                else:
                    candidates = set(balls[unique_words[0]])
                    for w in unique_words[1:]:
                        candidates.intersection_update(balls[w])
                        if not candidates:
                            break
                    
                    if candidates:
                        # Pick a candidate
                        # Heuristic: pick one that is not already in code?
                        # Or just random.
                        best_c_new = random.choice(list(candidates))
                        
                        if best_c_new != c:
                            for w in balls[c]:
                                cover_count[w] -= 1
                            for w in balls[best_c_new]:
                                cover_count[w] += 1
                            code[i] = best_c_new
                            improved = True
                            break

        return code, cover_count

    best_code = None
    best_len = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline:
        code, cover_count = greedy_construct()
        code, cover_count = local_search(code, cover_count)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
        if best_len <= 186:
            break

    if best_code is None:
        # Fallback to seed-like construction if something went wrong
        # But greedy should always produce something
        return []

    result = [idx_to_tuple(c) for c in best_code]
    return result
