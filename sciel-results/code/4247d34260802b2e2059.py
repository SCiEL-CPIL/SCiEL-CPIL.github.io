import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for large instances to avoid memory issues with matrix
    if total_words > 5000:
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        neighbors = []
        for i, w in enumerate(words):
            ball = [i]
            w_list = list(w)
            for pos in range(n):
                old_val = w_list[pos]
                for new_val in range(q):
                    if new_val != old_val:
                        w_list[pos] = new_val
                        idx = word_to_idx[tuple(w_list)]
                        ball.append(idx)
                        w_list[pos] = old_val
            neighbors.append(ball)
            
        uncovered = np.ones(total_words, dtype=bool)
        code_indices = []
        while np.any(uncovered):
            max_cov = -1
            best_cand = -1
            for c in range(total_words):
                cov = np.sum(uncovered[neighbors[c]])
                if cov > max_cov:
                    max_cov = cov
                    best_cand = c
            code_indices.append(best_cand)
            uncovered[neighbors[best_cand]] = False
        return [list(words[i]) for i in code_indices]

    # Fast path for small N (e.g., 3^6 = 729)
    start_time = time.time()
    deadline = start_time + 115.0
    
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors = []
    Ball = np.zeros((total_words, total_words), dtype=np.float32)
    
    for i, w in enumerate(words):
        ball_indices = [i]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    idx = word_to_idx[tuple(w_list)]
                    ball_indices.append(idx)
                    w_list[pos] = old_val
        neighbors.append(ball_indices)
        for idx in ball_indices:
            Ball[i, idx] = 1.0

    best_size = total_words
    best_code = None
    
    def reduce_code(code_indices):
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            for w in neighbors[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code_indices:
                essential = False
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        essential = True
                        break
                if not essential:
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code_indices = new_code
            else:
                break
        return code_indices

    max_restarts = 2000
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        
        uncovered = np.ones(total_words, dtype=np.float32)
        code_indices = []
        noise_scale = 0.01
        
        while np.any(uncovered > 0.5):
            coverages = Ball @ uncovered
            if restart_count > 1:
                coverages += np.random.random(total_words) * noise_scale
            
            best_cand = int(np.argmax(coverages))
            code_indices.append(best_cand)
            
            uncovered[Ball[best_cand] > 0.5] = 0.0
        
        reduced_code = reduce_code(code_indices)
        
        if len(reduced_code) < best_size:
            best_size = len(reduced_code)
            best_code = [list(words[i]) for i in reduced_code]
            
            if best_size <= 73:
                if time.time() > start_time + 100.0:
                    break

    if best_code is not None:
        return best_code
    else:
        return [list(words[i]) for i in range(total_words)]
