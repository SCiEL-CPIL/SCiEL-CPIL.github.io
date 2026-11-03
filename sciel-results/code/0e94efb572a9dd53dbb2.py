import time
import random
import itertools
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    balls = []
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = tuple(h + list(tail))
            initial_code.add(word_to_idx[word])
            
    in_code = [False] * N
    for idx in initial_code:
        in_code[idx] = True
        
    cover_count = [0] * N
    for idx in initial_code:
        for w in balls[idx]:
            cover_count[w] += 1
            
    uncovered = sum(1 for c in cover_count if c == 0)
    current_size = len(initial_code)
    best_size = current_size
    best_code = set(initial_code)
    
    lambda_val = 5.0
    T = 1.0
    cooling = 0.999
    
    steps = 0
    last_improve = 0
    
    while time.time() < deadline:
        i = random.randint(0, N - 1)
        
        if in_code[i]:
            newly_uncovered = 0
            affected = []
            for w in balls[i]:
                if cover_count[w] == 1:
                    newly_uncovered += 1
                affected.append(w)
            
            delta_E = -1.0 + lambda_val * newly_uncovered
            if random.random() < math.exp(-delta_E / T):
                in_code[i] = False
                for w in affected:
                    cover_count[w] -= 1
                uncovered += newly_uncovered
                current_size -= 1
                
                if uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_code = set(idx for idx in range(N) if in_code[idx])
                    last_improve = steps
        else:
            newly_covered = 0
            affected = []
            for w in balls[i]:
                if cover_count[w] == 0:
                    newly_covered += 1
                affected.append(w)
            
            delta_E = 1.0 - lambda_val * newly_covered
            if random.random() < math.exp(-delta_E / T):
                in_code[i] = True
                for w in affected:
                    cover_count[w] += 1
                uncovered -= newly_covered
                current_size += 1
                
                if uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_code = set(idx for idx in range(N) if in_code[idx])
                    last_improve = steps
        
        T *= cooling
        steps += 1
        
        if steps % 10000 == 0:
            if T < 0.05:
                T = 0.5
            if steps - last_improve > 50000:
                T = 1.0

    return [list(all_words[idx]) for idx in best_code]
