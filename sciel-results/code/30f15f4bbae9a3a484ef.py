import numpy as np
import time
import itertools

def construct(q, n, radius):
    total_words = q ** n
    if total_words > 100000:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    def index_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    def digits_to_index(digits):
        idx = 0
        for i in range(n):
            idx = idx * q + digits[i]
        return idx

    balls = []
    for i in range(total_words):
        d = index_to_digits(i)
        covered = [i]
        for pos in range(n):
            original = d[pos]
            for val in range(q):
                if val == original:
                    continue
                d[pos] = val
                covered.append(digits_to_index(d))
            d[pos] = original
        balls.append(covered)
    
    balls_np = [np.array(b) for b in balls]
    
    def prune_fast(selected_indices, balls_np):
        cover_count = np.zeros(total_words, dtype=int)
        for idx in selected_indices:
            cover_count[balls_np[idx]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_indices)):
                idx = selected_indices[i]
                if np.any(cover_count[balls_np[idx]] == 1):
                    continue
                cover_count[balls_np[idx]] -= 1
                selected_indices = selected_indices[:i] + selected_indices[i+1:]
                changed = True
                break
        return selected_indices

    def greedy_cover(random_seed=0):
        rng = np.random.default_rng(random_seed)
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            coverages = np.array([np.sum(uncovered[balls_np[i]]) for i in range(total_words)])
            max_cov = np.max(coverages)
            if max_cov == 0:
                break
            candidates = np.where(coverages == max_cov)[0]
            idx = rng.choice(candidates)
            selected.append(idx)
            uncovered[balls_np[idx]] = False
            
        selected = prune_fast(selected, balls_np)
        return selected

    best_solution = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 100
    
    for seed in range(10000): # Many restarts
        if time.time() > deadline:
            break
            
        sol = greedy_cover(seed)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
        if best_len == 71:
            break

    if best_solution is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    codewords = []
    for idx in best_solution:
        d = index_to_digits(idx)
        codewords.append(d)
        
    return codewords
