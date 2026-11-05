import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            pass
        return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute balls for all points
    balls = np.empty((total_points, ball_size), dtype=np.int32)
    neighbors = [None] * total_points
    
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        balls[idx, 0] = idx
        neighbor_idx = 1
        nbrs = []
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            
            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            balls[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            nbrs.append(n_idx)
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            balls[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            nbrs.append(n_idx)
            
        neighbors[idx] = nbrs

    def idx_to_word(idx):
        w = [0] * n
        temp_idx = idx
        for i in range(n):
            w[i] = temp_idx % q
            temp_idx //= q
        return w

    def run_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            uncovered_mask = ~is_covered
            counts = np.sum(uncovered_mask[balls], axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = rng.choice(candidates)
            
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True
            
        return selected

    def prune(selected, counts):
        if not selected:
            return selected, counts
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                c = selected[i]
                if np.min(counts[balls[c]]) > 1:
                    counts[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
                    break
        return selected, counts

    def local_search(selected, counts, deadline):
        best_len = len(selected)
        best_selected = selected[:]
        best_counts = counts.copy()
        
        rng = random.Random()
        
        # Use a set for fast membership checks
        selected_set = set(selected)
        
        while time.time() < deadline:
            # 1. Try to prune
            selected, counts = prune(selected, counts)
            if len(selected) < best_len:
                best_len = len(selected)
                best_selected = selected[:]
                best_counts = counts.copy()
                selected_set = set(selected)
            
            if len(selected) == 0:
                break
                
            # 2. Try a random swap
            i = rng.randint(0, len(selected) - 1)
            c = selected[i]
            
            # Pick a random neighbor
            nbrs = neighbors[c]
            c_prime = rng.choice(nbrs)
            
            if c_prime in selected_set:
                continue
                
            # Tentative swap
            counts[balls[c]] -= 1
            counts[balls[c_prime]] += 1
            
            if np.any(counts[balls[c]] == 0):
                counts[balls[c_prime]] -= 1
                counts[balls[c]] += 1
            else:
                selected[i] = c_prime
                selected_set.remove(c)
                selected_set.add(c_prime)
                selected, counts = prune(selected, counts)
                selected_set = set(selected)
                if len(selected) < best_len:
                    best_len = len(selected)
                    best_selected = selected[:]
                    best_counts = counts.copy()

        return best_selected

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_code = None
    best_len = float('inf')
    
    num_runs = 3
    time_per_run = (deadline - start_time) / num_runs
    
    for run_idx in range(num_runs):
        run_deadline = start_time + (run_idx + 1) * time_per_run
        
        seed = run_idx * 100 + 42
        selected = run_greedy(seed)
        
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in selected:
            counts[balls[idx]] += 1
            
        selected, counts = prune(selected, counts)
        
        selected = local_search(selected, counts, run_deadline)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
        if time.time() > deadline:
            break

    if best_code is None:
        selected = run_greedy(0)
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in selected:
            counts[balls[idx]] += 1
        selected, counts = prune(selected, counts)
        best_code = selected

    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
