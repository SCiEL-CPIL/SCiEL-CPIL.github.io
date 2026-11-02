import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    N = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((N, ball_size), dtype=np.int32)
    
    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors.append(new_w)
        balls[w] = neighbors

    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]
    
    def greedy_construct():
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            gains = uncovered[balls].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            ball = balls[best_idx]
            cover_count[ball] += 1
            uncovered[ball] = False
            
        return selected, cover_count, uncovered

    def init_state(selected):
        counts = np.zeros(N, dtype=np.int32)
        for c in selected:
            counts[balls[c]] += 1
        U = np.sum(counts == 0)
        return counts, U

    # 1. Initial Greedy
    selected, counts, uncovered = greedy_construct()
    if np.any(uncovered):
        return [idx_to_list(w) for w in range(N)]
        
    U = 0
    K = len(selected)
    best_K = K
    best_selected = selected.copy()
    selected_set = set(selected)
    
    # Simulated Annealing
    T = 10.0 # Start high to allow exploration
    cooling_rate = 0.9995
    min_T = 0.1
    lambda_U = 5.0 # Penalty for uncovered
    
    while time.time() < deadline - 5.0:
        if T < min_T:
            break
            
        r = random.random()
        
        if r < 0.3:
            # Remove
            if K == 0: continue
            i = random.randint(0, K-1)
            c = selected[i]
            
            ball = balls[c]
            old_counts = counts[ball]
            zero_mask = old_counts == 0
            dU = -np.sum(zero_mask)
            counts[ball] -= 1
            U += dU
            K -= 1
            selected.pop(i)
            selected_set.remove(c)
            
            dE = -1.0 + lambda_U * dU
            
            if dE < 0 or (dE / T < 50 and random.random() < np.exp(-dE / T)):
                if U == 0 and K < best_K:
                    best_K = K
                    best_selected = selected.copy()
            else:
                selected.insert(i, c)
                selected_set.add(c)
                counts[ball] += 1
                U -= dU
                K += 1
                
        elif r < 0.6:
            # Add
            c = random.randint(0, N-1)
            if c in selected_set:
                continue
                
            ball = balls[c]
            old_counts = counts[ball]
            zero_mask = old_counts == 0
            dU = -np.sum(zero_mask)
            counts[ball] += 1
            U += dU
            K += 1
            selected.append(c)
            selected_set.add(c)
            
            dE = 1.0 + lambda_U * dU
            
            if dE < 0 or (dE / T < 50 and random.random() < np.exp(-dE / T)):
                if U == 0 and K < best_K:
                    best_K = K
                    best_selected = selected.copy()
            else:
                selected.pop()
                selected_set.remove(c)
                counts[ball] -= 1
                U -= dU
                K -= 1
                
        elif r < 0.9:
            # Swap
            if K == 0: continue
            i = random.randint(0, K-1)
            c1 = selected[i]
            c2 = random.randint(0, N-1)
            if c2 == c1 or c2 in selected_set:
                continue
                
            ball1 = balls[c1]
            ball2 = balls[c2]
            
            old1 = counts[ball1]
            zero1 = old1 == 0
            dU1 = -np.sum(zero1)
            counts[ball1] -= 1
            
            old2 = counts[ball2]
            zero2 = old2 == 0
            dU2 = -np.sum(zero2)
            counts[ball2] += 1
            
            dU = dU1 + dU2
            U += dU
            
            dE = lambda_U * dU
            
            if dE < 0 or (dE / T < 50 and random.random() < np.exp(-dE / T)):
                selected[i] = c2
                selected_set.remove(c1)
                selected_set.add(c2)
                if U == 0 and K < best_K:
                    best_K = K
                    best_selected = selected.copy()
            else:
                counts[ball2] -= 1
                counts[ball1] += 1
                U -= dU
                
        else:
            # 2-for-1
            if K < 2: continue
            i1 = random.randint(0, K-1)
            i2 = random.randint(0, K-1)
            if i1 == i2: continue
            if i1 > i2: i1, i2 = i2, i1
            
            c1 = selected[i1]
            c2 = selected[i2]
            c3 = random.randint(0, N-1)
            if c3 in selected_set:
                continue
                
            ball1 = balls[c1]
            ball2 = balls[c2]
            ball3 = balls[c3]
            
            old1 = counts[ball1]
            zero1 = old1 == 0
            dU1 = -np.sum(zero1)
            counts[ball1] -= 1
            
            old2 = counts[ball2]
            zero2 = old2 == 0
            dU2 = -np.sum(zero2)
            counts[ball2] -= 1
            
            old3 = counts[ball3]
            zero3 = old3 == 0
            dU3 = -np.sum(zero3)
            counts[ball3] += 1
            
            dU = dU1 + dU2 + dU3
            U += dU
            K -= 1
            
            selected.pop(i2)
            selected.pop(i1)
            selected_set.remove(c1)
            selected_set.remove(c2)
            selected.append(c3)
            selected_set.add(c3)
            
            dE = -1.0 + lambda_U * dU
            
            if dE < 0 or (dE / T < 50 and random.random() < np.exp(-dE / T)):
                if U == 0 and K < best_K:
                    best_K = K
                    best_selected = selected.copy()
            else:
                selected.pop()
                selected_set.remove(c3)
                selected.insert(i1, c1)
                selected.insert(i2, c2)
                selected_set.add(c1)
                selected_set.add(c2)
                
                counts[ball3] -= 1
                counts[ball2] += 1
                counts[ball1] += 1
                U -= dU
                K += 1
                
        T *= cooling_rate
        
    return [idx_to_list(w) for w in best_selected]
