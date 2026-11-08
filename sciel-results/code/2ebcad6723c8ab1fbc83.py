import time
import math
import numpy as np
import random

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Compute sum set
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
    
    # Compute difference set
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
    
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return 0.0
    
    ratio = n_diff / n_sum
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_max = math.log(2 * maxU + 1)
    
    if log_max <= 0:
        return 1.0
    
    return 1.0 + log_ratio / log_max

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    def try_set(U):
        nonlocal best_score, best_set
        if time.time() > deadline:
            return
        # Ensure U is a set of unique non-negative integers containing 0
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
            U = sorted(U)
        if len(U) < 2:
            return
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = list(U)

    # 1. Exhaustive search for small sets (fast)
    # Size 4
    for a in range(1, 25):
        for b in range(a + 1, 25):
            for c in range(b + 1, 25):
                try_set([0, a, b, c])
                if time.time() > deadline: return best_set
    # Size 5
    for a in range(1, 18):
        for b in range(a + 1, 18):
            for c in range(b + 1, 18):
                for d in range(c + 1, 18):
                    try_set([0, a, b, c, d])
                    if time.time() > deadline: return best_set
    # Size 6
    for a in range(1, 14):
        for b in range(a + 1, 14):
            for c in range(b + 1, 14):
                for d in range(c + 1, 14):
                    for e in range(d + 1, 14):
                        try_set([0, a, b, c, d, e])
                        if time.time() > deadline: return best_set

    # 2. Structured Candidates
    # Geometric
    for base in [2, 3, 4, 5, 7, 10]:
        for n in range(3, 15):
            U = [0]
            v = 1
            for i in range(n):
                U.append(v)
                v *= base
            if U[-1] < 10**7:
                try_set(U)
                
    # Arithmetic with offset
    for start in [1, 2, 3, 5]:
        for step in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
            for n in range(3, 25):
                U = [0] + [start + i*step for i in range(n)]
                if U[-1] < 10**6:
                    try_set(U)
                    
    # "Gap" structures: {0, 1, M, 2M, ...}
    for M in [2, 3, 5, 10, 20, 50, 100, 200]:
        for n in range(2, 20):
            U = [0, 1] + [i*M for i in range(2, n+2)]
            if U[-1] < 10**6:
                try_set(U)
            
    # 3. Local Search / Simulated Annealing
    random.seed(42)
    
    def local_search(initial_U, max_steps=1000):
        current_U = sorted(set(initial_U))
        if 0 not in current_U:
            current_U.append(0)
            current_U = sorted(current_U)
        
        current_score = compute_score(current_U)
        best_local_score = current_score
        best_local_U = list(current_U)
        
        T = 0.05 # Initial temperature
        cooling = 0.995
        
        for step in range(max_steps):
            if time.time() > deadline:
                return best_local_U, best_local_score
            
            # Choose a move
            move_type = random.choice(['swap', 'add', 'remove'])
            new_U = list(current_U)
            
            if move_type == 'swap' and len(new_U) > 1:
                idx = random.randint(1, len(new_U) - 1) # Don't swap 0
                max_val = max(new_U)
                # Mix of local and global jumps
                if random.random() < 0.5:
                    new_val = random.randint(1, max_val * 2)
                else:
                    new_val = new_U[idx] + random.randint(-5, 5)
                    if new_val < 1: new_val = 1
                if new_val not in new_U:
                    new_U[idx] = new_val
                else:
                    continue
            elif move_type == 'add':
                if len(new_U) > 40: # Cap size
                    continue
                max_val = max(new_U)
                new_val = random.randint(1, max_val * 2)
                if new_val not in new_U:
                    new_U.append(new_val)
                else:
                    continue
            elif move_type == 'remove' and len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1) # Don't remove 0
                new_U.pop(idx)
            else:
                continue
                
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
                new_U = sorted(new_U)
                
            new_score = compute_score(new_U)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_U = new_U
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_U = list(current_U)
            
            T *= cooling
            
        return best_local_U, best_local_score

    # Run local search from several starting points
    start_points = [
        best_set,
        [0, 1, 3, 6],
        [0, 1, 2, 4, 8],
        [0, 1, 5, 10, 15],
        [0, 2, 5, 9, 14],
    ]
    
    # Add some random starts
    for _ in range(10):
        n = random.randint(4, 15)
        max_val = random.randint(50, 500)
        U = [0] + sorted(random.sample(range(1, max_val), n - 1))
        start_points.append(U)
        
    for sp in start_points:
        if time.time() > deadline:
            break
        local_best_U, local_best_score = local_search(sp)
        if local_best_score > best_score:
            best_score = local_best_score
            best_set = local_best_U
            
        # Also try to improve the global best directly
        if time.time() > deadline:
            break
        local_best_U, local_best_score = local_search(best_set, max_steps=500)
        if local_best_score > best_score:
            best_score = local_best_score
            best_set = local_best_U

    return best_set
