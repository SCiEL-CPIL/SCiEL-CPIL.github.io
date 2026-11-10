import random
import time
import math

def construct():
    n = 20
    max_val = 10000
    start_time = time.time()
    time_limit = 55  # Safety margin for 60s limit
    
    def get_sizes(A):
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return len(sums), len(prods)
    
    def objective(A):
        s, p = get_sizes(A)
        return max(s, p)
        
    best_set = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. Random
    for _ in range(5):
        A = random.sample(range(1, max_val + 1), n)
        candidates.append(A)
        
    # 2. Arithmetic Progressions
    for start in range(1, 100):
        for step in [1, 2, 5, 10, 50, 100, 500]:
            A = [start + i * step for i in range(n)]
            if A[-1] <= max_val:
                candidates.append(A)
                
    # 3. Geometric Progressions
    for base in [2, 3, 4, 5]:
        for start in [1, 2, 3]:
            A = [start * (base ** i) for i in range(n)]
            if A[-1] <= max_val:
                candidates.append(A)
            else:
                break

    # Evaluate initial candidates
    for A in candidates:
        if len(set(A)) != n: continue
        if any(x < 1 or x > max_val for x in A): continue
        score = objective(A)
        if score < best_score:
            best_score = score
            best_set = A[:]
            
    if best_set is None:
        best_set = list(range(1, n+1))
        best_score = objective(best_set)

    # Simulated Annealing / Local Search
    current_set = best_set[:]
    current_score = best_score
    
    # Precompute neighbors? No, just random perturbations.
    # Perturbation types:
    # 1. Replace one element with random
    # 2. Swap two elements with random
    # 3. Add/subtract small delta to one element
    
    T = 100.0
    T_min = 1.0
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        new_set = current_set[:]
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Replace
            idx = random.randint(0, n-1)
            new_val = random.randint(1, max_val)
            if new_val not in new_set or new_val == new_set[idx]:
                new_set[idx] = new_val
            else:
                continue # Collision in set, skip or retry? Just skip for simplicity in this loop iteration
        elif move_type == 1:
            # Swap with random
            idx = random.randint(0, n-1)
            new_val = random.randint(1, max_val)
            if new_val not in new_set or new_val == new_set[idx]:
                new_set[idx] = new_val
            else:
                continue
        else:
            # Perturb
            idx = random.randint(0, n-1)
            delta = random.choice([-1, 1, -2, 2, -5, 5, -10, 10])
            new_val = new_set[idx] + delta
            if 1 <= new_val <= max_val and (new_val not in new_set or new_val == new_set[idx]):
                new_set[idx] = new_val
            else:
                continue
        
        if len(set(new_set)) != n:
            continue
            
        new_score = objective(new_set)
        
        # Acceptance criteria
        if new_score < current_score:
            current_set = new_set
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            # Simulated Annealing acceptance
            delta = new_score - current_score
            if T > T_min:
                if random.random() < math.exp(-delta / T):
                    current_set = new_set
                    current_score = new_score
        T = max(T_min, T * alpha)
        
    return best_set
