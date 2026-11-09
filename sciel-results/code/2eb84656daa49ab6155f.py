import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 20
    max_val = 10000
    
    def objective(A):
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
        
    def is_valid(A):
        if len(set(A)) != n:
            return False
        if any(x < 1 or x > max_val for x in A):
            return False
        return True

    # Initial best
    best_A = list(range(1, n + 1))
    best_score = objective(best_A)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Geometric progressions
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for scale in range(1, 50):
            A = [scale * (base ** i) for i in range(n)]
            if max(A) <= max_val:
                candidates.append(A)
                
    # 2. Random sets
    for _ in range(100):
        A = random.sample(range(1, max_val + 1), n)
        candidates.append(A)
        
    # 3. Structured: Sparse APs
    for _ in range(100):
        A = set()
        while len(A) < n:
            start = random.randint(1, max_val)
            step = random.randint(1, 100)
            for k in range(random.randint(1, 5)):
                val = start + k * step
                if 1 <= val <= max_val:
                    A.add(val)
        A = list(A[:n])
        if is_valid(A):
            candidates.append(A)

    # Evaluate initial candidates
    for A in candidates:
        if not is_valid(A):
            continue
        score = objective(A)
        if score < best_score:
            best_score = score
            best_A = A[:]

    # Local Search
    current_A = best_A[:]
    current_score = best_score
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    # Count steps to manage time
    steps = 0
    
    while time.time() < deadline:
        # Random move: Replace or Swap
        if random.random() < 0.5:
            # Replace
            idx = random.randint(0, n - 1)
            old_val = current_A[idx]
            new_val = random.randint(1, max_val)
            current_A[idx] = new_val
            
            if not is_valid(current_A):
                current_A[idx] = old_val
                continue
                
            new_score = objective(current_A)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = current_A[:]
            else:
                if random.random() < math.exp(-(new_score - current_score) / T):
                    current_score = new_score
                else:
                    current_A[idx] = old_val
        else:
            # Swap
            i, j = random.sample(range(n), 2)
            old_i, old_j = current_A[i], current_A[j]
            current_A[i], current_A[j] = old_j, old_i
            
            # Swap is always valid if original was valid
            new_score = objective(current_A)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = current_A[:]
            else:
                if random.random() < math.exp(-(new_score - current_score) / T):
                    current_score = new_score
                else:
                    current_A[i], current_A[j] = old_i, old_j
                    
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_A
