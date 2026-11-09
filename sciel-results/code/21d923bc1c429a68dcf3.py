import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    def evaluate(S):
        """Fast evaluation using sorted list and set membership."""
        S_set = set(S)
        valid_diffs = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def local_search(S, max_iters=5000):
        """Hill climbing with random restarts within a single trajectory."""
        S = S[:]
        current_score = evaluate(S)
        S_set = set(S)
        
        for _ in range(max_iters):
            if time.time() >= deadline:
                break
            # Pick a random element to replace
            idx = random.randint(0, 31)
            old_val = S[idx]
            
            # Generate a new value
            new_val = random.randint(1, 512)
            if new_val in S_set:
                continue
            
            # Try the swap
            S[idx] = new_val
            S_set.remove(old_val)
            S_set.add(new_val)
            new_score = evaluate(S)
            
            if new_score >= current_score:
                current_score = new_score
            else:
                # Revert
                S[idx] = old_val
                S_set.remove(new_val)
                S_set.add(old_val)
        
        return S, current_score
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Consecutive integers
    candidates.append(list(range(1, 33)))
    
    # 2. Arithmetic progressions with various steps
    for step in [2, 3, 4, 5, 7, 10, 15, 20]:
        start = 1
        end = start + 31 * step
        if end <= 512:
            candidates.append([start + i * step for i in range(32)])
    
    # 3. Random sets
    random.seed(42)
    for _ in range(20):
        candidates.append(random.sample(range(1, 513), 32))
    
    # 4. Structured: mix of small and large values
    for _ in range(10):
        small = random.sample(range(1, 100), 16)
        large = random.sample(range(400, 513), 16)
        candidates.append(small + large)
    
    # 5. Quadratic-like sets
    for a in [1, 2, 3, 5]:
        vals = [a * i * i + 1 for i in range(32)]
        if max(vals) <= 512:
            candidates.append(vals)
    
    # Evaluate all candidates
    best_S = None
    best_score = 0
    for cand in candidates:
        if len(set(cand)) != 32:
            continue
        score = evaluate(cand)
        if score > best_score:
            best_score = score
            best_S = cand[:]
    
    # Multi-start local search
    num_starts = 15
    for s in range(num_starts):
        if time.time() >= deadline:
            break
        
        # Create a perturbed version of best_S
        if s == 0:
            start_S = best_S[:]
        else:
            start_S = best_S[:]
            # Perturb 5-15 random elements
            num_perturb = random.randint(5, 15)
            indices = random.sample(range(32), num_perturb)
            S_set = set(start_S)
            for idx in indices:
                new_val = random.randint(1, 512)
                while new_val in S_set:
                    new_val = random.randint(1, 512)
                S_set.remove(start_S[idx])
                S_set.add(new_val)
                start_S[idx] = new_val
        
        # Run local search
        result_S, result_score = local_search(start_S, max_iters=3000)
        
        if result_score > best_score:
            best_score = result_score
            best_S = result_S[:]
    
    return best_S
