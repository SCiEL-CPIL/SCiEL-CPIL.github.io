import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    def evaluate(S):
        """Fast evaluation: count distinct d>0 such that a,a+d,a+2d all in S for some a."""
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
    
    def evaluate_fast(S_set, S_list):
        """Evaluate using pre-sorted list and set."""
        valid_diffs = set()
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)
    
    def local_search(S, max_iters=10000):
        """Hill climbing with smarter move generation."""
        S = S[:]
        S_sorted = sorted(S)
        S_set = set(S)
        current_score = evaluate_fast(S_set, S_sorted)
        
        best_score = current_score
        best_S = S[:]
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            # Pick a random element to replace
            idx = random.randint(0, 31)
            old_val = S[idx]
            
            # Generate a new value - bias towards values that might form progressions
            if random.random() < 0.5:
                # Try values near existing elements
                base = random.choice(S)
                offset = random.randint(-50, 50)
                new_val = base + offset
                if not (1 <= new_val <= 512) or new_val in S_set:
                    new_val = random.randint(1, 512)
                    if new_val in S_set:
                        continue
            else:
                new_val = random.randint(1, 512)
                if new_val in S_set:
                    continue
            
            # Try the swap
            S[idx] = new_val
            S_set.remove(old_val)
            S_set.add(new_val)
            S_sorted = sorted(S)
            new_score = evaluate_fast(S_set, S_sorted)
            
            if new_score > best_score:
                best_score = new_score
                best_S = S[:]
            elif new_score < current_score:
                # Revert
                S[idx] = old_val
                S_set.remove(new_val)
                S_set.add(old_val)
                S_sorted = sorted(S)
            
            current_score = new_score
        
        return best_S, best_score
    
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
    for _ in range(30):
        candidates.append(random.sample(range(1, 513), 32))
    
    # 4. Structured: mix of small and large values
    for _ in range(15):
        small = random.sample(range(1, 100), 16)
        large = random.sample(range(400, 513), 16)
        candidates.append(small + large)
    
    # 5. Quadratic-like sets
    for a in [1, 2, 3, 5]:
        vals = [a * i * i + 1 for i in range(32)]
        if max(vals) <= 512:
            candidates.append(vals)
    
    # 6. Sets designed to have many 3-term progressions
    # Use union of multiple arithmetic progressions
    for _ in range(10):
        s = set()
        while len(s) < 32:
            start = random.randint(1, 500)
            step = random.randint(2, 50)
            for k in range(random.randint(3, 8)):
                v = start + k * step
                if 1 <= v <= 512:
                    s.add(v)
        candidates.append(list(s))
    
    # 7. Sets with dense small differences
    for _ in range(10):
        base = random.randint(1, 100)
        s = set()
        for i in range(32):
            v = base + random.randint(0, 400)
            if 1 <= v <= 512:
                s.add(v)
        while len(s) < 32:
            v = random.randint(1, 512)
            s.add(v)
        candidates.append(list(s))
    
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
    num_starts = 20
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
        result_S, result_score = local_search(start_S, max_iters=5000)
        
        if result_score > best_score:
            best_score = result_score
            best_S = result_S[:]
    
    # Final polish: try to improve best_S with more iterations
    if time.time() < deadline - 10:
        final_S, final_score = local_search(best_S, max_iters=8000)
        if final_score > best_score:
            best_S = final_S
            best_score = final_score
    
    return best_S
