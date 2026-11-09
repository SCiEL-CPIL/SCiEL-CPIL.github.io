import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def evaluate(S_set):
        """Count distinct d>0 such that a, a+d, a+2d all in S for some a."""
        valid_diffs = set()
        # We only need to check pairs (a, b) where b > a.
        # d = b - a, c = a + 2d = 2b - a.
        for a in S_set:
            for b in S_set:
                if b > a:
                    d = b - a
                    c = 2 * b - a
                    if c in S_set:
                        valid_diffs.add(d)
        return len(valid_diffs)

    def get_best_move(S_set):
        """
        For each element in S_set, try replacing it with every other value in 1..512.
        Return the (S_set, score) that yields the highest score.
        """
        best_S_set = None
        best_score = -1
        elements = list(S_set)
        
        for i, old_val in enumerate(elements):
            # Try replacing old_val with every other possible value
            for new_val in range(1, 513):
                if new_val == old_val or new_val in S_set:
                    continue
                
                # Efficient update of the set
                S_set.remove(old_val)
                S_set.add(new_val)
                
                score = evaluate(S_set)
                
                if score > best_score:
                    best_score = score
                    best_S_set = S_set.copy()
                    
                # Revert
                S_set.remove(new_val)
                S_set.add(old_val)
                
        if best_S_set is None:
            return S_set, evaluate(S_set)
            
        return best_S_set, best_score

    def local_search(S_set, max_iters=500):
        """
        Hill climbing: repeatedly find the best single-element replacement.
        """
        current_score = evaluate(S_set)
        
        for _ in range(max_iters):
            if time.time() >= deadline:
                break
                
            next_S_set, next_score = get_best_move(S_set)
            
            if next_score <= current_score:
                break # Local optimum reached
                
            S_set = next_S_set
            current_score = next_score
            
        return S_set, current_score

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Consecutive integers
    candidates.append(set(range(1, 33)))
    
    # 2. Arithmetic progressions with various steps
    for step in [2, 3, 4, 5, 7, 10, 15, 20]:
        start = 1
        end = start + 31 * step
        if end <= 512:
            candidates.append(set(start + i * step for i in range(32)))
            
    # 3. Random sets
    random.seed(42)
    for _ in range(10):
        candidates.append(set(random.sample(range(1, 513), 32)))
        
    # 4. Structured: mix of small and large values
    for _ in range(5):
        small = set(random.sample(range(1, 100), 16))
        large = set(random.sample(range(400, 513), 16))
        candidates.add(small | large)

    # 5. Quadratic-like sets
    for a in [1, 2, 3, 5]:
        vals = set(a * i * i + 1 for i in range(32))
        if max(vals) <= 512:
            candidates.add(vals)

    # Evaluate all candidates
    best_S_set = None
    best_score = 0
    for cand in candidates:
        if len(cand) != 32:
            continue
        score = evaluate(cand)
        if score > best_score:
            best_score = score
            best_S_set = cand

    # Multi-start local search
    num_starts = 10
    for s in range(num_starts):
        if time.time() >= deadline:
            break
        
        # Create a perturbed version of best_S
        if s == 0:
            start_S_set = best_S_set.copy()
        else:
            start_S_set = best_S_set.copy()
            # Perturb 5-10 random elements
            num_perturb = random.randint(5, 10)
            elements = list(start_S_set)
            indices = random.sample(range(32), num_perturb)
            
            for idx in indices:
                old_val = elements[idx]
                new_val = random.randint(1, 512)
                while new_val in start_S_set:
                    new_val = random.randint(1, 512)
                start_S_set.remove(old_val)
                start_S_set.add(new_val)
        
        # Run local search
        result_S_set, result_score = local_search(start_S_set, max_iters=200)
        
        if result_score > best_score:
            best_score = result_score
            best_S_set = result_S_set

    return list(best_S_set)
