import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    def score_set(s):
        """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
        s_set = set(s)
        diffs = set()
        for a in s:
            for d in range(1, 513 - a):
                if a + d in s_set and a + 2 * d in s_set:
                    diffs.add(d)
        return len(diffs)
    
    def greedy_polish(s, max_iters=2000):
        """Try to improve by swapping elements."""
        s = list(s)
        s_set = set(s)
        best_score = score_set(s)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Try swapping a random element with a random candidate
            i = random.randint(0, len(s) - 1)
            old = s[i]
            # Try a few random new values
            improved = False
            for _ in range(20):
                new = random.randint(1, 512)
                if new in s_set:
                    continue
                s[i] = new
                s_set.remove(old)
                s_set.add(new)
                sc = score_set(s)
                if sc > best_score:
                    best_score = sc
                    old = new  # keep this swap
                    improved = True
                    break
                else:
                    s[i] = old
                    s_set.remove(new)
                    s_set.add(old)
            if not improved:
                # Try removing and re-adding
                pass
        
        return s, best_score
    
    def simulated_annealing(s, max_iters=50000):
        """Simulated annealing to optimize the set."""
        s = list(s)
        s_set = set(s)
        current_score = score_set(s)
        best_score = current_score
        best_s = list(s)
        
        temp = 5.0
        cooling = 0.9999
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move: swap one element
            idx = random.randint(0, len(s) - 1)
            old_val = s[idx]
            new_val = random.randint(1, 512)
            
            if new_val in s_set:
                continue
            
            s[idx] = new_val
            s_set.remove(old_val)
            s_set.add(new_val)
            
            new_score = score_set(s)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_s = list(s)
            else:
                # Revert
                s[idx] = old_val
                s_set.remove(new_val)
                s_set.add(old_val)
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        return best_s, best_score
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. Consecutive range
    candidates.append(list(range(1, 33)))
    
    # 2. Sparse arithmetic progression
    candidates.append(list(range(1, 513, 16)))  # 32 elements: 1,17,33,...,497
    
    # 3. Random sets
    for _ in range(5):
        candidates.append(random.sample(range(1, 513), 32))
    
    # 4. Structured: mix of small and large
    half_small = random.sample(range(1, 129), 16)
    half_large = random.sample(range(385, 513), 16)
    candidates.append(half_small + half_large)
    
    # 5. Geometric-ish spread
    import math
    geo = []
    for i in range(32):
        val = int(1 + i * (511 / 31.0))
        geo.append(min(max(val, 1), 512))
    # Remove duplicates by adjusting
    seen = set()
    unique_geo = []
    for v in geo:
        while v in seen and v < 512:
            v += 1
        if v <= 512:
            seen.add(v)
            unique_geo.append(v)
    while len(unique_geo) < 32:
        v = random.randint(1, 512)
        if v not in seen:
            seen.add(v)
            unique_geo.append(v)
    candidates.append(unique_geo)
    
    best_overall = None
    best_overall_score = 0
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Greedy polish
        polished, score = greedy_polish(cand, max_iters=500)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = polished
        
        # Simulated annealing
        sa_result, sa_score = simulated_annealing(polished, max_iters=30000)
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall = sa_result
    
    # Final greedy polish on the best
    if best_overall is not None and time.time() < deadline:
        final, final_score = greedy_polish(best_overall, max_iters=3000)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall = final
    
    if best_overall is None:
        best_overall = list(range(1, 33))
    
    return best_overall
