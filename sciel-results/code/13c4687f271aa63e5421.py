import random
import math

def _fast_score(S):
    S_set = set(S)
    valid_diffs = set()
    lst = list(S)
    for i in range(len(lst)):
        x = lst[i]
        for j in range(i + 1, len(lst)):
            y = lst[j]
            d_val = y - x
            
            if (x - d_val) in S_set:
                valid_diffs.add(d_val)
            if (y + d_val) in S_set:
                valid_diffs.add(d_val)
            
            if d_val % 2 == 0:
                d_half = d_val // 2
                if (x + d_half) in S_set:
                    valid_diffs.add(d_half)
                    
    return len(valid_diffs)

def construct():
    # Initial solution: random or heuristic
    # Heuristic: Start with a dense interval to get many small differences
    # Then use SA to expand/optimize.
    
    # Let's start with a random set to avoid bias, or a structured one.
    # A structured start like [1..32] has score 15 (d=1..15).
    # Let's try to beat that.
    
    # We will run SA for a fixed number of steps or time.
    # Time limit is 120s. We can run for ~10s to be safe, leaving buffer.
    # But Python SA might be slow if not optimized.
    # Let's run for a fixed number of iterations, say 500,000.
    # Each step: pick random index, remove, pick random new val, add, score.
    # Scoring is O(N^2) = ~1000 ops.
    # 500k * 1000 = 500M ops. Too slow in Python.
    
    # We need a faster score update or fewer iterations.
    # Actually, the O(N^2) score is ~1000 set lookups.
    # 1000 lookups * 500k = 500M lookups. In Python, this is likely > 10s.
    # Let's reduce iterations to 50,000.
    # 50,000 * 1000 = 50M lookups. ~5-10s. Acceptable.
    
    # Better: Use a greedy initial construction and then SA.
    
    # Initial: Random set
    current = set(random.sample(range(1, 513), 32))
    current_score = _fast_score(current)
    
    best = set(current)
    best_score = current_score
    
    # Simulated Annealing
    # T starts high, decreases exponentially
    T_start = 10.0
    T_end = 0.1
    iterations = 50000
    alpha = (T_end / T_start) ** (1.0 / iterations)
    
    T = T_start
    
    # Pre-compute list of all possible values 1..512
    all_vals = list(range(1, 513))
    
    for i in range(iterations):
        # Pick a random element to replace
        idx = random.randrange(32)
        # We need to pick an element from the set.
        # Since we maintain a list for easy indexing, let's keep a list parallel to set?
        # Or just pick random element from set.
        # Removing from set and adding to set is O(1).
        
        # To pick random element from set efficiently, we can convert to list occasionally?
        # No, just use random.choice(list(current)) is O(N).
        # N=32, so it's fast.
        
        elem_to_remove = random.choice(list(current))
        current.remove(elem_to_remove)
        
        # Pick a new element
        # Ensure it's not in current
        while True:
            new_elem = random.choice(all_vals)
            if new_elem not in current:
                break
        
        current.add(new_elem)
        
        new_score = _fast_score(current)
        
        # Accept if better or with probability exp((new-old)/T)
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = set(current)
        else:
            # Accept worse solution with probability
            delta = new_score - current_score
            if random.random() < math.exp(delta / T):
                current_score = new_score
        
        T *= alpha
        
    return list(best)
