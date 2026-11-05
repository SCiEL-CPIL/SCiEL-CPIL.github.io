import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = list(range(1, 21))
    best_score = float('inf')
    
    # Helper to compute score
    def get_score(A):
        # A is a list of 20 distinct integers
        sums = set()
        prods = set()
        for i in range(20):
            ai = A[i]
            for j in range(i, 20):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
    
    # Initial seed score
    best_score = get_score(best_set)
    
    # Search over Arithmetic Progressions
    # A = {x, x+d, ..., x+19d}
    # x >= 1, d >= 1
    # x + 19d <= 10000
    
    # We can iterate d from 1 upwards
    # For each d, x can range from 1 to 10000 - 19d
    # However, checking all x for all d is too slow (10000 * 10000 / 19 ~ 5 million checks, each O(20^2) ~ 400 ops -> 2 billion ops. Too slow in Python).
    
    # We need to be smarter or limit the search.
    # Note: |A+A| is always 39 for AP.
    # So we only care about |A*A|.
    # Let's sample some promising d and x.
    
    # Heuristic: Small d and x might lead to more collisions?
    # Or specific structures.
    
    # Let's try a limited search with time guard.
    # We'll iterate d up to some limit, and x up to some limit.
    
    max_d = 500
    max_x = 10000
    
    for d in range(1, max_d + 1):
        if time.time() > deadline:
            break
        max_x_for_d = 10000 - 19 * d
        if max_x_for_d < 1:
            break
        
        # Step x to save time? Or check all?
        # Checking all x for d=1 is 10000 checks. 10000 * 400 = 4M ops. Fast.
        # d=100 is 10000 checks.
        # Total checks ~ sum(10000 - 19d for d in 1..500) ~ 500 * 10000 = 5M checks.
        # Each check is O(20^2) = 400 set insertions.
        # 5M * 400 = 2B ops. Might be too slow in Python (60s limit).
        
        # Optimization: We can precompute or use numpy?
        # Or just reduce search space.
        
        # Let's try a smaller range first.
        # d in 1..100, x in 1..10000
        # That's 1M checks. 1M * 400 = 400M ops. ~ 5-10 seconds in Python.
        
        for x in range(1, max_x_for_d + 1):
            if time.time() > deadline:
                break
            
            A = [x + i * d for i in range(20)]
            
            # Quick check: if last element > 10000, break inner loop
            # Already handled by max_x_for_d
            
            # Compute product set size only, since sum set is 39
            prods = set()
            for i in range(20):
                ai = A[i]
                for j in range(i, 20):
                    prods.add(ai * A[j])
            
            p_size = len(prods)
            score = max(39, p_size)
            
            if score < best_score:
                best_score = score
                best_set = A[:]
                
    return best_set
