import random
import time
import math

def construct():
    n = 20
    max_val = 10000
    time_limit = 55.0  # Leave 5s buffer
    
    def get_score(A):
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))

    # Initialize with a simple set, e.g., 1..20
    current_set = list(range(1, n + 1))
    current_score = get_score(current_set)
    best_set = current_set[:]
    best_score = current_score
    
    T = 10.0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick a random index to change
        i = random.randint(0, n - 1)
        
        # Pick a random new value
        # To avoid duplicates, we can generate and check
        new_val = random.randint(1, max_val)
        if new_val in current_set:
            # Try a few times or just pick a different one
            # For simplicity, loop until we get a unique one
            while new_val in current_set:
                new_val = random.randint(1, max_val)
        
        new_set = current_set[:]
        new_set[i] = new_val
        new_score = get_score(new_set)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_set = new_set
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        
        # Cool down
        T *= 0.999
        if T < 0.1:
            T = 0.1
            
    return best_set
