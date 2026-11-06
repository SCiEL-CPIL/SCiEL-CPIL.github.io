import random
import time

def construct():
    # Approach: Start with Arithmetic Progression {1, ..., 20}
    # This minimizes the sumset size (2n-1 = 39).
    # The productset size is the primary cost.
    # We use local search to optimize the productset by swapping elements.
    
    initial_A = list(range(1, 21))
    
    def evaluate(A):
        sum_set = set()
        prod_set = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        return max(len(sum_set), len(prod_set))
        
    current_A = initial_A[:]
    current_score = evaluate(current_A)
    
    best_A = current_A[:]
    best_score = current_score
    
    # Candidate pool: Small integers are best for product collisions.
    # We expand the pool slightly to allow for some flexibility.
    candidate_pool = list(range(1, 151))
    
    start_time = time.time()
    timeout = 50
    
    # Local Search with random restarts if stuck?
    # Simple hill climbing first.
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < timeout:
        # Randomly pick an index to replace
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Pick a new value
        new_val = random.choice(candidate_pool)
        
        if new_val in current_A and new_val != old_val:
            continue
            
        new_A = current_A[:]
        new_A[idx] = new_val
        
        new_score = evaluate(new_A)
        
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            no_improve_count += 1
            # If stuck for a long time, restart with a perturbation?
            # Or just keep going.
            
    return best_A
