import random
import math

def construct():
    # Generate 5-smooth numbers (Hamming numbers) up to 10000
    # A number is 5-smooth if its prime factors are a subset of {2, 3, 5}
    smooth_numbers = []
    i2 = i3 = i5 = 0
    next_2, next_3, next_5 = 2, 3, 5
    smooth_numbers.append(1)
    while True:
        min_val = min(next_2, next_3, next_5)
        if min_val > 10000:
            break
        smooth_numbers.append(min_val)
        if min_val == next_2:
            i2 += 1
            next_2 = smooth_numbers[i2] * 2
        if min_val == next_3:
            i3 += 1
            next_3 = smooth_numbers[i3] * 3
        if min_val == next_5:
            i5 += 1
            next_5 = smooth_numbers[i5] * 5
    
    # Start with the 20 smallest smooth numbers
    # This minimizes the product range and likely maximizes collisions
    initial_set = sorted(smooth_numbers[:20])
    
    def evaluate(A):
        A_set = set(A)
        sum_set = set()
        prod_set = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n): # i<=j for sums/prods? 
                # A+A includes x+y. If we iterate i<=j, we get all sums?
                # Yes, x+y = y+x.
                # A*A includes x*y. If we iterate i<=j, we get all products?
                # Yes, x*y = y*x.
                # Wait, the definition says "equal summands/factors allowed".
                # So we need all pairs (a,b) where a,b in A.
                # Using i<=j covers all unique pairs {a,b}.
                # The value a+b is the same as b+a.
                # The value a*b is the same as b*a.
                # So iterating i<=j is sufficient to generate the set values.
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        
        # Wait, does "equal summands allowed" mean we include a+a?
        # Yes, i=j covers a+a.
        # Does it mean we include a+b for a!=b?
        # Yes, i<j covers a+b.
        # So the set of values is correct.
        
        return max(len(sum_set), len(prod_set))

    current_A = initial_set[:]
    current_score = evaluate(current_A)
    
    # Local Search
    # We want to reduce the score.
    # Try swapping elements with other smooth numbers or small random integers.
    
    # Candidate pool: Smooth numbers + small integers
    # We restrict to small integers to keep product set small.
    candidate_pool = list(set(smooth_numbers[:50] + list(range(1, 101))))
    random.shuffle(candidate_pool)
    
    best_A = current_A[:]
    best_score = current_score
    
    # Simple Hill Climbing with Random Restarts
    # Time limit is 60s, but we should be fast.
    # 20 elements, pool size ~100.
    
    import time
    start_time = time.time()
    timeout = 50  # Leave 10s buffer
    
    while time.time() - start_time < timeout:
        # Randomly pick an index to replace
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Pick a new value from candidate pool
        new_val = random.choice(candidate_pool)
        
        if new_val in current_A and new_val != old_val:
            continue
            
        # Create new set
        new_A = current_A[:]
        new_A[idx] = new_val
        new_A = sorted(new_A) # Keep sorted for consistency, though not strictly needed
        
        new_score = evaluate(new_A)
        
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            # Occasionally accept worse or equal to escape local minima?
            # Given the landscape is discrete and small, simple hill climbing might get stuck.
            # But with random restarts or random walks, it's better.
            # Let's do a random walk step if stuck?
            # For simplicity, just try many random moves.
            pass

    return best_A
