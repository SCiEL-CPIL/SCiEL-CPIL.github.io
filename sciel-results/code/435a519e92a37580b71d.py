import numpy as np
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds to be safe
    
    # 1. Generate pool of candidate numbers
    # Using 5-smooth numbers (Hamming numbers) as they have many multiplicative relations
    primes = [2, 3, 5]
    max_val = 10000
    pool_set = {1}
    for p in primes:
        current = list(pool_set)
        for x in current:
            val = x
            while val * p <= max_val:
                val *= p
                pool_set.add(val)
    
    pool = sorted(pool_set)
    n_pool = len(pool)
    
    # 2. Objective Function
    def evaluate(A):
        # A is a list of ints
        s_set = set()
        p_set = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                s_set.add(ai + aj)
                p_set.add(ai * aj)
        return max(len(s_set), len(p_set))
    
    # 3. Simulated Annealing
    best_score = float('inf')
    best_A = None
    
    # Try multiple random restarts if time permits
    # Or just one long run with good initialization
    
    # Initialization: Pick 20 random numbers from the pool
    # But a structured start is better.
    # Start with the smallest 20 smooth numbers?
    # Or powers of 2 + small smooth?
    
    # Let's use a few initial states
    initial_states = []
    
    # State 1: Smallest 20
    initial_states.append(sorted(pool[:20]))
    
    # State 2: Powers of 2 + smallest others
    p2 = [2**i for i in range(14)] # 1 to 8192
    others = [x for x in pool if x not in p2]
    initial_states.append(sorted(p2 + others[:6]))
    
    # State 3: Random 20 from pool
    initial_states.append(sorted(random.sample(pool, 20)))
    
    # State 4: Random 20 from pool (another)
    initial_states.append(sorted(random.sample(pool, 20)))
    
    for init_A in initial_states:
        if time.time() > deadline:
            break
            
        current_A = list(init_A)
        current_score = evaluate(current_A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = list(current_A)
        
        temp = 100.0
        cooling_rate = 0.9995
        
        while temp > 0.1:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            # Replace one element with another from the pool
            idx = random.randint(0, 19)
            old_val = current_A[idx]
            
            # Choose new value
            # Bias towards values close to current to allow local search?
            # Or random from pool.
            new_val = random.choice(pool)
            
            if new_val == old_val:
                continue
            
            # Check distinctness
            if new_val in current_A:
                continue
            
            # Apply move
            current_A[idx] = new_val
            new_score = evaluate(current_A)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(current_A)
            else:
                # Revert
                current_A[idx] = old_val
            
            temp *= cooling_rate
    
    return best_A
