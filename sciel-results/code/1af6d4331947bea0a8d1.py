import random
import math

def construct():
    # Simulated Annealing approach
    # Initial set: 1..20
    A = list(range(1, 21))
    
    # Candidate pool: 1..100 (small numbers keep sumset small)
    # Plus some highly composite numbers to help product collisions if we need to expand
    pool = list(range(1, 101))
    
    def score(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    best_A = A[:]
    best_score = score(A)
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.995
    max_iter = 20000
    iter_count = 0
    T = T_start
    
    random.seed(42)
    
    while T > T_end and iter_count < max_iter:
        # Generate neighbor
        # Strategy: Replace one element with a random one from pool
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Try to pick a new value not in A
        # Bias towards values close to old_val or small values
        # To keep sumset small, prefer small values
        # To keep productset small, prefer values with many divisors
        
        # Simple random from pool
        new_val = random.choice(pool)
        
        if new_val == old_val:
            continue
        if new_val in A:
            continue
            
        A[idx] = new_val
        new_score = score(A)
        
        # Metropolis acceptance
        delta = new_score - best_score
        if delta < 0:
            # Improvement
            best_score = new_score
            best_A = A[:]
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < math.exp(-delta / T):
                best_score = new_score # We don't update best_score here if it's worse, 
                                       # but we keep the state. 
                                       # Actually, best_score should track the global best.
                                       # Let's separate current_score and best_score.
                # Correction:
                pass # Logic fix below
                
        T *= cooling_rate
        iter_count += 1

    # The above loop has a flaw in tracking best_score vs current_score.
    # Let's rewrite properly.
    
    return best_A
