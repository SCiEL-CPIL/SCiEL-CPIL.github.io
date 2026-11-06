import time
import random
import math

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    # Using Python sets is often faster for small N than np.unique due to overhead
    sums = set()
    prods = set()
    for a in A:
        for b in A:
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))

def generate_smooth_candidates(limit=10000):
    """
    Generate a pool of 'smooth' numbers (composed of small primes) <= limit.
    """
    candidates = set([1])
    primes = [2, 3, 5, 7, 11, 13, 17]
    
    def generate(idx, current_val):
        if current_val > limit:
            return
        if idx == len(primes):
            candidates.add(current_val)
            return
        
        p = primes[idx]
        val = current_val
        while val <= limit:
            candidates.add(val)
            generate(idx + 1, val)
            val *= p
        # Also skip this prime and go to next
        generate(idx + 1, current_val)

    generate(0, 1)
    return [x for x in candidates if 1 <= x <= limit]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Generate candidate pool
    # Smooth numbers are good for product collisions.
    # We bias towards smaller numbers to control sumset size.
    smooth_pool = generate_smooth_candidates(10000)
    # Filter to a more manageable size if needed, but 10000 limit is fine.
    # Let's ensure we have enough candidates.
    if len(smooth_pool) < 100:
        smooth_pool = list(range(1, 101))
    
    # 2. Initial Seeds
    # Seed 1: 1..20 (Min sumset, max product set)
    seed1 = list(range(1, 21))
    
    # Seed 2: Smooth numbers, biased small
    # Sort smooth pool and take the first 20? No, that's 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 basically.
    # Let's try a mix of small smooths and some larger ones to break product patterns.
    random.shuffle(smooth_pool)
    seed2 = []
    used = set()
    # Prioritize small numbers
    sorted_smooth = sorted(smooth_pool)
    for x in sorted_smooth:
        if x not in used:
            seed2.append(x)
            used.add(x)
        if len(seed2) == 20:
            break
    
    # Seed 3: Random smooth
    seed3 = random.sample(smooth_pool, 20)
    
    # Ensure validity
    for s in [seed1, seed2, seed3]:
        if len(set(s)) != 20 or any(x > 10000 or x < 1 for x in s):
            # Fallback
            s[:] = seed1[:]

    best_A = None
    best_score = float('inf')
    
    for seed in [seed1, seed2, seed3]:
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
            
    # 3. Simulated Annealing
    A = best_A[:]
    current_score = best_score
    
    T_start = 20.0
    T_min = 0.05
    cooling_rate = 0.999
    
    # To improve search, we can restart if stuck
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline:
        if time.time() > deadline - 1.0:
            break
            
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Pick a new value
        # Bias towards smaller values to keep sumset small?
        # Or random from pool?
        new_val = random.choice(smooth_pool)
        
        if new_val == old_val or new_val in A:
            continue
            
        A[idx] = new_val
        new_score = compute_metrics(A)
        delta = new_score - current_score
        
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A[:]
                restarts = 0 # Reset restart counter on improvement
        else:
            # Accept with probability exp(-delta / T)
            if T_start > T_min:
                prob = math.exp(-delta / T_start)
                if random.random() < prob:
                    current_score = new_score
                else:
                    A[idx] = old_val
            else:
                A[idx] = old_val
        
        T_start *= cooling_rate
        if T_start < T_min:
            # Restart
            restarts += 1
            if restarts > max_restarts:
                T_start = T_min
            else:
                T_start = 10.0
                A = best_A[:]
                current_score = best_score

    # 4. Final Polish
    if time.time() < deadline - 0.5:
        A = best_A[:]
        current_score = best_score
        polish_T = 0.1
        polish_deadline = min(deadline - 0.1, time.time() + 5.0)
        
        while time.time() < polish_deadline:
            idx = random.randint(0, 19)
            old_val = A[idx]
            
            # Try swapping with a random other element or a neighbor
            # Neighbor in value space
            new_val = old_val + random.randint(-10, 10)
            
            if new_val == old_val or new_val in A or new_val < 1 or new_val > 10000:
                continue
                
            A[idx] = new_val
            new_score = compute_metrics(A)
            delta = new_score - current_score
            
            if delta < 0:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = A[:]
            else:
                prob = math.exp(-delta / polish_T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    A[idx] = old_val

    return best_A
