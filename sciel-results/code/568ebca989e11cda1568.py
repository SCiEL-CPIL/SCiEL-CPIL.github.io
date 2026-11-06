import time
import random
import math

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    sum_set = set()
    prod_set = set()
    # Using explicit loops for clarity and slight speed optimization
    for a in A:
        for b in A:
            sum_set.add(a + b)
            prod_set.add(a * b)
            
    return max(len(sum_set), len(prod_set))

def generate_smooth_candidates(limit=10000, max_exponent=10):
    """
    Generate a pool of 'smooth' numbers (composed of small primes) <= limit.
    These numbers are more likely to create product collisions than random large primes.
    """
    candidates = set([1])
    primes = [2, 3, 5, 7, 11, 13]
    
    def generate(idx, current_val):
        if current_val > limit:
            return
        if idx == len(primes):
            candidates.add(current_val)
            return
        
        p = primes[idx]
        # Multiply by p^k
        val = current_val
        for _ in range(max_exponent):
            if val > limit:
                break
            candidates.add(val)
            generate(idx + 1, val)
            val *= p
        # Also skip this prime and go to next
        generate(idx + 1, current_val)

    generate(0, 1)
    # Filter valid candidates
    return [x for x in candidates if 1 <= x <= limit]

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin: 45s out of 60s limit
    
    # 1. Generate a pool of structurally relevant candidates
    # Smooth numbers are better for product set minimization.
    # We also include small integers 1..100 to help with sum set minimization.
    smooth_pool = generate_smooth_candidates(10000, max_exponent=15)
    small_ints = list(range(1, 101))
    
    # Combine and unique
    candidate_pool = list(set(smooth_pool) | set(small_ints))
    random.shuffle(candidate_pool)
    
    # 2. Initial Seeds
    # Seed 1: 1..20 (minimizes sumset)
    seed1 = list(range(1, 21))
    
    # Seed 2: Powers of 2 mixed with 3s (minimizes product set structure)
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 nums)
    # Add 3, 6, 12, 24, 48, 96? 
    # Let's just use a random sample from the smooth pool for a second seed
    random.shuffle(smooth_pool)
    seed2 = []
    used = set()
    for x in smooth_pool:
        if x not in used:
            seed2.append(x)
            used.add(x)
        if len(seed2) == 20:
            break
    
    # Ensure seeds are valid
    if len(set(seed2)) != 20 or any(x > 10000 or x < 1 for x in seed2):
        seed2 = seed1[:] # Fallback

    best_A = None
    best_score = float('inf')
    
    for seed in [seed1, seed2]:
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
            
    # 3. Simulated Annealing
    A = best_A[:]
    current_score = best_score
    
    # Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We will iterate until time limit.
    # Each iteration: pick a random index, pick a random candidate from pool, swap.
    
    while time.time() < deadline:
        # Check time frequently to avoid overshoot
        if time.time() > deadline - 0.5:
            break
            
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Pick a new value from the candidate pool
        # Bias towards values not already in A
        # To speed up, just pick a random candidate and check validity
        new_val = random.choice(candidate_pool)
        
        if new_val == old_val or new_val in A:
            continue
            
        # Ensure bounds (pool is already bounded, but just in case)
        if new_val < 1 or new_val > 10000:
            continue
            
        A[idx] = new_val
        new_score = compute_metrics(A)
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A[:]
        else:
            # Accept with probability exp(-delta / T)
            # delta is integer, T is float
            # If T is small, exp(-large/T) is ~0
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject, revert
                    A[idx] = old_val
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_A

# Keep the function name and signature exactly as requested.
# The function above is the implementation.
# Just to be safe, define construct as the entry point clearly.
# The code above defines construct().
