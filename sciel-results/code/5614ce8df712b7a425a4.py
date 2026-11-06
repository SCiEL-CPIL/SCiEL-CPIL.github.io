import time
import random
import math
import itertools

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    sums = set()
    prods = set()
    # Optimized loop: iterate over pairs (i, j) with i <= j to reduce iterations by half?
    # No, A+A includes a+b where a != b. Since addition is commutative, a+b = b+a.
    # However, the set of values is what matters.
    # Standard double loop is O(N^2). N=20, so 400 operations. Very fast.
    for a in A:
        for b in A:
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Generate candidate pool
    # We need a pool of numbers that are "smooth" (many small prime factors) to maximize product collisions.
    # Also, we want a mix of small and large numbers to control sumset size and range.
    
    def is_smooth(n, max_prime=100):
        if n <= 1:
            return True
        d = 2
        while d * d <= n and d <= max_prime:
            while n % d == 0:
                n //= d
            d += 1
        return n == 1 or n <= max_prime

    # Generate a diverse pool
    # Include 1, 2, ..., 100 (small numbers for sum control)
    # Include smooth numbers up to 10000
    pool = set(range(1, 101))
    
    # Add smooth numbers
    for i in range(101, 10001):
        if is_smooth(i, 100):
            pool.add(i)
    
    pool = list(pool)
    random.shuffle(pool)
    
    # 2. Initial Seeds
    # Seed 1: 1..20
    seed1 = list(range(1, 21))
    
    # Seed 2: First 20 smooth numbers (sorted)
    smooth_sorted = sorted([x for x in pool if x > 1])
    seed2 = smooth_sorted[:20]
    
    # Seed 3: Random sample from pool
    seed3 = random.sample(pool, 20)
    
    # Seed 4: Geometric progression based (2^0 .. 2^19 is too big, 2^0..2^13 is 8192)
    # Let's try a mix: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 numbers)
    # Fill rest with small primes?
    seed4 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # Add 6, 12, 24, 48, 96, 192 (multiples of 6 that are smooth)
    for x in [6, 12, 24, 48, 96, 192]:
        if x not in seed4:
            seed4.append(x)
    seed4 = seed4[:20]
    
    # Ensure validity
    for s in [seed1, seed2, seed3, seed4]:
        if len(set(s)) != 20 or any(x > 10000 or x < 1 for x in s):
            s[:] = seed1[:]

    best_A = None
    best_score = float('inf')
    
    for seed in [seed1, seed2, seed3, seed4]:
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]

    # 3. Simulated Annealing with Restart
    A = best_A[:]
    current_score = best_score
    
    # Parameters
    T_start = 15.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    restarts = 0
    max_restarts = 10
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if time.time() > deadline - 1.0:
            break
            
        # Move: Pick a random index and replace with a random value from pool
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Strategy for new value:
        # 80% chance: random from pool
        # 20% chance: local move (neighbor in value space)
        if random.random() < 0.8:
            new_val = random.choice(pool)
        else:
            # Local move: try to find a nearby smooth number
            offset = random.randint(-50, 50)
            new_val = old_val + offset
            if new_val < 1 or new_val > 10000:
                new_val = random.choice(pool)
            # Ensure it's in pool? If not, we can still use it if it's valid, 
            # but pool is biased towards smooth. Let's stick to pool for structural bias.
            if new_val not in pool:
                # Find closest in pool? Too slow. Just pick random from pool.
                new_val = random.choice(pool)
        
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
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Accept with probability exp(-delta / T)
            if T_start > T_min:
                prob = math.exp(-delta / T_start)
                if random.random() < prob:
                    current_score = new_score
                    no_improve_count += 1
                else:
                    A[idx] = old_val
            else:
                A[idx] = old_val
                no_improve_count += 1
        
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
        
        # Stagnation check
        if no_improve_count > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            T_start = 10.0
            A = best_A[:]
            current_score = best_score
            no_improve_count = 0

    # 4. Final Polish (Local Search)
    if time.time() < deadline - 0.5:
        A = best_A[:]
        current_score = best_score
        polish_deadline = min(deadline - 0.1, time.time() + 5.0)
        
        # Try swapping pairs to see if it helps
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(20):
                for j in range(i + 1, 20):
                    # Try swapping A[i] and A[j] with random pool elements
                    # This is too slow (20*20*pool_size).
                    # Instead, try single element replacement with a few candidates
                    pass
        
        # Better polish: try to replace each element with a "better" smooth number
        for i in range(20):
            if time.time() > polish_deadline:
                break
            old_val = A[i]
            # Try a few random candidates
            for _ in range(50):
                new_val = random.choice(pool)
                if new_val == old_val or new_val in A:
                    continue
                A[i] = new_val
                new_score = compute_metrics(A)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = A[:]
                    improved = True
                else:
                    A[i] = old_val
                    break # Revert and stop trying for this i if first fails? No, try more.
                    # Actually, just revert and continue to next candidate
                    A[i] = old_val

    return best_A
