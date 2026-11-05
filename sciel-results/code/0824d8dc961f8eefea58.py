import time
import random
import math

MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    # Precompute smooth numbers for various prime sets
    # We want to find 12 distinct integers a_1..a_12 <= 10000
    # such that a_i + a_j is P-smooth for all i != j.
    
    def generate_smooth(primes, limit):
        smooth = set()
        def gen(idx, current):
            if current > limit:
                return
            if idx == len(primes):
                smooth.add(current)
                return
            p = primes[idx]
            val = 1
            while val <= limit:
                gen(idx + 1, current * val)
                val *= p
        gen(0, 1)
        return smooth

    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    best_cost = float('inf')
    best_arr = None
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        smooth_nums = generate_smooth(primes, 20000)
        # Filter to valid sums [2, 20000]
        valid_sums = [s for s in smooth_nums if 2 <= s <= 20000]
        valid_sums_set = set(valid_sums)
        
        # For each x in 1..10000, find neighbors y such that x+y is smooth
        # neighbors[x] = set of y in 1..10000 such that x+y in valid_sums_set
        # We don't need to store all, we can check on the fly or precompute a bitset/list
        
        # To speed up clique search, let's precompute for each x, the list of y
        # This is 10000 * (number of smooth sums) which might be large, but smooth sums are sparse.
        # Number of 2,3,5,7-smooth numbers up to 20000 is roughly 500-1000.
        # So neighbors[x] has size ~500. Total memory 10000*500 = 5M entries. OK.
        
        neighbors = [set() for _ in range(10001)]
        for x in range(1, 10001):
            for s in valid_sums:
                y = s - x
                if 1 <= y <= 10000 and y != x:
                    neighbors[x].add(y)
        
        # Greedy clique search with restarts
        # We want a clique of size 12 in this graph.
        # Greedy: pick a random node, then iteratively pick a neighbor connected to all previous.
        
        found = False
        restarts = 5000
        for r in range(restarts):
            if time.time() > deadline - 2:
                break
                
            # Random start
            start_node = random.randint(1, 10000)
            clique = [start_node]
            candidates = set(neighbors[start_node])
            
            while len(clique) < 12 and candidates:
                if time.time() > deadline - 1:
                    break
                # Pick a random candidate
                # To improve success rate, pick one with high degree in the remaining graph?
                # Or just random.
                cand = random.choice(list(candidates))
                clique.append(cand)
                # Intersect candidates with neighbors of cand
                candidates = candidates.intersection(neighbors[cand])
                
            if len(clique) == 12:
                # Verify distinctness (guaranteed by clique logic? No, y!=x check was in neighbor gen)
                # Verify sums are smooth
                valid = True
                for i in range(12):
                    for j in range(i+1, 12):
                        if clique[i] + clique[j] not in valid_sums_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Calculate cost
                    cost = len(primes) # Since all sums are P-smooth, cost <= len(primes)
                    # Actually cost is number of distinct primes in the union of factors.
                    # It could be less than len(primes) if some primes are never used.
                    # But for the purpose of finding a good solution, this is a strong candidate.
                    # Let's compute exact cost to be sure.
                    exact_cost = count_distinct_primes(clique)
                    if exact_cost < best_cost:
                        best_cost = exact_cost
                        best_arr = clique[:]
                        found = True
                        # If we found a very good solution, we can stop early
                        if exact_cost <= 3:
                            return clique
        
        if found and best_cost <= 3:
            return best_arr
        if found and best_cost <= 4:
            # Try to improve with local search? Or just return if we have time.
            # Given the time constraint, returning a 4-prime solution is already a huge win over 9.
            pass
            
    if best_arr is not None:
        return best_arr
        
    # Fallback: Random search
    # Just return a random set or a simple structured one
    # [1, 2, 4, 8, ...] is bad.
    # Let's try the old fallback but faster.
    # Actually, if we reach here, we failed to find a clique.
    # Return a set of multiples of 6?
    arr = [6 * (i + 1) for i in range(12)]
    return arr
