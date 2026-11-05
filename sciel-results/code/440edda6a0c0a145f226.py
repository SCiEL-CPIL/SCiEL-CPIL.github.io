import math
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    LIMIT = 10**18
    NUM_PRIMES = len(primes)
    
    def calc_score(n):
        """
        Calculate the exact score tau_perp(n).
        """
        if n < 2:
            return 0
            
        # Factorize n
        temp = n
        factors = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate divisors
        divs = [1]
        for p, exp in factors:
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        
        divs.sort()
        
        score = 0
        # Optimization: use local variables for speed
        gcd = math.gcd
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    def get_n_from_exps(exps):
        """
        Calculate n from exponent vector.
        Returns None if n > LIMIT.
        """
        n = 1
        for i, e in enumerate(exps):
            if e > 0:
                n *= primes[i] ** e
                if n > LIMIT:
                    return None
        return n

    def estimate_priority(exps):
        """
        Heuristic to prioritize search.
        We want many primes and balanced exponents to maximize coprime transitions.
        A good proxy is the number of divisors, but weighted by the number of distinct primes.
        """
        tau = 1
        active_primes = 0
        for e in exps:
            if e > 0:
                tau *= (e + 1)
                active_primes += 1
        # Prioritize more primes and higher tau
        return active_primes * 1000 + tau

    # Initialize with a strong candidate: Primorial of first k primes
    # Find the largest k such that product of first k primes <= LIMIT
    best_exps = [0] * NUM_PRIMES
    current_n = 1
    for i in range(NUM_PRIMES):
        if current_n * primes[i] <= LIMIT:
            current_n *= primes[i]
            best_exps[i] = 1
        else:
            break
            
    # If we have room, increase exponents of smaller primes
    # Greedy fill: increase e_i for i=0,1,... as much as possible while maintaining n <= LIMIT
    # Actually, standard "highly composite" construction: e_0 >= e_1 >= ...
    # Let's refine the initial guess by trying to increase exponents of smaller primes
    # while keeping n <= LIMIT.
    
    # A better initial guess:
    # Start with all 1s for the first k primes.
    # Then, try to increase e_0, then e_1, etc., if it fits.
    # This creates a "staircase" of exponents.
    
    # Let's use a priority queue search from the initial state.
    heap = []
    initial_est = estimate_priority(best_exps)
    heapq.heappush(heap, (-initial_est, -1, current_n, tuple(best_exps)))
    
    visited = set()
    visited.add(tuple(best_exps))
    
    best_n = current_n
    best_score = calc_score(current_n)
    
    # Search
    # We will explore neighbors by changing one exponent by +/- 1.
    # To maintain the "staircase" property which is often optimal for tau(n),
    # we can also try to enforce exps[i] >= exps[i+1].
    # However, for coprime pairs, non-monotonic might be better?
    # Let's stick to monotonic non-increasing for the search space to reduce complexity,
    # as it's known to be optimal for maximizing tau(n). It's a good heuristic for this too.
    
    while heap and time.time() < deadline:
        neg_est, _, n, exps = heapq.heappop(heap)
        
        # Evaluate
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
        # Generate neighbors
        exps_list = list(exps)
        
        # Try increasing exponents
        for i in range(NUM_PRIMES):
            # Only increase if it maintains or improves the "shape"
            # For simplicity, just try +/- 1
            new_exps = exps_list[:]
            new_exps[i] += 1
            
            # Check monotonicity constraint (optional but helps focus search)
            # If i > 0 and new_exps[i] > new_exps[i-1], it's "worse" shape for tau,
            # but might be better for coprime? Let's allow it but penalize in heuristic?
            # Actually, let's just allow any change and let the score decide.
            
            new_n = n * primes[i]
            if new_n <= LIMIT:
                key = tuple(new_exps)
                if key not in visited:
                    visited.add(key)
                    est = estimate_priority(new_exps)
                    heapq.heappush(heap, (-est, -len(visited), new_n, key))
            
            # Try decreasing exponents
            if exps_list[i] > 0:
                new_exps = exps_list[:]
                new_exps[i] -= 1
                new_n = n // primes[i]
                if new_n >= 2:
                    key = tuple(new_exps)
                    if key not in visited:
                        visited.add(key)
                        est = estimate_priority(new_exps)
                        heapq.heappush(heap, (-est, -len(visited), new_n, key))
                        
    return best_n
