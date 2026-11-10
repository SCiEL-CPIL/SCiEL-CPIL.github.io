import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin before 150s kill
    
    N = 1000
    best_set = []
    best_size = 0
    
    def greedy_build(order):
        """Build a B2[2] set greedily from the given order."""
        S = []
        counts = [0] * (2 * N + 1)
        in_set = set()
        for x in order:
            if x in in_set:
                continue
            valid = True
            # Check sums x + y for y in S
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            # Check sum 2*x
            if valid:
                s = 2 * x
                if s <= 2 * N and counts[s] >= 2:
                    valid = False
            
            if valid:
                # Update counts
                for y in S:
                    s = x + y
                    counts[s] += 1
                s2 = 2 * x
                if s2 <= 2 * N:
                    counts[s2] += 1
                S.append(x)
                in_set.add(x)
        return S, counts
    
    def local_search(S, counts, max_time=5.0):
        """Try to improve the set by swapping elements."""
        if not S:
            return S, counts
        
        current_set = list(S)
        current_counts = list(counts)
        in_set = set(current_set)
        current_size = len(current_set)
        
        # Precompute which sums are "tight" (count == 2)
        # These are the constraints that limit us
        
        search_start = time.time()
        no_improve_count = 0
        
        while time.time() - search_start < max_time and no_improve_count < 20:
            improved = False
            # Try swapping each element
            for i in range(len(current_set)):
                if time.time() > deadline - 2.0:
                    return current_set, current_counts
                
                elem = current_set[i]
                
                # Try a few random candidates to replace elem
                for _attempt in range(5):
                    candidate = random.randint(1, N)
                    if candidate in in_set:
                        continue
                    
                    # Temporarily remove elem
                    new_set = list(current_set)
                    new_set.pop(i)
                    new_counts = list(current_counts)
                    
                    # Subtract elem's contributions
                    for y in new_set:
                        s = elem + y
                        if s <= 2 * N:
                            new_counts[s] -= 1
                    s2 = 2 * elem
                    if s2 <= 2 * N:
                        new_counts[s2] -= 1
                    
                    # Try to add candidate
                    valid = True
                    for y in new_set:
                        s = candidate + y
                        if s <= 2 * N and new_counts[s] >= 2:
                            valid = False
                            break
                    
                    if valid:
                        s2c = 2 * candidate
                        if s2c <= 2 * N and new_counts[s2c] >= 2:
                            valid = False
                    
                    if valid:
                        # Accept the swap
                        for y in new_set:
                            s = candidate + y
                            if s <= 2 * N:
                                new_counts[s] += 1
                        if s2c <= 2 * N:
                            new_counts[s2c] += 1
                        
                        current_set = new_set + [candidate]
                        current_counts = new_counts
                        in_set = set(current_set)
                        improved = True
                        break
                
                if improved:
                    no_improve_count = 0
                    break
            if not improved:
                no_improve_count += 1
        
        return current_set, current_counts
    
    def evaluate(order):
        nonlocal best_size, best_set
        S, counts = greedy_build(order)
        # Apply local search to improve
        S2, counts2 = local_search(S, counts, max_time=3.0)
        if len(S2) > best_size:
            best_size = len(S2)
            best_set = S2
    
    # 1. Natural order
    evaluate(list(range(1, N + 1)))
    
    # 2. Reverse order
    if time.time() < deadline - 10:
        evaluate(list(range(N, 0, -1)))
    
    # 3. Primes first, then composites
    if time.time() < deadline - 10:
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        primes = [x for x in range(1, N+1) if is_prime(x)]
        composites = [x for x in range(1, N+1) if not is_prime(x)]
        evaluate(primes + composites)
    
    # 4. Sparse pattern: powers of 2 first
    if time.time() < deadline - 10:
        powers = [2**k for k in range(10) if 2**k <= N]
        rest = [x for x in range(1, N+1) if x not in set(powers)]
        evaluate(powers + rest)
    
    # 5. Random shuffles with local search
    rng = random.Random(42)
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        evaluate(order)
        
        # If we've found a very good solution, break early
        if best_size >= 750:
            break
    
    return best_set
