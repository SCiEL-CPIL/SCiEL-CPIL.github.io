import math
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit
    
    N = 256
    
    # Precompute prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    prime_factors[1] = set()  # 1 has no prime factors
    for i in range(2, N + 1):
        x = i
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].add(x)
    
    # Check if number x is coprime to all in set S (using prime factor sets)
    def is_coprime_to_set(x, S):
        if x == 1:
            return True
        pf_x = prime_factors[x]
        for y in S:
            if y == 1:
                continue
            if pf_x & prime_factors[y]:
                return False
        return True
    
    # Get all primes up to N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(N)
    
    # Initial solution: all primes
    current_set = set(primes)
    current_sum = sum(current_set)
    
    all_numbers = list(range(1, N + 1))
    
    def try_add(S, s_sum, x):
        if x in S:
            return S, s_sum, False
        if is_coprime_to_set(x, S):
            S.add(x)
            return S, s_sum + x, True
        return S, s_sum, False
    
    def try_swap(S, s_sum, out_x, in_y):
        if out_x not in S or in_y in S:
            return S, s_sum, False
        if in_y == out_x:
            return S, s_sum, False
        # Check if in_y is coprime to S \ {out_x}
        temp_set = S - {out_x}
        if is_coprime_to_set(in_y, temp_set):
            S.discard(out_x)
            S.add(in_y)
            return S, s_sum - out_x + in_y, True
        return S, s_sum, False
    
    # Phase 1: Greedy add - add any number that's coprime to current set
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for x in all_numbers:
            if x not in current_set:
                new_set, new_sum, did_add = try_add(current_set, current_sum, x)
                if did_add:
                    current_set = new_set
                    current_sum = new_sum
                    improved = True
                    break  # Restart to re-evaluate
    
    # Phase 2: Single swap - replace x with y if y > x and valid
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        s_list = list(current_set)
        candidates_out = sorted(s_list, key=lambda x: x)  # Try removing small ones first
        candidates_in = [y for y in all_numbers if y not in current_set]
        candidates_in.sort(reverse=True)  # Try adding large ones first
        
        for x in candidates_out:
            if time.time() - start_time > time_limit - 3:
                break
            for y in candidates_in:
                if y <= x:
                    break  # No gain
                new_set, new_sum, did_swap = try_swap(current_set, current_sum, x, y)
                if did_swap:
                    current_set = new_set
                    current_sum = new_sum
                    improved = True
                    break
            if improved:
                break
    
    # Phase 3: Multi-swap - replace subset A with subset B if valid and gain > 0
    # Focus on replacing 2-3 small primes with 1-2 larger composites
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        s_list = list(current_set)
        
        # Try replacing 2 elements with 1
        for i in range(len(s_list)):
            if time.time() - start_time > time_limit - 3:
                break
            for j in range(i + 1, len(s_list)):
                x1, x2 = s_list[i], s_list[j]
                if x1 + x2 >= 256:
                    continue  # Can't beat with a single number <= 256
                # Try adding y that replaces {x1, x2}
                # y must be coprime to S \ {x1, x2} and y > x1 + x2
                temp_set = current_set - {x1, x2}
                for y in all_numbers:
                    if y in current_set or y <= x1 + x2:
                        continue
                    if is_coprime_to_set(y, temp_set):
                        # Valid multi-swap
                        current_set.discard(x1)
                        current_set.discard(x2)
                        current_set.add(y)
                        current_sum += y - x1 - x2
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Try replacing 3 elements with 1
        if not improved:
            for i in range(len(s_list)):
                if time.time() - start_time > time_limit - 3:
                    break
                for j in range(i + 1, len(s_list)):
                    for k in range(j + 1, len(s_list)):
                        x1, x2, x3 = s_list[i], s_list[j], s_list[k]
                        if x1 + x2 + x3 >= 256:
                            continue
                        temp_set = current_set - {x1, x2, x3}
                        for y in all_numbers:
                            if y in current_set or y <= x1 + x2 + x3:
                                continue
                            if is_coprime_to_set(y, temp_set):
                                current_set.discard(x1)
                                current_set.discard(x2)
                                current_set.discard(x3)
                                current_set.add(y)
                                current_sum += y - x1 - x2 - x3
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Phase 4: Random restarts with local search
    best_sum = current_sum
    best_set = set(current_set)
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        num_restarts = int(remaining_time / 5)  # ~5s per restart
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Create a perturbed solution
            random_set = set()
            random_sum = 0
            # Start with a random subset of primes
            shuffled_primes = primes[:]
            random.shuffle(shuffled_primes)
            for p in shuffled_primes:
                if is_coprime_to_set(p, random_set):
                    random_set.add(p)
                    random_sum += p
            
            # Greedy add
            improved = True
            while improved and time.time() - start_time < time_limit - 3:
                improved = False
                for x in all_numbers:
                    if x not in random_set:
                        new_set, new_sum, did_add = try_add(random_set, random_sum, x)
                        if did_add:
                            random_set = new_set
                            random_sum = new_sum
                            improved = True
                            break
            
            # Single swap
            improved = True
            while improved and time.time() - start_time < time_limit - 3:
                improved = False
                s_list = list(random_set)
                candidates_out = sorted(s_list, key=lambda x: x)
                candidates_in = [y for y in all_numbers if y not in random_set]
                candidates_in.sort(reverse=True)
                
                for x in candidates_out:
                    if time.time() - start_time > time_limit - 3:
                        break
                    for y in candidates_in:
                        if y <= x:
                            break
                        new_set, new_sum, did_swap = try_swap(random_set, random_sum, x, y)
                        if did_swap:
                            random_set = new_set
                            random_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
            
            # Update best
            if random_sum > best_sum:
                best_sum = random_sum
                best_set = set(random_set)
    
    current_set = best_set
    
    # Final validation and type check
    result = list(current_set)
    # Ensure all are Python int
    result = [int(x) for x in result]
    return result
