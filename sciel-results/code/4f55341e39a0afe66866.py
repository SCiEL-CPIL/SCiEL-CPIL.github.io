import math
import time

def construct():
    n = 256
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Precompute prime factors for each number 1..n
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)
    
    def get_factors(num):
        """Return set of prime factors of num"""
        return set(prime_factors[num])
    
    def is_pairwise_coprime(lst):
        """Check if all pairs in lst are coprime"""
        factors = [get_factors(x) for x in lst]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if factors[i] & factors[j]:
                    return False
        return True
    
    def sum_of(lst):
        return sum(lst)
    
    # Greedy initialization: take 1 and highest power of each prime
    result = [1]
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        result.append(power)
    
    best_result = list(result)
    best_sum = sum_of(result)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any missing element that doesn't conflict
        current_set = set(result)
        for candidate in range(2, n + 1):
            if candidate in current_set:
                continue
            cand_factors = get_factors(candidate)
            conflict = False
            for existing in result:
                if cand_factors & get_factors(existing):
                    conflict = True
                    break
            if not conflict:
                new_result = result + [candidate]
                new_sum = best_sum + candidate
                if new_sum > best_sum:
                    result = new_result
                    best_result = list(result)
                    best_sum = new_sum
                    improved = True
                    break  # restart loop with new state
        
        if time.time() >= deadline:
            break
        
        # Try 1-exchange: remove one element, add best non-conflicting
        for i in range(len(result)):
            if time.time() >= deadline:
                break
            removed = result[i]
            remaining = result[:i] + result[i+1:]
            removed_factors = get_factors(removed)
            
            # Find best candidate to add
            best_candidate = None
            best_gain = 0
            for candidate in range(2, n + 1):
                if candidate in set(remaining):
                    continue
                cand_factors = get_factors(candidate)
                # Must not conflict with remaining
                conflict = False
                for existing in remaining:
                    if cand_factors & get_factors(existing):
                        conflict = True
                        break
                if conflict:
                    continue
                gain = candidate - removed
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate
            
            if best_candidate is not None and best_gain > 0:
                new_result = remaining + [best_candidate]
                new_sum = best_sum + best_gain
                if new_sum > best_sum:
                    result = new_result
                    best_result = list(result)
                    best_sum = new_sum
                    improved = True
                    break  # restart loop with new state
        
        if time.time() >= deadline:
            break
        
        # Try 2-exchange: remove two elements, add best non-conflicting pair
        for i in range(len(result)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(result)):
                if time.time() >= deadline:
                    break
                removed1 = result[i]
                removed2 = result[j]
                remaining = result[:i] + result[i+1:j] + result[j+1:]
                
                # Find best pair to add
                best_pair = None
                best_gain = 0
                for c1 in range(2, n + 1):
                    if c1 in set(remaining) or c1 == removed1 or c1 == removed2:
                        continue
                    f1 = get_factors(c1)
                    # Check c1 against remaining
                    conflict1 = False
                    for existing in remaining:
                        if f1 & get_factors(existing):
                            conflict1 = True
                            break
                    if conflict1:
                        continue
                    
                    for c2 in range(c1 + 1, n + 1):
                        if c2 in set(remaining) or c2 == removed1 or c2 == removed2:
                            continue
                        f2 = get_factors(c2)
                        # Check c2 against remaining and c1
                        conflict2 = False
                        for existing in remaining:
                            if f2 & get_factors(existing):
                                conflict2 = True
                                break
                        if conflict2:
                            continue
                        if f1 & f2:
                            continue
                        
                        gain = c1 + c2 - removed1 - removed2
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (c1, c2)
                
                if best_pair is not None and best_gain > 0:
                    new_result = remaining + list(best_pair)
                    new_sum = best_sum + best_gain
                    if new_sum > best_sum:
                        result = new_result
                        best_result = list(result)
                        best_sum = new_sum
                        improved = True
                        break  # restart loop with new state
            
            if improved:
                break
        
        if time.time() >= deadline:
            break
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_result]
