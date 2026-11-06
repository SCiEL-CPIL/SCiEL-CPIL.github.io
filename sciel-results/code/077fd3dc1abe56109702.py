import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin

    # Precompute Smallest Prime Factor (SPF) up to 20000
    max_val = 20000
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_prime_factors(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def eval_b(arr_b):
        primes = set()
        n = len(arr_b)
        for i in range(n):
            for j in range(i+1, n):
                s = arr_b[i] + arr_b[j]
                if s <= max_val:
                    primes.update(get_prime_factors(s))
                else:
                    # Fallback for large sums
                    s_tmp = s
                    while s_tmp > 1:
                        p = spf[s_tmp] if s_tmp <= max_val else s_tmp
                        primes.add(p)
                        while s_tmp % p == 0:
                            s_tmp //= p
        return len(primes), primes

    # Greedy set-cover heuristic to find a good prime set
    def greedy_set_cover(target_sums, max_primes=500):
        """Find a small set of primes that covers all target sums."""
        uncovered = set(target_sums)
        if not uncovered:
            return set()
        
        # Get all primes that divide at least one target sum
        prime_to_sums = {}
        for s in target_sums:
            for p in get_prime_factors(s):
                if p not in prime_to_sums:
                    prime_to_sums[p] = set()
                prime_to_sums[p].add(s)
        
        selected_primes = set()
        while uncovered and len(selected_primes) < max_primes:
            best_prime = None
            best_count = -1
            for p, sums in prime_to_sums.items():
                if p in selected_primes:
                    continue
                count = len(sums & uncovered)
                if count > best_count:
                    best_count = count
                    best_prime = p
            
            if best_prime is None or best_count <= 0:
                break
            
            selected_primes.add(best_prime)
            uncovered -= prime_to_sums[best_prime]
        
        return selected_primes

    # Initial heuristic: try several starting configurations
    candidates = []
    
    # Configuration 1: Odd numbers scaled by 6
    b1 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    candidates.append(b1[:])
    
    # Configuration 2: Consecutive odd numbers starting from different base
    for base in [1, 3, 5, 7]:
        b = [base + 2*i for i in range(12)]
        if max(b) <= 1666:
            candidates.append(b)
    
    # Configuration 3: Powers of 2 based
    b3 = [2**i for i in range(1, 13)]
    if max(b3) <= 1666:
        candidates.append(b3[:])
    
    # Configuration 4: Fibonacci-like
    fib = [1, 1]
    for i in range(10):
        fib.append(fib[-1] + fib[-2])
    candidates.append(fib[1:13])
    
    # Configuration 5: Random with structure
    random.seed(42)
    for _ in range(5):
        b = sorted(random.sample(range(1, 1667), 12))
        candidates.append(b)
    
    # Evaluate all candidates
    best_b = None
    best_score = float('inf')
    for b in candidates:
        score, primes = eval_b(b)
        if score < best_score:
            best_score = score
            best_b = b[:]
    
    if best_b is None:
        best_b = candidates[0][:]
        best_score, _ = eval_b(best_b)
    
    current_score = best_score
    
    # Local Search with multiple strategies
    max_iters = 10000
    stall_count = 0
    max_stall = 200
    
    for it in range(max_iters):
        if time.time() - start_time > timeout - 5:
            break
        
        # Pick a random index to mutate
        k = random.randint(0, 11)
        old_val = best_b[k]
        
        # Try multiple mutation strategies
        improved = False
        for strategy in range(3):
            if improved:
                break
            
            for _ in range(10):
                if time.time() - start_time > timeout - 5:
                    break
                
                if strategy == 0:
                    # Random replacement
                    new_val = random.randint(1, 1666)
                elif strategy == 1:
                    # Local perturbation
                    delta = random.choice([-3, -2, -1, 1, 2, 3])
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 1666:
                        continue
                else:
                    # Swap with another element
                    if k == 0:
                        continue
                    j = random.randint(0, 11)
                    if j == k:
                        continue
                    new_val = best_b[j]
                    best_b[j] = old_val
                    # Evaluate swap
                    new_score, _ = eval_b(best_b)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    elif new_score == current_score:
                        if random.random() < 0.1:
                            improved = True
                            break
                    # Undo swap
                    best_b[j], best_b[k] = best_b[k], best_b[j]
                    continue
                
                if new_val in best_b and new_val != old_val:
                    continue
                if new_val == old_val:
                    continue
                
                # Swap
                best_b[k] = new_val
                new_score, _ = eval_b(best_b)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    stall_count = 0
                    break
                elif new_score == current_score:
                    # Tie-breaker: accept with small probability
                    if random.random() < 0.05:
                        improved = True
                        break
                    else:
                        best_b[k] = old_val
                else:
                    best_b[k] = old_val
        
        if not improved:
            stall_count += 1
            if stall_count >= max_stall:
                # Restart with perturbation
                stall_count = 0
                for _ in range(3):
                    k = random.randint(0, 11)
                    best_b[k] = random.randint(1, 1666)
                    # Ensure distinctness
                    while best_b[k] in best_b[:k] + best_b[k+1:]:
                        best_b[k] = random.randint(1, 1666)
                current_score, _ = eval_b(best_b)
        else:
            stall_count = 0
    
    # Final polish: tighter local search
    polish_iters = 2000
    for _ in range(polish_iters):
        if time.time() - start_time > timeout - 1:
            break
        
        k = random.randint(0, 11)
        old_val = best_b[k]
        
        for delta in range(-5, 6):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 1 or new_val > 1666:
                continue
            if new_val in best_b and new_val != old_val:
                continue
            
            best_b[k] = new_val
            new_score, _ = eval_b(best_b)
            
            if new_score < current_score:
                current_score = new_score
                old_val = new_val
            else:
                best_b[k] = old_val
                break
    
    # Final result
    a = [6 * x for x in best_b]
    
    return a
