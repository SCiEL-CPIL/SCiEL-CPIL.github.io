import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Precomputation: Prime Factors up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(math.isqrt(MAX_SUM)) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    prime_factors = [set() for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        n = i
        while n > 1:
            p = spf[n]
            prime_factors[i].add(p)
            while n % p == 0:
                n //= p

    # 2. Generate Candidate Pool
    pool = set()
    for i in range(1, 201):
        pool.add(i)
        
    # 7-smooth numbers
    for a in range(15):
        v2 = 2 ** a
        if v2 > 10000: break
        for b in range(11):
            v3 = 3 ** b
            if v2 * v3 > 10000: break
            for c in range(9):
                v5 = 5 ** c
                v235 = v2 * v3 * v5
                if v235 > 10000: break
                for d in range(6):
                    v7 = 7 ** d
                    val = v235 * v7
                    if val > 10000: break
                    pool.add(val)
                    
    pool = list(pool)
    random.shuffle(pool)
    
    # 3. Greedy Initialization
    current_set = []
    prime_count = {}
    active_prime_count = 0
    
    for _ in range(12):
        best_candidate = None
        best_cost = float('inf')
        
        for c in pool:
            if c in current_set:
                continue
                
            # Simulate adding c
            # We need to know how many NEW primes are introduced.
            # A prime p is new if prime_count.get(p, 0) == 0.
            
            new_primes = 0
            for x in current_set:
                s = c + x
                for p in prime_factors[s]:
                    if prime_count.get(p, 0) == 0:
                        new_primes += 1
                        # Note: if multiple pairs introduce the same new prime, 
                        # we only count it once for the "active" set increase.
                        # But here we are just estimating the cost increase.
                        # Actually, we need to be careful.
                        # If p is new, active count increases by 1.
                        # If p is already active, active count doesn't change.
                        
            # This logic is slightly flawed because if two pairs both introduce the same NEW prime p, 
            # we count it twice in `new_primes`, but active count only increases by 1.
            # However, for greedy, this is a good heuristic.
            # Better: Just compute the exact new active count.
            
            # Let's just compute exact cost for greedy. It's fast enough.
            temp_set = current_set + [c]
            temp_prime_set = set()
            for i in range(len(temp_set)):
                for j in range(i+1, len(temp_set)):
                    s = temp_set[i] + temp_set[j]
                    temp_prime_set.update(prime_factors[s])
            
            cost = len(temp_prime_set)
            if cost < best_cost:
                best_cost = cost
                best_candidate = c
                
        if best_candidate is None:
            for c in pool:
                if c not in current_set:
                    best_candidate = c
                    break
        
        current_set.append(best_candidate)
        
        # Update state
        prime_count = {}
        for i in range(len(current_set)):
            for j in range(i+1, len(current_set)):
                s = current_set[i] + current_set[j]
                for p in prime_factors[s]:
                    prime_count[p] = prime_count.get(p, 0) + 1
        active_prime_count = len(prime_count)
        
    # 4. Local Search
    best_set = current_set[:]
    best_cost = active_prime_count
    current_cost = active_prime_count
    
    T = 10.0
    cooling = 0.99
    
    while time.time() < deadline:
        k = random.randint(0, 11)
        old_val = current_set[k]
        
        new_val = None
        for _ in range(10):
            cand = random.choice(pool)
            if cand not in current_set:
                new_val = cand
                break
        if new_val is None:
            continue
            
        # Apply Move
        # Store changes to rollback
        changes = [] # List of (prime, delta)
        
        # Remove old_val
        for j in range(12):
            if j == k: continue
            s = old_val + current_set[j]
            for p in prime_factors[s]:
                prime_count[p] -= 1
                if prime_count[p] == 0:
                    active_prime_count -= 1
                changes.append((p, 1)) # Record that we decremented p (to increment back)
                
        # Add new_val
        for j in range(12):
            if j == k: continue
            s = new_val + current_set[j]
            for p in prime_factors[s]:
                if p not in prime_count:
                    prime_count[p] = 0
                    active_prime_count += 1
                prime_count[p] += 1
                changes.append((p, -1)) # Record that we incremented p (to decrement back)
                
        new_cost = active_prime_count
        delta = new_cost - current_cost
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0.01:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
                    
        if accept:
            current_set[k] = new_val
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = current_set[:]
        else:
            # Rollback
            for p, delta_change in changes:
                if delta_change == 1:
                    # We decremented, so increment back
                    prime_count[p] += 1
                    if prime_count[p] == 1:
                        active_prime_count += 1
                else:
                    # We incremented, so decrement back
                    prime_count[p] -= 1
                    if prime_count[p] == 0:
                        active_prime_count -= 1
            # Note: The rollback logic above is slightly tricky with the active count.
            # It's safer to just recalculate active_prime_count from prime_count if rollback fails?
            # Or better: Just don't use incremental updates for rollback.
            # Given the speed of full recalc, maybe just use full recalc for simplicity and correctness?
            # Let's stick to full recalc in the inner loop if incremental is buggy.
            # But 33 ops vs 432 ops is significant.
            # Let's trust the incremental logic but verify.
            
            # Actually, the rollback logic above has a flaw:
            # If I decrement p to 0, active decreases.
            # If I rollback by incrementing p to 1, active increases.
            # This works.
            # If I increment p from 0 to 1, active increases.
            # If I rollback by decrementing p to 0, active decreases.
            # This works.
            # The `changes` list records the net effect on `prime_count` values.
            # But `active_prime_count` is a scalar.
            # The rollback of `active_prime_count` is implicit in the `prime_count` updates?
            # No, I am explicitly updating `active_prime_count` in the rollback loop.
            # So it should be consistent.
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_set

# EVOLVE-BLOCK-END
