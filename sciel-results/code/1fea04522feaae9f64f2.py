import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_set = []
    best_size = 0
    
    def run_greedy(order):
        selected = []
        current_sums = set()
        
        for x in order:
            # Generate new sums involving x
            # We need to check if any sum involving x collides with existing sums
            # or with other new sums involving x.
            
            # Optimization: 
            # 1. Check 3x
            # 2. Check 2x+a for a in selected
            # 3. Check x+a+b for a,b in selected (a<=b)
            
            # To be fast, we can break early on conflict.
            
            new_sums = []
            conflict = False
            
            # 1. 3x
            s = 3 * x
            if s in current_sums:
                continue
            new_sums.append(s)
            
            # 2. 2x + a
            for a in selected:
                s = 2 * x + a
                if s in current_sums or s in new_sums:
                    conflict = True
                    break
                new_sums.append(s)
            if conflict:
                continue
            
            # 3. x + a + b
            for i in range(len(selected)):
                a = selected[i]
                for j in range(i, len(selected)):
                    b = selected[j]
                    s = x + a + b
                    if s in current_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
                if conflict:
                    break
            
            if not conflict:
                selected.append(x)
                current_sums.update(new_sums)
                
        return selected

    # Strategy 1: Ascending
    s1 = run_greedy(range(1, 4097))
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
        
    # Strategy 2: Descending
    if time.time() - start_time < time_limit:
        s2 = run_greedy(range(4096, 0, -1))
        if len(s2) > best_size:
            best_size = len(s2)
            best_set = s2
            
    # Strategy 3: Primes
    if time.time() - start_time < time_limit:
        # Simple sieve for primes up to 4096
        is_prime = [True] * 4097
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(4096**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, 4097, i):
                    is_prime[j] = False
        primes = [i for i, p in enumerate(is_prime) if p]
        s3 = run_greedy(primes)
        if len(s3) > best_size:
            best_size = len(s3)
            best_set = s3
            
    # Strategy 4: Powers of 2 (and nearby)
    if time.time() - start_time < time_limit:
        # This might be too sparse, but let's try a mix
        # Actually, let's try a "gap" strategy: pick 1, then 1+gap, etc.
        # Or just rely on the previous ones.
        
        # Let's try a greedy that prefers numbers with small digital root? 
        # Or just a second pass with a different random seed if we want to add randomness?
        # The directive says "new approach... do not reuse current designs".
        # Current design is random shuffle.
        # This design is deterministic structured.
        
        # Let's add one more: Greedy starting from the middle?
        mid = 2048
        order_mid = list(range(mid, 4097)) + list(range(1, mid))
        s4 = run_greedy(order_mid)
        if len(s4) > best_size:
            best_size = len(s4)
            best_set = s4

    return best_set
