import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    MAX_L = 20000
    
    counts = [0] * (MAX_L + 1)
    bad0 = list(range(1, MAX_L + 1))
    bad1 = []
    bad0_set = set(bad0)
    bad1_set = set(bad1)
    
    residues = []
    
    for p in primes:
        if time.time() > deadline:
            while len(residues) < len(primes):
                residues.append(0)
            return residues
            
        best_r = 0
        best_L = -1
        best_tie_breaker = -1
        
        m0 = bad0[0] if bad0 else MAX_L + 1
        
        for r in range(p):
            m1 = MAX_L + 1
            for x in bad1:
                if x > MAX_L: break
                if x % p != r:
                    m1 = x
                    break
            
            first_bad = min(m0, m1)
            new_L = first_bad - 1
            if new_L > MAX_L:
                new_L = MAX_L
                
            tie_breaker = 0
            if new_L < MAX_L:
                end = min(new_L + p, MAX_L)
                for i in range(new_L + 1, end + 1):
                    if i in bad1_set and i % p == r:
                        tie_breaker += 100
                    elif i in bad0_set and i % p == r:
                        tie_breaker += 1

            if new_L > best_L:
                best_L = new_L
                best_r = r
                best_tie_breaker = tie_breaker
            elif new_L == best_L:
                if tie_breaker > best_tie_breaker:
                    best_r = r
                    best_tie_breaker = tie_breaker

        # Update
        new_bad0 = []
        new_bad1 = []
        for x in bad0:
            if x % p == best_r:
                new_bad1.append(x)
                counts[x] += 1
            else:
                new_bad0.append(x)
                
        for x in bad1:
            if x % p == best_r:
                counts[x] += 1
                # moves to good
            else:
                new_bad1.append(x)
                
        bad0 = new_bad0
        bad1 = new_bad1
        bad0_set = set(bad0)
        bad1_set = set(bad1)
        
        residues.append(best_r)

    return residues
