import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # Sieve for Smallest Prime Factor up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
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

    # Candidate Pool
    pool = set()
    for i in range(1, 201):
        pool.add(i)
    p = 1
    while p <= 10000:
        pool.add(p)
        p *= 2
    # Add some randoms
    while len(pool) < 500:
        pool.add(random.randint(1, 10000))
    pool = list(pool)
    
    best_score = float('inf')
    best_set = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        if restart % 3 == 0:
            S = random.sample(pool, 12)
        elif restart % 3 == 1:
            S = list(range(1, 13))
        else:
            S = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        S = list(set(S))
        if len(S) < 12:
            while len(S) < 12:
                c = random.choice(pool)
                if c not in S:
                    S.append(c)
        
        prime_counts = {}
        for i in range(12):
            for j in range(i+1, 12):
                s = S[i] + S[j]
                factors = get_prime_factors(s)
                for p in factors:
                    prime_counts[p] = prime_counts.get(p, 0) + 1
        
        current_score = len(prime_counts)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old_a = S[i]
                for c in pool:
                    if c == old_a or c in S:
                        continue
                    
                    changes = []
                    new_score = current_score
                    # Decrement old
                    for j in range(12):
                        if j == i: continue
                        s = old_a + S[j]
                        factors = get_prime_factors(s)
                        for p in factors:
                            changes.append((p, -1))
                            prime_counts[p] -= 1
                            if prime_counts[p] == 0:
                                new_score -= 1
                                del prime_counts[p]
                    # Increment new
                    for j in range(12):
                        if j == i: continue
                        s = c + S[j]
                        factors = get_prime_factors(s)
                        for p in factors:
                            changes.append((p, 1))
                            if p not in prime_counts:
                                prime_counts[p] = 1
                                new_score += 1
                            else:
                                prime_counts[p] += 1
                    
                    if new_score < current_score:
                        S[i] = c
                        current_score = new_score
                        improved = True
                        break
                    else:
                        # Revert
                        for p, delta in changes:
                            if delta == -1:
                                if p not in prime_counts:
                                    prime_counts[p] = 1
                                else:
                                    prime_counts[p] += 1
                            else: # delta == 1
                                prime_counts[p] -= 1
                                if prime_counts[p] == 0:
                                    del prime_counts[p]
                
                if improved:
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_set = S[:]
            
    if best_set is None:
        return list(range(1, 13))
    return best_set
