import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute the set of residues for each prime
    # p is "active" for n if n % p is in residues[p]
    # n+i is divisible by p => n % p == (-i) % p
    # i ranges from 1 to 24
    residues = {}
    for p in primes:
        res_set = set()
        for i in range(1, 25):
            res_set.add((-i) % p)
        residues[p] = res_set
        
    best_n = 1000
    best_score = 10 # Max possible is 9
    
    # Scan the range
    # We can step by 1
    for n in range(1000, 1000001):
        # Check time
        if time.time() > deadline:
            break
            
        score = 0
        for p in primes:
            if n % p in residues[p]:
                score += 1
                if score >= best_score:
                    break # No need to check further if already worse or equal
            
        if score < best_score:
            best_score = score
            best_n = n
            
    return best_n
