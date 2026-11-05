import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    prime_set = set(primes)
    
    # Generate candidate sets of primes
    # Seed set: primes >= 13
    seed_set = [p for p in primes if p >= 13]
    
    # Helper to generate a set by swapping out 'remove' and adding 'add'
    # Ensures sum of reciprocals <= 1
    def make_candidate(base_set, to_remove, to_add):
        new_set = [p for p in base_set if p not in to_remove]
        # Check if we can add 'to_add'
        # Calculate current sum
        current_sum = sum(1.0 / p for p in new_set)
        valid_add = []
        for p in to_add:
            if p in new_set:
                continue
            if current_sum + 1.0 / p <= 1.0:
                valid_add.append(p)
                current_sum += 1.0 / p
        new_set.extend(valid_add)
        return new_set

    candidates = []
    candidates.append(seed_set)
    
    # Try adding 11
    # Remove some large primes to make room
    # 1/11 = 0.0909. Seed sum ~0.93. Slack 0.07. Need to remove ~0.02.
    # Remove 53, 59
    cand_11 = make_candidate(seed_set, [53, 59], [11])
    candidates.append(cand_11)
    
    # Try adding 7
    # 1/7 = 0.1428. Need to remove ~0.07.
    # Remove 13
    cand_7 = make_candidate(seed_set, [13], [7])
    candidates.append(cand_7)
    
    # Try adding 5
    # 1/5 = 0.2. Need to remove ~0.13.
    # Remove 13, 17
    cand_5 = make_candidate(seed_set, [13, 17], [5])
    candidates.append(cand_5)
    
    # Try adding 3
    # 1/3 = 0.333. Need to remove ~0.26.
    # Remove 13, 17, 19, 23, 29
    cand_3 = make_candidate(seed_set, [13, 17, 19, 23, 29], [3])
    candidates.append(cand_3)
    
    # Try adding 2
    # 1/2 = 0.5. Need to remove ~0.43.
    # Remove 13..47
    to_remove_2 = [p for p in seed_set if p <= 47]
    cand_2 = make_candidate(seed_set, to_remove_2, [2])
    candidates.append(cand_2)
    
    # Try a mix: Add 7 and 11?
    # Remove 13, 53, 59
    cand_7_11 = make_candidate(seed_set, [13, 53, 59], [7, 11])
    candidates.append(cand_7_11)
    
    # Try adding 5 and 11?
    # Remove 13, 17, 53, 59
    cand_5_11 = make_candidate(seed_set, [13, 17, 53, 59], [5, 11])
    candidates.append(cand_5_11)
    
    # Evaluate each candidate
    best_L = 0
    best_classes = []
    
    for P in candidates:
        if time.time() > deadline - 10:
            break
            
        # Sort primes ascending for greedy assignment
        P_sorted = sorted(P)
        
        # Greedy residue assignment
        # We want to cover 1..L
        # We assume L is at most 100000
        max_L = 100000
        covered = [False] * (max_L + 1)
        residues = {}
        used_primes = set()
        
        # We will iterate n from 1 to max_L
        # If n is not covered, we assign a residue to the smallest available prime
        # This heuristic tends to use small primes for small n
        
        # Optimization: Precompute multiples for each prime?
        # For each prime p, the numbers covered are r, r+p, r+2p...
        
        for n in range(1, max_L + 1):
            if time.time() > deadline - 5:
                break
                
            if covered[n]:
                continue
                
            # Find smallest prime in P_sorted not yet used
            assigned = False
            for p in P_sorted:
                if p in used_primes:
                    continue
                    
                # Use p to cover n
                r = n % p
                residues[p] = r
                used_primes.add(p)
                
                # Mark covered numbers
                # Start from the first number >= 1 congruent to r mod p
                # k = r if r >= 1 else r + p
                # Actually, n is the first one we care about?
                # We mark all k in [1, max_L] such that k % p == r
                start_k = r if r > 0 else p
                # If r == 0, start at p. If r > 0, start at r.
                # But we only care about 1..max_L.
                for k in range(start_k, max_L + 1, p):
                    covered[k] = True
                    
                assigned = True
                break
                
            if not assigned:
                # No more primes to assign
                break
                
        # Calculate L
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
                
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in P_sorted if p in residues]
            
    # Verify sum of reciprocals is <= 1
    # The make_candidate function ensured this, but double check
    s = 0
    for p, r in best_classes:
        s += Fraction(1, p)
    if s > 1:
        # Fallback to seed
        return [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
        
    return best_classes
