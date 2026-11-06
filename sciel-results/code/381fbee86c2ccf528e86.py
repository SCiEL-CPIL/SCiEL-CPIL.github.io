import random
import time

def construct():
    # Generate primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    n_primes = len(primes)
    LIMIT = 10000 
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in primes]
    
    def get_counts(residues):
        counts = [0] * (LIMIT + 1)
        for r, p in zip(residues, primes):
            start = r if r > 0 else p
            if start <= LIMIT:
                for x in range(start, LIMIT + 1, p):
                    counts[x] += 1
        return counts

    def get_score_from_counts(counts):
        L = 0
        for x in range(1, LIMIT + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L

    current_counts = get_counts(residues)
    current_score = get_score_from_counts(current_counts)
    best_residues = residues.copy()
    best_score = current_score
    
    start_time = time.time()
    max_time = 100
    
    while time.time() - start_time < max_time:
        # Strategy 1: Directed move to fix the first failure
        # Find first x with count < 2
        x_fail = current_score + 1
        if x_fail > LIMIT:
            break
            
        if current_counts[x_fail] < 2:
            # Find primes that do NOT cover x_fail
            # We want to change one of them to cover x_fail
            candidates = []
            for i, p in enumerate(primes):
                if (x_fail - residues[i]) % p != 0:
                    candidates.append(i)
            
            if candidates:
                # Pick a random candidate
                idx = random.choice(candidates)
                p = primes[idx]
                old_r = residues[idx]
                new_r = x_fail % p
                
                # Update counts incrementally
                # Remove old_r
                start_old = old_r if old_r > 0 else p
                if start_old <= LIMIT:
                    for x in range(start_old, LIMIT + 1, p):
                        current_counts[x] -= 1
                        
                # Add new_r
                start_new = new_r if new_r > 0 else p
                if start_new <= LIMIT:
                    for x in range(start_new, LIMIT + 1, p):
                        current_counts[x] += 1
                        
                residues[idx] = new_r
                
                new_score = get_score_from_counts(current_counts)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    # Revert
                    residues[idx] = old_r
                    # Revert counts
                    start_new = new_r if new_r > 0 else p
                    if start_new <= LIMIT:
                        for x in range(start_new, LIMIT + 1, p):
                            current_counts[x] -= 1
                    start_old = old_r if old_r > 0 else p
                    if start_old <= LIMIT:
                        for x in range(start_old, LIMIT + 1, p):
                            current_counts[x] += 1

        # Strategy 2: Random move to escape local minima
        if random.random() < 0.1: # 10% of the time
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Update counts
            start_old = old_r if old_r > 0 else p
            if start_old <= LIMIT:
                for x in range(start_old, LIMIT + 1, p):
                    current_counts[x] -= 1
            start_new = new_r if new_r > 0 else p
            if start_new <= LIMIT:
                for x in range(start_new, LIMIT + 1, p):
                    current_counts[x] += 1
            
            residues[idx] = new_r
            new_score = get_score_from_counts(current_counts)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert
                residues[idx] = old_r
                start_new = new_r if new_r > 0 else p
                if start_new <= LIMIT:
                    for x in range(start_new, LIMIT + 1, p):
                        current_counts[x] -= 1
                start_old = old_r if old_r > 0 else p
                if start_old <= LIMIT:
                    for x in range(start_old, LIMIT + 1, p):
                        current_counts[x] += 1

    return best_residues
