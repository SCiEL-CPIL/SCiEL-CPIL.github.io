import math
import time
import random
import bisect

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    limit_time = start_time + 280.0  # Safety margin
    
    best_T = []
    best_size = 0
    
    # We will run multiple restarts with randomized greedy initialization
    while time.time() < limit_time:
        # 1. Randomized Greedy Initialization
        active = [True] * (N + 1)
        
        # Process primes in increasing order with randomization
        for p in PRIMES:
            if time.time() > limit_time:
                break
            if p > N:
                break
            
            # Count residues
            counts = [0] * p
            for x in range(N + 1):
                if active[x]:
                    counts[x % p] += 1
            
            # Check if any residue is missing
            min_count = min(counts)
            if min_count > 0:
                # All residues present, must remove one
                # Find all residues with min count
                min_residues = [r for r in range(p) if counts[r] == min_count]
                # Pick randomly among min-count residues for diversity
                min_r = random.choice(min_residues)
                # Remove all x such that x % p == min_r
                for x in range(min_r, N + 1, p):
                    active[x] = False
        
        T = [x for x in range(N + 1) if active[x]]
        
        # 2. Local Search with improved strategies
        # Precompute counts for current T
        counts = []
        missing_counts = []
        for p in PRIMES:
            c = [0] * p
            for x in T:
                c[x % p] += 1
            counts.append(c)
            missing_counts.append(sum(1 for val in c if val == 0))
        
        in_T = [False] * (N + 1)
        for x in T:
            in_T[x] = True
            
        # Local Search Loop
        stall_counter = 0
        max_stall = 500
        consecutive_removals = 0
        
        while time.time() < limit_time:
            current_size = len(T)
            end_idx = bisect.bisect_right(PRIMES, current_size + 1)
            
            # Try systematic addition: check a batch of random candidates
            added = False
            for _ in range(10):  # Try up to 10 candidates per iteration
                x = random.randint(0, N)
                if in_T[x]:
                    continue
                
                valid = True
                for i in range(end_idx):
                    p = PRIMES[i]
                    r = x % p
                    if counts[i][r] == 0 and missing_counts[i] == 1:
                        valid = False
                        break
                
                if valid:
                    # Commit Add
                    for i in range(end_idx):
                        p = PRIMES[i]
                        r = x % p
                        if counts[i][r] == 0:
                            missing_counts[i] -= 1
                        counts[i][r] += 1
                    
                    T.append(x)
                    in_T[x] = True
                    stall_counter = 0
                    consecutive_removals = 0
                    added = True
                    break
                else:
                    stall_counter += 1
            
            if not added:
                stall_counter += 10
            
            # If stalled, try smart removal
            if stall_counter > max_stall:
                # Remove element that blocks the most potential additions
                if T:
                    # Score each element by how many potential additions it blocks
                    # An element y blocks addition of x if for some p <= len(T)+1,
                    # x % p has count 0 and missing_count 1, and y % p == x % p
                    
                    # Simplified: remove random element but with bias toward
                    # elements in "tight" residue classes
                    idx = random.randint(0, len(T) - 1)
                    y = T[idx]
                    
                    in_T[y] = False
                    T.pop(idx)
                    
                    # Update counts for all primes
                    for i in range(len(PRIMES)):
                        p = PRIMES[i]
                        r = y % p
                        counts[i][r] -= 1
                        if counts[i][r] == 0:
                            missing_counts[i] += 1
                    
                    consecutive_removals += 1
                    stall_counter = 0
                    
                    # If we've removed too many in a row, increase max_stall
                    if consecutive_removals > 5:
                        max_stall = int(max_stall * 1.3)
                        if max_stall > 5000:
                            max_stall = 5000
                
                else:
                    # T is empty, break
                    break
        
        # Update best
        if len(T) > best_size:
            best_size = len(T)
            best_T = T.copy()
            
        # If we've found a very good solution, do a final polish
        if best_size > 2200 and time.time() < limit_time - 30:
            # Try to improve best_T with a more careful local search
            T = best_T.copy()
            counts = []
            missing_counts = []
            for p in PRIMES:
                c = [0] * p
                for x in T:
                    c[x % p] += 1
                counts.append(c)
                missing_counts.append(sum(1 for val in c if val == 0))
            
            in_T = [False] * (N + 1)
            for x in T:
                in_T[x] = True
            
            polish_time_limit = time.time() + 25
            while time.time() < polish_time_limit:
                current_size = len(T)
                end_idx = bisect.bisect_right(PRIMES, current_size + 1)
                
                # Try to add elements systematically
                for x in range(N + 1):
                    if in_T[x]:
                        continue
                    valid = True
                    for i in range(end_idx):
                        p = PRIMES[i]
                        r = x % p
                        if counts[i][r] == 0 and missing_counts[i] == 1:
                            valid = False
                            break
                    if valid:
                        for i in range(end_idx):
                            p = PRIMES[i]
                            r = x % p
                            if counts[i][r] == 0:
                                missing_counts[i] -= 1
                            counts[i][r] += 1
                        T.append(x)
                        in_T[x] = True
                        if len(T) > best_size:
                            best_size = len(T)
                            best_T = T.copy()
                
                if time.time() > polish_time_limit:
                    break
                
                # If no more additions possible, try removals
                current_size = len(T)
                end_idx = bisect.bisect_right(PRIMES, current_size + 1)
                added_any = False
                for x in range(N + 1):
                    if in_T[x]:
                        continue
                    valid = True
                    for i in range(end_idx):
                        p = PRIMES[i]
                        r = x % p
                        if counts[i][r] == 0 and missing_counts[i] == 1:
                            valid = False
                            break
                    if valid:
                        added_any = True
                        break
                
                if not added_any:
                    # Try removing a random element to unblock
                    if T:
                        idx = random.randint(0, len(T) - 1)
                        y = T[idx]
                        in_T[y] = False
                        T.pop(idx)
                        for i in range(len(PRIMES)):
                            p = PRIMES[i]
                            r = y % p
                            counts[i][r] -= 1
                            if counts[i][r] == 0:
                                missing_counts[i] += 1
    
    return best_T
