import time
import math

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if _is_prime(p)]
    
    # We will use a greedy approach to select primes and residues
    # Goal: Maximize L such that 1..L are covered
    # Constraint: Sum(1/p) <= 1
    
    # To be safe and fast, we limit the search space for L to 2000
    # The score is capped at 10^5, but 999 is the target.
    # Using a larger limit allows us to see if we can go beyond 999.
    LIMIT = 2000
    
    # State
    used_primes = set()
    residues = {}
    total_cost = 0.0
    covered = [False] * (LIMIT + 1)  # 1-indexed
    current_max = 0
    
    # Pre-compute 1/p for speed
    inv_primes = {p: 1.0 / p for p in primes}
    
    # Greedy Selection
    while current_max < LIMIT:
        if time.time() > deadline:
            break
            
        next_n = current_max + 1
        best_p = None
        best_r = None
        best_new_max = current_max
        
        # Try all available primes
        for p in primes:
            if p in used_primes:
                continue
            if total_cost + inv_primes[p] > 1.0:
                continue
                
            # Residue must cover next_n
            r = next_n % p
            
            # Calculate new_max if we add (p, r)
            # We need to check coverage from next_n upwards
            # Optimization: We can just check up to a certain point
            # Since we add one prime, the new coverage is the union.
            # We can simulate by checking each number.
            
            # To speed up, we can check how far the coverage extends
            # given the current covered array and the new prime.
            
            temp_max = current_max
            # We know next_n is covered by the new prime.
            # Check subsequent numbers
            # We can stop if we exceed LIMIT
            
            # Efficient check:
            # A number x is covered if covered[x] is True OR (x % p == r)
            # We iterate x from next_n upwards.
            
            # To avoid O(LIMIT) for each prime, we can break early if we find a gap.
            # But we need the max contiguous run.
            
            # Let's just do a simple loop, it's fast enough for LIMIT=2000 and ~150 primes
            # 150 * 2000 = 300,000 ops per iteration.
            # Max iterations ~ 150.
            # Total 45,000,000 ops. Might be slow in Python (10-20s).
            # Let's optimize.
            
            # We can maintain a list of gaps? No, just check.
            
            # Optimization: Only check up to current_max + p (since prime p covers 1 in p)
            # Actually, it can extend further if other primes fill gaps.
            
            # Let's try a heuristic:
            # Check coverage for x in range(next_n, min(LIMIT, next_n + 50))
            # If we hit a gap, stop.
            # If we pass 50, assume it's good? No, we need exact max.
            
            # Let's stick to the loop but break early if we find a gap.
            
            x = next_n
            while x <= LIMIT:
                if covered[x] or (x % p == r):
                    x += 1
                else:
                    break
            
            new_max = x - 1
            
            if new_max > best_new_max:
                best_new_max = new_max
                best_p = p
                best_r = r
                
        if best_p is None:
            break
            
        # Commit the choice
        used_primes.add(best_p)
        residues[best_p] = best_r
        total_cost += inv_primes[best_p]
        
        # Update covered array
        for x in range(best_r, LIMIT + 1, best_p):
            if x > 0:
                covered[x] = True
                
        current_max = best_new_max
        
    # Local Search / Refinement (Optional if time permits)
    # Try to swap primes or adjust residues?
    # Given the deadline, we might skip complex local search.
    # But we can try to "fill" the next gap with a different prime if we missed it?
    # The greedy already does this.
    
    # Construct result
    result = [[p, residues[p]] for p in sorted(residues.keys())]
    
    return result
