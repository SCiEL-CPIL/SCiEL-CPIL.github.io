import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    prime_set = set(primes)

    # Initial set: primes >= 13
    initial_primes = [p for p in primes if p >= 13]
    
    def sum_reciprocals(prime_list):
        return sum(1.0 / p for p in prime_list)

    # Greedy check if 1..L can be covered by given primes
    def can_cover(L, prime_list):
        # Sort primes by size (smaller first) to prefer high density
        # But for greedy selection, we evaluate on the fly
        used_primes = set()
        residues = {}
        covered = [False] * (L + 1)
        
        # Pre-compute coverage for a prime p with residue r
        # We just iterate x from 1 to L
        
        for x in range(1, L + 1):
            if covered[x]:
                continue
            
            best_p = None
            best_gain = -1
            best_r = None
            
            for p in prime_list:
                if p in used_primes:
                    # Check if it already covers x
                    if residues[p] == x % p:
                        # It covers x, but we are here because covered[x] is False?
                        # No, if it covers x, covered[x] would be True.
                        # So this branch is never taken for uncovered x.
                        pass
                    continue
                
                # Calculate gain if we assign r = x % p
                r = x % p
                gain = 0
                # Count how many y in x, x+p, x+2p... <= L are currently uncovered
                # y = x + k*p
                # We only care about y <= L
                for k in range(0, (L - x) // p + 1):
                    y = x + k * p
                    if not covered[y]:
                        gain += 1
                    # Optimization: if gain is already high, we might break?
                    # No, we need exact gain to compare.
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            
            if best_p is None or best_gain <= 0:
                return False
            
            # Assign
            used_primes.add(best_p)
            residues[best_p] = best_r
            
            # Update covered
            for k in range(0, (L - x) // best_p + 1):
                y = x + k * best_p
                covered[y] = True
                
        return True

    # Start with seed configuration
    current_primes = initial_primes[:]
    current_L = 163
    
    # Verify seed
    if not can_cover(current_L, current_primes):
        # Fallback to seed if something went wrong
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]

    # Try to extend L
    # List of candidate primes to add (small ones first)
    candidates_to_add = [p for p in primes if p < 13]
    candidates_to_add.sort(reverse=True) # Try 11, 7, 5, 3, 2
    
    # List of primes to remove (largest first)
    # We will manage the set dynamically
    
    while time.time() < deadline:
        next_L = current_L + 1
        
        # Try current set
        if can_cover(next_L, current_primes):
            current_L = next_L
            continue
        
        # Try to modify the set
        success = False
        for p_add in candidates_to_add:
            if p_add in current_primes:
                continue
            
            # Calculate new set
            # We need to add p_add and remove enough large primes to keep sum <= 1
            # Current sum
            s_curr = sum_reciprocals(current_primes)
            s_add = 1.0 / p_add
            s_target = s_curr + s_add
            
            # We need to remove primes such that sum <= 1
            # Remove largest primes first (they contribute least to coverage density for 1..L)
            # But wait, for large L, large primes cover 1 integer. Small primes cover many.
            # So removing large primes is "cheaper" in terms of lost coverage potential.
            
            # Create a copy to modify
            temp_primes = current_primes[:]
            temp_primes.append(p_add)
            
            # Sort by prime value descending to remove largest
            temp_primes.sort(reverse=True)
            
            # Remove from end (largest) until sum <= 1
            s_temp = s_target
            removed = []
            for i in range(len(temp_primes) - 1, -1, -1):
                if s_temp <= 1.0:
                    break
                p_remove = temp_primes[i]
                # Don't remove the one we just added?
                # If p_add is large, it might be removed. But p_add is small.
                # We iterate from largest. p_add is small, so it's at the beginning of the sorted list (if sorted desc).
                # Wait, sort(reverse=True) puts largest first.
                # We iterate from index len-1 (smallest) to 0 (largest)?
                # No, we want to remove LARGEST.
                # So we should iterate from index 0.
                
            # Let's restart the removal logic
            temp_primes = current_primes[:]
            temp_primes.append(p_add)
            # Sort descending to pick largest for removal
            temp_primes.sort(reverse=True)
            
            s_temp = sum_reciprocals(temp_primes)
            # Remove largest while sum > 1
            while s_temp > 1.0 and len(temp_primes) > 0:
                p_remove = temp_primes.pop(0) # Largest
                s_temp -= 1.0 / p_remove
            
            if s_temp > 1.0:
                continue
                
            # Check if this new set covers next_L
            if can_cover(next_L, temp_primes):
                current_primes = temp_primes
                current_L = next_L
                success = True
                break
                
        if not success:
            # Try random swaps? Or just stop?
            # If we can't extend with adding small primes, we might be stuck.
            # We can try removing small primes and adding large ones?
            # Unlikely to help coverage of 1..L.
            break

    # Construct result
    # We need to assign residues.
    # The can_cover function finds a valid assignment.
    # We need to return that assignment.
    # Let's re-run a solver that returns the assignment.
    
    def get_assignment(L, prime_list):
        used_primes = set()
        residues = {}
        covered = [False] * (L + 1)
        
        for x in range(1, L + 1):
            if covered[x]:
                continue
            
            best_p = None
            best_gain = -1
            best_r = None
            
            for p in prime_list:
                if p in used_primes:
                    continue
                
                r = x % p
                gain = 0
                for k in range(0, (L - x) // p + 1):
                    y = x + k * p
                    if not covered[y]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            
            if best_p is None or best_gain <= 0:
                return None
            
            used_primes.add(best_p)
            residues[best_p] = best_r
            
            for k in range(0, (L - x) // best_p + 1):
                y = x + k * best_p
                covered[y] = True
                
        return residues

    final_assignment = get_assignment(current_L, current_primes)
    
    if final_assignment is None:
        # Fallback to seed
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]
    
    # Format output
    result = []
    for p in current_primes:
        if p in final_assignment:
            result.append([p, final_assignment[p]])
            
    return result
