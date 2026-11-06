import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    # Sort primes ascending. Smaller primes cover more numbers but cost more budget.
    # We might want to consider larger primes too if budget allows, but usually small primes are key for dense coverage.
    
    LIMIT = 100000
    # We only care about L up to 1000 for the "win", but score is L.
    # However, the task says "capped at 10^5".
    # Let's use a coverage array of size LIMIT + 1.
    
    best_L_found = 0
    best_classes_found = []
    
    # State variables for DFS
    used_primes = set()
    current_classes = []
    remaining_budget = Fraction(1)
    
    # Coverage array
    # Using a list of booleans or bytearray. Bytearray is faster for indexing.
    covered = bytearray(LIMIT + 1)
    
    # Helper to find the first uncovered number starting from 1
    def find_first_uncovered():
        # We can optimize this by keeping track of current_L, 
        # but since we backtrack, we need to be careful.
        # Actually, we always extend from the current L.
        # Let's pass current_L as a parameter to the DFS to avoid scanning from 1.
        pass
        
    def simulate_move(p, r, current_L):
        """
        Returns the new L if we add class (p, r) to the current coverage.
        Assumes [1, current_L] are covered.
        """
        n = current_L + 1
        if n > LIMIT:
            return LIMIT
        
        # Check from n upwards
        # We know covered[1..current_L] is 1.
        # We need to check if k is covered by existing classes OR (k % p == r).
        # The 'covered' array holds the state of existing classes.
        
        k = n
        while k <= LIMIT:
            # If k is already covered by existing classes, it's fine.
            # If not, it must be covered by the new class (p, r).
            if covered[k] == 0:
                if k % p != r:
                    return k - 1
            k += 1
        return LIMIT

    def dfs(current_L, depth):
        nonlocal best_L_found, best_classes_found, used_primes, current_classes, remaining_budget, covered
        
        if time.time() > deadline:
            return
        
        if current_L > best_L_found:
            best_L_found = current_L
            best_classes_found = current_classes.copy()
            
        if current_L >= LIMIT:
            return
            
        if current_L >= 999:
            # We achieved the target. We could try to go higher, but 999 is the "win" condition.
            # Let's keep searching a bit if we have time, but this is a strong signal.
            pass

        n = current_L + 1
        if n > LIMIT:
            return

        # Generate candidates
        candidates = []
        
        # Iterate through unused primes
        # Optimization: We can skip primes that are too small to help if we are stuck?
        # No, small primes are best for filling gaps.
        # We should prioritize primes that extend L the most.
        
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if remaining_budget < cost:
                continue
            
            r = n % p
            
            # Simulate to see how far this extends L
            # This is the most expensive part.
            new_L = simulate_move(p, r, current_L)
            
            # If new_L is not greater than current_L, it's useless (doesn't cover n? No, it covers n by definition of r=n%p).
            # Wait, if covered[n] was 1, then new_L would be > current_L.
            # But n is the FIRST uncovered, so covered[n] is 0.
            # So new_L >= n = current_L + 1.
            
            candidates.append((new_L, p, r))
            
            if time.time() > deadline:
                return

        if not candidates:
            return

        # Sort candidates by new_L descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Limit branching factor to keep search tractable
        # Trying all primes is too slow.
        # Let's try the top 5 or 10 best extending primes.
        max_branches = 10
        
        for new_L, p, r in candidates[:max_branches]:
            if time.time() > deadline:
                return
            
            # Apply move
            used_primes.add(p)
            current_classes.append([p, r])
            remaining_budget -= Fraction(1, p)
            
            # Update covered array
            # We need to mark all numbers congruent to r mod p as covered.
            # start = r if r > 0 else p
            start = r
            if start == 0:
                start = p
            if start <= LIMIT:
                for j in range(start, LIMIT + 1, p):
                    covered[j] = 1
            
            # Recurse
            dfs(new_L, depth + 1)
            
            # Undo move
            used_primes.remove(p)
            current_classes.pop()
            remaining_budget += Fraction(1, p)
            
            # Undo covered array
            for j in range(start, LIMIT + 1, p):
                covered[j] = 0
            
            # If we found a solution with L >= 999, we might want to stop early?
            # No, we want to maximize L. But if we are deep in time, we should stop.
            # The "best" is tracked globally.

    # Initial call
    # We start with L=0.
    # However, a pure DFS from L=0 might be too slow if it explores bad branches.
    # Let's add a greedy initialization to get a good starting point, then DFS from there?
    # Or just run DFS. The "smallest uncovered" heuristic is strong.
    
    # To ensure we have a valid result even if DFS is slow, let's run a quick greedy first.
    # Greedy:
    greedy_classes = []
    greedy_budget = Fraction(1)
    greedy_used = set()
    greedy_covered = bytearray(LIMIT + 1)
    greedy_L = 0
    
    # Simple greedy pass
    while time.time() < deadline - 10: # Leave 10s for DFS
        n = greedy_L + 1
        if n > LIMIT: break
        best_p_g = None
        best_r_g = None
        best_L_g = greedy_L
        
        for p in primes:
            if p in greedy_used: continue
            cost = Fraction(1, p)
            if greedy_budget < cost: continue
            r = n % p
            # Simulate
            k = n
            while k <= LIMIT:
                if greedy_covered[k] == 0:
                    if k % p != r:
                        break
                k += 1
            new_L = k - 1
            if new_L > best_L_g:
                best_L_g = new_L
                best_p_g = p
                best_r_g = r
                if best_L_g >= 999:
                    break
            elif new_L == best_L_g and best_p_g is not None and p > best_p_g:
                best_p_g = p
                best_r_g = r
        
        if best_p_g is None or best_L_g == greedy_L:
            break
            
        greedy_used.add(best_p_g)
        greedy_classes.append([best_p_g, best_r_g])
        greedy_budget -= Fraction(1, best_p_g)
        
        start = best_r_g
        if start == 0: start = best_p_g
        for j in range(start, LIMIT + 1, best_p_g):
            greedy_covered[j] = 1
            
        greedy_L = best_L_g
        
    if greedy_L > best_L_found:
        best_L_found = greedy_L
        best_classes_found = greedy_classes.copy()
        
    if best_L_found >= 999:
        return best_classes_found

    # If greedy didn't reach 999, try DFS starting from the greedy state?
    # Or restart DFS from scratch with the knowledge that we need to be smarter?
    # Restarting DFS from scratch is cleaner but might repeat work.
    # Let's try DFS from scratch but with a tighter time limit if greedy was fast.
    
    # Reset state for DFS
    used_primes = set()
    current_classes = []
    remaining_budget = Fraction(1)
    covered = bytearray(LIMIT + 1)
    
    # If we have significant time left, run DFS
    if time.time() < deadline - 5:
        dfs(0, 0)
        
    return best_classes_found
