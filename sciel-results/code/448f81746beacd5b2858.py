"""Rewritten program: Greedy Repair and Local Search for residue optimization."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    # Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    L_max = 5000
    
    # Helper to calculate coverage
    def calc_coverage(residues, primes, L_max):
        coverage = [0] * (L_max + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            x = r if r > 0 else p
            while x <= L_max:
                coverage[x] += 1
                x += p
        return coverage

    def find_L(coverage, L_max):
        for x in range(1, L_max + 1):
            if coverage[x] < 2:
                return x - 1
        return L_max

    # Phase 1: Greedy Initialization
    # We want to cover [1, L_max] densely.
    # Strategy: Iterate x from 1 to L_max. If cov[x] < 2, find a prime p 
    # such that changing r_p to x % p doesn't break previous coverage.
    
    residues = [0] * n_primes
    coverage = [0] * (L_max + 1)
    
    # Initial setup: r_p = 0
    for i, p in enumerate(primes):
        r = residues[i]
        x = r if r > 0 else p
        while x <= L_max:
            coverage[x] += 1
            x += p
            
    # Greedy Repair
    # We iterate through positions 1..L_max.
    # If a position is deficient, we try to fix it by adjusting a prime's residue.
    # To keep it fast, we only look at a subset of primes or use a heuristic.
    
    # A better initialization: 
    # Assign residues to cover the first K integers as much as possible.
    # Let's try to cover 1..200 with >= 2.
    
    # Reset
    residues = [0] * n_primes
    coverage = [0] * (L_max + 1)
    
    # For each prime, pick a residue that maximizes coverage in [1, 500]
    # This is the same as the old greedy, but let's make it smarter.
    # We'll use a "sweep" greedy.
    
    # Actually, let's start with a random-ish but structured init:
    # r_p = (p * 7) % p is 0. 
    # Let's use r_p = 0 for all, then repair.
    
    for i, p in enumerate(primes):
        # Try to find a residue that covers the most "deficient" spots in [1, 500]
        best_r = 0
        best_score = -1
        # Sample a few residues to save time? No, p <= 1000, 168 primes. 168k ops is fine.
        for r in range(p):
            score = 0
            x = r if r > 0 else p
            while x <= 500:
                if coverage[x] < 2:
                    score += 1
                x += p
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        # Update coverage
        x = best_r if best_r > 0 else p
        while x <= L_max:
            coverage[x] += 1
            x += p

    current_L = find_L(coverage, L_max)
    best_L = current_L
    best_residues = residues[:]
    
    # Phase 2: Local Search / Repair
    # Focus on extending L.
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < 110:
        if no_improve_count > 5000:
            break
            
        # Find the bottleneck
        # We look for the first x where coverage[x] < 2
        # We limit the search to current_L + 100 to keep it local
        
        search_limit = min(current_L + 100, L_max)
        bottleneck = None
        for x in range(current_L + 1, search_limit + 1):
            if coverage[x] < 2:
                bottleneck = x
                break
                
        if bottleneck is None:
            # We covered up to search_limit
            current_L = search_limit
            best_L = current_L
            best_residues = residues[:]
            no_improve_count = 0
            continue
            
        # We need to fix 'bottleneck'
        # Target: Increase coverage at 'bottleneck'
        
        # Strategy: Find a prime p that can cover 'bottleneck'
        # and whose change doesn't break [1, current_L]
        
        # Sort primes by size? 
        # Large primes are "sparse", so changing them is safer for the prefix [1, current_L]
        # because they have fewer hits in that range.
        # Small primes are "dense", so changing them is riskier but has higher impact.
        
        # Let's try primes in descending order of p (largest first)
        # This minimizes the risk of breaking the prefix.
        
        indices = list(range(n_primes))
        # Sort by prime value descending
        indices.sort(key=lambda i: primes[i], reverse=True)
        
        fixed = False
        for i in indices:
            p = primes[i]
            r_old = residues[i]
            r_new = bottleneck % p
            
            if r_old == r_new:
                continue
                
            # Check if changing r_old to r_new helps
            # 1. Does it cover bottleneck? Yes, by construction.
            # 2. Does it break [1, current_L]?
            
            # Calculate delta coverage for [1, current_L + 1]
            # We need to know if any x in [1, current_L] drops below 2.
            
            # Positions affected:
            # Old positions: x such that x % p == r_old
            # New positions: x such that x % p == r_new
            
            # Let's check the "loss" positions (old) and "gain" positions (new)
            # Loss: x in [1, current_L] where x % p == r_old
            #   If coverage[x] == 2, and we remove this prime, it becomes 1. BAD.
            #   If coverage[x] > 2, it becomes coverage[x]-1. OK.
            
            # Gain: x in [1, current_L] where x % p == r_new
            #   Increases coverage. OK.
            
            # Also, we care about 'bottleneck' (which is > current_L)
            # It will gain +1.
            
            # Let's iterate through the affected positions in [1, current_L + 1]
            
            valid = True
            
            # Check old positions in [1, current_L]
            x = r_old
            if x == 0: x = p
            # We need to check all x in [1, current_L]
            # Step by p
            while x <= current_L:
                if coverage[x] == 2:
                    # If we remove this prime, it becomes 1. Invalid.
                    valid = False
                    break
                x += p
                
            if not valid:
                continue
                
            # If valid, we can make the move
            # Update coverage
            # Remove old
            x = r_old
            if x == 0: x = p
            while x <= L_max:
                coverage[x] -= 1
                x += p
                
            # Add new
            x = r_new
            if x == 0: x = p
            while x <= L_max:
                coverage[x] += 1
                x += p
                
            residues[i] = r_new
            fixed = True
            no_improve_count = 0
            
            # Update current_L
            # It might have increased
            new_L = find_L(coverage, L_max)
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
            current_L = new_L
            break # Restart the bottleneck search with the new state
            
        if not fixed:
            # No single prime change could fix the bottleneck without breaking the prefix
            # Try a random perturbation to escape local minimum
            # Pick a random prime and change its residue to a random value
            # This is a "kicks" strategy
            
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            r_new = random.randint(0, p - 1)
            
            if r_old != r_new:
                # Update coverage
                x = r_old
                if x == 0: x = p
                while x <= L_max:
                    coverage[x] -= 1
                    x += p
                    
                x = r_new
                if x == 0: x = p
                while x <= L_max:
                    coverage[x] += 1
                    x += p
                    
                residues[i] = r_new
                
            no_improve_count += 1
            # Recalculate current_L after kick
            current_L = find_L(coverage, L_max)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
                
    return best_residues
# EVOLVE-BLOCK-END
