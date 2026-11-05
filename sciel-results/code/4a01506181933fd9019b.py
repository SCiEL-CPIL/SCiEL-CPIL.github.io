"""Rewritten program with a stronger algorithm for maximizing disjoint residue classes."""
import time
import math
import random
from math import gcd

def construct():
    """
    Return a list of [residue, modulus] pairs with distinct moduli in [2,128],
    canonical residues 0<=r<m, such that every pair of infinite residue classes
    is disjoint (verified by exact gcd criterion). Maximize the number of classes.
    
    Disjointness criterion: classes (r1,m1) and (r2,m2) are disjoint iff
    gcd(m1,m2) does NOT divide (r1-r2).
    """
    start_time = time.time()
    deadline = 85  # 5s margin before 90s kill
    
    def check_pair(r1, m1, r2, m2):
        """Check if two classes are disjoint."""
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def check_all(classes):
        """Check if all pairs in the list are disjoint."""
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            for j in range(i+1, n):
                r2, m2 = classes[j]
                if not check_pair(r1, m1, r2, m2):
                    return False
        return True
    
    # Seed solution: powers of 2 with residue = 2^k - 1
    def seed_solution():
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    best = seed_solution()
    best_count = len(best)
    
    # Precompute gcd table for all moduli pairs
    all_moduli = list(range(2, 129))
    n_mods = len(all_moduli)
    gcd_table = [[0]*n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i+1, n_mods):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def mod_idx(m):
        return m - 2  # since moduli start at 2
    
    # Strategy 1: Try to extend the seed by adding more moduli
    # For the seed, all moduli are powers of 2.
    # We can try to add odd moduli or other even moduli.
    
    # Strategy 2: Greedy construction with backtracking
    # Order moduli by some heuristic and try to assign residues
    
    def greedy_construct(mod_order, time_limit):
        """Greedy: for each modulus in order, find a residue compatible with all existing."""
        current = []
        used_mods = set()
        
        for m in mod_order:
            if time.time() > time_limit:
                break
            if m in used_mods:
                continue
            
            # Try all residues for this modulus
            found = False
            for r in range(m):
                ok = True
                for r_j, m_j in current:
                    g = gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    current.append([r, m])
                    used_mods.add(m)
                    found = True
                    break
            
            if not found:
                # Try different residues with some randomization
                for _ in range(20):
                    r = random.randint(0, m-1)
                    ok = True
                    for r_j, m_j in current:
                        g = gcd(m, m_j)
                        if (r - r_j) % g == 0:
                            ok = False
                            break
                    if ok:
                        current.append([r, m])
                        used_mods.add(m)
                        break
        
        return current
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        random.seed(trial * 12345 + 678)
        mod_order = all_moduli[:]
        random.shuffle(mod_order)
        
        # Bias: put smaller moduli first (they have fewer residues to try)
        # Or put moduli with many factors first (more constraints)
        result = greedy_construct(mod_order, deadline - 5)
        
        if len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Strategy 3: Local search - start from a good solution and try to improve
    # Try to swap residues or add/remove moduli
    
    def local_search(initial, time_limit):
        """Try to improve the solution by local moves."""
        current = [list(x) for x in initial]
        current_count = len(current)
        used_mods = set(m for _, m in current)
        available_mods = [m for m in all_moduli if m not in used_mods]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add a new modulus
            for m in available_mods[:]:
                if time.time() > time_limit:
                    break
                
                # Find a residue for m that works with all current
                for r in range(m):
                    ok = True
                    for r_j, m_j in current:
                        g = gcd(m, m_j)
                        if (r - r_j) % g == 0:
                            ok = False
                            break
                    if ok:
                        current.append([r, m])
                        available_mods.remove(m)
                        improved = True
                        break
            
            # Try to remove a modulus and re-add with different residue
            # (This is complex, skip for now)
        
        return current
    
    # Apply local search to best found so far
    if time.time() < deadline - 10:
        improved = local_search(best, deadline - 10)
        if len(improved) > best_count:
            best = improved
            best_count = len(improved)
    
    # Strategy 4: Try specific structured solutions
    # All even moduli with carefully chosen residues
    
    # For moduli that are all divisible by 2, we need gcd(m_i, m_j) | (r_i - r_j) to be false
    # If all moduli are even, gcd >= 2 for all pairs
    
    # Try: use all even moduli, assign residues based on a pattern
    even_moduli = [m for m in all_moduli if m % 2 == 0]
    
    # Try residue = m//2 for all even moduli? No, that might not work.
    # Try residue = 1 for all? Check: gcd(m1,m2) must not divide (1-1)=0. 
    # gcd always divides 0, so this fails.
    
    # Try: for each even modulus m, use residue r = m/2 - 1 (if m/2 is even) or similar
    # This is getting complex. Let's try a different structured approach.
    
    # Use moduli that are multiples of a fixed number, say 6
    # Then gcd >= 6 for pairs that share factor 6
    
    # Actually, let's try: all moduli divisible by 2, and use residues that are all odd
    # Then r_i - r_j is even, and gcd >= 2, so we need gcd to NOT divide the even difference
    # This requires gcd > |r_i - r_j| or gcd to have odd factors
    
    # Let's try a direct approach: for each subset of moduli, check if a valid assignment exists
    # This is NP-hard in general, but for small sets we can try
    
    # Strategy 5: Integer programming style - try to maximize using backtracking with pruning
    # Sort moduli by number of divisors (more constraints first)
    
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                count += 2
                if i*i != n:
                    count -= 1
        return count
    
    # Sort by number of divisors descending (more constrained first)
    sorted_mods = sorted(all_moduli, key=num_divisors, reverse=True)
    
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        random.seed(trial * 9999 + 777)
        # Perturb the ordering
        mod_order = sorted_mods[:]
        # Shuffle within groups of similar constraint
        for i in range(0, len(mod_order)-1, 5):
            chunk = mod_order[i:i+5]
            random.shuffle(chunk)
            mod_order[i:i+5] = chunk
        
        result = greedy_construct(mod_order, deadline - 5)
        if len(result) > best_count:
            best = result
            best_count = len(result)
    
    return best
