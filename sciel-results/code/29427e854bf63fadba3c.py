"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    def gcd(a, b):
        return math.gcd(a, b)
    
    def check_pair(r1, m1, r2, m2):
        """Check if two residue classes are disjoint."""
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def check_all(classes):
        """Check if all pairs are disjoint."""
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            for j in range(i + 1, n):
                r2, m2 = classes[j]
                if not check_pair(r1, m1, r2, m2):
                    return False
        return True
    
    def seed_solution():
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    best = seed_solution()
    best_count = len(best)
    
    # Try to build larger sets using structured approaches
    # Key: all moduli must share common prime factors with each other
    # Strategy: pick a prime p, use moduli that are all divisible by p
    
    # For each prime, try to find a large set of moduli all divisible by p
    # with appropriate residues
    
    # Approach 1: Use moduli that are all multiples of a fixed prime p
    # For a fixed p, moduli m1, m2 both divisible by p have gcd >= p
    # We need gcd(m1,m2) not to divide (r1-r2)
    
    # Best strategy: use moduli that are powers of 2 (seed) plus try to add more
    # Actually, let's try a different structured approach
    
    # For moduli all divisible by 2: use residues that differ mod gcd
    # If m1=2*a, m2=2*b, gcd >= 2
    # We need (r1-r2) not divisible by gcd(m1,m2)
    
    # Let's try: for each modulus m in [2,128], assign residue r such that
    # the set is maximally large and all pairs are disjoint
    
    # Greedy approach: sort moduli, add one by one, try all residues
    def greedy_build(moduli_order, time_limit=5.0):
        """Try to build a set by adding moduli in given order."""
        classes = []
        used_moduli = set()
        t0 = time.time()
        
        for m in moduli_order:
            if time.time() - t0 > time_limit:
                break
            if m in used_moduli:
                continue
            found = False
            for r in range(m):
                if all(check_pair(r, m, r2, m2) for r2, m2 in classes):
                    classes.append([r, m])
                    used_moduli.add(m)
                    found = True
                    break
            if not found:
                # Try a few more residues (maybe the first wasn't optimal)
                for r in range(m):
                    if all(check_pair(r, m, r2, m2) for r2, m2 in classes):
                        classes.append([r, m])
                        used_moduli.add(m)
                        found = True
                        break
        return classes
    
    # Try different orderings
    all_moduli = list(range(2, 129))
    
    # Order by descending number of prime factors (more shared factors = easier)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Try ordering by descending prime factor count
    moduli_by_pf = sorted(all_moduli, key=lambda x: -num_prime_factors(x))
    result = greedy_build(moduli_by_pf, time_limit=3.0)
    if len(result) > best_count:
        best = result
        best_count = len(result)
    
    if time.time() > deadline:
        return best
    
    # Try local search: start from seed, try to add more moduli
    def local_search_improve(initial_classes, time_limit=10.0):
        """Improve a solution by trying to add moduli and swap residues."""
        classes = [list(c) for c in initial_classes]
        used_moduli = set(m for _, m in classes)
        t0 = time.time()
        improved = True
        
        while improved and time.time() - t0 < time_limit:
            improved = False
            
            # Try to add each unused modulus
            for m in range(2, 129):
                if time.time() - t0 > time_limit:
                    break
                if m in used_moduli:
                    continue
                for r in range(m):
                    if all(check_pair(r, m, r2, m2) for r2, m2 in classes):
                        classes.append([r, m])
                        used_moduli.add(m)
                        improved = True
                        break
            
            # Try to swap residues to free up space for more moduli
            if time.time() - t0 > time_limit:
                break
            for i in range(len(classes)):
                if time.time() - t0 > time_limit:
                    break
                r_old, m_i = classes[i]
                for r_new in range(m_i):
                    if r_new == r_old:
                        continue
                    classes[i][0] = r_new
                    # Check if this allows adding a new modulus
                    for m in range(2, 129):
                        if m in used_moduli:
                            continue
                        for r in range(m):
                            if all(check_pair(r, m, r2, m2) for r2, m2 in classes):
                                classes.append([r, m])
                                used_moduli.add(m)
                                improved = True
                                break
                        if improved:
                            break
                    if not improved:
                        classes[i][0] = r_old
        
        return classes
    
    improved = local_search_improve(best, time_limit=10.0)
    if len(improved) > best_count:
        best = improved
        best_count = len(improved)
    
    if time.time() > deadline:
        return best
    
    # Random restarts with different initial seeds
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        # Random initial set
        random.seed(restart)
        init_moduli = random.sample(all_moduli, min(10, len(all_moduli)))
        init_classes = []
        for m in init_moduli:
            r = random.randint(0, m - 1)
            if all(check_pair(r, m, r2, m2) for r2, m2 in init_classes):
                init_classes.append([r, m])
        
        # Improve from this start
        improved = local_search_improve(init_classes, time_limit=0.05)
        if len(improved) > best_count:
            best = improved
            best_count = len(improved)
    
    # Final attempt: backtracking search for maximum set
    if time.time() < deadline - 5:
        def backtrack_search(available_moduli, current_classes, depth=0):
            if time.time() > deadline - 2:
                return current_classes
            if len(current_classes) >= best_count + 1:
                # Try to find even larger
                pass
            
            best_result = current_classes[:]
            for i, m in enumerate(available_moduli):
                if time.time() > deadline - 2:
                    break
                found_r = None
                for r in range(m):
                    if all(check_pair(r, m, r2, m2) for r2, m2 in current_classes):
                        found_r = r
                        break
                if found_r is not None:
                    new_classes = current_classes + [[found_r, m]]
                    new_available = available_moduli[:i] + available_moduli[i+1:]
                    result = backtrack_search(new_available, new_classes, depth + 1)
                    if len(result) > len(best_result):
                        best_result = result
                else:
                    # Skip this modulus
                    pass
            
            return best_result
        
        # Use a subset of moduli for backtracking (too many otherwise)
        # Focus on moduli with high prime factor counts
        candidate_moduli = [m for m in all_moduli if num_prime_factors(m) >= 2]
        candidate_moduli.sort(key=lambda x: -num_prime_factors(x))
        candidate_moduli = candidate_moduli[:30]  # Limit for tractability
        
        bt_result = backtrack_search(candidate_moduli, [])
        if len(bt_result) > best_count:
            best = bt_result
            best_count = len(bt_result)
    
    return best
# EVOLVE-BLOCK-END
