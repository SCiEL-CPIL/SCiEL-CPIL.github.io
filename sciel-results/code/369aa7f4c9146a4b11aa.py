"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def construct():
    """
    Return a list of [residue, modulus] pairs with distinct moduli in [2,128],
    canonical residues 0<=r<m, such that every pair of infinite residue classes
    is disjoint (verified by exact gcd criterion). Maximize the number of classes.
    
    Two classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1, m2) does not divide (r1 - r2).
    """
    start_time = time.time()
    deadline = 85  # 5s margin before 90s kill
    
    def check_disjoint(classes):
        """Check if all pairs are disjoint using exact gcd criterion."""
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            for j in range(i + 1, n):
                r2, m2 = classes[j]
                g = math.gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True
    
    # Seed solution: powers of 2 with residue 2^k - 1
    def seed_solution():
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    best = seed_solution()
    best_count = len(best)
    
    # Strategy 1: Try single-prime-power families with optimized residues
    # For prime p, powers p, p^2, ..., p^k <= 128
    # We need to find residues r_i for each p^i such that for all i<j:
    #   gcd(p^i, p^j) = p^i does not divide (r_i - r_j)
    # i.e., r_i != r_j (mod p^i) for all i < j
    
    # For a chain of moduli m_1 | m_2 | ... | m_k (each divides the next),
    # we need r_i != r_j (mod m_i) for all i < j.
    # This is equivalent to: for each i, r_i is determined mod m_i, and 
    # r_j (for j > i) must not be congruent to r_i mod m_i.
    
    # A simple construction: r_i = -1 mod m_i (i.e., r_i = m_i - 1)
    # Check: r_i - r_j = (m_i - 1) - (m_j - 1) = m_i - m_j
    # Since m_i | m_j, m_i - m_j = m_i(1 - m_j/m_i), so m_i divides (r_i - r_j).
    # That means they are NOT disjoint! So r_i = m_i - 1 doesn't work.
    
    # Try r_i = 0 for all: r_i - r_j = 0, m_i divides 0. Not disjoint.
    
    # Try r_i = 1 for all: r_i - r_j = 0. Not disjoint.
    
    # The seed uses r_i = 2^k - 1 where m_i = 2^(k+1). So r_i = m_i/2 - 1.
    # Let's verify the general pattern for prime powers.
    # For m_i = p^i, try r_i = p^(i-1) - 1.
    # r_i - r_j = (p^(i-1) - 1) - (p^(j-1) - 1) = p^(i-1) - p^(j-1) = p^(i-1)(1 - p^(j-i))
    # gcd(m_i, m_j) = p^min(i,j) = p^i (assuming i < j)
    # Does p^i divide p^(i-1)(1 - p^(j-i))?
    # p^(i-1)(1 - p^(j-i)) / p^i = (1 - p^(j-i)) / p
    # 1 - p^(j-i) is not divisible by p (since p^(j-i) ≡ 0 mod p, so 1 - 0 = 1 mod p)
    # So p does not divide (1 - p^(j-i)), hence p^i does not divide the difference.
    # Great! So r_i = p^(i-1) - 1 works for the chain p, p^2, ..., p^k.
    
    # But wait, we need 0 <= r_i < m_i = p^i.
    # r_i = p^(i-1) - 1. For i=1: r_1 = p^0 - 1 = 0. m_1 = p. 0 < p. OK.
    # For i=2: r_2 = p - 1. m_2 = p^2. p-1 < p^2 for p >= 2. OK.
    # For i=3: r_3 = p^2 - 1. m_3 = p^3. p^2-1 < p^3 for p >= 2. OK.
    # In general p^(i-1) - 1 < p^i for p >= 2, i >= 1. Always true.
    
    # So for each prime p, we can get a chain of length = number of powers of p <= 128.
    # p=2: 2,4,8,16,32,64,128 -> 7 classes
    # p=3: 3,9,27,81 -> 4 classes
    # p=5: 5,25 -> 2 classes
    # p=7: 7,49 -> 2 classes
    # p=11: 11 -> 1 class
    # etc.
    
    # But can we combine chains from different primes?
    # If we have m from prime p and n from prime q (p != q), gcd(m,n) = 1.
    # We need 1 ∤ (r - s), which is impossible. So we CANNOT mix different primes.
    
    # Therefore, the maximum is the maximum chain length over all primes.
    # p=2 gives 7. That's the seed.
    
    # Can we do better than 7 by NOT using a single prime chain?
    # What if we use moduli that are NOT all powers of the same prime,
    # but have gcd > 1 for every pair?
    
    # For example, all even numbers. But we need gcd(m_i, m_j) > 1 for all pairs,
    # AND gcd(m_i, m_j) ∤ (r_i - r_j).
    
    # Let's try a greedy/backtracking approach to find a larger set.
    # We'll try to build a set of moduli with assigned residues.
    
    all_moduli = list(range(2, 129))
    
    # Precompute gcd matrix
    n_mods = len(all_moduli)
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            g = math.gcd(all_moduli[i], all_moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Backtracking search with pruning
    # Order moduli by some heuristic (e.g., number of prime factors, or just value)
    # Try to add moduli in order, finding a compatible residue for each.
    
    def try_greedy(order, max_time):
        """Try greedy construction with given modulus order."""
        current = []  # list of [residue, modulus]
        used_mods = set()
        
        for m in order:
            if time.time() > max_time:
                break
            # Find a residue r in [0, m-1] compatible with all current
            found_r = None
            for r in range(m):
                ok = True
                for r_j, m_j in current:
                    g = math.gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            if found_r is not None:
                current.append([found_r, m])
        
        return current
    
    # Try several orderings
    # Ordering 1: by number of divisors (more constrained first)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    # Ordering by increasing number of divisors (more constrained first)
    order1 = sorted(all_moduli, key=lambda x: num_divisors(x))
    result1 = try_greedy(order1, deadline - 2)
    if len(result1) > best_count:
        best = result1
        best_count = len(result1)
    
    if time.time() > deadline - 5:
        return best
    
    # Ordering 2: by value
    order2 = all_moduli[:]
    result2 = try_greedy(order2, deadline - 2)
    if len(result2) > best_count:
        best = result2
        best_count = len(result2)
    
    if time.time() > deadline - 5:
        return best
    
    # Ordering 3: random with fixed seed
    random.seed(42)
    for attempt in range(5):
        if time.time() > deadline - 2:
            break
        order3 = all_moduli[:]
        random.shuffle(order3)
        result3 = try_greedy(order3, deadline - 2)
        if len(result3) > best_count:
            best = result3
            best_count = len(result3)
    
    if time.time() > deadline - 3:
        return best
    
    # Try a more sophisticated backtracking for small subsets
    # Focus on moduli with many common factors
    
    # Group moduli by their prime factorization
    # Try to find a clique where every pair has gcd > 1
    
    # For each prime p, consider moduli divisible by p
    # Within that set, we can potentially build a larger chain
    
    # Let's try: pick a prime p, consider all moduli in [2,128] divisible by p
    # Then within that set, try to build a maximum set with the disjointness constraint
    
    # But this is still complex. Let's try a local search / hill climbing approach.
    
    # Start from best, try to swap out one modulus for another to increase count
    
    def try_improve(current, max_time):
        """Try to improve current solution by adding moduli or swapping."""
        current = [list(x) for x in current]
        used_mods = set(m for _, m in current)
        
        # Try adding each unused modulus
        for m in all_moduli:
            if m in used_mods:
                continue
            if time.time() > max_time:
                break
            found_r = None
            for r in range(m):
                ok = True
                for r_j, m_j in current:
                    g = math.gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            if found_r is not None:
                new_sol = current + [[found_r, m]]
                if check_disjoint(new_sol):
                    return new_sol
        
        # Try swapping: remove one, try to add two
        for i in range(len(current)):
            if time.time() > max_time:
                break
            removed = current.pop(i)
            used_mods.remove(removed[1])
            
            # Try to add two new moduli
            added = 0
            for m in all_moduli:
                if m in used_mods:
                    continue
                if added >= 2:
                    break
                found_r = None
                for r in range(m):
                    ok = True
                    for r_j, m_j in current:
                        g = math.gcd(m, m_j)
                        if (r - r_j) % g == 0:
                            ok = False
                            break
                    if ok:
                        found_r = r
                        break
                if found_r is not None:
                    current.append([found_r, m])
                    used_mods.add(m)
                    added += 1
            
            if added > 1:
                if check_disjoint(current):
                    return current
            else:
                # Restore
                current.insert(i, removed)
                used_mods.add(removed[1])
        
        return current
    
    improved = try_improve(best, deadline - 2)
    if len(improved) > best_count:
        best = improved
        best_count = len(improved)
    
    return best
# EVOLVE-BLOCK-END
