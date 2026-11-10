"""Improved program for disjoint residue classes; stronger greedy + local search."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    def check_pair(r1, m1, r2, m2):
        """Check if two residue classes are disjoint."""
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def is_valid(classes):
        """Check if all pairs are disjoint."""
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                if not check_pair(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                    return False
        return True
    
    def find_residue(m, selected):
        """Find a valid residue for modulus m given already selected classes."""
        for r in range(m):
            valid = True
            for rp, mp in selected:
                if not check_pair(r, m, rp, mp):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def greedy(order):
        """Greedy: add moduli in given order, picking first valid residue."""
        selected = []
        for m in order:
            if time.time() - start_time > DEADLINE:
                break
            r = find_residue(m, selected)
            if r is not None:
                selected.append([r, m])
        return selected
    
    def count_prime_factors(n):
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
    
    def local_search(selected, max_time):
        """Improve by reassigning residues and adding unused moduli."""
        if not selected:
            return selected
        current = [list(c) for c in selected]
        used_moduli = set(m for _, m in current)
        unused_moduli = [m for m in all_moduli if m not in used_moduli]
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            
            # Try adding each unused modulus
            for m_new in list(unused_moduli):
                r_new = find_residue(m_new, current)
                if r_new is not None:
                    current.append([r_new, m_new])
                    unused_moduli.remove(m_new)
                    improved = True
                    break
            
            # Try reassigning residues to free space
            if not improved:
                for i in range(len(current)):
                    if time.time() - start_time > max_time:
                        break
                    m_i = current[i][1]
                    old_r = current[i][0]
                    
                    # Try different residues for this modulus
                    for r_try in range(m_i):
                        if r_try == old_r:
                            continue
                        current[i][0] = r_try
                        
                        if is_valid(current):
                            # Try to add an unused modulus
                            for m_new in unused_moduli:
                                r_new = find_residue(m_new, current)
                                if r_new is not None:
                                    current.append([r_new, m_new])
                                    unused_moduli.remove(m_new)
                                    improved = True
                                    break
                            
                            if not improved:
                                current[i][0] = old_r
                        else:
                            current[i][0] = old_r
                    
                    if improved:
                        break
        
        return current
    
    def random_restart_greedy(n_restarts):
        """Try multiple random orderings."""
        best = []
        for _ in range(n_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Create a random ordering with some structure
            order = all_moduli[:]
            
            # Sort by some random heuristic
            strategy = random.randint(0, 3)
            if strategy == 0:
                random.shuffle(order)
            elif strategy == 1:
                # Prefer coprime-friendly moduli
                order.sort(key=lambda m: (count_prime_factors(m), m))
            elif strategy == 2:
                # Prefer larger moduli first
                order.sort(reverse=True)
            else:
                # Mix
                random.shuffle(order)
                order.sort(key=lambda m: (m % 3, m))
            
            result = greedy(order)
            if len(result) > len(best):
                best = result
        
        return best
    
    # Phase 1: Deterministic greedy with different orderings
    best = []
    
    # Ordering 1: ascending
    cand = greedy(all_moduli)
    if len(cand) > len(best):
        best = cand
    
    # Ordering 2: descending
    cand = greedy(list(reversed(all_moduli)))
    if len(cand) > len(best):
        best = cand
    
    # Ordering 3: by number of prime factors
    sorted_by_pf = sorted(all_moduli, key=lambda m: count_prime_factors(m))
    cand = greedy(sorted_by_pf)
    if len(cand) > len(best):
        best = cand
    
    # Ordering 4: powers of 2 first
    powers_of_2 = [2**k for k in range(1, 8) if 2**k <= MAX_MOD]
    other_mods = [m for m in all_moduli if m not in set(powers_of_2)]
    order4 = powers_of_2 + other_mods
    cand = greedy(order4)
    if len(cand) > len(best):
        best = cand
    
    # Phase 2: Random restarts
    if time.time() - start_time < DEADLINE - 10:
        random.seed(42)
        n_restarts = min(200, int((DEADLINE - 10 - (time.time() - start_time)) / 0.05))
        cand = random_restart_greedy(n_restarts)
        if len(cand) > len(best):
            best = cand
    
    # Phase 3: Local search
    if time.time() - start_time < DEADLINE - 5:
        best = local_search(best, DEADLINE - 2)
    
    # Final verification
    if not is_valid(best):
        # Fallback: try a simple valid set
        best = []
        for m in [2, 3, 5, 7, 11, 13]:
            r = find_residue(m, best)
            if r is not None:
                best.append([r, m])
        if not best:
            best = [[1, 2]]
    
    return best
# EVOLVE-BLOCK-END
