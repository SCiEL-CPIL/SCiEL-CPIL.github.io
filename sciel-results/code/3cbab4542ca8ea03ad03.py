"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _find_max_classes(max_mod=128, deadline=None):
    """
    Multi-start randomized greedy search for maximum pairwise-disjoint residue classes.
    
    Strategy:
    1. Generate candidate moduli (2..max_mod).
    2. Try many random orderings of these moduli.
    3. For each ordering, greedily add a modulus if there exists a residue that is 
       compatible with all currently selected classes.
    4. Keep the best result found within the time limit.
    """
    if deadline is None:
        deadline = time.time() + 80  # safety margin
    
    best_set = []
    best_size = 0
    
    # Pre-compute gcds for all pairs of moduli up to max_mod
    # gcd_table[m1][m2] = gcd(m1, m2)
    # This is a 129x129 matrix, very fast to build and access.
    gcd_table = [[0]*(max_mod+1) for _ in range(max_mod+1)]
    for m1 in range(2, max_mod+1):
        for m2 in range(2, max_mod+1):
            gcd_table[m1][m2] = gcd(m1, m2)
            
    # Candidate moduli
    all_moduli = list(range(2, max_mod+1))
    
    # Try multiple random orderings
    # We can also try some heuristic orderings (e.g., sorted by number of divisors, etc.)
    # but random is often effective for this type of CSP.
    
    # To speed up the inner loop, we can pre-check compatibility.
    # For a new modulus m_new and current set S, we need to find r_new such that
    # for all (r_old, m_old) in S: (r_new - r_old) % gcd(m_new, m_old) != 0.
    
    # This is equivalent to: r_new must avoid certain residue classes modulo g = gcd(m_new, m_old).
    # Specifically, r_new != r_old (mod g).
    
    # We can use a bitset or a simple check. Since m_new <= 128, we can just iterate r_new in 0..m_new-1.
    
    # Optimization: If we sort moduli, maybe smaller moduli are easier to place? Or harder?
    # Let's just do random restarts.
    
    start_time = time.time()
    max_iters = 10000 # Cap iterations to prevent infinite loop if time check is slow
    
    # Heuristic: Start with a few promising orderings
    # 1. Random
    # 2. Sorted by modulus
    # 3. Sorted by number of prime factors (Omega function) - maybe?
    # Let's stick to random for robustness, but maybe bias towards "composite" numbers?
    # Actually, numbers with more factors have more constraints (larger gcds with others? No, gcd depends on shared factors).
    # Numbers with many small prime factors are likely to share factors with many others.
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        # Shuffle moduli
        moduli = all_moduli[:]
        random.shuffle(moduli)
        
        current_set = []
        
        for m in moduli:
            # Check if m is coprime with any in current_set? 
            # If gcd(m, m_old) == 1, they are always disjoint? 
            # No! If gcd=1, the classes are always disjoint ONLY IF the residues are compatible?
            # Wait. Two residue classes r1 mod m1 and r2 mod m2 intersect iff r1 ≡ r2 (mod gcd(m1, m2)).
            # If gcd(m1, m2) = 1, then r1 ≡ r2 (mod 1) is ALWAYS true.
            # So if gcd(m1, m2) = 1, the classes ALWAYS intersect.
            # Therefore, we CANNOT have two coprime moduli in the set.
            # This is a huge pruning rule!
            
            # Check coprime constraint first
            compatible_moduli = True
            for r_old, m_old in current_set:
                if gcd_table[m][m_old] == 1:
                    compatible_moduli = False
                    break
            
            if not compatible_moduli:
                continue
            
            # Now find a residue r for m that works with all current_set
            # We need r such that for all (r_old, m_old) in current_set:
            # (r - r_old) % gcd(m, m_old) != 0
            
            # Let's try all residues 0..m-1
            found_r = -1
            for r in range(m):
                ok = True
                for r_old, m_old in current_set:
                    g = gcd_table[m][m_old]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            
            if found_r != -1:
                current_set.append([found_r, m])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = [list(c) for c in current_set]
            
            # If we found a very large set, we can stop early? 
            # Theoretical max? 
            # If we pick all multiples of 2, there are 63 such moduli.
            # But they must be pairwise non-coprime. Multiples of 2 are pairwise non-coprime.
            # Can we assign residues to all 63 even moduli?
            # If yes, the max is at least 63.
            # The seed was 7. 63 is a big jump.
            # Let's see if the greedy can find it.
            
            # If best_size > 50, we are doing well.
            # If we reach a limit, maybe break? No, let's keep going until time runs out.
    
    return best_set

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
