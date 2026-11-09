import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd[m1_idx][m2_idx]
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def can_add(classes, m_idx, r):
        """Check if adding (m_idx, r) to classes keeps them disjoint."""
        for mi, ri in classes:
            g = gcd_table[mi][m_idx]
            if g == 1:
                return False # Coprime moduli cannot be disjoint
            if ri % g == r % g:
                return False
        return True

    def score(classes):
        """Sum of 1/m for the classes."""
        total = Fraction(0, 1)
        for m_idx, r in classes:
            total += Fraction(1, moduli[m_idx])
        return total

    def is_disjoint_set(classes):
        """Check if all pairs are disjoint. classes: list of (m_idx, r)"""
        n = len(classes)
        for i in range(n):
            mi, ri = classes[i]
            for j in range(i + 1, n):
                mj, rj = classes[j]
                g = gcd_table[mi][mj]
                if g == 1:
                    return False
                if ri % g == rj % g:
                    return False
        return True

    # Strategy: Focus on clusters of moduli sharing a common prime factor.
    # The best cluster is likely multiples of 2.
    
    best_classes = []
    best_score = Fraction(0, 1)
    
    # Try different primes as the "base" for the cluster
    primes_to_check = [2, 3, 5, 7, 11, 13, 17, 19]
    
    for p in primes_to_check:
        if time.time() > deadline - 10.0:
            break
            
        # Get moduli divisible by p
        cluster_mods = [i for i in range(n_mod) if moduli[i] % p == 0]
        
        if not cluster_mods:
            continue
            
        # Sort by modulus ascending (prefer small m)
        cluster_mods.sort(key=lambda idx: moduli[idx])
        
        # Greedy construction for this cluster
        classes = []
        for m_idx in cluster_mods:
            m = moduli[m_idx]
            # Try residues in a smart order: small residues first?
            # Or random?
            # Try all residues, pick the first valid one
            found = False
            # To improve diversity, we can shuffle the residue order
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                if can_add(classes, m_idx, r):
                    classes.append((m_idx, r))
                    found = True
                    break
            
        cur_score = score(classes)
        if cur_score > best_score:
            best_score = cur_score
            best_classes = list(classes)
            
        # Local Search on this cluster's result
        # We restrict swaps to moduli within the same cluster to maintain the "common prime" property?
        # Actually, swapping with a modulus outside the cluster might break the "all pairs non-coprime" invariant if not careful.
        # But local search can try any swap.
        
        def local_search(classes, max_iter=5000):
            classes = list(classes)
            current_score = score(classes)
            improved = True
            iter_count = 0
            
            while improved and iter_count < max_iter and time.time() < deadline - 5.0:
                improved = False
                iter_count += 1
                
                # Try to add a missing modulus (from the whole range, but likely same cluster)
                # Prioritize small moduli
                # To speed up, only consider moduli that share a factor with at least one existing modulus?
                # Or just iterate all.
                
                # 1. Add move
                for m_idx in range(n_mod):
                    if any(c[0] == m_idx for c in classes):
                        continue
                    # Optimization: If m_idx is coprime with all existing, it can't be added (unless set is empty)
                    # But we check can_add which handles g=1.
                    
                    m = moduli[m_idx]
                    # Try a few random residues to speed up
                    # If greedy failed, random might work
                    for _ in range(5):
                        r = random.randint(0, m - 1)
                        if can_add(classes, m_idx, r):
                            new_classes = classes + [(m_idx, r)]
                            new_score = score(new_classes)
                            if new_score > current_score:
                                classes = new_classes
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                    if time.time() > deadline - 5.0:
                        break

                # 2. Swap move
                # Try to replace a class with a better one (smaller m)
                for i in range(len(classes)):
                    mi, ri = classes[i]
                    m_old = moduli[mi]
                    
                    for m_idx in range(n_mod):
                        if m_idx == mi:
                            continue
                        if any(c[0] == m_idx for c in classes if c[0] != mi):
                            continue
                        m_new = moduli[m_idx]
                        if m_new >= m_old:
                            continue
                        
                        # Try to find a residue for m_new that fits
                        found_swap = False
                        # Try random residues
                        for _ in range(10):
                            r_new = random.randint(0, m_new - 1)
                            # Check if replacing classes[i] with (m_idx, r_new) is valid
                            # We need to check disjointness with all OTHER classes
                            valid = True
                            for j, (mj, rj) in enumerate(classes):
                                if j == i:
                                    continue
                                g = gcd_table[mj][m_idx]
                                if g == 1:
                                    valid = False
                                    break
                                if rj % g == r_new % g:
                                    valid = False
                                    break
                            
                            if valid:
                                test_classes = classes[:i] + [(m_idx, r_new)] + classes[i+1:]
                                new_score = score(test_classes)
                                if new_score > current_score:
                                    classes = test_classes
                                    current_score = new_score
                                    improved = True
                                    found_swap = True
                                    break
                        
                        if found_swap:
                            break
                    if improved:
                        break
                            
            return classes, current_score

        improved_classes, improved_score = local_search(classes)
        if improved_score > best_score:
            best_score = improved_score
            best_classes = improved_classes

    # Final check
    if not best_classes:
        # Fallback: simple greedy on all moduli
        classes = []
        for m_idx in range(n_mod):
            m = moduli[m_idx]
            for r in range(m):
                if can_add(classes, m_idx, r):
                    classes.append((m_idx, r))
                    break
        best_classes = classes
        best_score = score(classes)

    result = [[moduli[m_idx], r] for m_idx, r in best_classes]
    
    # Verify validity
    assert is_disjoint_set(best_classes), "Result is not pairwise disjoint!"
    assert len(set(moduli[m_idx] for m_idx, r in best_classes)) == len(best_classes), "Moduli not distinct!"
    
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of classes: {len(result)}")
    print(f"Score: {sum(1/m for m, r in result)}")
    print(f"Classes: {result[:10]}...")
