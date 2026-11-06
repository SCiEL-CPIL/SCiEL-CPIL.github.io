"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
import sys

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    Uses a model-and-optimize approach: precompute constraints, use structured search.
    """
    start_time = time.time()
    time_limit = 85.0  # Leave buffer for overhead
    
    MAX_M = 128
    MODULI = list(range(2, MAX_M + 1))
    NUM_MODULI = len(MODULI)
    MODULI_INDEX = {m: i for i, m in enumerate(MODULI)}
    
    # Precompute gcd table as 2D list for fast access
    gcd_table = [[0] * NUM_MODULI for _ in range(NUM_MODULI)]
    for i, m1 in enumerate(MODULI):
        for j, m2 in enumerate(MODULI):
            gcd_table[i][j] = math.gcd(m1, m2)
    
    def is_compatible(r, m_idx, chosen):
        """Check if (r, m) is compatible with all pairs in chosen list [(r2, m2_idx), ...]"""
        m = MODULI[m_idx]
        for r2, m2_idx in chosen:
            g = gcd_table[m_idx][m2_idx]
            if g == 1:
                return False
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_residue(m_idx, chosen):
        """Find a compatible residue for modulus m_idx given current chosen set.
        Returns residue or None if no compatible residue exists."""
        m = MODULI[m_idx]
        # Precompute the set of forbidden residues for each gcd value
        # For each existing (r2, m2), we need r != r2 mod gcd(m, m2)
        # Collect all forbidden residue classes
        forbidden = set()
        for r2, m2_idx in chosen:
            g = gcd_table[m_idx][m2_idx]
            if g == 1:
                return None  # Impossible
            # r must NOT be congruent to r2 mod g
            # So r mod g != r2 mod g
            # Add all residues r in [0,m) where r % g == r2 % g
            target = r2 % g
            for r in range(target, m, g):
                forbidden.add(r)
        # Find first residue not in forbidden
        for r in range(m):
            if r not in forbidden:
                return r
        return None
    
    # Precompute which moduli are gcd-compatible (gcd > 1) with each other
    # compatible_mask[m_idx] = bitmask of moduli that have gcd > 1 with m
    # But we'll use a simpler approach: precompute for each pair
    
    # Build adjacency: m1 and m2 are adjacent if gcd(m1,m2) > 1
    adj = [set() for _ in range(NUM_MODULI)]
    for i in range(NUM_MODULI):
        for j in range(i+1, NUM_MODULI):
            if gcd_table[i][j] > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute prime factors for each modulus
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    prime_factor_sets = [prime_factors(m) for m in MODULI]
    
    # Strategy: focus on even moduli first (they form a natural clique)
    even_indices = [i for i, m in enumerate(MODULI) if m % 2 == 0]
    odd_indices = [i for i, m in enumerate(MODULI) if m % 2 == 1]
    
    def solve_subset(indices, num_random_orderings=200):
        """Try to find a large compatible subset using random orderings."""
        best = []
        for attempt in range(num_random_orderings):
            if time.time() - start_time > time_limit:
                break
            order = list(indices)
            random.shuffle(order)
            chosen = []
            for m_idx in order:
                r = find_residue(m_idx, chosen)
                if r is not None:
                    chosen.append((r, m_idx))
            if len(chosen) > len(best):
                best = list(chosen)
        return best
    
    # Start with even moduli - they form a natural clique
    best_solution = solve_subset(even_indices, num_random_orderings=500)
    best_size = len(best_solution)
    used_moduli = {m_idx for r, m_idx in best_solution}
    
    # Now try to add odd moduli to the best solution
    for m_idx in odd_indices:
        if m_idx in used_moduli:
            continue
        # Check if m can be added: it needs gcd(m, m2) > 1 for all m2 in used_moduli
        possible = True
        for r2, m2_idx in best_solution:
            if gcd_table[m_idx][m2_idx] == 1:
                possible = False
                break
        if not possible:
            continue
        r = find_residue(m_idx, best_solution)
        if r is not None:
            best_solution.append((r, m_idx))
            used_moduli.add(m_idx)
            best_size = len(best_solution)
    
    # Local search: try to improve by swapping or adding
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Phase 1: Try to add any unused modulus
        added = False
        unused = [m_idx for m_idx in range(NUM_MODULI) if m_idx not in used_moduli]
        random.shuffle(unused)
        for m_idx in unused:
            # Check if m is compatible with all used moduli (gcd > 1 requirement)
            possible = True
            for r2, m2_idx in best_solution:
                if gcd_table[m_idx][m2_idx] == 1:
                    possible = False
                    break
            if not possible:
                continue
            r = find_residue(m_idx, best_solution)
            if r is not None:
                best_solution.append((r, m_idx))
                used_moduli.add(m_idx)
                added = True
                break
        
        if added:
            best_size = len(best_solution)
            continue
        
        # Phase 2: Try to swap - remove one and add one or two
        if len(best_solution) >= 2:
            # Remove a random element
            idx = random.randint(0, len(best_solution) - 1)
            removed = best_solution.pop(idx)
            used_moduli.discard(removed[1])
            
            # Try to add back a different modulus
            unused = [m_idx for m_idx in range(NUM_MODULI) if m_idx not in used_moduli]
            random.shuffle(unused)
            added_back = False
            for m_idx in unused:
                possible = True
                for r2, m2_idx in best_solution:
                    if gcd_table[m_idx][m2_idx] == 1:
                        possible = False
                        break
                if not possible:
                    continue
                r = find_residue(m_idx, best_solution)
                if r is not None:
                    best_solution.append((r, m_idx))
                    used_moduli.add(m_idx)
                    added_back = True
                    break
            
            if not added_back:
                # Put it back
                best_solution.append(removed)
                used_moduli.add(removed[1])
            else:
                # Try to add one more
                unused = [m_idx for m_idx in range(NUM_MODULI) if m_idx not in used_moduli]
                random.shuffle(unused)
                for m_idx in unused:
                    possible = True
                    for r2, m2_idx in best_solution:
                        if gcd_table[m_idx][m2_idx] == 1:
                            possible = False
                            break
                    if not possible:
                        continue
                    r = find_residue(m_idx, best_solution)
                    if r is not None:
                        best_solution.append((r, m_idx))
                        used_moduli.add(m_idx)
                        break
                    if len(best_solution) > best_size:
                        best_size = len(best_solution)
                    break
    
    # Final phase: try all possible additions exhaustively for remaining time
    while time.time() - start_time < time_limit:
        added = False
        unused = [m_idx for m_idx in range(NUM_MODULI) if m_idx not in used_moduli]
        random.shuffle(unused)
        for m_idx in unused:
            possible = True
            for r2, m2_idx in best_solution:
                if gcd_table[m_idx][m2_idx] == 1:
                    possible = False
                    break
            if not possible:
                continue
            r = find_residue(m_idx, best_solution)
            if r is not None:
                best_solution.append((r, m_idx))
                used_moduli.add(m_idx)
                best_size = len(best_solution)
                added = True
                break
        if not added:
            break
    
    return [[r, MODULI[m_idx]] for r, m_idx in best_solution]

# EVOLVE-BLOCK-END
