"""Rewritten program: randomized greedy maximization of pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = time.time() + 85.0  # 5s safety margin
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    best_count = 0
    best_result = None
    
    # Precompute gcds for speed? Not strictly necessary for 127 items, but good practice.
    # gcd_cache = {}
    # def get_gcd(a, b):
    #    if a > b: a, b = b, a
    #    key = (a, b)
    #    if key not in gcd_cache:
    #        gcd_cache[key] = math.gcd(a, b)
    #    return gcd_cache[key]
    
    # We will try several strategies:
    # 1. Random order of moduli, random order of residues
    # 2. Heuristic order of moduli (e.g., descending, ascending, prime-first, composite-first)
    # 3. Heuristic order of residues (e.g., 0-first, random, coprime-first)
    
    # To maximize chances, we run many iterations of randomized greedy
    # Each iteration:
    # - Shuffle moduli (or use a fixed heuristic)
    # - For each modulus, try residues in a shuffled order (or heuristic)
    # - Add if disjoint from all currently added
    
    num_iterations = 200
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        mo = moduli_list[:]
        random.shuffle(mo)
        
        chosen = []
        for m in mo:
            valid_residue = None
            candidates = list(range(m))
            random.shuffle(candidates)
            
            for r in candidates:
                ok = True
                for (r2, m2) in chosen:
                    if not _is_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    valid_residue = r
                    break
            
            if valid_residue is not None:
                chosen.append((valid_residue, m))
        
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
            
    # Fallback if random search failed or found very few
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
        
    return best_result
# EVOLVE-BLOCK-END
