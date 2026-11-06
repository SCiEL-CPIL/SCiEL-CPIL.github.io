"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

MAX_MOD = 128
DEADLINE_MARGIN = 5.0  # seconds before the 90s kill

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _valid(pairs):
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if not _disjoint(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]):
                return False
    return True

def _greedy_build(moduli_order, time_limit, randomize_residues=False):
    """Given an ordered list of moduli, greedily assign residues."""
    pairs = []
    used_mods = set()
    for m in moduli_order:
        if m in used_mods:
            continue
        if time.time() > time_limit:
            break
        best_r = None
        # Try residues 0..m-1
        # If randomize, shuffle the order of residues
        residues = list(range(m))
        if randomize_residues:
            random.shuffle(residues)
        
        for r in residues:
            ok = True
            for (r2, m2) in pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                best_r = r
                break
        
        if best_r is not None:
            pairs.append((best_r, m))
            used_mods.add(m)
    return pairs

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN  # stay well under 90s

    all_mods = list(range(2, MAX_MOD + 1))
    
    # Pre-compute GCD matrix for speed? 
    # gcd[i][j] for i,j in 2..128
    # This might be overkill, math.gcd is fast.
    
    best = []
    best_len = 0

    # Strategy 1: Greedy with different orderings
    # Small moduli first (Most Constrained Variable First)
    orderings = [
        all_mods, # Ascending
        list(reversed(all_mods)), # Descending
        sorted(all_mods, key=lambda m: -m.bit_count()), # Many factors first?
        sorted(all_mods, key=lambda m: m.bit_count()), # Few factors first?
    ]
    
    # Add some random shuffles
    for _ in range(10):
        shuffled = all_mods[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for order in orderings:
        if time.time() > deadline:
            break
        # Try deterministic and randomized residue selection
        for rand_res in [False, True]:
            if time.time() > deadline:
                break
            pairs = _greedy_build(order, deadline, randomize_residues=rand_res)
            if len(pairs) > best_len:
                best = pairs
                best_len = len(pairs)
                
    # Strategy 2: Local Search / Iterated Local Search
    # Start from best found
    current_best = [[r, m] for r, m in best]
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Perturbation: Remove a few random pairs
        perturbed = current_best[:]
        if perturbed:
            # Remove 1 to 3 pairs
            k = min(len(perturbed), random.randint(1, 3))
            indices = random.sample(range(len(perturbed)), k)
            indices.sort(reverse=True)
            for idx in indices:
                perturbed.pop(idx)
        
        # Optimization: Greedily add pairs from remaining moduli
        used_mods = set(m for _, m in perturbed)
        remaining_mods = [m for m in all_mods if m not in used_mods]
        
        # Sort remaining by ascending value (heuristic)
        remaining_mods.sort()
        
        new_pairs = perturbed[:]
        for m in remaining_mods:
            if time.time() > deadline:
                break
            # Try to find a residue for m
            found = False
            # Randomize residue order to escape local optima
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                ok = True
                for (r2, m2) in new_pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    new_pairs.append([r, m])
                    found = True
                    break
        
        if len(new_pairs) > len(current_best):
            current_best = new_pairs
            best_len = len(new_pairs)
            best = [[r, m] for r, m in current_best]
            
    return best
# EVOLVE-BLOCK-END
