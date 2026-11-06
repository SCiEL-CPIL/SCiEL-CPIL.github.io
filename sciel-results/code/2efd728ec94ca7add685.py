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

def _get_valid_residues(m, pairs):
    """Returns list of residues r in [0, m) that are disjoint from all pairs."""
    valid = []
    for r in range(m):
        ok = True
        for (r2, m2) in pairs:
            if not _disjoint(r, m, r2, m2):
                ok = False
                break
        if ok:
            valid.append(r)
    return valid

def _greedy_mcv(all_mods, time_limit):
    """Greedy build using Most Constrained Variable heuristic."""
    pairs = []
    used_mods = set()
    remaining = set(all_mods)
    
    while remaining:
        if time.time() > time_limit:
            break
        
        # Find the modulus with the fewest valid residues
        best_m = None
        min_count = float('inf')
        best_residues = []
        
        # To avoid O(N^2) in the inner loop if N is large, we can just iterate.
        # N is at most 127.
        for m in remaining:
            valid_res = _get_valid_residues(m, pairs)
            if len(valid_res) < min_count:
                min_count = len(valid_res)
                best_m = m
                best_residues = valid_res
                if min_count == 0:
                    # If any modulus has 0 valid residues, we might be stuck?
                    # No, we just skip it for now? 
                    # Actually, if min_count is 0, no other modulus can have < 0.
                    # But we should check if ALL are 0.
                    # If best_m has 0, we can't add it.
                    # We should continue to see if others have > 0?
                    # No, if one has 0, it's the minimum. 
                    # But we want to add a pair. If best_m has 0, we can't add it.
                    # We should look for the one with minimum > 0?
                    # Or if all are 0, we stop.
                    pass
        
        # If the best_m has 0 valid residues, we need to check if any other has > 0.
        # The above loop finds the global minimum. If global min is 0, we can't add best_m.
        # We should find the minimum among those with count > 0.
        
        best_m = None
        min_count = float('inf')
        best_residues = []
        
        for m in remaining:
            valid_res = _get_valid_residues(m, pairs)
            cnt = len(valid_res)
            if 0 < cnt < min_count:
                min_count = cnt
                best_m = m
                best_residues = valid_res
        
        if best_m is None:
            break
            
        # Pick a residue. Random selection to diversify?
        # Or pick the one that is 'most central'? 
        # Let's pick random to allow different paths.
        r = random.choice(best_residues)
        pairs.append((r, best_m))
        used_mods.add(best_m)
        remaining.remove(best_m)
        
    return pairs

def _repair(pairs, all_mods, time_limit):
    """Add pairs to the existing set using MCV."""
    current_pairs = list(pairs)
    used_mods = set(m for _, m in current_pairs)
    remaining = [m for m in all_mods if m not in used_mods]
    
    while remaining:
        if time.time() > time_limit:
            break
        
        best_m = None
        min_count = float('inf')
        best_residues = []
        
        for m in remaining:
            valid_res = _get_valid_residues(m, current_pairs)
            cnt = len(valid_res)
            if 0 < cnt < min_count:
                min_count = cnt
                best_m = m
                best_residues = valid_res
        
        if best_m is None:
            break
            
        r = random.choice(best_residues)
        current_pairs.append((r, best_m))
        remaining.remove(best_m)
        
    return current_pairs

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN  # stay well under 90s

    all_mods = list(range(2, MAX_MOD + 1))
    
    best = []
    best_len = 0
    
    # Strategy 1: Greedy MCV with multiple restarts
    for i in range(20):
        if time.time() > deadline:
            break
        pairs = _greedy_mcv(all_mods, deadline)
        if len(pairs) > best_len:
            best = pairs
            best_len = len(pairs)
            
    # Strategy 2: Iterated Local Search
    current_best = [[r, m] for r, m in best]
    
    # Run ILS until time runs out
    while time.time() < deadline:
        # Perturbation: Remove 1 to 5 random pairs
        perturbed = current_best[:]
        if perturbed:
            k = min(len(perturbed), random.randint(1, 5))
            indices = random.sample(range(len(perturbed)), k)
            indices.sort(reverse=True)
            for idx in indices:
                perturbed.pop(idx)
        
        # Repair
        new_pairs = _repair(perturbed, all_mods, deadline)
        
        if len(new_pairs) > len(current_best):
            current_best = new_pairs
            best_len = len(new_pairs)
            best = [[r, m] for r, m in current_best]
        elif len(new_pairs) == len(current_best) and random.random() < 0.1:
            # Accept equal with small probability to escape plateaus?
            # Or just keep best.
            pass
            
    return best
# EVOLVE-BLOCK-END
