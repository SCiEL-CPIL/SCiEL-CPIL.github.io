import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _construct_greedy(moduli_order, time_limit, rng):
    pairs = []
    # pairs is a list of (r, m)
    
    for m in moduli_order:
        if time.time() > time_limit:
            break
        
        valid_residues = []
        for r in range(m):
            ok = True
            for (r2, m2) in pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                valid_residues.append(r)
        
        if valid_residues:
            # Pick a random valid residue
            r = rng.choice(valid_residues)
            pairs.append((r, m))
            
    return pairs

def _local_search_1for1_swap(pairs, all_mods, time_limit, rng):
    """Try 1-for-1 swaps to improve the set size"""
    pairs = list(pairs)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        used_mods = set(m for _, m in pairs)
        missed = [m for m in all_mods if m not in used_mods]
        
        if not missed:
            break
        
        # Try each missed modulus
        for m_new in missed:
            if time.time() > time_limit:
                break
            
            # Try to find a residue for m_new that's disjoint with all existing pairs
            valid_residues = []
            for r in range(m_new):
                ok = True
                for (r2, m2) in pairs:
                    if not _disjoint(r, m_new, r2, m2):
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if valid_residues:
                # We can add this modulus without removing anything
                r = rng.choice(valid_residues)
                pairs.append((r, m_new))
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps: remove one existing pair, add one new pair
        for i, (r_old, m_old) in enumerate(pairs):
            if time.time() > time_limit:
                break
            
            # Try each missed modulus as replacement
            for m_new in missed:
                if m_new == m_old:
                    continue
                
                # Try to find a residue for m_new that's disjoint with all pairs except the one we're removing
                valid_residues = []
                for r in range(m_new):
                    ok = True
                    for j, (r2, m2) in enumerate(pairs):
                        if j == i:
                            continue
                        if not _disjoint(r, m_new, r2, m2):
                            ok = False
                            break
                    if ok:
                        valid_residues.append(r)
                
                if valid_residues:
                    # Perform the swap
                    r = rng.choice(valid_residues)
                    pairs[i] = (r, m_new)
                    improved = True
                    break
            
            if improved:
                break
    
    return pairs

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Precompute GCDs for ordering heuristic
    # score[m] = sum of gcd(m, k) for k in all_mods, k != m
    # Higher score means more constrained (shares more common factors)
    scores = {}
    for m in all_mods:
        s = 0
        for k in all_mods:
            if m != k:
                s += math.gcd(m, k)
        scores[m] = s
        
    # Most constrained first: sort by descending score
    order_most_constrained = sorted(all_mods, key=lambda m: scores[m], reverse=True)
    # Also try ascending (least constrained first)
    order_least_constrained = sorted(all_mods, key=lambda m: scores[m])
    
    best_pairs = []
    best_len = 0
    
    # Run randomized greedy multiple times with local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts)
        
        # Choose an ordering strategy
        # 60% most constrained, 20% least constrained, 20% random shuffle
        r = rng.random()
        if r < 0.6:
            order = order_most_constrained[:]
        elif r < 0.8:
            order = order_least_constrained[:]
        else:
            order = all_mods[:]
            rng.shuffle(order)
            
        # Check if we have time for local search
        local_search_time = deadline - 2.0  # Leave 2s margin
        
        pairs = _construct_greedy(order, local_search_time, rng)
        
        # Apply local search if we have time
        if time.time() < local_search_time - 0.5:
            pairs = _local_search_1for1_swap(pairs, all_mods, local_search_time, rng)
        
        if len(pairs) > best_len:
            best_len = len(pairs)
            best_pairs = pairs
        
        # Early exit if we found a perfect solution
        if best_len == len(all_mods):
            break

    return [[r, m] for r, m in best_pairs]
