import time
import math
import random
from fractions import Fraction

def _compatible(assign, m, r, ignore=None):
    """Check if residue r for modulus m is compatible with all placed (except ignore)."""
    for m2, r2 in assign.items():
        if m2 == ignore:
            continue
        g = math.gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True

def _score(assign):
    return sum(1.0 / m for m in assign)

def _greedy_construct(all_moduli, order_fn=None):
    """Greedy: place moduli in given order, first compatible residue."""
    assign = {}
    for m in all_moduli:
        if order_fn:
            residues = order_fn(m)
        else:
            residues = range(m)
        placed = False
        for r in residues:
            if _compatible(assign, m, r):
                assign[m] = r
                placed = True
                break
        if not placed:
            # Skip this modulus
            pass
    return assign

def _local_search(assign, time_limit=5.0):
    """Local search: try swapping residues, adding missing moduli, removing conflicting ones."""
    deadline = time.time() + time_limit
    current_score = _score(assign)
    assign = dict(assign)
    
    all_moduli = set(range(20, 201))
    current_moduli = set(assign.keys())
    
    while time.time() < deadline:
        improved = False
        
        # Try to add a missing modulus
        missing = list(all_moduli - current_moduli)
        random.shuffle(missing)
        for m in missing:
            if time.time() > deadline:
                break
            # Try all residues
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if _compatible(assign, m, r):
                    assign[m] = r
                    current_moduli.add(m)
                    new_score = _score(assign)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert if it doesn't help (or keep if same?)
                        del assign[m]
                        current_moduli.discard(m)
                    break
        
        # Try swapping residues of existing moduli to allow additions
        if not improved and len(assign) < 180:
            items = list(assign.items())
            for _ in range(100):
                if time.time() > deadline:
                    break
                idx1, idx2 = random.sample(range(len(items)), 2)
                m1, r1 = items[idx1]
                m2, r2 = items[idx2]
                
                # Try swapping
                if r1 % math.gcd(m1, m2) == r2 % math.gcd(m1, m2):
                    continue  # Would conflict after swap? No, we're swapping values.
                
                old_r1, old_r2 = r1, r2
                assign[m1] = r2
                assign[m2] = r1
                
                if _compatible(assign, m1, r2, ignore=m1) and _compatible(assign, m2, r1, ignore=m2):
                    # Check pairwise
                    ok = True
                    for m3, r3 in assign.items():
                        if m3 == m1 or m3 == m2:
                            continue
                        if not _compatible(assign, m1, assign[m1], ignore=m1) or \
                           not _compatible(assign, m2, assign[m2], ignore=m2):
                            ok = False
                            break
                    
                    if ok:
                        new_score = _score(assign)
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            items = list(assign.items())
                        else:
                            assign[m1] = old_r1
                            assign[m2] = old_r2
                    else:
                        assign[m1] = old_r1
                        assign[m2] = old_r2
                else:
                    assign[m1] = old_r1
                    assign[m2] = old_r2
        
        if not improved:
            break
            
    return assign

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_assign = {}
    best_score = 0.0
    
    all_moduli = list(range(20, 201))
    
    # Strategy 1: Greedy with small-first order (small moduli have more constraints? Or less?)
    # Actually, small moduli have fewer residues, so they are harder to place.
    # Large moduli have many residues, so they are easier.
    # Let's try large-first.
    if time.time() < deadline:
        assign = _greedy_construct(sorted(all_moduli, reverse=True))
        score = _score(assign)
        if score > best_score:
            best_score = score
            best_assign = dict(assign)
    
    # Strategy 2: Randomized Greedy with restarts
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        random.shuffle(all_moduli)
        # Bias: smaller moduli are harder, so maybe place them first?
        # Or place them last to have more flexibility?
        # Let's try a mix.
        
        def random_residue_order(m):
            return random.sample(range(m), m)
        
        assign = _greedy_construct(all_moduli, random_residue_order)
        score = _score(assign)
        if score > best_score:
            best_score = score
            best_assign = dict(assign)
    
    # Strategy 3: Local Search from best found so far
    if time.time() < deadline - 10:
        remaining_time = min(20, deadline - time.time() - 5)
        if remaining_time > 1:
            assign = _local_search(best_assign, remaining_time)
            score = _score(assign)
            if score > best_score:
                best_score = score
                best_assign = dict(assign)
    
    # Strategy 4: Try to fill gaps with specific moduli
    if time.time() < deadline - 2:
        current_moduli = set(best_assign.keys())
        missing = list(set(range(20, 201)) - current_moduli)
        random.shuffle(missing)
        for m in missing:
            if time.time() > deadline - 1:
                break
            for r in range(m):
                if _compatible(best_assign, m, r):
                    best_assign[m] = r
                    break
    
    result = [[m, r] for m, r in best_assign.items()]
    return result
