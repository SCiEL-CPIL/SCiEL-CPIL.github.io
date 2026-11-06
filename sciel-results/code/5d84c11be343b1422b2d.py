"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # All candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    n = len(candidates)
    
    # Precompute gcd for all pairs
    gcd_table = {}
    for i, m1 in enumerate(candidates):
        for j, m2 in enumerate(candidates):
            gcd_table[(m1, m2)] = gcd(m1, m2)
    
    # Precompute 1/m as Fraction for each modulus
    inv_mod = {m: Fraction(1, m) for m in candidates}
    
    # Precompute for each modulus m and each possible gcd g:
    # count of residues r in [0, m) with r % g == t for each t in [0, g)
    # This helps in evaluating how "blocking" a residue choice is
    
    def count_residues_mod(m, g, t):
        """Count residues r in [0, m) with r % g == t"""
        if t >= m:
            return 0
        # r = t, t+g, t+2g, ... < m
        # Number of terms: floor((m-1-t)/g) + 1
        if t < m:
            return (m - 1 - t) // g + 1
        return 0
    
    # Precompute blocking cost: for modulus m with residue r, 
    # the total blocking cost against all other moduli m2 is sum over m2 of 1/gcd(m,m2)
    # This is a proxy for how many future options we block
    
    def is_compatible(m_new, r_new, selected):
        for (m_old, r_old) in selected:
            g = gcd_table[(m_new, m_old)]
            if r_new % g == r_old % g:
                return False
        return True
    
    def find_valid_residues(m_new, selected):
        valid = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected):
                valid.append(r)
        return valid
    
    def compute_blocking_cost(m_new, r_new, all_mods, selected_mods_set):
        """Estimate how much this choice blocks future moduli"""
        cost = 0.0
        for m2 in all_mods:
            if m2 in selected_mods_set or m2 == m_new:
                continue
            g = gcd_table[(m_new, m2)]
            # This residue blocks exactly the g-class r_new % g for m2
            # The fraction of m2's domain blocked is ~1/g
            cost += 1.0 / g
        return cost
    
    def greedy_construct(order, selected_mods_set=None, max_time=None):
        selected = []
        if selected_mods_set is None:
            selected_mods_set = set()
        if max_time is None:
            max_time = deadline - 20
        
        for m in order:
            if time.time() > max_time:
                break
            if m in selected_mods_set:
                continue
            valid_residues = find_valid_residues(m, selected)
            if valid_residues:
                # Choose residue that minimizes blocking cost
                best_r = valid_residues[0]
                best_cost = float('inf')
                for r in valid_residues:
                    cost = compute_blocking_cost(m, r, candidates, selected_mods_set | {m})
                    if cost < best_cost:
                        best_cost = cost
                        best_r = r
                selected.append((m, best_r))
                selected_mods_set.add(m)
        return selected
    
    # Phase 1: Multiple greedy strategies
    states = []
    ordered_mods = sorted(candidates)  # ascending = highest weight first
    
    # Strategy 1: Ascending order (greedy by weight)
    s1 = greedy_construct(ordered_mods)
    sc1 = sum(inv_mod[m] for m, r in s1)
    states.append((s1, sc1))
    
    # Strategy 2: Descending order
    s2 = greedy_construct(list(reversed(ordered_mods)))
    sc2 = sum(inv_mod[m] for m, r in s2)
    states.append((s2, sc2))
    
    # Strategy 3: Randomized orders with multiple seeds
    rng = random.Random(12345)
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        order = list(ordered_mods)
        rng.shuffle(order)
        s = greedy_construct(order)
        sc = sum(inv_mod[m] for m, r in s)
        states.append((s, sc))
    
    # Pick best initial state
    best_state = max(states, key=lambda x: x[1])
    best_selected = best_state[0]
    best_score = best_state[1]
    
    # Phase 2: Local search - iteratively add missing moduli
    def local_search_add(selected, max_time):
        selected = list(selected)
        score = sum(inv_mod[m] for m, r in selected)
        end_time = time.time() + max_time
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            used_mods = set(m for m, r in selected)
            
            # Find all moduli that can be added
            addable = []
            for m in ordered_mods:
                if m in used_mods:
                    continue
                valid_residues = find_valid_residues(m, selected)
                if valid_residues:
                    # Pick the best residue (min blocking)
                    best_r = valid_residues[0]
                    best_cost = float('inf')
                    for r in valid_residues:
                        cost = compute_blocking_cost(m, r, candidates, used_mods | {m})
                        if cost < best_cost:
                            best_cost = cost
                            best_r = r
                    addable.append((m, best_r, inv_mod[m]))
            
            if not addable:
                break
            
            # Add the one with highest weight (smallest m)
            addable.sort(key=lambda x: x[0])  # sort by m ascending
            m_add, r_add, weight = addable[0]
            
            selected.append((m_add, r_add))
            score += weight
            improved = True
        
        return selected, score
    
    best_selected, best_score = local_search_add(best_selected, max_time=20)
    
    # Phase 3: Local search with swaps - remove one, try to add two
    def local_search_swap(selected, max_time):
        selected = list(selected)
        score = sum(inv_mod[m] for m, r in selected)
        end_time = time.time() + max_time
        improved_any = False
        
        for i in range(len(selected)):
            if time.time() > end_time:
                break
            
            m_remove, r_remove = selected[i]
            without = selected[:i] + selected[i+1:]
            score_without = score - inv_mod[m_remove]
            used_mods = set(m for m, r in without)
            
            # Find all addable moduli
            addable = []
            for m in ordered_mods:
                if m in used_mods:
                    continue
                valid_residues = find_valid_residues(m, without)
                if valid_residues:
                    best_r = valid_residues[0]
                    best_cost = float('inf')
                    for r in valid_residues:
                        cost = compute_blocking_cost(m, r, candidates, used_mods | {m})
                        if cost < best_cost:
                            best_cost = cost
                            best_r = r
                    addable.append((m, best_r, inv_mod[m]))
            
            if not addable:
                continue
            
            # Try adding the best single modulus
            addable.sort(key=lambda x: x[0])
            m_add1, r_add1, w1 = addable[0]
            new_selected = without + [(m_add1, r_add1)]
            new_score = score_without + w1
            
            if new_score > score:
                selected = new_selected
                score = new_score
                improved_any = True
                continue
            
            # Try adding two moduli
            if len(addable) >= 2:
                best_pair_score = -1
                best_pair = None
                for idx1 in range(min(len(addable), 10)):
                    m1, r1, w1 = addable[idx1]
                    new_sel1 = without + [(m1, r1)]
                    for idx2 in range(min(len(addable), 10)):
                        if idx1 == idx2:
                            continue
                        m2, r2, w2 = addable[idx2]
                        if m2 in set(m for m, _ in new_sel1):
                            continue
                        if is_compatible(m2, r2, new_sel1):
                            pair_score = score_without + w1 + w2
                            if pair_score > best_pair_score:
                                best_pair_score = pair_score
                                best_pair = (m1, r1, m2, r2)
                
                if best_pair and best_pair_score > score:
                    m1, r1, m2, r2 = best_pair
                    selected = without + [(m1, r1), (m2, r2)]
                    score = best_pair_score
                    improved_any = True
        
        return selected, score, improved_any
    
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        best_selected, best_score, improved = local_search_swap(best_selected, max_time=15)
        if not improved:
            break
    
    # Phase 4: Final greedy fill
    used_mods = set(m for m, r in best_selected)
    for m in ordered_mods:
        if m in used_mods:
            continue
        if time.time() > deadline - 5:
            break
        valid_residues = find_valid_residues(m, best_selected)
        if valid_residues:
            # Pick best residue
            best_r = valid_residues[0]
            best_cost = float('inf')
            for r in valid_residues:
                cost = compute_blocking_cost(m, r, candidates, used_mods | {m})
                if cost < best_cost:
                    best_cost = cost
                    best_r = r
            best_selected.append((m, best_r))
            best_score += inv_mod[m]
            used_mods.add(m)
    
    # Convert to required format
    result = [[m, r] for m, r in best_selected]
    return result
# EVOLVE-BLOCK-END
