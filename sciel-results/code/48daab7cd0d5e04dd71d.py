import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return time.time() < deadline
    
    def score(classes):
        return sum(Fraction(1, m) for m, r in classes)
    
    def is_compatible(m, r, chosen):
        for (m2, r2) in chosen:
            g = gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def greedy(order, residue_strategy='first'):
        """Build a set by processing moduli in given order."""
        chosen = []
        for m in order:
            if not time_ok():
                break
            if residue_strategy == 'first':
                for r in range(m):
                    if is_compatible(m, r, chosen):
                        chosen.append((m, r))
                        break
            elif residue_strategy == 'random':
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if is_compatible(m, r, chosen):
                        chosen.append((m, r))
                        break
            elif residue_strategy == 'spread':
                for r in range(0, m, max(1, m // 10)):
                    if is_compatible(m, r, chosen):
                        chosen.append((m, r))
                        break
                else:
                    for r in range(m):
                        if is_compatible(m, r, chosen):
                            chosen.append((m, r))
                            break
        return chosen
    
    def local_search(chosen, max_iters=500):
        """Try to improve by swapping or reassigning residues."""
        chosen = list(chosen)
        chosen_score = score(chosen)
        
        for _ in range(max_iters):
            if not time_ok():
                break
            
            improved = False
            used_moduli = set(m for m, r in chosen)
            
            # Try adding a new modulus that wasn't included
            for m in range(20, 201):
                if m in used_moduli:
                    continue
                for r in range(m):
                    if is_compatible(m, r, chosen):
                        new_chosen = chosen + [(m, r)]
                        new_score = score(new_chosen)
                        if new_score > chosen_score:
                            chosen = new_chosen
                            chosen_score = new_score
                            used_moduli.add(m)
                            improved = True
                            break
                if improved:
                    break
            
            # Try removing a large modulus to add multiple smaller ones
            if not improved and len(chosen) > 10:
                for idx in range(len(chosen)):
                    m, r = chosen[idx]
                    if m < 50:
                        continue
                    temp_chosen = [x for i, x in enumerate(chosen) if i != idx]
                    temp_score = score(temp_chosen)
                    used_moduli = set(m2 for m2, r2 in temp_chosen)
                    added = 0
                    for m2 in range(20, 201):
                        if m2 in used_moduli:
                            continue
                        for r2 in range(m2):
                            if is_compatible(m2, r2, temp_chosen):
                                temp_chosen.append((m2, r2))
                                used_moduli.add(m2)
                                added += 1
                                break
                    if score(temp_chosen) > chosen_score:
                        chosen = temp_chosen
                        chosen_score = score(chosen)
                        improved = True
                        break
            
            if not improved:
                break
        
        return chosen
    
    # Strategy 1: Greedy increasing order
    best = greedy(list(range(20, 201)))
    best_score = score(best)
    
    # Strategy 2: Greedy with spread residue strategy
    if time_ok():
        candidate = greedy(list(range(20, 201)), 'spread')
        if score(candidate) > best_score:
            best = candidate
            best_score = score(best)
    
    # Strategy 3: Local search on best
    if time_ok():
        candidate = local_search(best, max_iters=200)
        if score(candidate) > best_score:
            best = candidate
            best_score = score(best)
    
    # Strategy 4: Randomized restarts
    for trial in range(20):
        if not time_ok():
            break
        random_order = list(range(20, 201))
        random.shuffle(random_order)
        candidate = greedy(random_order, 'random')
        if score(candidate) > best_score:
            best = candidate
            best_score = score(candidate)
    
    # Final local search
    if time_ok():
        candidate = local_search(best, max_iters=300)
        if score(candidate) > best_score:
            best = candidate
            best_score = score(best)
    
    # Ensure we have a valid result
    if not best:
        best = [(20, 0), (21, 0), (22, 0), (23, 0), (25, 0)]
    
    return [[m, r] for m, r in best]
