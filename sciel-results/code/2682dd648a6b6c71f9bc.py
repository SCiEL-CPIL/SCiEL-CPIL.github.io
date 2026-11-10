import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    def score(sel):
        return sum(1.0 / m for m, r in sel)
    
    def can_add(m, r, sel):
        for em, er in sel:
            g = math.gcd(m, em)
            if g == 1:
                return False
            if r % g == er % g:
                return False
        return True
    
    def find_valid_residues(m, sel):
        valid = []
        for r in range(m):
            ok = True
            for em, er in sel:
                g = math.gcd(m, em)
                if g == 1:
                    ok = False
                    break
                if r % g == er % g:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid
    
    def greedy_with_random(order, rng):
        sel = []
        for m in order:
            if time.time() > deadline:
                break
            valid = find_valid_residues(m, sel)
            if valid:
                r = rng.choice(valid)
                sel.append((m, r))
        return sel
    
    def local_search(sel, time_limit):
        best = list(sel)
        best_s = score(best)
        rng = random.Random()
        
        while time.time() < time_limit:
            improved = False
            
            # Try reassigning residue for each selected modulus
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                m_i, r_i = best[i]
                remaining = [x for j, x in enumerate(best) if j != i]
                valid = find_valid_residues(m_i, remaining)
                if valid:
                    # Try a random different residue
                    if len(valid) > 1:
                        candidates = [r for r in valid if r != r_i]
                        if candidates:
                            r_new = rng.choice(candidates)
                            new_sel = remaining + [(m_i, r_new)]
                            s = score(new_sel)
                            if s > best_s + 1e-12:
                                best = new_sel
                                best_s = s
                                improved = True
            
            # Try adding an unselected modulus
            selected_m = set(m for m, r in best)
            for m_new in range(20, 201):
                if time.time() > time_limit:
                    break
                if m_new in selected_m:
                    continue
                valid = find_valid_residues(m_new, best)
                if valid:
                    r_new = rng.choice(valid)
                    new_sel = best + [(m_new, r_new)]
                    s = score(new_sel)
                    if s > best_s + 1e-12:
                        best = new_sel
                        best_s = s
                        improved = True
                        break  # Re-loop to allow more additions
            
            # Try removing one and replacing with two smaller ones
            if len(best) >= 2:
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                    m_remove, _ = best[i]
                    remaining = [x for j, x in enumerate(best) if j != i]
                    # Try adding a smaller modulus that was previously blocked
                    for m_try in range(20, m_remove):
                        if m_try in set(m for m, r in remaining):
                            continue
                        valid = find_valid_residues(m_try, remaining)
                        if valid:
                            r_try = rng.choice(valid)
                            new_sel = remaining + [(m_try, r_try)]
                            s = score(new_sel)
                            if s > best_s + 1e-12:
                                best = new_sel
                                best_s = s
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best, best_s
    
    best_solution = []
    best_score = 0.0
    
    # Phase 1: Many random restarts with greedy construction
    base_order = list(range(20, 201))
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(restart * 7919 + 13)
        sel = greedy_with_random(base_order, rng)
        s = score(sel)
        
        if s > best_score:
            best_score = s
            best_solution = sel
    
    # Phase 2: Local search on the best solution found
    if best_solution and time.time() < deadline - 5:
        refined, refined_score = local_search(best_solution, deadline - 5)
        if refined_score > best_score:
            best_score = refined_score
            best_solution = refined
    
    # Phase 3: A few more random restarts if time allows, then refine
    if time.time() < deadline - 15:
        for restart in range(200, 400):
            if time.time() > deadline - 15:
                break
            rng = random.Random(restart * 104729 + 37)
            sel = greedy_with_random(base_order, rng)
            s = score(sel)
            if s > best_score:
                best_score = s
                best_solution = sel
        
        if best_solution and time.time() < deadline - 5:
            refined, refined_score = local_search(best_solution, deadline - 5)
            if refined_score > best_score:
                best_score = refined_score
                best_solution = refined
    
    # Final local search if time remains
    if best_solution and time.time() < deadline - 2:
        refined, refined_score = local_search(best_solution, deadline - 2)
        if refined_score > best_score:
            best_score = refined_score
            best_solution = refined
    
    return [[m, r] for m, r in best_solution]
