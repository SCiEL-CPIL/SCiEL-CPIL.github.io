import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute gcd table
    gcd_table = {}
    for m1 in range(20, 201):
        for m2 in range(20, 201):
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def score(sel):
        return sum(1.0 / m for m, r in sel)
    
    def can_add(m, r, sel):
        for em, er in sel:
            g = gcd_table[(m, em)]
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
                g = gcd_table[(m, em)]
                if g == 1:
                    ok = False
                    break
                if r % g == er % g:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid
    
    def greedy_with_order(order, rng):
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
        rng = random.Random(42)
        
        while time.time() < time_limit:
            improved = False
            
            # Try reassigning residue for each selected modulus
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                m_i, r_i = best[i]
                remaining = [x for j, x in enumerate(best) if j != i]
                valid = find_valid_residues(m_i, remaining)
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
            # Prioritize smaller moduli
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
            
            # Try swap: remove one, add two smaller ones
            if len(best) >= 2:
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                    m_remove, _ = best[i]
                    remaining = [x for j, x in enumerate(best) if j != i]
                    rem_score = sum(1.0 / m for m, r in remaining)
                    
                    # Try adding a smaller modulus
                    for m_try in range(20, min(m_remove, 201)):
                        if time.time() > time_limit:
                            break
                        if m_try in set(m for m, r in remaining):
                            continue
                        if 1.0 / m_try <= 1.0 / m_remove:
                            break
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
            
            # Try 2-swap: remove two, add one or two others
            if len(best) >= 3:
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                    for j in range(i+1, len(best)):
                        if time.time() > time_limit:
                            break
                        m1, _ = best[i]
                        m2, _ = best[j]
                        remaining = [x for k, x in enumerate(best) if k != i and k != j]
                        removed_score = 1.0/m1 + 1.0/m2
                        
                        # Try adding a single modulus that was blocked
                        for m_new in range(20, 201):
                            if time.time() > time_limit:
                                break
                            if m_new in set(m for m, r in remaining):
                                continue
                            if 1.0/m_new <= removed_score:
                                continue
                            valid = find_valid_residues(m_new, remaining)
                            if valid:
                                r_new = rng.choice(valid)
                                new_sel = remaining + [(m_new, r_new)]
                                s = score(new_sel)
                                if s > best_s + 1e-12:
                                    best = new_sel
                                    best_s = s
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best, best_s
    
    best_solution = []
    best_score = 0.0
    
    # Phase 1: Multiple greedy strategies
    # Strategy A: Ascending order (small moduli first)
    order_asc = list(range(20, 201))
    # Strategy B: Descending order
    order_desc = list(range(200, 19, -1))
    # Strategy C: Random shuffle
    rng_shuf = random.Random(12345)
    order_shuf = list(range(20, 201))
    rng_shuf.shuffle(order_shuf)
    
    orders = [order_asc, order_desc, order_shuf]
    
    for restart in range(800):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(restart * 7919 + 13)
        # Cycle through different orderings
        order = orders[restart % len(orders)]
        
        # Sometimes use a shuffled version
        if restart % 3 == 0:
            temp_order = list(order)
            rng.shuffle(temp_order)
            order = temp_order
        
        sel = greedy_with_order(order, rng)
        s = score(sel)
        
        if s > best_score:
            best_score = s
            best_solution = sel
    
    # Phase 2: Local search on the best solution found
    if best_solution and time.time() < deadline - 10:
        refined, refined_score = local_search(best_solution, deadline - 10)
        if refined_score > best_score:
            best_score = refined_score
            best_solution = refined
    
    # Phase 3: More restarts if time allows
    if time.time() < deadline - 20:
        for restart in range(800, 1200):
            if time.time() > deadline - 20:
                break
            rng = random.Random(restart * 104729 + 37)
            order = orders[restart % len(orders)]
            if restart % 2 == 0:
                temp_order = list(order)
                rng.shuffle(temp_order)
                order = temp_order
            sel = greedy_with_order(order, rng)
            s = score(sel)
            if s > best_score:
                best_score = s
                best_solution = sel
        
        if best_solution and time.time() < deadline - 10:
            refined, refined_score = local_search(best_solution, deadline - 10)
            if refined_score > best_score:
                best_score = refined_score
                best_solution = refined
    
    # Final local search if time remains
    if best_solution and time.time() < deadline - 3:
        refined, refined_score = local_search(best_solution, deadline - 3)
        if refined_score > best_score:
            best_score = refined_score
            best_solution = refined
    
    return [[m, r] for m, r in best_solution]
