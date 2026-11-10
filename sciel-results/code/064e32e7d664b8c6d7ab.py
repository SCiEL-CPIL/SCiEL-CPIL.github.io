import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Precompute GCDs for all pairs of moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    m_to_idx = {m: i for i, m in enumerate(moduli)}
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute which moduli are coprime (conflict if gcd==1)
    coprime_with = [set() for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(n_mod):
            if gcd_table[i][j] == 1:
                coprime_with[i].add(j)
    
    def score(sel):
        return sum(1.0 / m for m, r in sel)
    
    def find_valid_residues(m, sel):
        """Find all valid residues for modulus m given current selection."""
        idx = m_to_idx[m]
        # Build constraint: for each selected (em, er), r % gcd(m, em) != er % gcd(m, em)
        # If gcd(m, em) == 1, no valid residue exists
        for em, er in sel:
            eidx = m_to_idx[em]
            if gcd_table[idx][eidx] == 1:
                return []
        
        valid = []
        for r in range(m):
            ok = True
            for em, er in sel:
                eidx = m_to_idx[em]
                g = gcd_table[idx][eidx]
                if r % g == er % g:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid
    
    def greedy_with_order(order, rng, residue_strategy="random"):
        """Greedy construction with given modulus order and residue strategy."""
        sel = []
        for m in order:
            if time.time() > deadline:
                break
            valid = find_valid_residues(m, sel)
            if valid:
                if residue_strategy == "random":
                    r = rng.choice(valid)
                elif residue_strategy == "min_block":
                    # Choose residue that blocks fewest future moduli
                    # Heuristic: pick residue that is "generic" - not aligned with common gcds
                    # Simplified: just pick the middle residue
                    r = valid[len(valid) // 2]
                else:
                    r = rng.choice(valid)
                sel.append((m, r))
        return sel
    
    def local_search(sel, time_limit):
        best = list(sel)
        best_s = score(best)
        rng = random.Random()
        
        while time.time() < time_limit:
            improved = False
            
            # Move 1: Try reassigning residue for each selected modulus
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                m_i, r_i = best[i]
                remaining = [x for j, x in enumerate(best) if j != i]
                valid = find_valid_residues(m_i, remaining)
                if valid:
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
            
            # Move 2: Try adding an unselected modulus
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
            
            # Move 3: Try removing one and adding two smaller ones
            if len(best) >= 2:
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                    m_remove, _ = best[i]
                    remaining = [x for j, x in enumerate(best) if j != i]
                    rem_m = set(m for m, r in remaining)
                    # Try adding two smaller moduli
                    added = 0
                    new_sel = list(remaining)
                    for m_try in range(20, m_remove):
                        if time.time() > time_limit:
                            break
                        if m_try in rem_m or m_try == m_remove:
                            continue
                        valid = find_valid_residues(m_try, new_sel)
                        if valid:
                            r_try = rng.choice(valid)
                            new_sel.append((m_try, r_try))
                            rem_m.add(m_try)
                            added += 1
                            if added >= 2:
                                break
                    if added >= 1:
                        s = score(new_sel)
                        if s > best_s + 1e-12:
                            best = new_sel
                            best_s = s
                            improved = True
                            break
            
            # Move 4: 2-swap - remove two, add two different
            if len(best) >= 2:
                if time.time() > time_limit:
                    break
                i1 = rng.randint(0, len(best) - 1)
                i2 = rng.randint(0, len(best) - 1)
                if i1 == i2:
                    i2 = (i2 + 1) % len(best)
                remaining = [x for j, x in enumerate(best) if j not in (i1, i2)]
                rem_m = set(m for m, r in remaining)
                # Try to add two moduli not in remaining
                new_sel = list(remaining)
                added = 0
                for m_try in range(20, 201):
                    if time.time() > time_limit:
                        break
                    if m_try in rem_m:
                        continue
                    valid = find_valid_residues(m_try, new_sel)
                    if valid:
                        r_try = rng.choice(valid)
                        new_sel.append((m_try, r_try))
                        rem_m.add(m_try)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    s = score(new_sel)
                    if s > best_s + 1e-12:
                        best = new_sel
                        best_s = s
                        improved = True
            
            if not improved:
                break
        
        return best, best_s
    
    best_solution = []
    best_score = 0.0
    
    # Phase 1: Greedy with multiple orderings
    ascending = list(range(20, 201))
    descending = list(range(200, 19, -1))
    
    for restart in range(300):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Try ascending order
        sel = greedy_with_order(ascending, rng, "random")
        s = score(sel)
        if s > best_score:
            best_score = s
            best_solution = sel
        
        # Try descending order
        sel = greedy_with_order(descending, rng, "random")
        s = score(sel)
        if s > best_score:
            best_score = s
            best_solution = sel
        
        # Try random order
        random_order = list(range(20, 201))
        rng.shuffle(random_order)
        sel = greedy_with_order(random_order, rng, "random")
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
    
    # Phase 3: More random restarts if time allows, then refine
    if time.time() < deadline - 20:
        for restart in range(300, 500):
            if time.time() > deadline - 20:
                break
            rng = random.Random(restart * 104729 + 37)
            
            # Try different orderings
            for order_type in range(3):
                if time.time() > deadline - 20:
                    break
                if order_type == 0:
                    order = ascending
                elif order_type == 1:
                    order = descending
                else:
                    order = list(range(20, 201))
                    rng.shuffle(order)
                
                sel = greedy_with_order(order, rng, "random")
                s = score(sel)
                if s > best_score:
                    best_score = s
                    best_solution = sel
            
            # Also try with min_block strategy
            if time.time() > deadline - 20:
                break
            order = ascending
            sel = greedy_with_order(order, rng, "min_block")
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
