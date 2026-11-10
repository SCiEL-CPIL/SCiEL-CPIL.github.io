import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Precompute GCD table for moduli 20-200
    max_m = 200
    gcd_table = {}
    for m1 in range(20, max_m + 1):
        for m2 in range(20, max_m + 1):
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def is_valid_fast(selected, m, r, m_set):
        """Check if (m, r) is disjoint from all (m_j, r_j) in selected using precomputed GCD."""
        for m_j, r_j in selected:
            g = gcd_table[(m, m_j)]
            if r % g == r_j % g:
                return False
        return True
    
    def greedy_pass(random_seed=None, strategy="ascending"):
        """Generates a solution using a greedy approach."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        selected = []
        m_set = set()
        
        if strategy == "ascending":
            m_order = list(range(20, 201))
        elif strategy == "descending":
            m_order = list(range(200, 19, -1))
        elif strategy == "random":
            m_order = list(range(20, 201))
            rng.shuffle(m_order)
        else:
            m_order = list(range(20, 201))
        
        for m in m_order:
            if time.time() > deadline:
                break
            rs_to_check = list(range(m))
            if random_seed is not None:
                rng.shuffle(rs_to_check)
            for r in rs_to_check:
                if is_valid_fast(selected, m, r, m_set):
                    selected.append((m, r))
                    m_set.add(m)
                    break
        score = sum(1.0 / m for m, r in selected)
        return selected, score
    
    def local_search(selected, max_iterations=10000):
        """Improves a given solution using local search with multiple move types."""
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        best_score = current_score
        best_solution = list(selected)
        
        iteration = 0
        improved = True
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if time.time() > deadline - 5:
                break
                
            m_set = {m for m, r in selected}
            
            # Move type 1: Add a new modulus (try all valid residues)
            for m_add in range(20, 201):
                if m_add in m_set:
                    continue
                found = False
                for r in range(m_add):
                    if is_valid_fast(selected, m_add, r, m_set):
                        new_selected = list(selected) + [(m_add, r)]
                        new_score = current_score + 1.0 / m_add
                        if new_score > best_score + 1e-12:
                            selected = new_selected
                            current_score = new_score
                            best_score = new_score
                            improved = True
                            m_set.add(m_add)
                        found = True
                        break
                # If no valid residue, skip
                
            # Move type 2: 2-opt swap - remove one modulus, add a different one
            for idx in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                m_remove, r_remove = selected[idx]
                others = selected[:idx] + selected[idx+1:]
                others_m_set = {m for m, r in others}
                base_score = current_score - 1.0 / m_remove
                
                # Try adding any modulus not in others
                for m_add in range(20, 201):
                    if m_add in others_m_set:
                        continue
                    for r in range(m_add):
                        if is_valid_fast(others, m_add, r, others_m_set):
                            new_score = base_score + 1.0 / m_add
                            if new_score > best_score + 1e-12:
                                new_selected = others + [(m_add, r)]
                                selected = new_selected
                                current_score = new_score
                                best_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                
            # Move type 3: Change residue of existing modulus to enable more additions
            for idx in range(len(selected)):
                if time.time() > deadline - 5:
                    break
                m, r = selected[idx]
                others = selected[:idx] + selected[idx+1:]
                others_m_set = {m for m, r in others}
                
                # Try each possible new residue
                for new_r in range(m):
                    if new_r == r:
                        continue
                    if not is_valid_fast(others, m, new_r, others_m_set):
                        continue
                    
                    # Check if this change enables adding new moduli
                    test_selected = others + [(m, new_r)]
                    test_m_set = others_m_set | {m}
                    added_score = 0.0
                    added_moduli = []
                    
                    for m_add in range(20, 201):
                        if m_add in test_m_set:
                            continue
                        for r_add in range(m_add):
                            if is_valid_fast(test_selected, m_add, r_add, test_m_set):
                                added_score += 1.0 / m_add
                                added_moduli.append((m_add, r_add))
                                test_m_set.add(m_add)
                                break
                    
                    if added_score > 1e-12:
                        new_score = current_score + added_score
                        if new_score > best_score + 1e-12:
                            new_selected = test_selected + added_moduli
                            selected = new_selected
                            current_score = new_score
                            best_score = new_score
                            improved = True
                            break
        
        return best_solution, best_score
    
    best_score = 0.0
    best_solution = []
    
    # 1. Run multiple greedy strategies
    strategies = ["ascending", "descending", "random"]
    for strat in strategies:
        sol, score = greedy_pass(random_seed=None, strategy=strat)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # 2. Run Randomized Greedy Passes with different strategies
    num_passes = 150
    for i in range(num_passes):
        if time.time() > deadline - 40:  # Reserve 40s for local search
            break
        seed = i
        strat = strategies[i % 3]
        sol, score = greedy_pass(random_seed=seed, strategy=strat)
        if score > best_score:
            best_score = score
            best_solution = sol

    # 3. Apply Local Search to the best solution found
    if time.time() < deadline - 5:
        improved_sol, improved_score = local_search(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol
    
    return [[m, r] for m, r in best_solution]
