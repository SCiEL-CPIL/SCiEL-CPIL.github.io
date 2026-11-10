"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def is_valid(selected, m, r):
        """
        Check if (m, r) is disjoint from all (m_j, r_j) in selected.
        """
        for m_j, r_j in selected:
            g = math.gcd(m, m_j)
            if r % g == r_j % g:
                return False
        return True

    def greedy_pass(random_seed=None):
        """
        Generates a solution using a greedy approach.
        random_seed: if provided, randomizes the choice of residue among valid ones.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        selected = []
        # We process moduli in increasing order.
        for m in range(20, 201):
            if time.time() > deadline:
                break
                
            # Find all valid residues for m
            valid_rs = []
            rs_to_check = list(range(m))
            if random_seed is not None:
                rng.shuffle(rs_to_check)
            else:
                # Deterministic: smallest r
                pass
                
            for r in rs_to_check:
                if is_valid(selected, m, r):
                    valid_rs.append(r)
                    break
            
            if valid_rs:
                selected.append([m, valid_rs[0]])
        
        score = sum(1.0 / m for m, r in selected)
        return selected, score

    def local_search(selected):
        """
        Improves a given solution by trying to remove one element and re-adding others.
        Returns the improved list and score.
        """
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        
        improved = True
        while improved:
            improved = False
            # Try removing each selected modulus
            for idx in range(len(selected)):
                if time.time() > deadline:
                    return selected, current_score
                    
                m_to_remove, r_to_remove = selected[idx]
                
                others = selected[:idx] + selected[idx+1:]
                
                new_selected = list(others)
                new_score = current_score - (1.0 / m_to_remove)
                
                # Attempt to add any modulus from 20 to 200 not in new_selected
                m_set = {m for m, r in new_selected}
                
                for m_add in range(20, 201):
                    if m_add in m_set:
                        continue
                    
                    valid_r = None
                    for r in range(m_add):
                        if is_valid(new_selected, m_add, r):
                            valid_r = r
                            break
                    
                    if valid_r is not None:
                        new_selected.append((m_add, valid_r))
                        new_score += 1.0 / m_add
                
                if new_score > current_score + 1e-12:
                    selected = new_selected
                    current_score = new_score
                    improved = True
                    break
        
        return selected, current_score

    best_score = 0.0
    best_solution = []
    
    # 1. Run Deterministic Greedy
    sol, score = greedy_pass(random_seed=None)
    if score > best_score:
        best_score = score
        best_solution = sol
        
    # 2. Run Randomized Greedy Passes
    num_passes = 50
    for i in range(num_passes):
        if time.time() > deadline - 20.0: # Reserve 20s for local search
            break
            
        seed = i
        sol, score = greedy_pass(random_seed=seed)
        
        if score > best_score:
            best_score = score
            best_solution = sol

    # 3. Apply Local Search to the best solution found
    if time.time() < deadline - 5.0:
        improved_sol, improved_score = local_search(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol

    return [list(x) for x in best_solution]
# EVOLVE-BLOCK-END
