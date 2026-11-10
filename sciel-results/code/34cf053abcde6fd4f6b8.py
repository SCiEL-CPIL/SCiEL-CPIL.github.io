import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    def is_valid_pair(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
        
    def find_valid_residue(m, selected, randomize=True, num_trials=100):
        """Find a valid residue for m given selected moduli/residues."""
        if not selected:
            return 0
        
        if randomize:
            candidates = list(range(m))
            random.shuffle(candidates)
            # Limit trials to avoid long loops if no solution exists quickly
            # But we must ensure we find one if it exists. 
            # If random fails, fallback to sequential.
            for _ in range(min(num_trials, m)):
                if candidates:
                    r = candidates.pop()
                    ok = True
                    for m_prime, r_prime in selected:
                        if not is_valid_pair(m, r, m_prime, r_prime):
                            ok = False
                            break
                    if ok:
                        return r
            
            # Fallback: Sequential search
            for r in range(m):
                ok = True
                for m_prime, r_prime in selected:
                    if not is_valid_pair(m, r, m_prime, r_prime):
                        ok = False
                        break
                if ok:
                    return r
            return None
        else:
            for r in range(m):
                ok = True
                for m_prime, r_prime in selected:
                    if not is_valid_pair(m, r, m_prime, r_prime):
                        ok = False
                        break
                if ok:
                    return r
            return None

    def greedy_build(order, randomize_residues=True):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected, randomize_residues)
            if r is not None:
                selected.append([m, r])
        return selected

    def try_add_missing(solution, max_attempts=50):
        """Try to add missing moduli, prioritizing small ones (high weight)."""
        selected_moduli = set(m for m, r in solution)
        # We want to add small moduli first to maximize score
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort() # Ascending order for high weight
        
        added = 0
        for m in missing:
            if time.time() > deadline:
                break
            if added >= max_attempts:
                break
            
            # Try to find a residue for this missing modulus
            r = find_valid_residue(m, solution, randomize=True, num_trials=50)
            if r is not None:
                solution.append([m, r])
                added += 1
        return solution

    def score_solution(solution):
        if not solution:
            return 0.0
        return sum(1.0 / m for m, r in solution)

    best_result = []
    best_score = 0.0
    
    # All possible moduli
    all_moduli = list(range(20, 201))
    
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # Strategy: Ascending order is best for maximizing sum(1/m)
        # Randomize residue selection to explore different feasible sets
        order = all_moduli[:]
        
        # Occasionally shuffle to escape potential local minima in residue assignment
        if num_restarts % 10 == 0:
             random.shuffle(order)
        
        solution = greedy_build(order, randomize_residues=True)
        
        # Repair: Try to add missing moduli (especially small ones)
        solution = try_add_missing(solution, max_attempts=50)
        
        current_score = score_solution(solution)
        if current_score > best_score:
            best_score = current_score
            best_result = [list(x) for x in solution]
            
        if time.time() > deadline - 1:
            break

    # Final validation
    if best_result:
        for i in range(len(best_result)):
            for j in range(i + 1, len(best_result)):
                m1, r1 = best_result[i]
                m2, r2 = best_result[j]
                if not is_valid_pair(m1, r1, m2, r2):
                    return [[20, 0]]
    
    return best_result
