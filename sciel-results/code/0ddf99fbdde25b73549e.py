import math
import time
import random
from fractions import Fraction

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
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()
        
        added = 0
        for m in missing:
            if time.time() > deadline:
                break
            if added >= max_attempts:
                break
            
            r = find_valid_residue(m, solution, randomize=True, num_trials=50)
            if r is not None:
                solution.append([m, r])
                added += 1
        return solution

    def score_solution(solution):
        if not solution:
            return 0.0
        return sum(1.0 / m for m, r in solution)

    def local_search_improve(solution, max_iterations=200):
        """Local search: try to add missing moduli and swap residues to free up space."""
        if not solution:
            return solution
        
        selected_moduli = set(m for m, r in solution)
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()  # Small moduli first
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            
            # Try to add each missing modulus
            for m in missing[:]:
                if time.time() > deadline:
                    break
                if m in selected_moduli:
                    continue
                
                # Try multiple random residues
                for _ in range(20):
                    r = random.randint(0, m - 1)
                    ok = True
                    for m_prime, r_prime in solution:
                        if not is_valid_pair(m, r, m_prime, r_prime):
                            ok = False
                            break
                    if ok:
                        solution.append([m, r])
                        selected_moduli.add(m)
                        missing.remove(m)
                        improved = True
                        break
            
            # Try to swap residues to free up space for smaller moduli
            for i in range(len(solution)):
                if time.time() > deadline:
                    break
                m_i, r_i = solution[i]
                for j in range(len(solution)):
                    if i == j:
                        continue
                    m_j, r_j = solution[j]
                    
                    # Try swapping residues if it allows adding a missing modulus
                    for m_missing in missing[:3]:  # Only try top 3 missing
                        if m_missing in selected_moduli:
                            continue
                        
                        # Try swapping r_i and r_j
                        ok_swap = True
                        for k in range(len(solution)):
                            if k == i or k == j:
                                continue
                            m_k, r_k = solution[k]
                            if not is_valid_pair(m_i, r_j, m_k, r_k) or not is_valid_pair(m_j, r_i, m_k, r_k):
                                ok_swap = False
                                break
                        
                        if ok_swap:
                            # Check if we can add m_missing with swapped residues
                            for r_new in range(min(5, m_missing)):
                                ok_add = True
                                for k in range(len(solution)):
                                    if k == i:
                                        m_test, r_test = m_i, r_j
                                    elif k == j:
                                        m_test, r_test = m_j, r_i
                                    else:
                                        m_test, r_test = solution[k]
                                    if not is_valid_pair(m_missing, r_new, m_test, r_test):
                                        ok_add = False
                                        break
                                if ok_add:
                                    # Perform the swap
                                    solution[i][1] = r_j
                                    solution[j][1] = r_i
                                    solution.append([m_missing, r_new])
                                    selected_moduli.add(m_missing)
                                    missing.remove(m_missing)
                                    improved = True
                                    break
        
        return solution

    best_result = []
    best_score = 0.0
    
    all_moduli = list(range(20, 201))
    
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # Use ascending order for best reciprocal sum
        order = all_moduli[:]
        
        # Occasionally shuffle to explore different residue assignments
        if num_restarts % 5 == 0:
            random.shuffle(order)
        
        solution = greedy_build(order, randomize_residues=True)
        solution = try_add_missing(solution, max_attempts=50)
        
        # Apply local search to improve
        solution = local_search_improve(solution, max_iterations=100)
        
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
