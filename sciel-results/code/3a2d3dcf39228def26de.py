import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 1.0

def _check_pair(r1, m1, r2, m2):
    """Check if two residue classes are disjoint."""
    g = gcd(m1, m2)
    return r1 % g != r2 % g

def _is_valid_solution(solution):
    """Check if a solution is valid: distinct moduli in [2,128], canonical residues, pairwise disjoint."""
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            r1, m1 = solution[i]
            r2, m2 = solution[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _count_valid_residues(m, solution):
    """Count how many residues r in [0, m) are compatible with all elements in solution."""
    count = 0
    for r in range(m):
        compatible = True
        for (r_prev, m_prev) in solution:
            if not _check_pair(r, m, r_prev, m_prev):
                compatible = False
                break
        if compatible:
            count += 1
    return count

def _get_valid_residues(m, solution):
    """Get all residues r in [0, m) that are compatible with all elements in solution."""
    valid = []
    for r in range(m):
        compatible = True
        for (r_prev, m_prev) in solution:
            if not _check_pair(r, m, r_prev, m_prev):
                compatible = False
                break
        if compatible:
            valid.append(r)
    return valid

def _backtrack_csp(assigned, unassigned, best_solution, deadline, max_depth=200):
    """
    Backtracking CSP solver with most-constrained-first heuristic.
    
    assigned: list of [r, m] pairs already assigned
    unassigned: list of moduli not yet assigned
    best_solution: current best solution found (will be updated)
    deadline: time limit
    max_depth: maximum recursion depth to prevent excessive computation
    """
    if time.time() > deadline:
        return
    
    # If we have a better solution, update best
    if len(assigned) > len(best_solution):
        best_solution.clear()
        best_solution.extend([list(x) for x in assigned])
    
    # Base case: no more moduli to assign
    if not unassigned:
        return
    
    # Select the most constrained modulus (fewest valid residues)
    best_idx = -1
    best_count = float('inf')
    best_valid_residues = None
    
    for i, m in enumerate(unassigned):
        valid_residues = _get_valid_residues(m, assigned)
        count = len(valid_residues)
        if count == 0:
            # This modulus cannot be assigned, prune
            return
        if count < best_count:
            best_count = count
            best_idx = i
            best_valid_residues = valid_residues
            if count == 1:
                break
    
    if best_idx == -1:
        return
    
    m = unassigned[best_idx]
    new_unassigned = unassigned[:best_idx] + unassigned[best_idx+1:]
    
    # Try each valid residue for the selected modulus
    # Sort residues to try "better" ones first (heuristic: prefer residues that leave more options)
    for r in best_valid_residues:
        if time.time() > deadline:
            return
        
        new_assigned = assigned + [[r, m]]
        _backtrack_csp(new_assigned, new_unassigned, best_solution, deadline, max_depth)
    
    # Additional heuristic: after trying all residues for this modulus, 
    # try removing some already-assigned moduli to explore different branches
    # This is a limited form of backtracking with restarts
    if len(assigned) > 0 and len(assigned) < 15:
        # Try removing the last assigned modulus and continuing
        pass  # Standard backtracking already handles this via recursion

def _csp_solve(all_moduli, deadline):
    """
    Solve the CSP using backtracking with most-constrained-first heuristic.
    Returns the best solution found within the time limit.
    """
    best_solution = []
    
    # Start with an empty assignment
    _backtrack_csp([], all_moduli, best_solution, deadline)
    
    return best_solution

def _greedy_construct(mod_order, deadline):
    """Greedy construction: for each modulus in order, pick the first valid residue."""
    solution = []
    used_moduli = set()
    
    for m in mod_order:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        found = False
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                if not _check_pair(r, m, r_prev, m_prev):
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                used_moduli.add(m)
                found = True
                break
    
    return solution

def _local_search_improve(solution, deadline):
    """Try to improve the solution by reassigning residues or adding new moduli."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        
        # Try to add a new modulus
        used_moduli = set(m for _, m in best)
        for m_new in range(2, MAXIMUM + 1):
            if m_new in used_moduli:
                continue
            valid_residues = _get_valid_residues(m_new, best)
            if valid_residues:
                # Pick the first valid residue
                r_new = valid_residues[0]
                new_solution = [list(x) for x in best]
                new_solution.append([r_new, m_new])
                if len(new_solution) > len(best):
                    best = new_solution
                    improved = True
                break
        
        # Try reassigning residues for each element to free up space for new moduli
        if not improved:
            for i in range(n):
                if time.time() > deadline:
                    break
                r_i, m_i = best[i]
                valid_residues = _get_valid_residues(m_i, [(r, m) for j, (r, m) in enumerate(best) if j != i])
                
                for r in valid_residues:
                    if r == r_i:
                        continue
                    
                    new_solution = [list(x) for x in best]
                    new_solution[i] = [r, m_i]
                    
                    # Try to add a new modulus with this new configuration
                    used_moduli = set(m for _, m in new_solution)
                    for m_new in range(2, MAXIMUM + 1):
                        if m_new in used_moduli:
                            continue
                        valid_new = _get_valid_residues(m_new, new_solution)
                        if valid_new:
                            new_solution.append([valid_new[0], m_new])
                            if len(new_solution) > len(best):
                                best = new_solution
                                improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Phase 1: CSP backtracking with most-constrained-first heuristic
    # Start with a subset of moduli to keep the search space manageable
    # We'll try different subsets of increasing size
    
    # First, try a greedy solution to get a baseline
    greedy_solution = _greedy_construct(all_moduli, deadline)
    if len(greedy_solution) > len(best_solution):
        best_solution = [list(x) for x in greedy_solution]
    
    # Phase 2: CSP on the full set of moduli
    if time.time() < deadline:
        csp_solution = _csp_solve(all_moduli, deadline)
        if len(csp_solution) > len(best_solution):
            best_solution = [list(x) for x in csp_solution]
    
    # Phase 3: Local search improvement
    if best_solution and time.time() < deadline:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Phase 4: Try CSP on subsets that include the best solution's moduli
    if best_solution and time.time() < deadline:
        used_moduli = set(m for _, m in best_solution)
        
        # Try to add more moduli using CSP
        remaining_moduli = [m for m in all_moduli if m not in used_moduli]
        
        # Try CSP with the current solution as a starting point
        # We'll do this by trying to extend the solution
        for m_new in remaining_moduli:
            if time.time() > deadline:
                break
            
            valid_residues = _get_valid_residues(m_new, best_solution)
            if valid_residues:
                # Try each residue
                for r_new in valid_residues[:5]:  # Limit to first 5 to save time
                    new_solution = [list(x) for x in best_solution]
                    new_solution.append([r_new, m_new])
                    
                    # Try to extend further with local search
                    if time.time() < deadline:
                        extended = _local_search_improve(new_solution, deadline)
                        if len(extended) > len(best_solution):
                            best_solution = [list(x) for x in extended]
                            break
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
