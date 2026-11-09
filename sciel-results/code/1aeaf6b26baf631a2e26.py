import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

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

def _backtracking_construct(mod_order, deadline, max_depth=127):
    """Backtracking search to find a large valid set."""
    best_solution = []
    solution = []
    used_moduli = set()
    
    def backtrack(idx):
        nonlocal best_solution
        if time.time() > deadline:
            return
        if idx == len(mod_order):
            if len(solution) > len(best_solution):
                best_solution = [list(x) for x in solution]
            return
        if len(solution) + (len(mod_order) - idx) <= len(best_solution):
            return  # Prune: can't beat best
        
        m = mod_order[idx]
        # Try to add this modulus
        for r in range(m):
            compatible = True
            for (r_prev, m_prev) in solution:
                if not _check_pair(r, m, r_prev, m_prev):
                    compatible = False
                    break
            if compatible:
                solution.append([r, m])
                used_moduli.add(m)
                backtrack(idx + 1)
                solution.pop()
                used_moduli.discard(m)
                if time.time() > deadline:
                    return
        # Also try skipping this modulus
        backtrack(idx + 1)
    
    backtrack(0)
    return best_solution

def _local_search_improve(solution, deadline):
    """Try to improve the solution by reassigning residues or adding new moduli."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        
        # Try reassigning residues for each element to enable adding a new modulus
        for i in range(n):
            if time.time() > deadline:
                break
            r_i, m_i = best[i]
            # Try other residues for this modulus
            for r in range(m_i):
                if r == r_i:
                    continue
                # Check if new residue works with all others
                compatible = True
                for j in range(n):
                    if i == j:
                        continue
                    r_j, m_j = best[j]
                    if not _check_pair(r, m_i, r_j, m_j):
                        compatible = False
                        break
                if compatible:
                    new_solution = [list(x) for x in best]
                    new_solution[i] = [r, m_i]
                    
                    # Try to add a new modulus
                    used_moduli = set(m for _, m in new_solution)
                    for m_new in range(2, MAXIMUM + 1):
                        if m_new in used_moduli:
                            continue
                        for r_new in range(m_new):
                            compatible_new = True
                            for (r_prev, m_prev) in new_solution:
                                if not _check_pair(r_new, m_new, r_prev, m_prev):
                                    compatible_new = False
                                    break
                            if compatible_new:
                                new_solution.append([r_new, m_new])
                                if len(new_solution) > len(best):
                                    best = new_solution
                                    improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # Try adding a new modulus directly
        used_moduli = set(m for _, m in best)
        for m_new in range(2, MAXIMUM + 1):
            if time.time() > deadline:
                break
            if m_new in used_moduli:
                continue
            for r_new in range(m_new):
                compatible_new = True
                for (r_prev, m_prev) in best:
                    if not _check_pair(r_new, m_new, r_prev, m_prev):
                        compatible_new = False
                        break
                if compatible_new:
                    new_solution = [list(x) for x in best]
                    new_solution.append([r_new, m_new])
                    if len(new_solution) > len(best):
                        best = new_solution
                        improved = True
                    break
            if improved:
                break
    
    return best

def _smarter_greedy(moduli_list, deadline):
    """Greedy with dynamic ordering: at each step, pick the modulus that has the most remaining options."""
    solution = []
    remaining = list(moduli_list)
    
    while remaining and time.time() < deadline:
        best_m = None
        best_r = None
        best_score = -1
        
        for m in remaining:
            # Count how many valid residues exist
            valid_count = 0
            first_valid = None
            for r in range(m):
                compatible = True
                for (r_prev, m_prev) in solution:
                    if not _check_pair(r, m, r_prev, m_prev):
                        compatible = False
                        break
                if compatible:
                    valid_count += 1
                    if first_valid is None:
                        first_valid = r
            if valid_count > 0:
                # Prefer moduli with fewer valid options (more constrained first)
                score = valid_count
                if score < best_score or best_m is None:
                    best_score = score
                    best_m = m
                    best_r = first_valid
        
        if best_m is None:
            break
        
        solution.append([best_r, best_m])
        remaining.remove(best_m)
    
    return solution

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Phase 1: Multiple greedy strategies
    # Strategy 1: Simple greedy with various orderings
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: m.bit_count()),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
        sorted(all_moduli, key=lambda m: min(p for p in range(2, m+1) if m%p==0)),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        solution = _greedy_construct(mod_order, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Strategy 2: Smarter greedy with dynamic ordering
    if time.time() < deadline:
        solution = _smarter_greedy(all_moduli, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Phase 2: Local search improvement
    if best_solution and time.time() < deadline:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = improved
    
    # Phase 3: Randomized multi-start
    if time.time() < deadline:
        random.seed(42)
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            mod_order = all_moduli[:]
            random.shuffle(mod_order)
            
            solution = _greedy_construct(mod_order, deadline)
            if len(solution) > len(best_solution):
                best_solution = [list(x) for x in solution]
            
            if time.time() < deadline and len(solution) > 0:
                improved = _local_search_improve(solution, deadline)
                if len(improved) > len(best_solution):
                    best_solution = improved
    
    # Phase 4: Final local search
    if best_solution and time.time() < deadline:
        final = _local_search_improve(best_solution, deadline)
        if len(final) > len(best_solution):
            best_solution = final
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
