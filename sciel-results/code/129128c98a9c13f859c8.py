import time
import math
from math import gcd
from itertools import combinations
import random

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

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
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True

def _backtracking_construct(deadline):
    """Backtracking with constraint propagation to maximize the number of residue classes."""
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Precompute gcds
    gcd_table = {}
    for i in range(len(all_moduli)):
        for j in range(i + 1, len(all_moduli)):
            m1, m2 = all_moduli[i], all_moduli[j]
            gcd_table[(m1, m2)] = gcd(m1, m2)
            gcd_table[(m2, m1)] = gcd(m1, m2)
    
    best_solution = []
    
    def backtrack(current_solution, used_moduli, start_idx):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if len(current_solution) > len(best_solution):
            best_solution = [list(x) for x in current_solution]
        
        # Try to add more moduli
        for idx in range(start_idx, len(all_moduli)):
            if time.time() > deadline:
                return
            
            m = all_moduli[idx]
            if m in used_moduli:
                continue
            
            # Try all residues for this modulus
            for r in range(m):
                compatible = True
                for (r_prev, m_prev) in current_solution:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    current_solution.append([r, m])
                    used_moduli.add(m)
                    backtrack(current_solution, used_moduli, idx + 1)
                    current_solution.pop()
                    used_moduli.remove(m)
    
    # Try different orderings
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: m.bit_count(), reverse=True),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        best_solution = []
        backtrack([], set(), 0)
    
    return best_solution

def _greedy_with_backtracking(deadline):
    """Greedy approach with local backtracking to escape local optima."""
    all_moduli = list(range(2, MAXIMUM + 1))
    best_solution = []
    
    # Multiple random restarts with different orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Create a shuffled ordering with bias toward even numbers
        even_moduli = [m for m in all_moduli if m % 2 == 0]
        odd_moduli = [m for m in all_moduli if m % 2 != 0]
        
        # Shuffle
        random.shuffle(even_moduli)
        random.shuffle(odd_moduli)
        
        # Interleave or just use even first
        mod_order = even_moduli + odd_moduli
        
        solution = []
        used_moduli = set()
        
        for m in mod_order:
            if time.time() > deadline:
                break
            
            # Try residues in a smart order
            # Prefer residues that are "spread out"
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                compatible = True
                for (r_prev, m_prev) in solution:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    solution.append([r, m])
                    used_moduli.add(m)
                    break
        
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    return best_solution

def _local_search_improve(solution, deadline):
    """Improve a solution by trying to add more moduli or reassign residues."""
    if not solution:
        return solution
    
    current = [list(x) for x in solution]
    best = [list(x) for x in current]
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Try to add unused moduli
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        used_moduli = set(m for _, m in best)
        
        for m in all_moduli:
            if time.time() > deadline:
                break
            if m in used_moduli:
                continue
            
            for r in range(m):
                candidate = best + [[r, m]]
                if _is_valid_solution(candidate):
                    best = [list(x) for x in candidate]
                    improved = True
                    break
        
        if not improved:
            # Try reassigning residues of existing moduli
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                m_i = best[i][1]
                for r_new in range(m_i):
                    if r_new == best[i][0]:
                        continue
                    candidate = [list(x) for x in best]
                    candidate[i][0] = r_new
                    if _is_valid_solution(candidate):
                        # Try to add more
                        used_m = set(x[1] for x in candidate)
                        for m_add in all_moduli:
                            if m_add in used_m:
                                continue
                            for r_add in range(m_add):
                                candidate2 = candidate + [[r_add, m_add]]
                                if _is_valid_solution(candidate2):
                                    if len(candidate2) > len(best):
                                        best = [list(x) for x in candidate2]
                                        improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    
    # Strategy 1: Greedy with backtracking (multiple restarts)
    solution = _greedy_with_backtracking(deadline)
    if len(solution) > len(best_solution):
        best_solution = [list(x) for x in solution]
    
    # Strategy 2: Local search improvement
    if len(best_solution) > 0:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = [list(x) for x in improved]
    
    # Strategy 3: More random restarts
    for trial in range(20):
        if time.time() > deadline:
            break
        
        all_moduli = list(range(2, MAXIMUM + 1))
        random.shuffle(all_moduli)
        
        solution = []
        used_moduli = set()
        
        for m in all_moduli:
            if time.time() > deadline:
                break
            
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                compatible = True
                for (r_prev, m_prev) in solution:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    solution.append([r, m])
                    used_moduli.add(m)
                    break
        
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Final local search
    if len(best_solution) > 0:
        improved = _local_search_improve(best_solution, deadline)
        if len(improved) > len(best_solution):
            best_solution = [list(x) for x in improved]
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
