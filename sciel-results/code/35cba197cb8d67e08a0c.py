import math
import time
import random
from itertools import combinations
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    moduli = list(range(2, 129))
    n = len(moduli)
    
    # Precompute GCDs
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = _gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    best_count = 0
    best_result = []
    
    def is_valid(i, r, selected):
        """Check if residue r for modulus index i is disjoint from all in selected."""
        for (j, rj) in selected:
            g = gcds[i][j]
            if (r - rj) % g == 0:
                return False
        return True
    
    def greedy_solve(order):
        """Greedy: process moduli in given order, pick first valid residue."""
        selected = []
        for i in order:
            m = moduli[i]
            for r in range(m):
                if is_valid(i, r, selected):
                    selected.append((i, r))
                    break
        return selected
    
    def solve_even_subproblem(time_budget=10.0):
        """Dedicated solver for even moduli using constraint propagation."""
        end_time = time.time() + time_budget
        # Even moduli: 2, 4, 6, 8, ..., 128
        even_indices = [i for i in range(n) if moduli[i] % 2 == 0]
        even_moduli = [moduli[i] for i in even_indices]
        k = len(even_moduli)
        
        # For even moduli 2*a_i, gcd(2*a_i, 2*a_j) = 2*gcd(a_i, a_j)
        # Disjoint iff (r_i - r_j) % (2*gcd(a_i, a_j)) != 0
        
        # Precompute gcds for the half-moduli
        half_moduli = [m // 2 for m in even_moduli]
        half_gcds = [[0]*k for _ in range(k)]
        for i in range(k):
            for j in range(i+1, k):
                g = _gcd(half_moduli[i], half_moduli[j])
                half_gcds[i][j] = g * 2  # actual gcd of full moduli
                half_gcds[j][i] = g * 2
        
        # Greedy with backtracking for even subproblem
        def backtrack_even(order, selected, depth=0):
            nonlocal best_even_count, best_even_result
            if time.time() > end_time:
                return
            if len(selected) > best_even_count:
                best_even_count = len(selected)
                best_even_result = list(selected)
            
            if depth >= len(order):
                return
            
            i = order[depth]
            m = even_moduli[i]
            
            # Try residues in order of how many constraints they satisfy
            for r in range(m):
                valid = True
                for (j, rj) in selected:
                    g = half_gcds[i][j]
                    if (r - rj) % g == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                selected.append((i, r))
                backtrack_even(order, selected, depth + 1)
                selected.pop()
                
                if time.time() > end_time:
                    return
            
        best_even_count = 0
        best_even_result = []
        
        # Try different orderings
        for _ in range(20):
            if time.time() > end_time:
                break
            order = list(range(k))
            random.shuffle(order)
            backtrack_even(order, [], 0)
        
        # Convert back to original indices
        result = []
        for (i, r) in best_even_result:
            result.append((even_indices[i], r))
        
        return result
    
    def solve_odd_subproblem(time_budget=5.0):
        """Solver for odd moduli."""
        end_time = time.time() + time_budget
        odd_indices = [i for i in range(n) if moduli[i] % 2 != 0]
        
        def backtrack_odd(order, selected, depth=0):
            nonlocal best_odd_count, best_odd_result
            if time.time() > end_time:
                return
            if len(selected) > best_odd_count:
                best_odd_count = len(selected)
                best_odd_result = list(selected)
            
            if depth >= len(order):
                return
            
            i = order[depth]
            m = moduli[i]
            
            for r in range(m):
                valid = True
                for (j, rj) in selected:
                    g = gcds[i][j]
                    if (r - rj) % g == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                selected.append((i, r))
                backtrack_odd(order, selected, depth + 1)
                selected.pop()
                
                if time.time() > end_time:
                    return
            
        best_odd_count = 0
        best_odd_result = []
        
        for _ in range(10):
            if time.time() > end_time:
                break
            order = list(odd_indices)
            random.shuffle(order)
            backtrack_odd(order, [], 0)
        
        return best_odd_result
    
    def combine_and_validate(even_result, odd_result):
        """Combine even and odd results, ensuring cross-disjointness."""
        combined = list(even_result) + list(odd_result)
        
        # Check cross-disjointness between even and odd
        valid_combined = []
        for item in combined:
            i, r = item
            valid = True
            for (j, rj) in valid_combined:
                g = gcds[i][j]
                if (r - rj) % g == 0:
                    valid = False
                    break
            if valid:
                valid_combined.append(item)
        
        return valid_combined
    
    def local_search(selected, time_budget=3.0):
        """Improve by trying to add missing moduli with different residues, or swapping."""
        selected = list(selected)
        end_time = time.time() + time_budget
        improved = True
        while improved and time.time() < end_time:
            improved = False
            in_set = set(i for i, _ in selected)
            
            # Try to add a missing modulus
            for i in range(n):
                if i in in_set:
                    continue
                m = moduli[i]
                for r in range(m):
                    if is_valid(i, r, selected):
                        selected.append((i, r))
                        improved = True
                        break
                if improved:
                    break
            
            # Try to swap residues to enable adding more
            if not improved and time.time() < end_time:
                for idx, (i, r) in enumerate(selected):
                    m = moduli[i]
                    found_swap = False
                    for r_new in range(m):
                        if r_new == r:
                            continue
                        valid = True
                        for j, (k, rk) in enumerate(selected):
                            if j == idx:
                                continue
                            g = gcds[i][k]
                            if (r_new - rk) % g == 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        old_r = selected[idx][1]
                        selected[idx] = (i, r_new)
                        
                        in_set_after_swap = set(i for i, _ in selected)
                        for i2 in range(n):
                            if i2 in in_set_after_swap:
                                continue
                            m2 = moduli[i2]
                            for r2 in range(m2):
                                if is_valid(i2, r2, selected):
                                    selected.append((i2, r2))
                                    found_swap = True
                                    break
                            if found_swap:
                                break
                        
                        if not found_swap:
                            selected[idx] = (i, old_r)
                    
                    if found_swap:
                        improved = True
                        break
        
        return selected
    
    # Strategy 1: Solve even subproblem separately
    if time.time() < deadline - 15:
        even_result = solve_even_subproblem(time_budget=min(10.0, deadline - time.time() - 15))
        if len(even_result) > best_count:
            best_count = len(even_result)
            best_result = [[r, moduli[i]] for i, r in even_result]
    
    # Strategy 2: Solve odd subproblem separately  
    if time.time() < deadline - 10:
        odd_result = solve_odd_subproblem(time_budget=min(5.0, deadline - time.time() - 10))
        if len(odd_result) > best_count:
            best_count = len(odd_result)
            best_result = [[r, moduli[i]] for i, r in odd_result]
    
    # Strategy 3: Combine and polish
    if time.time() < deadline - 8:
        even_result = solve_even_subproblem(time_budget=min(8.0, deadline - time.time() - 8))
        odd_result = solve_odd_subproblem(time_budget=min(4.0, deadline - time.time() - 6))
        
        combined = combine_and_validate(even_result, odd_result)
        if time.time() < deadline - 4:
            combined_polished = local_search(combined, time_budget=min(3.0, deadline - time.time() - 3))
            if len(combined_polished) > len(combined):
                combined = combined_polished
        
        if len(combined) > best_count:
            best_count = len(combined)
            best_result = [[r, moduli[i]] for i, r in combined]
    
    # Strategy 4: Generic greedy with local search
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            indices = list(range(n))
            random.shuffle(indices)
            selected = greedy_solve(indices)
            selected_polished = local_search(selected, time_budget=min(2.0, max(0.5, deadline - time.time() - 2.0)))
            if len(selected_polished) > best_count:
                best_count = len(selected_polished)
                best_result = [[r, moduli[i]] for i, r in selected_polished]
    
    if not best_result:
        return [[1, 2]]
        
    return best_result
