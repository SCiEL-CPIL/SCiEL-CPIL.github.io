import time
import numpy as np
from math import lcm, gcd
from fractions import Fraction

def _omega(n):
    """Total number of prime factors (with multiplicity)."""
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute lcm table for fast lookup
    lcm_table = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        vi = candidates[i]
        for j in range(i+1, n):
            vj = candidates[j]
            l = vi * vj // gcd(vi, vj)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
    
    # Conflict: lcm <= 256
    conflict = lcm_table <= 256
    np.fill_diagonal(conflict, False)
    
    # Precompute conflict counts
    conflict_counts = conflict.sum(axis=1).astype(np.int32)
    
    def score_of(selected):
        """Compute exact reciprocal sum using Fraction."""
        s = Fraction(0, 1)
        for v in selected:
            s += Fraction(1, v)
        return s
    
    def is_valid(selected):
        """Check if selected is a valid independent set."""
        sel_idx = [val_to_idx[v] for v in selected]
        for i in range(len(sel_idx)):
            for j in range(i+1, len(sel_idx)):
                if conflict[sel_idx[i]][sel_idx[j]]:
                    return False
        return True
    
    def can_add(v, selected_set):
        """Check if v can be added to selected_set."""
        vi = val_to_idx[v]
        for y in selected_set:
            if conflict[vi][val_to_idx[y]]:
                return False
        return True
    
    # 2. Generate multiple diverse initial solutions using greedy with different orderings
    best_selected = None
    best_score = Fraction(0, 1)
    
    def greedy_solution(order):
        """Greedy: add candidates in given order if they don't conflict."""
        selected = []
        selected_set = set()
        for v in order:
            if can_add(v, selected_set):
                selected.append(v)
                selected_set.add(v)
        return selected
    
    # Generate several greedy solutions with different orderings
    greedy_solutions = []
    
    # Order 1: ascending (small numbers first)
    greedy_solutions.append(greedy_solution(sorted(candidates)))
    
    # Order 2: descending (large numbers first)
    greedy_solutions.append(greedy_solution(sorted(candidates, reverse=True)))
    
    # Order 3: by Omega parity
    even_cand = [v for v in candidates if _omega(v) % 2 == 0]
    odd_cand = [v for v in candidates if _omega(v) % 2 == 1]
    greedy_solutions.append(greedy_solution(even_cand + odd_cand))
    greedy_solutions.append(greedy_solution(odd_cand + even_cand))
    
    # Order 4: Sort by number of conflicts (fewest conflicts first)
    by_conflicts = sorted(candidates, key=lambda v: conflict_counts[val_to_idx[v]])
    greedy_solutions.append(greedy_solution(by_conflicts))
    
    # Order 5: Sort by reciprocal weight / conflict_count
    by_weight_conflict = sorted(candidates, key=lambda v: (1.0/v) / max(1, conflict_counts[val_to_idx[v]]), reverse=True)
    greedy_solutions.append(greedy_solution(by_weight_conflict))
    
    # Evaluate all greedy solutions
    for sol in greedy_solutions:
        if sol and is_valid(sol):
            s = score_of(sol)
            if s > best_score:
                best_score = s
                best_selected = sol
    
    # 3. Local search: aggressive hill climbing
    if best_selected is not None and time.time() - start_time < deadline - 5:
        selected = list(best_selected)
        selected_set = set(selected)
        current_score = best_score
        
        improved = True
        iteration = 0
        while improved and time.time() - start_time < deadline - 2:
            improved = False
            iteration += 1
            
            # Phase 1: Try adding each unselected candidate
            for v in candidates:
                if v in selected_set:
                    continue
                if can_add(v, selected_set):
                    new_score = current_score + Fraction(1, v)
                    if new_score > current_score:
                        selected.append(v)
                        selected_set.add(v)
                        current_score = new_score
                        improved = True
            
            # Phase 2: Try 1-removal + multiple additions
            for v in sorted(selected, key=lambda x: x):
                if len(selected) <= 1:
                    break
                
                remaining = [x for x in selected if x != v]
                remaining_set = set(remaining)
                removed_score = current_score - Fraction(1, v)
                
                new_selected = list(remaining)
                new_set = set(remaining)
                new_score = removed_score
                added = 0
                
                for u in sorted(candidates, key=lambda x: 1.0/x, reverse=True):
                    if u in new_set:
                        continue
                    if can_add(u, new_set):
                        new_selected.append(u)
                        new_set.add(u)
                        new_score += Fraction(1, u)
                        added += 1
                
                if added >= 2 and new_score > current_score:
                    selected = new_selected
                    selected_set = new_set
                    current_score = new_score
                    improved = True
            
            # Phase 3: Try 2-removal + multiple additions
            if time.time() - start_time < deadline - 3 and len(selected) >= 3:
                if iteration % 3 == 0:
                    sorted_sel = sorted(selected, key=lambda x: x)
                    if len(sorted_sel) >= 3:
                        v1, v2 = sorted_sel[0], sorted_sel[1]
                        remaining = [x for x in selected if x != v1 and x != v2]
                        remaining_set = set(remaining)
                        removed_score = current_score - Fraction(1, v1) - Fraction(1, v2)
                        
                        new_selected = list(remaining)
                        new_set = set(remaining)
                        new_score = removed_score
                        
                        for u in sorted(candidates, key=lambda x: 1.0/x, reverse=True):
                            if u in new_set:
                                continue
                            if can_add(u, new_set):
                                new_selected.append(u)
                                new_set.add(u)
                                new_score += Fraction(1, u)
                        
                        if new_score > current_score:
                            selected = new_selected
                            selected_set = new_set
                            current_score = new_score
                            improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_selected = selected
    
    # 4. Final verification
    if best_selected is None or len(best_selected) == 0 or not is_valid(best_selected):
        selected = []
        for v in sorted(candidates):
            if all(not conflict[val_to_idx[v]][val_to_idx[y]] for y in selected):
                selected.append(v)
        best_selected = selected
    
    return best_selected
