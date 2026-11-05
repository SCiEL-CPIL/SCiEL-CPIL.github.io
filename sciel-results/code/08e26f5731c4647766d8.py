import time
import random
import numpy as np
from math import lcm, gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 90s kill

    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # 2. Precompute conflict relationships
    # conflict[i] = set of indices j where lcm(candidates[i], candidates[j]) <= 256
    conflict = [set() for _ in range(n)]
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            g = gcd(vi, vj)
            l = vi * vj // g
            if l <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # 3. Helper functions
    def score(selected_indices):
        """Compute exact reciprocal sum as Fraction"""
        s = Fraction(0, 1)
        for idx in selected_indices:
            s += Fraction(1, candidates[idx])
        return s
    
    def is_valid(selected_indices):
        """Check if no two selected indices conflict"""
        sel_set = set(selected_indices)
        for idx in sel_set:
            if sel_set & conflict[idx]:
                return False
        return True
    
    def greedy_select(order):
        """Greedy selection following given order"""
        selected = []
        sel_set = set()
        for idx in order:
            if not (sel_set & conflict[idx]):
                selected.append(idx)
                sel_set.add(idx)
        return selected
    
    def local_search(selected, max_iters=500):
        """Local search: try additions, removals, and 1-for-1 swaps"""
        selected_set = set(selected)
        current_score = score(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() - start_time > deadline:
                break
            
            # Try adding each unselected element
            for idx in range(n):
                if idx not in selected_set:
                    if not (selected_set & conflict[idx]):
                        new_score = current_score + Fraction(1, candidates[idx])
                        if new_score > current_score:
                            selected_set.add(idx)
                            current_score = new_score
                            improved = True
            
            # Try 1-for-1 swaps: remove one, add one
            for remove_idx in list(selected_set):
                for add_idx in range(n):
                    if add_idx in selected_set:
                        continue
                    # Check if add_idx conflicts with (selected_set - {remove_idx})
                    remaining = selected_set - {remove_idx}
                    if remaining & conflict[add_idx]:
                        continue
                    new_score = current_score - Fraction(1, candidates[remove_idx]) + Fraction(1, candidates[add_idx])
                    if new_score > current_score:
                        selected_set.remove(remove_idx)
                        selected_set.add(add_idx)
                        current_score = new_score
                        improved = True
                        break  # restart swap loop after improvement
            
            # Try removing low-value elements that block many additions
            # (This is implicit in the above, but we can also try 2-for-1 swaps)
            # Try 2-for-1: remove two, add one (if it increases score)
            sel_list = list(selected_set)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    r1, r2 = sel_list[i], sel_list[j]
                    remaining = selected_set - {r1, r2}
                    for add_idx in range(n):
                        if add_idx in selected_set:
                            continue
                        if remaining & conflict[add_idx]:
                            continue
                        new_score = current_score - Fraction(1, candidates[r1]) - Fraction(1, candidates[r2]) + Fraction(1, candidates[add_idx])
                        if new_score > current_score:
                            selected_set.discard(r1)
                            selected_set.discard(r2)
                            selected_set.add(add_idx)
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return list(selected_set), current_score
    
    # 4. Multiple restarts with different greedy orders
    best_result = None
    best_score = Fraction(0, 1)
    
    # Strategy 1: Sort by value ascending (smallest values first - highest reciprocal)
    order1 = sorted(range(n), key=lambda i: candidates[i])
    sel1 = greedy_select(order1)
    sel1, score1 = local_search(sel1)
    if score1 > best_score:
        best_score = score1
        best_result = sel1
    
    # Strategy 2: Sort by value descending
    order2 = sorted(range(n), key=lambda i: -candidates[i])
    sel2 = greedy_select(order2)
    sel2, score2 = local_search(sel2)
    if score2 > best_score:
        best_score = score2
        best_result = sel2
    
    # Strategy 3: Random shuffles
    random.seed(42)
    num_restarts = 0
    while time.time() - start_time < deadline - 5:
        order = list(range(n))
        random.shuffle(order)
        sel = greedy_select(order)
        sel, sc = local_search(sel, max_iters=300)
        if sc > best_score:
            best_score = sc
            best_result = sel
        num_restarts += 1
        if num_restarts >= 20:
            break
    
    # Strategy 4: Sort by number of conflicts (fewest conflicts first)
    order4 = sorted(range(n), key=lambda i: len(conflict[i]))
    sel4 = greedy_select(order4)
    sel4, score4 = local_search(sel4)
    if score4 > best_score:
        best_score = score4
        best_result = sel4
    
    # Strategy 5: Sort by reciprocal * (1/conflict_count) - heuristic
    order5 = sorted(range(n), key=lambda i: -1.0 / (candidates[i] * (len(conflict[i]) + 1)))
    sel5 = greedy_select(order5)
    sel5, score5 = local_search(sel5)
    if score5 > best_score:
        best_score = score5
        best_result = sel5
    
    # Final local search on best result with more iterations
    if time.time() - start_time < deadline - 2:
        best_result, best_score = local_search(best_result, max_iters=1000)
    
    # Convert indices back to values
    result = [candidates[i] for i in best_result]
    
    # Verify validity
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                valid = False
                break
        if not valid:
            break
    
    if not valid or not result:
        # Fallback: simple greedy
        selected = []
        for v in sorted(candidates):
            if all(lcm(v, y) > 256 for y in selected):
                selected.append(v)
        return selected
    
    return result
