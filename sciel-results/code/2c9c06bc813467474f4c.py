import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    all_vals = list(range(1, max_val + 1))
    
    # Precompute conflict graph for MWIS approach
    # Two numbers x, y conflict if they cannot both be in a Sidon set
    # This happens if there exist a, b in current set such that x+y = a+b (with a!=b or a=b and x!=a)
    # Actually, for the full conflict graph, x and y conflict if 2x = 2y (impossible) or x+y = x+y (always, but that's the same pair)
    # The real conflict is: x and y can't be in the set together if there exist a, b in the set with a+b = x+y and {a,b} != {x,y}
    # But for a static conflict graph, we need: x and y conflict if including both would create a duplicate sum
    # This is tricky because it depends on the rest of the set.
    
    # Instead, let's use a different approach:
    # For each pair (x, y) with x < y, they "conflict" if there exist (a, b) with a < b, a != x, b != y, 
    # and a+b = x+y. But this is also not quite right.
    
    # Let me think differently. The Sidon condition is that all pairwise sums are distinct.
    # So if we have a set S, for any two distinct pairs {a,b} and {c,d} in S (with a<=b, c<=d),
    # we need a+b != c+d unless {a,b} = {c,d}.
    
    # For the conflict graph approach, let's define:
    # Two elements x, y (x < y) are "incompatible" if there exist a, b such that:
    #   - a, b are in the prescribed set or could be in the set
    #   - a + b = x + y
    #   - {a, b} != {x, y}
    # But this is still set-dependent.
    
    # A better approach for MWIS:
    # We can define a conflict between x and y if x + y equals 2*z for some z in the set,
    # or if x + y equals a + b for some other pair (a, b).
    
    # Actually, let's use the following insight:
    # If we fix the prescribed elements, then for any candidate element x,
    # it conflicts with existing elements y if x + y is already a sum in the set.
    
    # For a static conflict graph, let's define conflicts between all pairs (x, y) where x < y:
    # x and y conflict if there exist a, b (a < b, a != x, b != y) such that a + b = x + y
    # AND a, b are "likely to be in the set" (e.g., in prescribed or small values)
    
    # This is getting complex. Let me use a hybrid approach:
    # 1. Use the current greedy + local search as the base
    # 2. Add a branch-and-bound for the final polish phase
    # 3. Use bitsets for fast conflict checking
    
    def init_sums(S):
        Sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                Sums.add(S[i] + S[j])
        return Sums
    
    def can_add(x, S, Sums):
        if 2 * x in Sums:
            return False
        for y in S:
            if x + y in Sums:
                return False
        return True
    
    def add_element(x, S, Sums):
        Sums.add(2 * x)
        for y in S:
            Sums.add(x + y)
        S.append(x)
        S.sort()
    
    def remove_element(x, S, Sums):
        S.remove(x)
        Sums.discard(2 * x)
        for y in S:
            Sums.discard(x + y)
    
    def greedy_fill(S, Sums, order='ascending'):
        S_set = set(S)
        if order == 'ascending':
            candidates = list(range(1, max_val + 1))
        elif order == 'descending':
            candidates = list(range(max_val, 0, -1))
        else:
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            if x in S_set:
                continue
            if can_add(x, S, Sums):
                add_element(x, S, Sums)
                S_set.add(x)
        return S, Sums
    
    def local_search(S, Sums, max_iters=200, aggressive=False):
        S_set = set(S)
        improved = False
        iters = 0
        stuck_count = 0
        
        while iters < max_iters and time.time() < deadline:
            iters += 1
            
            candidates = [x for x in all_vals if x not in S_set]
            if not candidates:
                break
            
            random.shuffle(candidates)
            
            added = False
            for x in candidates[:min(50, len(candidates))]:
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    S_set.add(x)
                    improved = True
                    added = True
                    stuck_count = 0
                    break
            
            if added:
                continue
            
            stuck_count += 1
            
            removable = [x for x in S if x not in prescribed_set]
            if not removable:
                break
            
            random.shuffle(removable)
            swap_done = False
            
            for r in removable[:min(10, len(removable))]:
                remove_element(r, S, Sums)
                S_set.discard(r)
                
                candidates = [x for x in all_vals if x not in S_set]
                random.shuffle(candidates)
                
                for x in candidates[:min(50, len(candidates))]:
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        swap_done = True
                        improved = True
                        stuck_count = 0
                        break
                
                if swap_done:
                    break
                else:
                    add_element(r, S, Sums)
                    S_set.add(r)
            
            if swap_done:
                continue
            
            if aggressive and len(removable) >= 2:
                random.shuffle(removable)
                r1, r2 = removable[0], removable[1]
                
                remove_element(r1, S, Sums)
                S_set.discard(r1)
                remove_element(r2, S, Sums)
                S_set.discard(r2)
                
                candidates = [x for x in all_vals if x not in S_set]
                random.shuffle(candidates)
                
                added_count = 0
                added_elems = []
                for x in candidates[:min(80, len(candidates))]:
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        added_elems.append(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 2:
                    improved = True
                    stuck_count = 0
                else:
                    for x in added_elems:
                        remove_element(x, S, Sums)
                        S_set.discard(x)
                    add_element(r1, S, Sums)
                    S_set.add(r1)
                    add_element(r2, S, Sums)
                    S_set.add(r2)
            
            if stuck_count > 50:
                break
        
        return S, Sums, improved
    
    def branch_and_bound_polish(S, Sums, target_size):
        """
        Try to extend S to target_size using branch and bound.
        Returns the best set found.
        """
        S_set = set(S)
        Sums_copy = Sums.copy()
        S_copy = S.copy()
        
        best_size = len(S_copy)
        best_S = S_copy.copy()
        
        # Candidates to try adding
        candidates = [x for x in all_vals if x not in S_set]
        
        # Sort candidates by a heuristic: prefer those that conflict less
        def conflict_score(x):
            score = 0
            if 2 * x in Sums_copy:
                score += 10
            for y in S_copy:
                if x + y in Sums_copy:
                    score += 5
            return score
        
        candidates.sort(key=conflict_score)
        
        # Backtracking search
        def backtrack(current_S, current_Sums, current_set, depth, max_depth):
            nonlocal best_size, best_S
            
            if time.time() > deadline - 2:
                return
            
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = current_S.copy()
            
            if len(current_S) >= target_size or depth >= max_depth:
                return
            
            # Try adding each candidate
            for x in candidates:
                if x in current_set:
                    continue
                
                if can_add(x, current_S, current_Sums):
                    # Add x
                    new_Sums = current_Sums.copy()
                    new_Sums.add(2 * x)
                    for y in current_S:
                        new_Sums.add(x + y)
                    
                    new_S = current_S.copy()
                    new_S.append(x)
                    new_S.sort()
                    
                    new_set = current_set.copy()
                    new_set.add(x)
                    
                    backtrack(new_S, new_Sums, new_set, depth + 1, max_depth)
                    
                    # Backtrack
                    # (already using copies, so no explicit removal needed)
        
        # Limit the depth to keep it tractable
        max_depth = min(10, target_size - len(S_copy))
        backtrack(S_copy, Sums_copy, S_set, 0, max_depth)
        
        return best_S, best_size
    
    def hill_climbing_polish(S, Sums):
        S_set = set(S)
        improved = True
        polish_iters = 0
        
        while improved and polish_iters < 100 and time.time() < deadline:
            improved = False
            polish_iters += 1
            
            for x in all_vals:
                if x in S_set:
                    continue
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            removable = [x for x in S if x not in prescribed_set]
            for r in removable:
                remove_element(r, S, Sums)
                S_set.discard(r)
                
                for x in all_vals:
                    if x in S_set:
                        continue
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        S_set.add(x)
                        improved = True
                        break
                
                if improved:
                    break
                else:
                    add_element(r, S, Sums)
                    S_set.add(r)
        
        return S, Sums
    
    best_size = 0
    best_S = []
    
    # Strategy 1: Multiple greedy + local search restarts
    restart_count = 0
    
    while time.time() < deadline - 15:
        restart_count += 1
        
        if restart_count % 3 == 1:
            order = 'ascending'
        elif restart_count % 3 == 2:
            order = 'descending'
        else:
            order = 'random'
        
        S = list(prescribed)
        S.sort()
        Sums = init_sums(S)
        S, Sums = greedy_fill(S, Sums, order=order)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        if best_size >= 20:
            break
        
        aggressive = restart_count % 2 == 0
        S, Sums, improved = local_search(S, Sums, max_iters=150, aggressive=aggressive)
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
        
        if best_size >= 20:
            break
    
    # Final polish with hill climbing
    if best_S and time.time() < deadline - 5:
        S = best_S.copy()
        Sums = init_sums(S)
        S, Sums = hill_climbing_polish(S, Sums)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    # Try branch and bound if we have time and haven't reached 20
    if best_S and best_size < 20 and time.time() < deadline - 10:
        S = best_S.copy()
        Sums = init_sums(S)
        target = min(20, best_size + 3)
        Bb_S, Bb_size = branch_and_bound_polish(S, Sums, target)
        if Bb_size > best_size:
            best_size = Bb_size
            best_S = Bb_S
    
    return best_S
