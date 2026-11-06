"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid_set(s):
    """Check if set s has no three distinct elements a,b,c with 1/a = 1/b + 1/c."""
    s_list = sorted(s)
    n = len(s_list)
    # For each pair (b,c) with b < c, check if a = bc/(b+c) is an integer in s and distinct from b,c
    # 1/a = 1/b + 1/c => a = bc/(b+c) => a(b+c) = bc => a = bc/(b+c)
    # We need bc % (b+c) == 0 and a = bc//(b+c) in s and a != b and a != c
    for i in range(n):
        for j in range(i+1, n):
            b = s_list[i]
            c = s_list[j]
            bc = b * c
            bc_sum = b + c
            if bc % bc_sum == 0:
                a = bc // bc_sum
                if a != b and a != c and a in s:
                    return False
    return True

def _build_triples_conflicts(s):
    """For a set s, return list of conflicting triples (a,b,c) where 1/a=1/b+1/c."""
    s_list = sorted(s)
    n = len(s_list)
    conflicts = []
    for i in range(n):
        for j in range(i+1, n):
            b = s_list[i]
            c = s_list[j]
            bc = b * c
            bc_sum = b + c
            if bc % bc_sum == 0:
                a = bc // bc_sum
                if a != b and a != c and a in s:
                    conflicts.append((a, b, c))
    return conflicts

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    random.seed(42)
    
    # Start with all numbers 1..180
    best_set = set(range(1, 181))
    
    # Greedy removal: while there's a conflict, remove one element from the conflict
    # Use local search: try removing elements that are in most conflicts
    while time.time() < deadline:
        conflicts = _build_triples_conflicts(best_set)
        if not conflicts:
            break
        
        # Count how many times each element appears in conflicts
        conflict_count = {}
        for (a, b, c) in conflicts:
            conflict_count[a] = conflict_count.get(a, 0) + 1
            conflict_count[b] = conflict_count.get(b, 0) + 1
            conflict_count[c] = conflict_count.get(c, 0) + 1
        
        # Remove the element with highest conflict count
        if conflict_count:
            worst = max(conflict_count, key=conflict_count.get)
            best_set.remove(worst)
        else:
            break
    
    # Try to add back elements
    for elem in range(1, 181):
        if time.time() > deadline - 5:
            break
        if elem not in best_set:
            trial_set = best_set | {elem}
            if _is_valid_set(trial_set):
                best_set = trial_set
    
    # Local search: try swapping
    while time.time() < deadline - 10:
        improved = False
        for to_add in range(1, 181):
            if to_add in best_set:
                continue
            for to_remove in list(best_set):
                if time.time() > deadline - 10:
                    break
                trial_set = (best_set - {to_remove}) | {to_add}
                if len(trial_set) > len(best_set) and _is_valid_set(trial_set):
                    best_set = trial_set
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Final greedy add
    for elem in range(1, 181):
        if time.time() > deadline - 2:
            break
        if elem not in best_set:
            trial_set = best_set | {elem}
            if _is_valid_set(trial_set):
                best_set = trial_set
    
    return list(best_set)
# EVOLVE-BLOCK-END
