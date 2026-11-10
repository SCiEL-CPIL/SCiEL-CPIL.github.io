import time
import sys
import random
import numpy as np

# Set recursion limit just in case
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin for 90s limit

def _is_valid_addition_fast(c, S):
    """
    Fast check if c can be added to S.
    S is a list of distinct integers.
    Returns True if valid.
    
    Optimizations:
    1. For "c divides sum of 3 in S":
       - Precompute residues of S mod c.
       - Check if there exist distinct i,j,k such that (S[i]+S[j]+S[k]) % c == 0.
       - We can do this by iterating pairs (i,j) and checking if (target - S[i] - S[j]) % c 
         is present in S with index k distinct from i,j.
       - To handle distinctness efficiently, we can maintain a count of residues?
       - Actually, for small N (<=48), a direct triple loop with early exit is O(N^3) ~ 18k.
       - A faster way: 
         Let R be a dictionary mapping residue -> list of indices in S with that residue.
         For each pair (i, j) in S:
           target_res = (-S[i] - S[j]) % c
           Look up target_res in R.
           If found, check if there is an index k in R[target_res] such that k != i and k != j.
           If yes, violation.
         This is O(N^2) lookups.
         
    2. For "a in S divides (c + x + y) for distinct x,y in S\{a}":
       - For each a in S:
         target = (-c) % a
         We need x, y in S \ {a} such that (x + y) % a == target.
         Let R_a be a dictionary mapping residue -> list of indices in S \ {a} with that residue.
         For each x in S \ {a}:
           needed_res = (target - x) % a
           Look up needed_res in R_a.
           If found, check if there is an index y in R_a[needed_res] such that y != index_of_x.
           If yes, violation.
         This is O(N) lookups per a, so O(N^2) total.
    """
    n = len(S)
    if n < 3:
        return True
        
    # Check 1: c divides sum of 3 distinct in S
    # Build residue map for S mod c
    res_map = {}
    for idx, val in enumerate(S):
        r = val % c
        if r not in res_map:
            res_map[r] = []
        res_map[r].append(idx)
        
    for i in range(n):
        for j in range(i + 1, n):
            target_res = (-S[i] - S[j]) % c
            if target_res in res_map:
                # Check if there's a k distinct from i and j
                # res_map[target_res] contains indices.
                # We need any k in that list such that k != i and k != j.
                # Since the list is small, we can just check.
                # Optimization: if len(list) > 2, it must contain a valid k.
                # If len == 2, check if both are i or j?
                # Actually, i and j are distinct.
                # If i and j have residue target_res, then res_map[target_res] contains i and j.
                # If len(res_map[target_res]) >= 3, there is a third one.
                # If len == 2, the only candidates are i and j (if they are in the list) or others.
                # Wait, i and j are in S. Their residues might be target_res.
                # Let's just iterate the small list.
                found = False
                for k in res_map[target_res]:
                    if k != i and k != j:
                        found = True
                        break
                if found:
                    return False
                    
    # Check 2: For each a in S, a divides (c + x + y)
    for idx_a, a in enumerate(S):
        # Build residue map for S \ {a} mod a
        res_map_a = {}
        for idx in range(n):
            if idx == idx_a:
                continue
            val = S[idx]
            r = val % a
            if r not in res_map_a:
                res_map_a[r] = []
            res_map_a[r].append(idx)
            
        target = (-c) % a
        # We need x, y in S \ {a} distinct such that (x+y) % a == target
        # Iterate x in S \ {a}
        for idx_x in range(n):
            if idx_x == idx_a:
                continue
            x = S[idx_x]
            needed_res = (target - x) % a
            if needed_res in res_map_a:
                # Check if there is y distinct from x
                found = False
                for idx_y in res_map_a[needed_res]:
                    if idx_y != idx_x:
                        found = True
                        break
                if found:
                    return False
                    
    return True

def _greedy_construct(order):
    """
    Construct a set by iterating through 'order' and adding if valid.
    """
    S = []
    for c in order:
        if _is_valid_addition_fast(c, S):
            S.append(c)
    return S

def _local_search(S, time_limit):
    """
    Local search with swaps and additions.
    """
    current_set = S[:]
    best_set = current_set[:]
    start = time.time()
    
    all_nums = set(range(1, 193))
    
    while time.time() - start < time_limit:
        improved = False
        
        # 1. Try to add an element if possible
        if len(current_set) < 48:
            # Sort candidates by "freedom" (e.g., descending value) or random
            candidates = [x for x in range(192, 0, -1) if x not in current_set]
            random.shuffle(candidates)
            for elem_in in candidates:
                if _is_valid_addition_fast(elem_in, current_set):
                    current_set.append(elem_in)
                    improved = True
                    break
        
        # 2. Try swaps if we want to improve quality or if we are stuck
        # Even if we can add, swapping might allow adding more later.
        # But for now, let's focus on maximizing size first.
        # If we are at 48, we can't add. But we might swap to a "better" set?
        # The goal is just cardinality. So if we are at 48, we are done?
        # No, we might be at 47 and stuck. Swapping might allow adding the 48th.
        
        if not improved and len(current_set) < 48:
            # Try swaps
            if not current_set:
                break
            idx_out = random.randint(0, len(current_set) - 1)
            elem_out = current_set[idx_out]
            
            # Try to find an element to add
            candidates = [x for x in range(192, 0, -1) if x not in current_set]
            random.shuffle(candidates)
            
            for elem_in in candidates:
                # Check if swapping is valid
                new_set_base = [x for i, x in enumerate(current_set) if i != idx_out]
                
                # We need to check if elem_in is valid against new_set_base
                # AND that the removal of elem_out doesn't break anything (it doesn't, as constraints are "no divide", removing elements only relaxes constraints for the remaining set? 
                # Wait. 
                # Constraint: a does not divide sum of 3 others.
                # If we remove an element, the set of "others" shrinks. 
                # If a was NOT dividing any sum in the old set, it won't divide any sum in the subset.
                # So removing elements preserves validity of the remaining set.
                # Therefore, we only need to check if elem_in is valid against new_set_base.
                
                if _is_valid_addition_fast(elem_in, new_set_base):
                    current_set = new_set_base + [elem_in]
                    improved = True
                    break
        
        if not improved:
            # Stuck. 
            # Try a more aggressive perturbation? 
            # Remove 2, add 2? 
            # Or just break and rely on multiple restarts.
            # For now, break.
            break
            
    return current_set

def construct():
    start_time = time.time()
    
    best_set = []
    
    # Strategy: 
    # 1. Greedy with descending order.
    s1 = _greedy_construct(range(192, 0, -1))
    if len(s1) > len(best_set):
        best_set = s1[:]
        
    # 2. Local search on s1
    if time.time() - start_time < _DEADLINE - 10:
        s1_improved = _local_search(s1, 5.0)
        if len(s1_improved) > len(best_set):
            best_set = s1_improved[:]

    # 3. Randomized restarts
    # Use remaining time for multiple random greedy + local search
    time_per_restart = 2.0
    
    while time.time() - start_time < _DEADLINE - 5:
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        s = _greedy_construct(candidates)
        
        # Local search
        s_improved = _local_search(s, time_per_restart)
        
        if len(s_improved) > len(best_set):
            best_set = s_improved[:]
            
        # If we found 48, we can stop early? 
        # The max is 48. If we reach 48, we can't do better.
        if len(best_set) == 48:
            break

    return best_set

# EVOLVE-BLOCK-END
