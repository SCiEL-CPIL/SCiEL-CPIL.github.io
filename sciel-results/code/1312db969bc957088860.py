import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit (90s wall-clock, safety margin)
    
    N = 192
    
    def can_add_fast(S, x):
        """Check if adding x to S keeps validity. O(n^2) using residue maps."""
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: x does not divide sum of any 3 distinct elements in S
        if n >= 3:
            # Build residue map for S mod x
            res_map = {}
            for val in S_list:
                r = val % x
                if r not in res_map:
                    res_map[r] = []
                res_map[r].append(val)
            
            # For each pair (u,v) in S, check if there exists w in S\{u,v}
            # such that (u+v+w) % x == 0, i.e., w % x == (-u-v) % x
            for i in range(n):
                u = S_list[i]
                for j in range(i+1, n):
                    v = S_list[j]
                    target_r = (-u - v) % x
                    if target_r in res_map:
                        for w in res_map[target_r]:
                            if w != u and w != v:
                                return False
        
        # Check 2: No a in S divides (x + u + v) for any distinct u,v in S
        if n >= 2:
            for a in S_list:
                # Need: (x + u + v) % a != 0 for all distinct u,v in S
                # i.e., (u + v) % a != (-x) % a
                target = (-x) % a
                # Build set of pair sums mod a
                # Check all pairs
                found = False
                for i in range(n):
                    u = S_list[i]
                    for j in range(i+1, n):
                        v = S_list[j]
                        if (u + v) % a == target:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False
        
        return True
    
    def find_one_violation(S):
        """Find one violation if exists. Returns (a, u, v, w) or None."""
        S_list = list(S)
        n = len(S_list)
        
        for idx_a, a in enumerate(S_list):
            # T = S without a
            T = [x for x in S_list if x != a]
            if len(T) < 3:
                continue
            
            # Build residue map for T mod a
            res_map = {}
            for val in T:
                r = val % a
                if r not in res_map:
                    res_map[r] = []
                res_map[r].append(val)
            
            # For each pair (u,v) in T, check if there exists w in T\{u,v}
            # such that (u+v+w) % a == 0
            for i in range(len(T)):
                u = T[i]
                for j in range(i+1, len(T)):
                    v = T[j]
                    target_r = (-u - v) % a
                    if target_r in res_map:
                        for w in res_map[target_r]:
                            if w != u and w != v:
                                return (a, u, v, w)
        
        return None
    
    def is_valid(S):
        return find_one_violation(S) is None
    
    # Multiple initial strategies
    strategies = [
        lambda: set(range(96, 193)),    # (96, 192] - 96 elements
        lambda: set(range(80, 193)),    # (80, 192] - 112 elements
        lambda: set(range(64, 193)),    # (64, 192] - 128 elements
        lambda: set(range(48, 193)),    # (48, 192] - 144 elements
        lambda: set(range(1, 193)),     # all
    ]
    
    best_S = set()
    best_size = 0
    
    for strat_idx, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        
        S = strategy()
        
        # Repair phase: remove elements causing violations
        max_repair = 300
        for step in range(max_repair):
            if time.time() > deadline:
                break
            
            viol = find_one_violation(S)
            if viol is None:
                break
            
            a, u, v, w = viol
            # Remove the element that is the "divider" a, or one of the summands
            # Heuristic: remove the smallest element in the violation (likely causes more issues)
            candidates = [a, u, v, w]
            # Prefer removing a (the divider) as it's the "active" constraint
            # But also consider removing summands
            # Remove the one that appears in most violations - approximate by removing a
            S.remove(a)
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
    
    S = set(best_S)
    
    # Backfill phase: try to add removed elements
    removed = set(range(1, N+1)) - S
    removed_list = sorted(list(removed), reverse=True)  # Try larger ones first (safer)
    
    # Multiple passes of backfill
    for p in range(15):
        if time.time() > deadline:
            break
        progress = False
        for x in removed_list:
            if x in S:
                continue
            if can_add_fast(S, x):
                S.add(x)
                removed.discard(x)
                progress = True
        if not progress:
            break
    
    # Local search: try swapping elements
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Try removing one element and adding another
            S_list = list(S)
            if not S_list:
                break
            
            # Try each removal
            improved = False
            for i in range(len(S_list)):
                if time.time() > deadline:
                    break
                elem = S_list[i]
                S.remove(elem)
                
                # Try to add a better element
                added = False
                for x in removed_list:
                    if x in S:
                        continue
                    if can_add_fast(S, x):
                        S.add(x)
                        removed.discard(x)
                        added = True
                        break
                
                if not added:
                    # Put it back
                    S.add(elem)
                    removed.add(elem)
                else:
                    improved = True
                    break
            
            if not improved:
                break
    
    # Final validation and cleanup
    while time.time() < deadline - 2:
        viol = find_one_violation(S)
        if viol is None:
            break
        a, u, v, w = viol
        S.remove(a)
        removed.add(a)
    
    return list(S)
