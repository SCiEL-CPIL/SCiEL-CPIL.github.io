import random
from math import gcd
import time

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    def is_bad_triple(a, b, c):
        """Check if three numbers form a bad triple (all pairwise GCDs equal)."""
        return gcd_mat[a][b] == gcd_mat[b][c] == gcd_mat[a][c]
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without creating a bad triple."""
        # A bad triple containing x would be {x, u, v} where u, v are in current_set
        # and gcd(x,u) == gcd(x,v) == gcd(u,v)
        # So for each pair (u,v) in current_set, check if gcd(x,u) == gcd(x,v) == gcd(u,v)
        # Optimization: group current_set elements by their gcd with x
        # If any gcd value g appears at least twice among elements in current_set,
        # and those two elements also have gcd g with each other, we have a bad triple.
        
        # Actually, simpler: for each pair (u, v) in current_set, check if
        # gcd(x, u) == gcd(x, v) == gcd(u, v)
        # This is O(|current_set|^2) which is too slow for large sets.
        
        # Better: For each u in current_set, compute g_u = gcd(x, u).
        # Then for each v in current_set with v != u, if gcd(x, v) == g_u and gcd(u, v) == g_u,
        # we have a bad triple.
        # Still O(|current_set|^2).
        
        # Even better: For each u in current_set, let g_u = gcd(x, u).
        # We need to check if there exists v in current_set (v != u) such that
        # gcd(x, v) == g_u AND gcd(u, v) == g_u.
        # So for a fixed g, let S_g = {u in current_set : gcd(x, u) == g}.
        # For any u in S_g, we need to check if there's v in S_g (v != u) with gcd(u, v) == g.
        # But if gcd(u, v) == g and gcd(x, u) == g and gcd(x, v) == g, then g divides u, v, x.
        # Actually, gcd(u, v) == g means u/g and v/g are coprime.
        
        # Let's just do the O(|current_set|^2) check but with early termination.
        # For |current_set| up to ~80, this is 80^2 = 6400 checks, which is fast.
        
        elems = list(current_set)
        for i in range(len(elems)):
            u = elems[i]
            g_u = gcd_mat[x][u]
            for j in range(i + 1, len(elems)):
                v = elems[j]
                if gcd_mat[x][v] == g_u and gcd_mat[u][v] == g_u:
                    return False
        return True
    
    def greedy_build(order):
        s = set()
        for x in order:
            if can_add(x, s):
                s.add(x)
        return s
    
    def local_search(s, deadline):
        """
        Improve set s with local search moves.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any single element
            candidates = [x for x in range(1, N + 1) if x not in s]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    return s
                if can_add(x, s):
                    s.add(x)
                    improved = True
                    break
            
            if improved or time.time() > deadline:
                continue
            
            # 2. Try 1-for-1 swap: remove one, add one
            list_s = list(s)
            random.shuffle(list_s)
            addable = [x for x in range(1, N + 1) if x not in s]
            
            for u in list_s:
                if time.time() > deadline:
                    return s
                s_without_u = s - {u}
                
                # Shuffle addable for this u
                addable_shuffled = addable[:]
                random.shuffle(addable_shuffled)
                
                for x in addable_shuffled:
                    if time.time() > deadline:
                        return s
                    if can_add(x, s_without_u):
                        s.remove(u)
                        s.add(x)
                        improved = True
                        break
                
                if improved:
                    break
            
            # 3. Try 1-for-2 swap: remove one, add two (can increase size by 1)
            if not improved and time.time() < deadline:
                list_s = list(s)
                random.shuffle(list_s)
                max_removals_to_try = min(len(list_s), 15)
                
                for u in list_s[:max_removals_to_try]:
                    if time.time() > deadline:
                        return s
                    s_without_u = s - {u}
                    addable = [x for x in range(1, N + 1) if x not in s]
                    random.shuffle(addable)
                    
                    found_pair = False
                    for i in range(len(addable)):
                        if time.time() > deadline:
                            return s
                        x1 = addable[i]
                        if can_add(x1, s_without_u):
                            s_temp = s_without_u | {x1}
                            for j in range(i + 1, len(addable)):
                                x2 = addable[j]
                                if can_add(x2, s_temp):
                                    s.remove(u)
                                    s.add(x1)
                                    s.add(x2)
                                    found_pair = True
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                        
        return s
    
    best_set = set()
    
    # Strategy 1: Greedy with natural order
    s = greedy_build(list(range(1, N + 1)))
    s = local_search(s, deadline)
    if len(s) > len(best_set):
        best_set = s.copy()
    
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline:
        s = greedy_build(list(range(N, 0, -1)))
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Strategy 3: Greedy by fewest conflicts
    if time.time() < deadline:
        # Count how many bad triples each element is part of
        conflict_count = [0] * (N + 1)
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                for c in range(b + 1, N + 1):
                    if is_bad_triple(a, b, c):
                        conflict_count[a] += 1
                        conflict_count[b] += 1
                        conflict_count[c] += 1
        order = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
        s = greedy_build(order)
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Strategy 4: Random restarts
    while time.time() < deadline:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        s = greedy_build(random_order)
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    return list(best_set)
