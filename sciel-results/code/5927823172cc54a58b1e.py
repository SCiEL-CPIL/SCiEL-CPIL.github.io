import time
import random

def _precompute_conflicts(n):
    # potential_conflicts[a] = list of (b, c) such that a < b <= c and a | (b+c)
    potential = {a: [] for a in range(1, n + 1)}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b, n + 1):
                if (b + c) % a == 0:
                    potential[a].append((b, c))
    return potential

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    potential = _precompute_conflicts(n)
    
    # 1. Greedy
    S = set()
    for x in range(n, 0, -1):
        valid = True
        for b, c in potential[x]:
            if b in S and c in S:
                valid = False
                break
        if valid:
            S.add(x)
            
    best_set = set(S)
    best_size = len(S)
    
    # 2. Local Search
    while time.time() < deadline:
        improved = False
        
        # A. Add
        for v in range(1, n + 1):
            if v in S:
                continue
            valid = True
            for b, c in potential[v]:
                if b in S and c in S:
                    valid = False
                    break
            if valid:
                S.add(v)
                improved = True
                break
        
        if improved:
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            continue
        
        # B. Swap
        # Find v not in S that is blocked by exactly one "critical" element u
        # i.e., all conflicts (b,c) for v contain u.
        
        # To speed up, we can just try all v and find the remover.
        best_swap = None
        max_adds = 0
        
        for v in range(1, n + 1):
            if v in S:
                continue
            # Find conflicts in S
            active_conflicts = []
            for b, c in potential[v]:
                if b in S and c in S:
                    active_conflicts.append((b, c))
            
            if not active_conflicts:
                # Should have been added in A
                continue
                
            # Find common element
            common = set(active_conflicts[0])
            for p in active_conflicts[1:]:
                common.intersection_update(set(p))
                if not common:
                    break
            
            if common:
                # Each u in common can remove v's conflicts
                # We want to group by u
                # We can just pick one u, say the one that allows most adds?
                # For now, just record that u allows v
                # We need to count how many v's are allowed by the same u
                
                pass # Logic needs to be: for each u, count v's.
        
        # Redo Swap logic correctly
        remover_counts = {}
        for v in range(1, n + 1):
            if v in S:
                continue
            active_conflicts = []
            for b, c in potential[v]:
                if b in S and c in S:
                    active_conflicts.append((b, c))
            
            if not active_conflicts:
                continue
                
            common = set(active_conflicts[0])
            for p in active_conflicts[1:]:
                common.intersection_update(set(p))
                if not common:
                    break
            
            if common:
                for u in common:
                    if u not in remover_counts:
                        remover_counts[u] = 0
                    remover_counts[u] += 1
                    # Store v for this u?
                    # We need to know which v's to add.
                    
        # We need to know the list of v's for each u.
        remover_to_adds = {}
        for v in range(1, n + 1):
            if v in S:
                continue
            active_conflicts = []
            for b, c in potential[v]:
                if b in S and c in S:
                    active_conflicts.append((b, c))
            
            if not active_conflicts:
                continue
                
            common = set(active_conflicts[0])
            for p in active_conflicts[1:]:
                common.intersection_update(set(p))
                if not common:
                    break
            
            if common:
                for u in common:
                    if u not in remover_to_adds:
                        remover_to_adds[u] = []
                    remover_to_adds[u].append(v)
        
        best_swap = None
        max_adds = 0
        for u, adds in remover_to_adds.items():
            if len(adds) > max_adds:
                max_adds = len(adds)
                best_swap = (u, adds)
                
        if best_swap and max_adds > 1:
            u, adds = best_swap
            S.remove(u)
            for v in adds:
                S.add(v)
            improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
        
        if not improved:
            break
            
    return sorted(best_set)
