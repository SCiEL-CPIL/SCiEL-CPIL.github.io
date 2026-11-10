import time
import random

N = 512

# Precompute all 4-term arithmetic progressions
ap_list = []
vertex_to_aps = [[] for _ in range(N + 1)]

for a in range(1, N + 1):
    for d in range(1, N + 1):
        b = a + d
        c = b + d
        e = c + d
        if e > N:
            break
        idx = len(ap_list)
        ap_list.append((a, b, c, e))
        for v in (a, b, c, e):
            vertex_to_aps[v].append(idx)

def greedy_construct():
    """
    Generate a good initial solution using a greedy heuristic with restarts.
    """
    best_S = set()
    start = time.time()
    while time.time() - start < 5:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = set()
        for x in order:
            can_add = True
            for idx in vertex_to_aps[x]:
                a, b, c, d = ap_list[idx]
                others = [v for v in (a, b, c, d) if v != x]
                if all(v in S for v in others):
                    can_add = False
                    break
            if can_add:
                S.add(x)
        
        if len(S) > len(best_S):
            best_S = S.copy()
            
    return best_S

def construct():
    global best_size, best_set, start_time, ap_list, vertex_to_aps
    
    start_time = time.time()
    best_size = 0
    best_set = set()
    
    # 1. Get a good initial solution
    init_S = greedy_construct()
    best_size = len(init_S)
    best_set = init_S.copy()
    
    all_vertices = set(range(1, N + 1))
    
    # Precompute degree for branching heuristic
    # degree[v] = number of APs v is in
    degree = [0] * (N + 1)
    for v in range(1, N + 1):
        degree[v] = len(vertex_to_aps[v])
        
    def search(remaining, current):
        nonlocal best_size, best_set
        
        # Time check
        if time.time() - start_time > 58:
            return
            
        # Bound
        if len(current) + len(remaining) <= best_size:
            return
            
        # Propagation
        # Find APs with 3 in current, 1 in remaining.
        # The 1 in remaining MUST be excluded.
        changed = True
        while changed:
            changed = False
            for idx in range(len(ap_list)):
                a, b, c, d = ap_list[idx]
                count_in_current = 0
                remaining_in_ap = []
                
                for v in (a, b, c, d):
                    if v in current:
                        count_in_current += 1
                    elif v in remaining:
                        remaining_in_ap.append(v)
                
                if count_in_current == 3 and len(remaining_in_ap) == 1:
                    v_to_remove = remaining_in_ap[0]
                    remaining.discard(v_to_remove)
                    changed = True
                    break
                
                if count_in_current == 4:
                    return

        if len(remaining) == 0:
            if len(current) > best_size:
                best_size = len(current)
                best_set = current.copy()
            return
        
        if len(current) + len(remaining) <= best_size:
            return

        # Branching
        # Find an AP with vertices in remaining
        # Heuristic: Pick AP with max number of vertices in remaining, 
        # and among those, pick the one with highest "degree" sum?
        
        target_ap = -1
        max_rem_count = 0
        max_degree_sum = 0
        
        for idx in range(len(ap_list)):
            a, b, c, d = ap_list[idx]
            rem_count = 0
            deg_sum = 0
            for v in (a, b, c, d):
                if v in remaining:
                    rem_count += 1
                    deg_sum += degree[v]
            
            if rem_count > 0:
                if rem_count > max_rem_count or (rem_count == max_rem_count and deg_sum > max_degree_sum):
                    max_rem_count = rem_count
                    max_degree_sum = deg_sum
                    target_ap = idx
                    
        if target_ap == -1:
            if len(current) + len(remaining) > best_size:
                best_size = len(current) + len(remaining)
                best_set = current.union(remaining)
            return
            
        a, b, c, d = ap_list[target_ap]
        candidates = [v for v in (a, b, c, d) if v in remaining]
        
        # Sort candidates by degree descending (exclude high degree first to prune more?)
        candidates.sort(key=lambda x: degree[x], reverse=True)
        
        for v in candidates:
            new_remaining = remaining.copy()
            new_remaining.discard(v)
            search(new_remaining, current)
            
            if time.time() - start_time > 58:
                return

    search(all_vertices, set())
    
    return sorted(best_set)
