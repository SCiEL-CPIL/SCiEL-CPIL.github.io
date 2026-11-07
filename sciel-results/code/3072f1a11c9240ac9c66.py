import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Deadline guard: 85s out of 90s limit
    deadline = time.time() + 85.0

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    all_mask = (1 << n) - 1
    coadj = [0] * n
    for u in range(n):
        coadj[u] = all_mask ^ adj[u] ^ (1 << u)

    CL = 0
    IND = 1
    BOTH = 2
    
    def can_add(v, part_mask, part_type):
        if part_type == CL:
            return (part_mask & ~adj[v]) == 0
        elif part_type == IND:
            return (part_mask & adj[v]) == 0
        elif part_type == BOTH:
            return True
        return False

    def get_new_type(v, part_mask, part_type):
        if part_type == BOTH:
            if (part_mask & adj[v]) != 0:
                return CL
            else:
                return IND
        return part_type

    best_solution = None
    
    def dfs_sol(unassigned, parts, k_rem):
        nonlocal best_solution, deadline
        if time.time() > deadline:
            return None
        
        if unassigned == 0:
            return parts[:]
        
        if k_rem == 0:
            return None
        
        # Pick vertex with fewest valid options among existing parts + new part
        v_best = -1
        min_options = n + 1
        v_bits = unassigned
        while v_bits:
            v = (v_bits & -v_bits).bit_length() - 1
            v_bits ^= (1 << v)
            opts = 0
            for i in range(len(parts)):
                p_mask, p_type = parts[i]
                if can_add(v, p_mask, p_type):
                    opts += 1
            if k_rem > 0:
                opts += 1
            if opts < min_options:
                min_options = opts
                v_best = v
            if min_options == 1:
                break
        
        v = v_best
        new_unassigned = unassigned ^ (1 << v)
        
        # Try adding v to existing parts
        for i in range(len(parts)):
            p_mask, p_type = parts[i]
            if can_add(v, p_mask, p_type):
                new_type = get_new_type(v, p_mask, p_type)
                new_mask = p_mask | (1 << v)
                
                new_parts = parts[:]
                new_parts[i] = (new_mask, new_type)
                
                res = dfs_sol(new_unassigned, new_parts, k_rem)
                if res is not None:
                    return res
        
        # Try creating a new part
        if k_rem > 0:
            new_parts = parts[:] + [(1 << v, BOTH)]
            res = dfs_sol(new_unassigned, new_parts, k_rem - 1)
            if res is not None:
                return res
        
        return None

    for k in range(1, n + 1):
        res = dfs_sol((1 << n) - 1, [], k)
        if res is not None:
            final_parts = []
            for mask, _ in res:
                verts = []
                for v in range(n):
                    if mask & (1 << v):
                        verts.append(v)
                final_parts.append(verts)
            return final_parts
        if time.time() > deadline:
            break

    # Fallback: greedy approach if deadline hit
    remaining = set(range(n))
    greedy_parts = []
    while remaining:
        best_set = None
        best_size = 0
        rem_list = list(remaining)
        for u in rem_list:
            cand = [u]
            for w in rem_list:
                if w == u:
                    continue
                if all(adj[w] & (1 << x) for x in cand):
                    cand.append(w)
            if len(cand) > best_size:
                best_size = len(cand)
                best_set = set(cand)
        for u in rem_list:
            cand = [u]
            for w in rem_list:
                if w == u:
                    continue
                if all(not (adj[w] & (1 << x)) for x in cand):
                    cand.append(w)
            if len(cand) > best_size:
                best_size = len(cand)
                best_set = set(cand)
        if best_set:
            greedy_parts.append(sorted(best_set))
            remaining -= best_set
        else:
            greedy_parts.append([min(remaining)])
            remaining.discard(min(remaining))
    return greedy_parts
