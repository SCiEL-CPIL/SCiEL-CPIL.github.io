"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Adjacency for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_clique(part):
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                if part[j] not in adj[part[i]]:
                    return False
        return True
    
    def is_independent(part):
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                if part[j] in adj[part[i]]:
                    return False
        return True
    
    def can_extend_clique(part, v):
        for u in part:
            if v not in adj[u]:
                return False
        return True
    
    def can_extend_independent(part, v):
        for u in part:
            if v in adj[u]:
                return False
        return True
    
    # Order vertices by degree (descending) for better pruning
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    assigned = [-1] * n  # assigned[v] = part index or -1
    parts = []  # list of lists
    best = [n]  # best number of parts found
    
    # Try to find a good initial solution using greedy
    def greedy_solution():
        parts_g = []
        assigned_g = [-1] * n
        for v in vertex_order:
            placed = False
            for pi, part in enumerate(parts_g):
                trial = part + [v]
                if is_clique(trial) or is_independent(trial):
                    part.append(v)
                    assigned_g[v] = pi
                    placed = True
                    break
            if not placed:
                parts_g.append([v])
                assigned_g[v] = len(parts_g) - 1
        return parts_g, assigned_g, len(parts_g)
    
    init_parts, init_assigned, init_count = greedy_solution()
    best[0] = init_count
    
    # Backtracking search
    def backtrack(idx):
        if idx == n:
            if len(parts) < best[0]:
                best[0] = len(parts)
            return
        
        v = vertex_order[idx]
        
        # Pruning: if we already have at least best[0] parts and can't do better
        if len(parts) >= best[0]:
            return
        
        # Try placing v in each existing part
        for pi, part in enumerate(parts):
            if can_extend_clique(part, v):
                part.append(v)
                assigned[v] = pi
                backtrack(idx + 1)
                part.pop()
                assigned[v] = -1
            elif can_extend_independent(part, v):
                part.append(v)
                assigned[v] = pi
                backtrack(idx + 1)
                part.pop()
                assigned[v] = -1
        
        # Try creating a new part with v
        if len(parts) + 1 < best[0]:
            parts.append([v])
            assigned[v] = len(parts) - 1
            backtrack(idx + 1)
            parts.pop()
            assigned[v] = -1
    
    # Initialize with greedy solution for bounding
    parts = [list(p) for p in init_parts]
    for v in range(n):
        assigned[v] = init_assigned[v]
    
    backtrack(0)
    
    # Reconstruct the best solution
    # We need to re-run to get the actual partition, not just the count
    # Let's do a final greedy with the known optimal count as bound
    def find_optimal_partition(target_count):
        parts_f = []
        assigned_f = [-1] * n
        
        def bt(idx):
            if idx == n:
                return len(parts_f) <= target_count
            v = vertex_order[idx]
            
            if len(parts_f) >= target_count:
                return False
            
            for pi, part in enumerate(parts_f):
                if can_extend_clique(part, v):
                    part.append(v)
                    assigned_f[v] = pi
                    if bt(idx + 1):
                        return True
                    part.pop()
                    assigned_f[v] = -1
                elif can_extend_independent(part, v):
                    part.append(v)
                    assigned_f[v] = pi
                    if bt(idx + 1):
                        return True
                    part.pop()
                    assigned_f[v] = -1
            
            if len(parts_f) + 1 <= target_count:
                parts_f.append([v])
                assigned_f[v] = len(parts_f) - 1
                if bt(idx + 1):
                    return True
                parts_f.pop()
                assigned_f[v] = -1
            
            return False
        
        if bt(0):
            return parts_f
        return None
    
    result = find_optimal_partition(best[0])
    if result is None:
        result = [list(p) for p in init_parts]
    
    # Convert to list of lists of Python ints
    final_result = []
    for part in result:
        final_result.append([int(v) for v in part])
    
    return final_result
# EVOLVE-BLOCK-END
