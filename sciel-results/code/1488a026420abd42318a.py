"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 85.0
    
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute clique/indep for all masks
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        verts = [i for i in range(n) if (mask >> i) & 1]
        if len(verts) <= 1:
            is_clique[mask] = True
            is_indep[mask] = True
            continue
        clique_ok = True
        indep_ok = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] not in adj[verts[i]]:
                    clique_ok = False
                else:
                    indep_ok = False
                if not clique_ok and not indep_ok:
                    break
            if not clique_ok and not indep_ok:
                break
        is_clique[mask] = clique_ok
        is_indep[mask] = indep_ok
    
    all_mask = (1 << n) - 1
    best_count = n + 1
    best_partition = None
    
    # Greedy initialization
    def greedy():
        nonlocal best_count, best_partition
        parts = []  # list of (mask, is_clique_type)
        for v in range(n):
            placed = False
            for i in range(len(parts)):
                m, ct = parts[i]
                nm = m | (1 << v)
                if ct and is_clique[nm]:
                    parts[i] = (nm, True)
                    placed = True
                    break
                elif not ct and is_indep[nm]:
                    parts[i] = (nm, False)
                    placed = True
                    break
            if not placed:
                parts.append((1 << v, True))
        cnt = len(parts)
        if cnt < best_count:
            best_count = cnt
            best_partition = parts
        return cnt
    
    greedy()
    
    # Branch and bound
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    def dfs(assigned_mask, parts):
        nonlocal best_count, best_partition, deadline
        
        if time.time() > deadline:
            return
        
        if len(parts) >= best_count:
            return
        
        if assigned_mask == all_mask:
            if len(parts) < best_count:
                best_count = len(parts)
                best_partition = [(m, ct) for m, ct in parts]
            return
        
        # Next vertex to place
        next_v = None
        for v in order:
            if not (assigned_mask >> v) & 1:
                next_v = v
                break
        if next_v is None:
            return
        
        # Try existing parts
        for i in range(len(parts)):
            m, ct = parts[i]
            nm = m | (1 << next_v)
            if ct:
                if is_clique[nm]:
                    parts[i] = (nm, True)
                    dfs(assigned_mask | (1 << next_v), parts)
                    parts[i] = (m, True)
            else:
                if is_indep[nm]:
                    parts[i] = (nm, False)
                    dfs(assigned_mask | (1 << next_v), parts)
                    parts[i] = (m, False)
        
        # New part
        if len(parts) + 1 < best_count:
            parts.append((1 << next_v, True))
            dfs(assigned_mask | (1 << next_v), parts)
            parts.pop()
    
    dfs(0, [])
    
    # Build result
    if best_partition is not None:
        result = []
        for m, ct in best_partition:
            verts = [i for i in range(n) if (m >> i) & 1]
            result.append(verts)
        return result
    else:
        return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
