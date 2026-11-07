"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    es = set()
    for e in P['edges']:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))
    
    adj = [set() for _ in range(n)]
    for u, v in es:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_mat = [[False]*n for _ in range(n)]
    for u, v in es:
        edge_mat[u][v] = True
        edge_mat[v][u] = True
    
    def is_clique(part):
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                if not edge_mat[part[i]][part[j]]:
                    return False
        return True
    
    def is_independent(part):
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                if edge_mat[part[i]][part[j]]:
                    return False
        return True
    
    def valid_add(part, v):
        if len(part) == 0:
            return True
        clique = True
        for u in part:
            if not edge_mat[u][v]:
                clique = False
                break
        if clique:
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    if not edge_mat[part[i]][part[j]]:
                        clique = False
                        break
                if not clique:
                    break
            if clique:
                return True
        indep = True
        for u in part:
            if edge_mat[u][v]:
                indep = False
                break
        if indep:
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    if edge_mat[part[i]][part[j]]:
                        indep = False
                        break
                if not indep:
                    break
            if indep:
                return True
        return False
    
    deadline = time.time() + 85
    
    orderings = [list(range(n))]
    deg_order = sorted(range(n), key=lambda x: -len(adj[x]))
    orderings.append(deg_order)
    deg_order_asc = sorted(range(n), key=lambda x: len(adj[x]))
    orderings.append(deg_order_asc)
    
    for k in range(1, n+1):
        if time.time() > deadline:
            break
        for order in orderings:
            if time.time() > deadline:
                break
            parts = [[] for _ in range(k)]
            
            def bt(idx):
                if time.time() > deadline:
                    return None
                if idx == len(order):
                    for p in parts:
                        if len(p) == 0:
                            return None
                        if not (is_clique(p) or is_independent(p)):
                            return None
                    return [list(p) for p in parts]
                
                v = order[idx]
                seen_empty = False
                for i in range(k):
                    if valid_add(parts[i], v):
                        if len(parts[i]) == 0:
                            if seen_empty:
                                continue
                            seen_empty = True
                        parts[i].append(v)
                        result = bt(idx + 1)
                        if result is not None:
                            return result
                        parts[i].pop()
                return None
            
            result = bt(0)
            if result is not None:
                return result
    
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            if valid_add(part, v):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
