"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _has_q3(adj):
    """Check if graph contains a Q3 subgraph using bitmask adjacency."""
    n = 16
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    def has_edge(u, v):
        return (adj_mask[u] >> v) & 1
    
    # Enumerate 4-cycles and try to extend to cubes
    for a in range(16):
        for b in range(a + 1, 16):
            if not has_edge(a, b):
                continue
            for c in range(b + 1, 16):
                if not has_edge(b, c) or has_edge(a, c):
                    continue
                for d in range(c + 1, 16):
                    if not has_edge(c, d) or has_edge(a, d) or has_edge(b, d):
                        continue
                    # 4-cycle a-b-c-d-a
                    a_neigh = [x for x in range(16) if x != a and has_edge(a, x) and x not in (b, d)]
                    if not a_neigh:
                        continue
                    b_neigh = [x for x in range(16) if x != b and has_edge(b, x) and x not in (a, c)]
                    if not b_neigh:
                        continue
                    c_neigh = [x for x in range(16) if x != c and has_edge(c, x) and x not in (b, d)]
                    if not c_neigh:
                        continue
                    d_neigh = [x for x in range(16) if x != d and has_edge(d, x) and x not in (c, a)]
                    if not d_neigh:
                        continue
                    
                    for e in a_neigh:
                        for f in b_neigh:
                            if not has_edge(e, f):
                                continue
                            for g in c_neigh:
                                if not has_edge(f, g):
                                    continue
                                for h in d_neigh:
                                    if not has_edge(g, h) or not has_edge(h, e):
                                        continue
                                    if len({a, b, c, d, e, f, g, h}) < 8:
                                        continue
                                    return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    best_edges = None
    best_count = 0
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 31)
        
        # Random greedy: shuffle edges, add if Q3-free
        adj = [set() for _ in range(n)]
        edges = []
        edge_set = set()
        
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for (u, v) in shuffled:
            if time.time() > deadline:
                break
            adj[u].add(v)
            adj[v].add(u)
            if _has_q3(adj):
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                edge_set.add((u, v))
                edges.append([u, v])
        
        # Local refinement: try to swap edges
        for _pass in range(3):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(edges)):
                if time.time() > deadline:
                    break
                u, v = edges[i]
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((min(u, v), max(u, v)))
                
                # Try adding random missing edges
                missing = [(a, b) for a, b in all_pairs if (a, b) not in edge_set]
                random.shuffle(missing)
                added = []
                for (x, y) in missing[:20]:
                    if len(added) >= 2:
                        break
                    adj[x].add(y)
                    adj[y].add(x)
                    if _has_q3(adj):
                        adj[x].discard(y)
                        adj[y].discard(x)
                    else:
                        edge_set.add((x, y))
                        added.append([x, y])
                
                if len(added) > 1:
                    for e in added:
                        edges.append(e)
                    edges.pop(i)
                    improved = True
                    break
            
            if not improved:
                break
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
    
    if best_edges is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return best_edges
# EVOLVE-BLOCK-END
