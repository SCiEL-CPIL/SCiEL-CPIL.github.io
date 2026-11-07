"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def _check_validity_fast(adj, deg):
    """Optimized validity check: for each edge (u,v), check all s in adj[u]\{v}, t in adj[v]\{u}."""
    n = len(adj)
    for u in range(n):
        adj_u = adj[u]
        for v in adj_u:
            if v <= u:
                continue
            adj_v = adj[v]
            for s in adj_u:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj_v:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _check_validity(adj, deg):
    """Check validity given adjacency lists and degrees. Returns True if valid."""
    return _check_validity_fast(adj, deg)

def _build_adj_deg(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    return adj, deg

def _try_add_edge(adj, deg, u, v):
    """Try adding edge (u,v). Returns True if valid, False otherwise."""
    adj[u].add(v)
    adj[v].add(u)
    deg[u] += 1
    deg[v] += 1
    valid = _check_validity(adj, deg)
    if not valid:
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
    return valid

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    
    # Base: K_{7,8}
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    base_edges = []
    for a in part_a:
        for b in part_b:
            base_edges.append((a, b))
    
    base_adj, base_deg = _build_adj_deg(base_edges, n)
    assert _check_validity(base_adj, base_deg)
    
    best_edges = base_edges[:]
    best_count = len(best_edges)
    
    # Candidate edges to add: within A, within B
    candidates = []
    for i in range(7):
        for j in range(i + 1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i + 1, 15):
            candidates.append((i, j))
    
    # Greedy with multiple random restarts
    random.seed(12345)
    
    # Multiple greedy passes with different orderings
    for trial in range(2000):
        if time.time() > deadline - 5:
            break
        
        cur_edges = best_edges[:]
        cur_adj, cur_deg = _build_adj_deg(cur_edges, n)
        
        cands = candidates[:]
        random.shuffle(cands)
        
        for ce in cands:
            if time.time() > deadline - 5:
                break
            u, v = ce
            if _try_add_edge(cur_adj, cur_deg, u, v):
                cur_edges.append(ce)
        
        if len(cur_edges) > best_count:
            best_edges = cur_edges[:]
            best_count = len(best_edges)
    
    # Try different bipartitions too
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        # Random bipartition
        parts = list(range(n))
        random.shuffle(parts)
        k = random.randint(5, 10)
        pa = parts[:k]
        pb = parts[k:]
        
        te = []
        for a in pa:
            for b in pb:
                te.append((a, b))
        
        ta, td = _build_adj_deg(te, n)
        if _check_validity(ta, td):
            if len(te) > best_count:
                best_edges = te[:]
                best_count = len(te)
            
            # Try to add intra-part edges
            cands2 = []
            for i in range(len(pa)):
                for j in range(i+1, len(pa)):
                    cands2.append((pa[i], pa[j]))
            for i in range(len(pb)):
                for j in range(i+1, len(pb)):
                    cands2.append((pb[i], pb[j]))
            
            random.shuffle(cands2)
            for ce in cands2:
                if time.time() > deadline - 5:
                    break
                u, v = ce
                if _try_add_edge(ta, td, u, v):
                    te.append(ce)
            
            if len(te) > best_count:
                best_edges = te[:]
                best_count = len(te)
    
    # Final result
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
