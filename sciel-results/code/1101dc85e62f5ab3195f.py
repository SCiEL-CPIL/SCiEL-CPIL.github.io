"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Start with K_{7,8}: A={0..6}, B={7..14}
    A = list(range(7))
    B = list(range(7, 15))
    
    edge_set = set()
    for a in A:
        for b in B:
            edge_set.add((a, b))
    
    def get_degrees(es, n):
        deg = [0] * n
        for (u, v) in es:
            deg[u] += 1
            deg[v] += 1
        return deg

    def is_valid(es, n):
        """Check if no simple three-edge path has endpoints of equal degree."""
        if not es:
            return True
        deg = get_degrees(es, n)
        adj = [set() for _ in range(n)]
        for (u, v) in es:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge (u,v): check if exists s in N(u)\{v}, t in N(v)\{u}, s!=t, deg[s]==deg[t]
        for (u, v) in es:
            nu = adj[u] - {v}
            nv = adj[v] - {u}
            if not nu or not nv:
                continue
            # Build degree->count map for nv
            deg_count = {}
            for t in nv:
                d = deg[t]
                deg_count[d] = deg_count.get(d, 0) + 1
            for s in nu:
                ds = deg[s]
                if ds in deg_count:
                    # Need t in nv with deg[t]==ds and t != s
                    if s in nv:
                        # s is common neighbor; need another t with same degree
                        if deg_count[ds] > 1:
                            return False
                    else:
                        # s not in nv, so any t works
                        return False
        return True

    def is_valid_fast(es, n):
        """Faster validity check using degree grouping."""
        if not es:
            return True
        deg = get_degrees(es, n)
        adj = [set() for _ in range(n)]
        for (u, v) in es:
            adj[u].add(v)
            adj[v].add(u)
        
        for (u, v) in es:
            nu = adj[u] - {v}
            nv = adj[v] - {u}
            if not nu or not nv:
                continue
            # Collect degrees in nv with counts
            nv_deg = {}
            for t in nv:
                d = deg[t]
                nv_deg[d] = nv_deg.get(d, 0) + 1
            for s in nu:
                ds = deg[s]
                if ds in nv_deg:
                    cnt = nv_deg[ds]
                    if s in nv:
                        if cnt > 1:
                            return False
                    else:
                        return False
        return True

    # Verify base is valid
    assert is_valid(edge_set, n), "Base K_{7,8} should be valid"
    
    best_edges = set(edge_set)
    best_count = len(best_edges)
    
    # Candidate edges: all missing edges (internal to A or internal to B)
    all_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                all_pairs.append((i, j))
    
    # Greedy: try adding edges one by one
    # Order candidates by potential benefit
    # A-internal edges first (smaller partition), then B-internal
    a_internal = [(i, j) for i in range(7) for j in range(i+1, 7)]
    b_internal = [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
    
    # Try all single additions first to find what works
    single_valid = []
    for e in a_internal + b_internal:
        test_es = set(edge_set)
        test_es.add(e)
        if is_valid_fast(test_es, n):
            single_valid.append(e)
    
    # Now greedily add from single_valid
    current = set(edge_set)
    for e in single_valid:
        if time.time() > deadline:
            break
        test_es = set(current)
        test_es.add(e)
        if is_valid_fast(test_es, n):
            current = test_es
    
    if len(current) > best_count:
        best_edges = current
        best_count = len(best_edges)
    
    # Try to add more edges that weren't in single_valid (they might work in combination)
    remaining = [e for e in a_internal + b_internal if e not in current]
    for e in remaining:
        if time.time() > deadline:
            break
        test_es = set(best_edges)
        test_es.add(e)
        if is_valid_fast(test_es, n):
            best_edges = test_es
            best_count = len(best_edges)
    
    # Try pairs of edges from remaining for better combinations
    if time.time() < deadline - 10:
        rem2 = [e for e in a_internal + b_internal if e not in best_edges]
        for i in range(len(rem2)):
            if time.time() > deadline:
                break
            e1 = rem2[i]
            for j in range(i+1, len(rem2)):
                if time.time() > deadline:
                    break
                e2 = rem2[j]
                test_es = set(best_edges)
                test_es.add(e1)
                test_es.add(e2)
                if is_valid_fast(test_es, n):
                    best_edges = test_es
                    best_count = len(best_edges)
    
    # Convert to required format
    result = [list(e) for e in best_edges]
    result.sort()
    
    # Ensure types are Python int
    for i, edge in enumerate(result):
        result[i] = [int(edge[0]), int(edge[1])]
    
    return result
# EVOLVE-BLOCK-END
