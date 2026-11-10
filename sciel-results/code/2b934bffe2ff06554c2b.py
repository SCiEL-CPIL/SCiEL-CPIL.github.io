"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

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

    def build_structures(es, n):
        """Build adjacency list and degree array."""
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in es:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj

    def is_valid_fast(es, n):
        """Fast validity check: no 3-path s-u-v-t with deg[s]==deg[t], s!=t."""
        if not es:
            return True
        deg, adj = build_structures(es, n)
        
        # For each edge (u,v), check if there exist s in N(u)\{v}, t in N(v)\{u}, s!=t, deg[s]==deg[t]
        for (u, v) in es:
            nu = adj[u] - {v}
            nv = adj[v] - {u}
            if not nu or not nv:
                continue
            # Build set of degrees in nv, with count per degree
            nv_deg_counts = {}
            for t in nv:
                d = deg[t]
                nv_deg_counts[d] = nv_deg_counts.get(d, 0) + 1
            for s in nu:
                ds = deg[s]
                if ds in nv_deg_counts:
                    cnt = nv_deg_counts[ds]
                    if s in nv:
                        # s is also in nv; need another t with same degree
                        if cnt > 1:
                            return False
                    else:
                        return False
        return True

    # Verify base is valid
    assert is_valid_fast(edge_set, n), "Base K_{7,8} should be valid"
    
    best_edges = set(edge_set)
    best_count = len(best_edges)
    
    # Candidate edges: all missing edges (internal to A or internal to B)
    a_internal = [(i, j) for i in range(7) for j in range(i+1, 7)]
    b_internal = [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
    all_candidates = a_internal + b_internal
    
    # Greedy: try adding edges one by one, sorted by vertex pair
    current = set(edge_set)
    
    # Multiple passes to find edges that work after others are added
    for pass_num in range(10):
        if time.time() > deadline - 5:
            break
        progress = False
        for e in all_candidates:
            if time.time() > deadline - 5:
                break
            if e in current:
                continue
            test_es = set(current)
            test_es.add(e)
            if is_valid_fast(test_es, n):
                current = test_es
                progress = True
        if not progress:
            break
    
    if len(current) > best_count:
        best_edges = current
        best_count = len(best_edges)
    
    # Local search: try removing one edge and adding two
    if time.time() < deadline - 10:
        current = set(best_edges)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try each existing edge for removal
            for e_remove in list(current):
                if time.time() > deadline - 5:
                    break
                current_without = set(current)
                current_without.discard(e_remove)
                
                # Try adding pairs of missing edges
                missing = [e for e in all_candidates if e not in current_without]
                for i in range(len(missing)):
                    if time.time() > deadline - 5:
                        break
                    e1 = missing[i]
                    for j in range(i+1, len(missing)):
                        if time.time() > deadline - 5:
                            break
                        e2 = missing[j]
                        test_es = set(current_without)
                        test_es.add(e1)
                        test_es.add(e2)
                        if is_valid_fast(test_es, n):
                            current = test_es
                            improved = True
                            break
                    if improved:
                        break
            if len(current) > best_count:
                best_edges = current
                best_count = len(best_edges)
    
    # Final greedy pass
    if time.time() < deadline - 3:
        current = set(best_edges)
        for e in all_candidates:
            if time.time() > deadline - 3:
                break
            if e in current:
                continue
            test_es = set(current)
            test_es.add(e)
            if is_valid_fast(test_es, n):
                current = test_es
        if len(current) > best_count:
            best_edges = current
            best_count = len(best_edges)
    
    # Convert to required format
    result = [list(e) for e in best_edges]
    result.sort()
    
    # Ensure types are Python int
    for i, edge in enumerate(result):
        result[i] = [int(edge[0]), int(edge[1])]
    
    return result
# EVOLVE-BLOCK-END
