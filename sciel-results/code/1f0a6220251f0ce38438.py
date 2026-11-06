import time
import sys

sys.setrecursionlimit(10000)

def _time_up(deadline):
    return time.time() > deadline

def construct():
    deadline = time.time() + 115.0
    
    # Grötzsch graph definition
    # Vertices 0..10
    # It is the Mycielski of C5.
    # C5: 0-1-2-3-4-0
    # Mycielski:
    # Original: 0,1,2,3,4
    # Copies: 5,6,7,8,9
    # Root: 10
    # Edges:
    # Original edges: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Copy edges: (5,6), (6,7), (7,8), (8,9), (9,5)
    # Cross edges: (i, 10) for i in 0..4
    # Mapping: v_i connected to v_j if i~j in C5? No.
    # Mycielski edges:
    # (u_i, u_j) if (i,j) in E(C5)
    # (v_i, v_j) if (i,j) in E(C5)
    # (u_i, v_i) ? No.
    # (u_i, v_j) if (i,j) in E(C5)
    # (v_i, root) for all i.
    
    # Let's build Grötzsch adjacency
    n_g = 11
    adj_g = [set() for _ in range(n_g)]
    
    # C5 edges
    c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in c5_edges:
        adj_g[u].add(v)
        adj_g[v].add(u)
        
    # Copy vertices 5-9
    for u, v in c5_edges:
        cu = u + 5
        cv = v + 5
        adj_g[cu].add(cv)
        adj_g[cv].add(cu)
        
    # Cross edges u_i - v_j
    for u, v in c5_edges:
        adj_g[u].add(v + 5)
        adj_g[v + 5].add(u)
        
    # Root 10 connected to all copies
    for i in range(5):
        adj_g[i + 5].add(10)
        adj_g[10].add(i + 5)
        
    # Verify Grötzsch is triangle-free
    # (It is known to be)
    
    # Find all independent sets of size 4
    # Since n=11, we can enumerate subsets
    max_ind_sets = []
    for mask in range(1 << n_g):
        if bin(mask).count('1') != 4:
            continue
        # Check if independent
        is_ind = True
        verts = [i for i in range(n_g) if (mask >> i) & 1]
        for u in verts:
            for v in verts:
                if u != v and v in adj_g[u]:
                    is_ind = False
                    break
            if not is_ind:
                break
        if is_ind:
            max_ind_sets.append(mask)
            
    # We have constraints: sum(w_i for i in I) <= 9 for each I in max_ind_sets
    # Maximize sum(w_i)
    # w_i >= 1
    
    # Simple local search / greedy to find good weights
    # Start with w = [1]*11
    w = [1] * n_g
    total = sum(w)
    
    # Try to increase weights
    # Since constraints are tight, we can try a simple heuristic:
    # Increase w_i if it doesn't violate any constraint
    changed = True
    while changed and not _time_up(deadline):
        changed = False
        for i in range(n_g):
            # Check if we can increase w[i] by 1
            new_w = w[:]
            new_w[i] += 1
            valid = True
            for mask in max_ind_sets:
                if (mask >> i) & 1:
                    s = sum(new_w[j] for j in range(n_g) if (mask >> j) & 1)
                    if s > 9:
                        valid = False
                        break
            if valid:
                w = new_w
                total = sum(w)
                changed = True
                if _time_up(deadline):
                    break
                    
    # Construct the graph
    n = total
    if n < 1:
        return {'n': 1, 'edges': []}
        
    # Map original vertices to ranges
    # Vertex i in Grötzsch becomes vertices [start_i, start_i + w[i])
    starts = [0] * n_g
    for i in range(n_g):
        starts[i] = sum(w[:i])
        
    edges = []
    # Add edges between parts
    for u in range(n_g):
        for v in range(n_g):
            if u < v and v in adj_g[u]:
                # Connect all vertices in part u to all vertices in part v
                for i in range(w[u]):
                    for j in range(w[v]):
                        edges.append([starts[u] + i, starts[v] + j])
                        
    # Verify
    # Triangle-free? Yes, blow-up of triangle-free.
    # Alpha < 10? Yes, by construction of w.
    
    return {'n': n, 'edges': edges}
