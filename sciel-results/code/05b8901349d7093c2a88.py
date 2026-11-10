import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    
    # Precompute all 8-subsets
    all_8subsets = list(itertools.combinations(range(n), 8))
    
    # Q3 structure: bipartite with parts A={0,3,5,6}, B={1,2,4,7}
    # Edges: each a in A connected to 3 in B
    q3_A = (0, 3, 5, 6)
    q3_B = (1, 2, 4, 7)
    q3_edges = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,4),(5,7),(5,2),(6,4),(6,7),(6,1)]
    
    def has_q3(edge_set, adj):
        """Fast check using precomputed adjacency"""
        for verts in all_8subsets:
            vset = set(verts)
            # Check if each vertex has degree >= 3 within vset
            for v in verts:
                if len(adj[v] & vset) < 3:
                    break
            else:
                # Check if induced subgraph contains Q3
                # Build local adjacency
                local_adj = {}
                for v in verts:
                    local_adj[v] = adj[v] & vset
                
                # Check 3-regular
                regular = True
                for v in verts:
                    if len(local_adj[v]) != 3:
                        regular = False
                        break
                if not regular:
                    continue
                
                # Check bipartite
                color = {}
                ok = True
                for start in verts:
                    if start not in color:
                        color[start] = 0
                        stack = [start]
                        while stack and ok:
                            x = stack.pop()
                            for y in local_adj[x]:
                                if y not in color:
                                    color[y] = 1 - color[x]
                                    stack.append(y)
                                elif color[y] == color[x]:
                                    ok = False
                                    break
                    if not ok:
                        break
                if ok and len(color) == 8:
                    return True
        return False
    
    # Use a different approach: start with dense random graph, remove edges to make Q3-free
    # Or: greedy add with fast Q3 check
    
    def greedy(seed_val):
        random.seed(seed_val)
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        adj = [set() for _ in range(n)]
        edge_set = set()
        
        for e in all_edges:
            u, v = e
            # Tentatively add
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(e)
            
            if has_q3(edge_set, adj):
                # Remove
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.discard(e)
        
        return edge_set
    
    best = set()
    best_count = 0
    seed = 1
    
    while time.time() < deadline:
        edges = greedy(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best = edges
        seed += 1
    
    return [[u, v] for u, v in best]
