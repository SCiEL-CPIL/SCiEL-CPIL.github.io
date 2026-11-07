import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 15
    # Partition: A = 0..6, B = 7..14
    # Base: K_{7,8}
    
    def build_base():
        edges = []
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u in range(7):
            for v in range(7, 15):
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        return edges, adj, deg

    def can_add(u, v, adj, deg):
        # Check if adding (u,v) to graph with adj/deg is valid
        # New degrees: d'(u)=d(u)+1, d'(v)=d(v)+1, others same.
        
        # Case 1: s-u-v-t (middle edge)
        # s in N(u)\{v}, t in N(v)\{u,s}
        # Check d(s) != d(t)
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
        
        # Case 2: u-v-x-y (first edge)
        # x in N(v)\{u}, y in N(x)\{u,v}
        # Check d'(u) != d'(y) => d(u)+1 != d(y)
        du_new = deg[u] + 1
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[y] == du_new:
                    return False
        
        # Case 3: x-y-v-u (last edge)
        # x in N(u)\{v}, y in N(x)\{u,v}
        # Check d'(x) != d'(u) => d(x) != d(u)+1
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[x] == du_new:
                    return False
            
        return True

    def add_edge(u, v, edges, adj, deg):
        edges.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def remove_edge(u, v, edges, adj, deg):
        edges.remove((u, v))
        adj[u].remove(v)
        adj[v].remove(u)
        deg[u] -= 1
        deg[v] -= 1

    # Generate all possible intra-part candidates
    candidates = []
    for i in range(7):
        for j in range(i+1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidates.append((i, j))
    
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
            
    best_edges = []
    best_count = 0
    
    # Multiple randomized restarts
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        base_edges, adj, deg = build_base()
        current_edges = list(base_edges)
        current_count = len(current_edges)
        
        # Greedy phase: try to add edges in random order
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        for u, v in shuffled_candidates:
            if time.time() > deadline:
                break
            if can_add(u, v, adj, deg):
                add_edge(u, v, current_edges, adj, deg)
                current_count += 1
        
        # Local search phase: try swaps (remove one, add another)
        improved = True
        max_swap_iters = 50
        for _ in range(max_swap_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try removing each current edge and adding a non-edge
            edge_list = current_edges[:]
            random.shuffle(edge_list)
            
            for rem_u, rem_v in edge_list:
                if time.time() > deadline:
                    break
                if (rem_u, rem_v) not in adj[rem_u]:
                    continue  # edge was already removed in this iteration
                
                # Try adding various non-edges
                shuffled_candidates = candidates[:]
                random.shuffle(shuffled_candidates)
                
                for add_u, add_v in shuffled_candidates:
                    if add_u == rem_u and add_v == rem_v:
                        continue
                    if (add_u, add_v) in adj[add_u]:
                        continue
                    if (add_v, add_u) in adj[add_v]:
                        continue
                    
                    # Temporarily remove rem edge
                    remove_edge(rem_u, rem_v, current_edges, adj, deg)
                    
                    # Check if we can add the new edge
                    if can_add(add_u, add_v, adj, deg):
                        add_edge(add_u, add_v, current_edges, adj, deg)
                        improved = True
                        break
                    else:
                        # Restore
                        add_edge(rem_u, rem_v, current_edges, adj, deg)
                
                if improved:
                    break
            
            if not improved:
                break
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges[:]
            
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
