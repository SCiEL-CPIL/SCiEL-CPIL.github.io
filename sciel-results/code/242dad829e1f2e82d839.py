import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    n = 20
    num_colours = 20
    
    # Precompute all 4-cycles
    cycles = []
    vertices = list(range(n))
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
    
    # For fast lookup: map edge (u,v) with u<v to an index
    edge_index = {}
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            idx = len(all_edges)
            edge_index[(u, v)] = idx
            all_edges.append((u, v))
    
    # For each edge, find all 4-cycles that contain it
    edge_cycles = [[] for _ in range(len(all_edges))]
    for cycle_idx, cycle in enumerate(cycles):
        for edge in cycle:
            u, v = edge
            if u > v:
                u, v = v, u
            ei = edge_index.get((u, v), -1)
            if ei != -1:
                edge_cycles[ei].append(cycle_idx)
    
    def count_violations(adj):
        """Count the number of 4-cycles that are not rainbow."""
        violations = 0
        for cycle in cycles:
            e1, e2, e3, e4 = cycle
            c1 = adj[e1[0], e1[1]]
            c2 = adj[e2[0], e2[1]]
            c3 = adj[e3[0], e3[1]]
            c4 = adj[e4[0], e4[1]]
            
            if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                continue
            
            if len({c1, c2, c3, c4}) < 4:
                violations += 1
        return violations
    
    def is_valid(adj):
        return count_violations(adj) == 0
    
    def count_edges(adj):
        return int(np.sum(adj != -1)) // 2
    
    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added with some colour without creating violations."""
        if adj[u, v] != -1:
            return None
        
        # Find forbidden colours due to proper edge colouring
        forbidden_proper = set()
        for w in range(n):
            if w != u and w != v and adj[u, w] != -1:
                forbidden_proper.add(adj[u, w])
            if w != u and w != v and adj[v, w] != -1:
                forbidden_proper.add(adj[v, w])
        
        # Find forbidden colours due to 4-cycle constraints
        forbidden_cycle = set()
        
        # 4-cycles containing (u,v): u-v-w-x-u
        # w neighbor of v, x neighbor of u, (w,x) exists
        for w in range(n):
            if w == u or w == v or adj[v, w] == -1:
                continue
            c_vw = adj[v, w]
            for x in range(n):
                if x == u or x == v or x == w or adj[u, x] == -1 or adj[w, x] == -1:
                    continue
                c_xu = adj[u, x]
                c_wx = adj[w, x]
                
                # Check if the 3 existing edges have distinct colours
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    # This would create a non-rainbow 4-cycle if we add (u,v)
                    # Actually, we need to check: does adding (u,v) with colour c create a violation?
                    # The 4-cycle is u-v-w-x-u with edges (u,v), (v,w), (w,x), (x,u)
                    # For it to be rainbow, c, c_vw, c_wx, c_xu must be distinct
                    # So c cannot be c_vw, c_wx, or c_xu
                    forbidden_cycle.add(c_vw)
                    forbidden_cycle.add(c_wx)
                    forbidden_cycle.add(c_xu)
        
        all_forbidden = forbidden_proper.union(forbidden_cycle)
        
        for c in range(num_colours):
            if c not in all_forbidden:
                return c
        return None
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) with a valid colour. Returns True if successful."""
        if adj[u, v] != -1:
            return False
        c = can_add_edge(adj, u, v)
        if c is not None:
            adj[u, v] = c
            adj[v, u] = c
            return True
        return False
    
    def remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        if adj[u, v] != -1:
            adj[u, v] = -1
            adj[v, u] = -1
            return True
        return False
    
    def swap_colours(adj, e1, e2):
        """Swap colours of two edges. e1, e2 are (u,v) tuples."""
        u1, v1 = e1
        u2, v2 = e2
        if adj[u1, v1] == -1 or adj[u2, v2] == -1:
            return False
        c1 = adj[u1, v1]
        c2 = adj[u2, v2]
        if c1 == c2:
            return False
        
        # Check if swap is valid (proper edge colouring)
        # For e1=(u1,v1): after swap, colour becomes c2
        # c2 must not conflict with edges incident to u1 or v1 (other than e2 if it shares a vertex)
        # This is complex; let's just do the swap and check violations
        adj[u1, v1] = c2
        adj[v1, u1] = c2
        adj[u2, v2] = c1
        adj[v2, u2] = c1
        
        return True
    
    # Strategy: Start with greedy, then use local search with edge addition/removal
    best_adj = None
    best_edges = 0
    
    all_edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Multiple restarts with different strategies
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Initialize with greedy
        adj = np.full((n, n), -1, dtype=int)
        edges = all_edges_list[:]
        random.shuffle(edges)
        
        for u, v in edges:
            try_add_edge(adj, u, v)
        
        # Second pass
        for u, v in edges:
            try_add_edge(adj, u, v)
        
        edge_count = count_edges(adj)
        
        # Local search: try to add more edges by removing and re-adding
        # We'll use a simple hill-climbing with restarts
        for step in range(200):
            if time.time() > deadline:
                break
            
            # Try to add a random missing edge
            missing_edges = [(u, v) for u, v in all_edges_list if adj[u, v] == -1]
            if not missing_edges:
                break
            
            u, v = random.choice(missing_edges)
            if try_add_edge(adj, u, v):
                edge_count = count_edges(adj)
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_adj = adj.copy()
            else:
                # Try to remove a random existing edge and then add the missing one
                existing_edges = [(u, v) for u, v in all_edges_list if adj[u, v] != -1]
                if existing_edges:
                    u_rem, v_rem = random.choice(existing_edges)
                    remove_edge(adj, u_rem, v_rem)
                    if try_add_edge(adj, u, v):
                        # Also try to re-add the removed edge
                        try_add_edge(adj, u_rem, v_rem)
                        edge_count = count_edges(adj)
                        if edge_count > best_edges:
                            best_edges = edge_count
                            best_adj = adj.copy()
                    else:
                        # Re-add the removed edge
                        try_add_edge(adj, u_rem, v_rem)
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
    
    # Final verification
    if best_adj is not None and is_valid(best_adj):
        adj = best_adj
    else:
        # Fallback: greedy result
        adj = np.full((n, n), -1, dtype=int)
        edges = all_edges_list[:]
        random.shuffle(edges)
        for u, v in edges:
            try_add_edge(adj, u, v)
        for u, v in edges:
            try_add_edge(adj, u, v)
    
    # If still invalid, use a simple valid construction
    if not is_valid(adj):
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
    
    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] != -1:
                edges_list.append([u, v, int(adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
