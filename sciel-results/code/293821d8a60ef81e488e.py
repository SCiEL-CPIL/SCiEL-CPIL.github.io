import time
import random
import itertools

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy:
    1. Use Simulated Annealing to find a dense C4-free graph.
    2. Start with a random graph or a greedy construction result.
    3. Moves: Add a random missing edge, or remove a random existing edge.
    4. Acceptance: 
       - If the move maintains C4-freeness: Accept if it increases edge count (always) 
         or with probability exp(-delta/T) if it decreases (to escape local minima).
       - Actually, we want to MAXIMIZE edges. So "energy" = -num_edges.
       - We only care about valid states (C4-free).
       - If a move results in a C4, it is invalid. We can reject it immediately.
       - To escape local minima where no single edge addition is valid, we allow 
         temporary edge removals (which reduce edge count) to open up space for 
         better configurations.
       
    5. Since C4-free implies rainbow 4-cycles (vacuously), we just enforce C4-free.
    """
    n = 20
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(all_edges)
    
    # Helper: Check if adding edge (u,v) to current graph creates a C4.
    # A graph is C4-free iff no two vertices have >1 common neighbor.
    # Adding (u,v) creates a C4 if:
    # 1. u and v already have >1 common neighbors in the current graph.
    # 2. OR, adding (u,v) causes some other pair (x,y) to have >1 common neighbors.
    #    This happens if:
    #    - x is a neighbor of u, y is a neighbor of v, and x-y is an edge? 
    #      No, the standard check is simpler:
    #      Adding (u,v) creates a C4 iff there exist distinct x,y such that 
    #      (u,x), (x,y), (y,v) are edges. 
    #      This is equivalent to: The number of length-2 paths between u and v 
    #      in the graph WITHOUT edge (u,v) is >= 1? No, that's a triangle if we add (u,v)? 
    #      Wait. If there is a path u-x-y-v, adding (v,u) creates u-v-y-x-u? 
    #      Edges: (u,v) [new], (v,y), (y,x), (x,u). Yes, that is a C4.
    #      So, adding (u,v) creates a C4 iff there is a path of length 3 between u and v.
    #      Path of length 3: u-x-y-v.
    #      This requires (u,x), (x,y), (y,v) to be in E.
    #      So, we need to check if there exist x, y such that x in N(u), y in N(v), and (x,y) in E.
    #      Note: x and y must be distinct from u,v and x!=y.
    
    def can_add_edge(u, v, adj):
        # Check if adding (u,v) creates a C4.
        # We need to check if there is a path u-x-y-v in the current graph.
        # x is neighbor of u, y is neighbor of v, and (x,y) is an edge.
        # Also x != y.
        # If x == y, it's a path u-x-v (length 2). Adding (u,v) creates a triangle, not a C4.
        # So we only care about x != y.
        
        # Optimization: 
        # If u and v have a common neighbor z, that's fine (triangle).
        # But if they have two common neighbors z1, z2, then u-z1-v-z2-u is a C4?
        # Edges: (u,z1), (z1,v), (v,z2), (z2,u). Yes.
        # So if |N(u) ∩ N(v)| >= 2, we cannot add (u,v).
        
        common = adj[u].intersection(adj[v])
        if len(common) >= 2:
            return False
            
        # Check for path of length 3: u-x-y-v with x!=y.
        # x in N(u), y in N(v), (x,y) in E.
        # Iterate over neighbors of u.
        for x in adj[u]:
            if x == v: continue # v is not in adj[u] yet
            # For each x, check if any neighbor y of x is in N(v) and y != x?
            # Actually, if y is in N(x) and y is in N(v), and y != x (since x,y distinct vertices in path),
            # and x != v (checked), y != u (y is in N(v), if y==u then (u,v) edge? no, we are checking before adding).
            # Wait, if y == u, then (x,u) is edge (since x in N(u)) and (u,v) is not edge yet.
            # The path is u-x-u-v? No, simple path.
            # The cycle would be u-v-y-x-u.
            # Edges: (u,v) [new], (v,y), (y,x), (x,u).
            # If y == u, edges: (u,v), (v,u) [same], (u,x), (x,u) [same]. This is a 2-cycle (multi-edge) or triangle?
            # In simple graph, u-v-y-x-u with y=u becomes u-v-u-x-u, which is not a simple cycle.
            # So we require x, y distinct and not u, v.
            
            for y in adj[x]:
                if y == x: continue
                if y == u: continue
                if y == v: continue
                if y in adj[v]:
                    # We found u-x-y-v.
                    # Check if this forms a C4 with (u,v).
                    # Cycle: u-v-y-x-u.
                    # Vertices: u, v, y, x.
                    # Are they distinct? 
                    # u != v (given).
                    # x != u (x in N(u), simple graph).
                    # x != v (checked above? No, x is in adj[u]. If x==v, v in adj[u], but we are adding (u,v) so v not in adj[u] yet. So x!=v is guaranteed).
                    # y != v (y in adj[x], if y==v then v in adj[x]. Is it possible? Yes. If y==v, then path is u-x-v. Length 2. Adding (u,v) makes triangle. Not C4. So we must exclude y==v).
                    # y != u (checked above).
                    # x != y (checked above).
                    # So if y in adj[v] and y != v and y != u and x != y, we have a C4.
                    return False
        return True

    def is_c4_free(adj):
        # Check if current graph is C4-free.
        # Equivalent to: no two vertices have >1 common neighbor.
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u].intersection(adj[v])) > 1:
                    return False
        return True

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    # Initial state: Greedy construction to get a decent start
    def initial_greedy():
        adj = [set() for _ in range(n)]
        edges_list = all_edges[:]
        random.shuffle(edges_list)
        for u, v in edges_list:
            if can_add_edge(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
        return adj

    best_adj = initial_greedy()
    best_edges_count = count_edges(best_adj)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    current_adj = best_adj
    current_edges_count = best_edges_count
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Choose a move
        # 50% chance add edge, 50% chance remove edge
        if random.random() < 0.5:
            # Try to add a random missing edge
            # Find a missing edge
            # Optimization: Pick random u, v. If edge exists, skip.
            # Better: Keep a list of missing edges? Or just try random pairs.
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if (u, v) in edge_set and v not in adj[u]:
                # Check if we can add
                if can_add_edge(u, v, current_adj):
                    # Always accept improvement (increase in edges)
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_edges_count += 1
                    if current_edges_count > best_edges_count:
                        best_edges_count = current_edges_count
                        best_adj = [s.copy() for s in current_adj]
                else:
                    # Move invalid. Reject.
                    pass
            else:
                pass
        else:
            # Try to remove a random existing edge
            if current_edges_count == 0: continue
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if v in current_adj[u]:
                # Remove edge
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                current_edges_count -= 1
                
                # Acceptance:
                # We lost an edge. This is "worse" in terms of objective.
                # Accept with probability exp(-delta/T) where delta = 1 (loss of 1 edge)
                # We accept to escape local minima.
                delta = 1
                if random.random() < pow(2.71828, -delta / T):
                    # Keep the removal
                    pass
                else:
                    # Reject, put it back
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_edges_count += 1
        
        T *= cooling_rate
        
        # Optional: If we hit a dead end (no moves possible?), reset?
        # SA should handle it.
        
        # Check time
        if time.time() > deadline - 5:
            break

    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in current_adj[u]:
            if u < v:
                best_edges.append([u, v, 0])
                
    # Verify validity just in case (should be fast for n=20)
    # Check C4-free
    adj_check = [set() for _ in range(n)]
    for u, v, c in best_edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
        
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            if len(adj_check[u].intersection(adj_check[v])) > 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid structure or greedy result if SA broke something
        # Re-run greedy
        adj = initial_greedy()
        best_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    best_edges.append([u, v, 0])
                    
    return best_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
