import time
import sys
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Adjacency as bitmasks
    adj = [0] * n
    
    def is_valid_add(u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        # After adding u-v, we need:
        # 1. For any common neighbor w of u and v: the edge (u,v) plus w creates a K2,3 only if
        #    there exist two other vertices x,y such that x,y are common neighbors of u and w,
        #    AND x,y are common neighbors of v and w.
        #    Actually simpler: adding edge (u,v) creates a K2,3 if there exists a vertex w 
        #    (common neighbor of u and v) such that u and w share at least 2 other common neighbors 
        #    that are also common neighbors of v and w.
        #    
        #    Let me think more carefully. A K2,3 consists of two vertices {a,b} and three 
        #    vertices {c,d,e} all connected to both a and b.
        #    
        #    If we add edge (u,v), new K2,3s involving this edge must use (u,v) as the 
        #    "part of size 2" or as part of the "part of size 3".
        #    
        #    Case 1: {u,v} is the pair. Then we need 3 common neighbors of u and v.
        #    But wait - u and v were NOT connected before. Their common neighbors are 
        #    N(u) ∩ N(v). If |N(u) ∩ N(v)| >= 3, then adding edge (u,v) creates a K2,3
        #    with vertex set {u,v} × (any 3 from N(u)∩N(v)).
        #    
        #    Case 2: u is in the pair, v is in the triple. Then we need another vertex w 
        #    such that w is adjacent to u (already true since we're checking) and w is 
        #    adjacent to all 3 vertices in the triple which includes v and two others.
        #    So we need: exists w adjacent to u, and v, and two other vertices c,d that are 
        #    also adjacent to u. I.e., N(u) ∩ N(w) contains {v, c, d} for some c,d.
        #    This means: exists w in N(u) such that |N(u) ∩ N(w)| >= 3 and v ∈ N(w).
        #    Equivalently: exists w in N(u) ∩ N(v) such that |N(u) ∩ N(w) \ {v}| >= 2.
        #    Which is: exists w in N(u) ∩ N(v) such that |N(u) ∩ N(w)| >= 3.
        #    But since v ∈ N(w), this is |N(u) ∩ N(w)| >= 3.
        #    
        #    Wait, this is the same as case 1 essentially. Let me re-examine.
        #    
        #    Actually, the constraint is: no two vertices have 3 common neighbors.
        #    Adding edge (u,v) changes the common neighbor counts:
        #    - For pair (u,v): common neighbors = N(u) ∩ N(v) (unchanged by adding u-v itself)
        #      Wait no, u and v are now adjacent, but that doesn't change their common neighbors.
        #      Common neighbors of u and v are vertices w such that w~u and w~v. Adding u~v 
        #      doesn't affect which vertices are common neighbors of u and v.
        #      
        #    - For pair (u, w) where w was a common neighbor of u and v: 
        #      Before: common neighbors of u and w = N(u) ∩ N(w)
        #      After adding u-v: if v is adjacent to w (i.e., v ∈ N(w)), then v becomes a new 
        #      common neighbor of u and w. So new common count = |N(u) ∩ N(w)| + (1 if v∈N(w) else 0)
        #      But v ∈ N(w) iff w ∈ N(v) iff w is a common neighbor of u and v.
        #      So for w ∈ N(u) ∩ N(v): new common neighbors of u and w = |N(u) ∩ N(w)| + 1
        #      We need this <= 2, so |N(u) ∩ N(w)| <= 1 for all w ∈ N(u) ∩ N(v).
        #      
        #    - For pair (v, w) where w was a common neighbor of u and v:
        #      Similarly, new common neighbors of v and w = |N(v) ∩ N(w)| + 1
        #      We need |N(v) ∩ N(w)| <= 1 for all w ∈ N(v) ∩ N(u).
        #      
        #    - For pair (w1, w2) where w1, w2 ∈ N(u) ∩ N(v):
        #      Adding u-v makes u and v both new common neighbors of w1 and w2 (if they weren't 
        #      already). u is already adjacent to both w1 and w2 (since they're in N(u)), so u was 
        #      already a common neighbor. v is adjacent to both w1 and w2 (since they're in N(v)), 
        #      so v was already a common neighbor. So no change.
        #      
        #    - For other pairs: no change.
        #    
        #    So the validity check for adding edge (u,v):
        #    1. |N(u) ∩ N(v)| <= 2  (the new pair u,v can't have 3 common neighbors)
        #    2. For all w in N(u) ∩ N(v): |N(u) ∩ N(w)| <= 1  (u and w can't gain a 3rd common neighbor v)
        #    3. For all w in N(u) ∩ N(v): |N(v) ∩ N(w)| <= 1  (v and w can't gain a 3rd common neighbor u)
        
        common = adj[u] & adj[v]
        if common.bit_count() > 2:
            return False
        
        # Check conditions 2 and 3
        temp = common
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            # Condition 2: |N(u) ∩ N(w)| <= 1
            if (adj[u] & adj[w]).bit_count() > 1:
                return False
            # Condition 3: |N(v) ∩ N(w)| <= 1
            if (adj[v] & adj[w]).bit_count() > 1:
                return False
        
        return True
    
    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def try_add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def try_remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    # Phase 1: Greedy construction with random restarts
    best_edges = []
    best_count = 0
    
    max_restarts = 500
    restart = 0
    
    while time.time() < deadline - 30.0 and restart < max_restarts:
        restart += 1
        random.shuffle(adj)  # This won't work, let me restructure
        
    # Better approach: greedy with randomized ordering
    while time.time() < deadline - 30.0:
        # Reset
        adj = [0] * n
        
        # Randomize vertex ordering for edge addition
        edge_order = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edge_order)
        
        count = 0
        for u, v in edge_order:
            if is_valid_add(u, v):
                try_add_edge(u, v)
                count += 1
        
        if count > best_count:
            best_count = count
            best_edges = []
            for u in range(n):
                mask = adj[u]
                k = mask
                while k:
                    lsb = k & -k
                    w = lsb.bit_length() - 1
                    k -= lsb
                    if u < w:
                        best_edges.append([u, w])
        
        if best_count >= 155:  # Reasonable upper bound target
            break
    
    # Phase 2: Local search (edge swap) to improve
    if time.time() < deadline - 5.0:
        # Work with the best solution found
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            count = count_edges()
            
            # Try removing one edge and adding two edges
            edges_list = []
            for u in range(n):
                mask = adj[u]
                k = mask
                while k:
                    lsb = k & -k
                    v = lsb.bit_length() - 1
                    k -= lsb
                    if u < v:
                        edges_list.append((u, v))
            
            for i in range(len(edges_list)):
                if time.time() > deadline - 5.0:
                    break
                u1, v1 = edges_list[i]
                
                # Try removing this edge
                try_remove_edge(u1, v1)
                
                # Try adding two new edges
                for j in range(len(edges_list)):
                    if j == i:
                        continue
                    # We need to consider non-edges. Let's just try all pairs.
                    # This is expensive. Let's be smarter.
                    
                # Restore
                try_add_edge(u1, v1)
            
            # Simpler local search: try adding any missing edge
            for u in range(n):
                if time.time() > deadline - 5.0:
                    break
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        if is_valid_add(u, v):
                            try_add_edge(u, v)
                            improved = True
    
    # Phase 3: More aggressive local search with edge swaps
    if time.time() < deadline - 3.0:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        current_count = count_edges()
        
        # Try all single edge swaps: remove one, add another
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        current_edges_set = set(best_edges)
        
        for _ in range(200):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            # Try adding any edge
            for u, v in all_edges:
                if (u, v) in current_edges_set:
                    continue
                if is_valid_add(u, v):
                    try_add_edge(u, v)
                    current_edges_set.add((u, v))
                    current_count += 1
                    improved = True
            
            # Try removing one and adding two
            for u1, v1 in list(current_edges_set):
                if time.time() > deadline - 3.0:
                    break
                
                try_remove_edge(u1, v1)
                current_edges_set.discard((u1, v1))
                
                added = []
                for u2, v2 in all_edges:
                    if (u2, v2) in current_edges_set:
                        continue
                    if is_valid_add(u2, v2):
                        try_add_edge(u2, v2)
                        current_edges_set.add((u2, v2))
                        added.append((u2, v2))
                        if len(added) == 2:
                            break
                
                if len(added) >= 2:
                    current_count += 1
                    improved = True
                else:
                    # Undo
                    for u2, v2 in added:
                        try_remove_edge(u2, v2)
                        current_edges_set.discard((u2, v2))
                    try_add_edge(u1, v1)
                    current_edges_set.add((u1, v1))
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(current_edges_set)
    
    # Convert to output format
    result = []
    for u, v in best_edges:
        if u > v:
            u, v = v, u
        result.append([u, v])
    
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Edges: {len(edges)}")
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = (adj[u] & adj[v]).bit_count()
            if common >= 3:
                valid = False
                print(f"Invalid: {u}, {v} have {common} common neighbors")
    print(f"Valid: {valid}")
