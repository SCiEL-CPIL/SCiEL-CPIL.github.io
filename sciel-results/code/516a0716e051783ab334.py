import time
import random
import itertools

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy: 
    1. Use a greedy randomized construction to build a C4-free graph.
    2. Start with an empty graph and try to add edges in a randomized order.
    3. An edge (u, v) can be added if it does not create a 4-cycle.
       A 4-cycle is created if there exists a vertex w such that w is adjacent to both u and v.
       Wait, that's a triangle. A 4-cycle u-v-x-y-u requires u and v to have a common neighbor? No.
       A 4-cycle involves vertices u, v, x, y. Edges (u,v), (v,x), (x,y), (y,u).
       If we add edge (u, v), it creates a 4-cycle if there are two distinct vertices x, y 
       such that (u,x), (x,y), (y,v) are already in the graph? No, that's a path of length 3.
       
       Let's re-verify the condition for adding an edge (u, v) to a C4-free graph.
       Adding (u, v) creates a 4-cycle if there exists a vertex w such that 
       dist(u, w) = 1 and dist(v, w) = 1? No, that's a triangle (3-cycle).
       
       A 4-cycle is u-v-x-y-u.
       If we add (u, v), we need to check if there are two other vertices x, y such that 
       (u,x), (x,y), (y,v) exist. This forms u-x-y-v plus the new edge (v,u).
       So, we need to check if there are two distinct vertices x, y such that 
       x is a neighbor of u, y is a neighbor of v, and x is connected to y.
       Actually, simpler: 
       Adding (u, v) creates a 4-cycle if there is a path of length 3 between u and v 
       that does not use the edge (u, v). 
       A path of length 3 is u-x-y-v.
       So, we need to check if there exist distinct x, y such that 
       (u,x) in E, (x,y) in E, (y,v) in E.
       
       Alternatively, we can maintain the number of common neighbors for every pair.
       If we add (u, v), it creates a 4-cycle if and only if there are at least two common 
       "paths of length 2" between u and v? No.
       
       Let's use the standard characterization:
       A graph is C4-free if and only if for every pair of distinct vertices u, v, 
       the number of common neighbors is at most 1.
       Wait, if u and v have two common neighbors x and y, then u-x-v and u-y-v are 
       paths of length 2. The cycle would be u-x-v-y-u? No, that requires edges (x,v) and (y,u) which exist, 
       and (x,y)? No.
       The cycle is u-x-v-y-u. Edges: (u,x), (x,v), (v,y), (y,u).
       If x and y are common neighbors of u and v, then (u,x), (x,v), (u,y), (y,v) are all edges.
       This forms a 4-cycle u-x-v-y-u ONLY IF x and y are distinct.
       Yes, if u and v have two common neighbors x and y, then u-x-v-y-u is a 4-cycle.
       
       Therefore, the condition "no two vertices have more than 1 common neighbor" is equivalent to C4-free.
       
       So, when adding edge (u, v):
       It does NOT create a 4-cycle if and only if 
       for every other vertex w, it is NOT the case that w is a common neighbor of u and v?
       No. The condition "common neighbors <= 1" applies to the FINAL graph.
       
       Let's check the impact of adding (u, v) on the common neighbor counts.
       1. The pair (u, v) itself: They now have a new edge. Do they have common neighbors?
          Let C = N(u) ∩ N(v). If |C| >= 1, then adding (u, v) creates a 4-cycle?
          If w is a common neighbor of u and v, then u-w-v is a path of length 2.
          With edge (u, v), we have triangle u-w-v. That's a 3-cycle, not a 4-cycle.
          So adding (u, v) does NOT create a 4-cycle involving w directly as a triangle.
          
          However, if u and v have TWO common neighbors w1, w2, then u-w1-v-w2-u is a 4-cycle.
          So, we can add (u, v) only if |N(u) ∩ N(v)| <= 1.
          
       2. Does adding (u, v) create a 4-cycle for OTHER pairs?
          Consider a pair (x, y) where x != u, y != v.
          Adding (u, v) increases the common neighbor count of (u, x) by 1 if v is a neighbor of x?
          If (v, x) is an edge, then v becomes a common neighbor of u and x.
          So we must ensure that for all x, if (v, x) is an edge, then |N(u) ∩ N(x)| < 1 BEFORE adding?
          No, after adding (u, v), v is in N(u). So v is in N(u) ∩ N(x) iff (v, x) is an edge.
          So the new common neighbor count for (u, x) is |Old_N(u) ∩ N(x)| + (1 if (v,x) in E else 0).
          We need this to be <= 1.
          So if (v, x) is an edge, we require |Old_N(u) ∩ N(x)| == 0.
          
          Similarly for (u, y) where y is neighbor of v.
          
          And for pairs (x, y) where neither is u or v?
          Adding (u, v) doesn't change N(x) or N(y) directly, so their common neighbors don't change?
          Wait, common neighbors of x and y are vertices z such that (x,z) and (y,z) are edges.
          Adding (u, v) adds u to N(v) and v to N(u).
          Does it affect N(x) ∩ N(y)?
          Only if u or v is a common neighbor of x and y.
          u is a common neighbor of x and y if (u,x) and (u,y) are edges.
          This status doesn't change when we add (u, v).
          v is a common neighbor of x and y if (v,x) and (v,y) are edges.
          This status doesn't change.
          
          So the only pairs whose common neighbor count changes are those involving u or v.
          Specifically:
          - Pair (u, v): New count is |N(u) ∩ N(v)|. Must be <= 1.
          - Pair (u, x) for x != v: 
            New count = |N(u) ∩ N(x)| + (1 if (v, x) in E else 0).
            Must be <= 1.
            Since counts are non-negative, if (v, x) in E, we need |N(u) ∩ N(x)| == 0.
            If (v, x) not in E, we need |N(u) ∩ N(x)| <= 1 (which is already true in a C4-free graph).
          - Pair (v, x) for x != u:
            New count = |N(v) ∩ N(x)| + (1 if (u, x) in E else 0).
            Must be <= 1.
            If (u, x) in E, we need |N(v) ∩ N(x)| == 0.
            If (u, x) not in E, we need |N(v) ∩ N(x)| <= 1 (already true).
          
       So the check to add (u, v) is:
       1. |N(u) ∩ N(v)| <= 1
       2. For all x such that (v, x) is an edge: |N(u) ∩ N(x)| == 0
       3. For all x such that (u, x) is an edge: |N(v) ∩ N(x)| == 0
       
       This is O(n^2) per edge addition if done naively. With n=20, n^2=400.
       Total edges ~80. 80 * 400 = 32000 operations. Very fast.
       
       We can run this greedy algorithm multiple times with different random orderings of edges to find a dense graph.
    """
    n = 20
    deadline = time.time() + 110
    
    best_edge_count = 0
    best_edges = []
    
    def try_greedy(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Generate all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                return 0, []
            
            # Check if (u, v) can be added
            # 1. Common neighbors of u and v
            common_uv = adj[u].intersection(adj[v])
            if len(common_uv) > 1:
                continue
            
            # 2. Check neighbors of v. If x is neighbor of v, then v becomes common neighbor of u and x.
            #    So N(u) and N(x) must be disjoint (0 common neighbors) BEFORE adding?
            #    Wait, the condition was: if (v, x) in E, then |N(u) ∩ N(x)| == 0.
            #    Note: N(u) does not contain v yet. N(x) does not contain u yet.
            #    So we just check intersection of current adj[u] and adj[x].
            valid = True
            for x in adj[v]:
                if x == u: continue # u is not in adj[v] yet
                # Check if adj[u] and adj[x] have any common neighbor
                if not adj[u].isdisjoint(adj[x]):
                    valid = False
                    break
            if not valid:
                continue
            
            # 3. Check neighbors of u. If x is neighbor of u, then u becomes common neighbor of v and x.
            #    So N(v) and N(x) must be disjoint.
            for x in adj[u]:
                if x == v: continue
                if not adj[v].isdisjoint(adj[x]):
                    valid = False
                    break
            if not valid:
                continue
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            
        return len(edges), edges
    
    # Run multiple restarts
    # We want to maximize edges.
    # Known C4-free graphs on 20 vertices can have up to ~70-80 edges.
    # Random greedy usually finds decent ones.
    
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        m, edges = try_greedy(i)
        if m > best_edge_count:
            best_edge_count = m
            best_edges = edges
    
    if best_edge_count == 0:
        # Fallback
        return [[0, v, 0] for v in range(1, 20)]
    
    # Assign colours. Since graph is C4-free, any colouring works.
    # To be safe and "fancy", we can just use 0.
    # Or we can try to use distinct colours if m <= 20, but m is likely > 20.
    # The problem allows colours 0..19. It does not require distinct colours globally.
    # It only requires distinct colours on 4-cycles.
    # Since there are no 4-cycles, we can use colour 0 for all edges.
    
    coloured_edges = [[u, v, 0] for u, v in best_edges]
    return coloured_edges
