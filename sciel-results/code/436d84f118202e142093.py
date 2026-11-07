"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Pre-compute adjacency lists for all possible edges? No, dynamic.
    # We will use a backtracking search.
    # State: current edges, current degrees, current adjacency.
    
    # To make backtracking efficient, we need a fast validity check for incremental changes.
    # A path s-u-v-t is invalid if deg(s) == deg(t).
    # This means: For any edge (u, v) in the graph, the set of degrees of N(u)\{v} 
    # and the set of degrees of N(v)\{u} must be disjoint.
    
    # Let's maintain:
    # 1. deg[i]: degree of vertex i
    # 2. adj[i]: set of neighbors of i
    # 3. For each edge (u, v), we need to ensure no conflict.
    # Instead of checking all edges on every step, we can check only the edges 
    # affected by the new edge (u, v).
    # Adding edge (u, v) affects:
    # - The path constraint for the edge (u, v) itself.
    # - The path constraint for any edge (s, u) where s is a neighbor of u (s != v). 
    #   Why? Because t can now be v (if v is neighbor of u? No, t is neighbor of v).
    #   Wait. Path s-a-b-t.
    #   If we add (u, v), it can serve as:
    #   1. The middle edge (u, v) in path s-u-v-t.
    #   2. The first edge (s, u) where s=u, u=a, v=b, t=v? No, s-u-v-t.
    #      If new edge is (u, v), it can be:
    #      - Edge 1 (s-u): s=u, u=a... no, edge is (u,v). So s=u, a=v? 
    #        Path u-v-b-t. Here (u,v) is edge 1. b is neighbor of v. t is neighbor of b.
    #        Condition: deg(u) != deg(t).
    #      - Edge 2 (u-v): Path s-u-v-t.
    #        Condition: deg(s) != deg(t) for s in N(u)\{v}, t in N(v)\{u}.
    #      - Edge 3 (v-t): Path s-u-v-t where t=v? No, edge is (u,v).
    #        Path s-u-v-t. If (u,v) is edge 3, then v=u, t=v? No.
    #        Path s-u-a-v. Here (a,v) is edge 3. If we add (u,v), it's edge (a,v) with a=u.
    #        So path s-u-u-v? No, distinct vertices.
    #        So (u,v) can be edge 1: u-v-b-t.
    #        Or edge 2: s-u-v-t.
    #        Or edge 3: s-u-v-t where t is connected to v via (u,v)? No, edge is (u,v).
    #        If (u,v) is edge 3, the path is s-u-a-v with a=u? No, a must be neighbor of v.
    #        If a=u, then edge is (u,v). Path s-u-u-v? No.
    #        So (u,v) can be edge 1 or edge 2.
    
    # Let's refine the check for adding edge (u, v):
    # 1. Check if (u, v) as middle edge causes conflict:
    #    Set S1 = {deg(s) for s in N(u) if s != v}
    #    Set S2 = {deg(t) for t in N(v) if t != u}
    #    If S1 intersects S2, conflict.
    
    # 2. Check if (u, v) as first edge (u-v-b-t) causes conflict:
    #    Here u is 's', v is 'a'.
    #    Path: u - v - b - t.
    #    b must be in N(v) \ {u}.
    #    t must be in N(b) \ {v}.
    #    Condition: deg(u) != deg(t).
    #    So for all b in N(v)\{u}, for all t in N(b)\{v}, we need deg(t) != deg(u).
    #    Equivalently: deg(u) is not in the set of degrees of all "distance 2" neighbors of u via v.
    
    # 3. Check if (u, v) as last edge (s-u-v-t) causes conflict:
    #    Here v is 't', u is 'b'.
    #    Path: s - a - v - u. (Wait, edge is (u,v). So b=v, t=u? No, edge is (b,t) = (v,u)).
    #    Path s - a - v - u.
    #    a must be in N(v) \ {u}.
    #    s must be in N(a) \ {v}.
    #    Condition: deg(s) != deg(u).
    #    So for all a in N(v)\{u}, for all s in N(a)\{v}, we need deg(s) != deg(u).
    #    This is symmetric to case 2, just swapping u and v in the "endpoint" role.
    #    Actually, case 2 checks deg(u) against deg(t) where t is dist 2 from u via v.
    #    Case 3 checks deg(s) against deg(u) where s is dist 2 from u via v.
    #    So we just need to ensure deg(u) is not in the set of degrees of vertices at distance 2 from u through v,
    #    AND deg(v) is not in the set of degrees of vertices at distance 2 from v through u.
    #    AND the middle edge check.

    def is_valid_addition(u, v, deg, adj):
        # Check 1: (u, v) is middle edge s-u-v-t
        # S1: degrees of N(u) \ {v}
        # S2: degrees of N(v) \ {u}
        deg_u_neighs = [deg[s] for s in adj[u] if s != v]
        deg_v_neighs = [deg[t] for t in adj[v] if t != u]
        
        # If intersection is non-empty, invalid
        if set(deg_u_neighs) & set(deg_v_neighs):
            return False
            
        # Check 2: (u, v) is first edge u-v-b-t
        # Endpoints are u and t.
        # t is in N(b) \ {v} for b in N(v) \ {u}.
        # We need deg(u) != deg(t) for all such t.
        # Collect all degrees of such t's.
        t_degs = set()
        for b in adj[v]:
            if b == u:
                continue
            for t in adj[b]:
                if t == v:
                    continue
                t_degs.add(deg[t])
        
        if deg[u] in t_degs:
            return False
            
        # Check 3: (u, v) is last edge s-a-v-u (where u is t, v is b? No, edge is (v,u))
        # Path s-a-v-u.
        # Endpoints s and u.
        # s is in N(a) \ {v} for a in N(v) \ {u}.
        # We need deg(s) != deg(u) for all such s.
        # This is the same set of s's as the t's in Check 2?
        # In Check 2, t was in N(b)\{v} for b in N(v)\{u}.
        # In Check 3, s is in N(a)\{v} for a in N(v)\{u}.
        # Yes, it's the same set of vertices (distance 2 from u via v).
        # So we just need deg(u) not in that set.
        # Wait, in Check 2, we checked deg(u) against deg(t).
        # In Check 3, we check deg(s) against deg(u).
        # Since s and t are from the same set of vertices, and we are comparing against deg(u),
        # the condition is identical: deg(u) must not be in the set of degrees of vertices at dist 2 from u via v.
        
        # Is there a symmetric check for deg(v)?
        # What if (u, v) is first edge v-u-b-t?
        # Then endpoints are v and t.
        # t is dist 2 from v via u.
        # So we need deg(v) not in set of degrees of vertices at dist 2 from v via u.
        
        t_degs_v = set()
        for b in adj[u]:
            if b == v:
                continue
            for t in adj[b]:
                if t == u:
                    continue
                t_degs_v.add(deg[t])
                
        if deg[v] in t_degs_v:
            return False
            
        return True

    # Backtracking search
    # Order edges to find a solution quickly. 
    # Maybe prefer adding edges that connect high-degree vertices? Or low?
    # Let's try a heuristic: add edges that are "safe" (pass is_valid_addition).
    
    best_edges = []
    best_count = 0
    
    # We can try multiple random orders of edges to escape local optima.
    # Or use a deterministic greedy with backtracking.
    
    # Let's implement a recursive backtracking with a depth limit or time check.
    # Given n=15, max edges 105.
    # A simple greedy might get stuck. Backtracking is better.
    
    def solve(idx, current_edges, deg, adj):
        nonlocal best_edges, best_count, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
        if idx >= len(all_edges):
            return
            
        u, v = all_edges[idx]
        
        # Option 1: Add edge (u, v)
        # Check if it's already added (shouldn't be if we process sequentially)
        # Check validity
        if is_valid_addition(u, v, deg, adj):
            # Add
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append((u, v))
            
            solve(idx + 1, current_edges, deg, adj)
            
            # Undo
            current_edges.pop()
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
            
        # Option 2: Skip edge (u, v)
        solve(idx + 1, current_edges, deg, adj)

    # To improve performance, we can shuffle the edge order.
    # But shuffling changes the "idx" logic.
    # Let's create a list of edge indices and shuffle it.
    
    # Since 105 edges, 2^105 is too big for pure backtracking without pruning.
    # The pruning in is_valid_addition is strong, but might not be enough for pure DFS.
    # Let's use a Greedy-Backtracking hybrid:
    # 1. Shuffle edges.
    # 2. Try to add if valid.
    # 3. If stuck (no more edges can be added), backtrack to last decision point?
    #    Or just do random restarts of the greedy algorithm, but with a smarter check.
    
    # Actually, the "is_valid_addition" check is O(degree^2).
    # With n=15, degrees are small.
    # Let's try 100 random restarts of a greedy algorithm that uses is_valid_addition.
    # This is much faster than full backtracking and often finds good solutions.
    
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        cur_edges = []
        deg = [0] * n
        adj = [set() for _ in range(n)]
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if is_valid_addition(u, v, deg, adj):
                cur_edges.append((u, v))
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
                
        if len(cur_edges) > best_count:
            best_count = len(cur_edges)
            best_edges = cur_edges[:]
            
        # Optional: Local search on the result
        # Try to swap edges to allow more additions?
        # For now, let's just do random restarts.
        
    # Return best found
    # Ensure types are correct: list of lists of ints
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
