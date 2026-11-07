import time
import sys
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85

    # Deterministic search over disjoint union of complete bipartite components K_{a, b}
    # where a != b.
    # We partition 15 vertices into pairs (a_i, b_i).
    # Sum(a_i + b_i) = 15.
    # Maximize Sum(a_i * b_i).
    
    best_edges_count = 0
    best_partition = []
    
    # Since n=15 is small, we can iterate through possible component sizes.
    # Components are defined by (a, b) with a+b = size, a != b.
    # Possible sizes: 2..15.
    # We need a set of components that sum to 15.
    
    # Recursive search for partitions
    def search_partitions(remaining, current_components):
        nonlocal best_edges_count, best_partition
        
        # Pruning: even if we use the max possible edges for remaining, can we beat best?
        # Max edges for size S is floor(S^2/4) if S is even (but a!=b, so (S/2-1)(S/2+1) = S^2/4 - 1)
        # If S is odd, floor(S^2/4).
        
        if remaining == 0:
            count = sum(a*b for a,b in current_components)
            if count > best_edges_count:
                best_edges_count = count
                best_partition = current_components[:]
            return
        
        # Try all possible component sizes for the next component
        # Min size 2 (1,1 is a=b invalid? K_1,1 has deg 1,1. Path length 3? No, K_1,1 has no path of length 3.
        # But K_1,1 is a component. Is it valid?
        # "no simple three-edge path". K_1,1 has no such path. So it is valid.
        # But a=1, b=1 => a=b.
        # Does a=b matter for K_1,1?
        # The constraint is on paths of length 3.
        # If a component has < 4 vertices, it has no path of length 3.
        # So K_1,1, K_1,2, K_2,2 are valid *locally* if they don't form paths of length 3.
        # K_2,2 has 4 vertices. Path 1-3-2-4. Endpoints 1,4. deg(1)=2, deg(4)=2. Invalid.
        # So K_2,2 is invalid.
        # K_1,1 is valid (no path len 3).
        # K_1,2 is valid (no path len 3).
        # K_1,3 is valid (no path len 3).
        # K_2,3: Path 1(L)-3(R)-2(L)-4(R). deg(1)=3, deg(4)=2. OK.
        # So a != b is sufficient for a,b >= 2?
        # If a=1, b=1, no path len 3. Valid.
        # If a=1, b=2, no path len 3. Valid.
        # If a=1, b=3, no path len 3. Valid.
        # If a=2, b=2, path exists. deg equal. Invalid.
        # So we must avoid a=b if a+b >= 4.
        
        max_size = remaining
        min_size = 2
        
        for size in range(min_size, max_size + 1):
            # Generate all (a, b) for this size
            for a in range(1, size):
                b = size - a
                if a > b: continue # symmetry
                # Check validity for this component
                if a == b and size >= 4:
                    continue
                # Estimate max possible edges for rest
                # This is a weak bound, but n=15 is small.
                search_partitions(remaining - size, current_components + [(a, b)])

    search_partitions(15, [])
    
    # Construct the graph from the best partition
    edges = set()
    vertex_id = 0
    for a, b in best_partition:
        # Component vertices: vertex_id ... vertex_id+a-1 (Part A)
        # vertex_id+a ... vertex_id+a+b-1 (Part B)
        for u in range(vertex_id, vertex_id + a):
            for v in range(vertex_id + a, vertex_id + a + b):
                edges.add((u, v))
        vertex_id += a + b
    
    current_edges = list(edges)
    
    # Local improvement: try to add edges
    # (This is likely to fail for K_7,8, but we keep it as per "investigate improvements")
    
    # Build adjacency and degree from current edges
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    def check_edge_add(u, v):
        """Check if adding edge (u,v) keeps the graph valid."""
        # Adding (u,v) increases deg[u] and deg[v] by 1
        # New deg: deg[u]+1, deg[v]+1
        # Paths that could be violated: any path s-u-v-t or s-v-u-t
        # where s != t, s in adj[u]\{v}, t in adj[v]\{u}, s != t
        # After adding (u,v): deg[s] and deg[t] are unchanged (unless s or t is u or v, but s!=u and t!=v by construction)
        # Wait: s is in adj[u] and s != v. t is in adj[v] and t != u. s != t.
        # deg[s] is unchanged, deg[t] is unchanged.
        # So we just need: for all s in adj[u]\{v}, t in adj[v]\{u}, s != t: deg[s] != deg[t]
        # But also, the new edge (u,v) creates new paths where u or v is an endpoint.
        # Path s-v-u-t: s in adj[v]\{u}, t in adj[u]\{v}, s != t. deg[s] vs deg[t]. Same check.
        # Path u-...-v-...: no, paths are length 3: s-u-v-t.
        # Also paths where one endpoint is u or v:
        #   u-x-y-t: x in adj[u], y in adj[x], t in adj[y], t != u. But this was already checked before adding (u,v) unless it involves the new edge.
        #   The new edge only creates new paths that USE the new edge (u,v).
        #   Paths using (u,v): s-u-v-t where s in adj[u]\{v}, t in adj[v]\{u}, s != t.
        #   And s-v-u-t where s in adj[v]\{u}, t in adj[u]\{v}, s != t. (same set)
        #   So we just need to check: for all s in adj[u]\{v}, t in adj[v]\{u}, s != t: deg[s] != deg[t]
        #   BUT: deg[u] and deg[v] change, but s and t are not u or v, so their degrees don't change.
        #   However, we also need to check that no OTHER path is violated. Adding an edge doesn't change degrees of other vertices,
        #   and doesn't create new paths that don't use the new edge. So only paths using (u,v) are new.
        #   Wait: adding (u,v) also means that u and v are now adjacent. This could create new paths like s-x-u-v where x is a common neighbor? No, that's still using edge (u,v) in a sense, but the path is s-x-u-v, endpoints s and v.
        #   Hmm, I need to think more carefully.
        
        # Actually, adding edge (u,v) creates new 3-edge paths that include this edge:
        # 1. s-u-v-t: s in N(u)\{v}, t in N(v)\{u}, s!=t. Endpoints s,t. Need deg(s)!=deg(t).
        # 2. u-v-t-s: same as above, just reversed direction.
        # 3. s-u-v-w where w... no, it's s-u-v-t, 4 vertices.
        # Also paths where (u,v) is not the middle edge but one of the outer edges:
        # 4. u-v-x-y: u is endpoint. x in N(v)\{u}, y in N(x), y!=u. Endpoints u,y. Need deg(u)!=deg(y).
        #    But deg(u) is now deg[u]+1. y's degree is unchanged.
        # 5. y-x-v-u: same as 4.
        # 6. x-u-v-y: x in N(u)\{v}, y in N(v)\{u}. Endpoints x,y. Need deg(x)!=deg(y). Same as case 1.
        # 7. x-v-u-y: same as 6.
        
        # So we need to check:
        # A. For all s in N(u)\{v}, t in N(v)\{u}, s!=t: deg(s) != deg(t)
        # B. For all x in N(u)\{v}, y in N(x), y!=u, y!=v: deg(u)+1 != deg(y)  [path x-u-v-y? No, x-u is edge, u-v is new edge, v-y is edge. So path x-u-v-y. Endpoints x,y. Need deg(x)!=deg(y). That's case A.]
        #    Wait, I'm confusing myself. Let me re-examine.
        
        # Path of length 3: vertices a-b-c-d with edges (a,b),(b,c),(c,d). a!=d.
        # Adding edge (u,v) creates new such paths where at least one of the three edges is (u,v).
        # Case 1: (u,v) is the middle edge. Path: s-u-v-t. s in N(u)\{v}, t in N(v)\{u}, s!=t.
        # Case 2: (u,v) is the first edge. Path: u-v-x-y. x in N(v)\{u}, y in N(x), y!=u.
        #    Endpoints: u and y. Need deg(u) != deg(y). deg(u) is now deg[u]+1.
        # Case 3: (u,v) is the last edge. Path: x-y-u-v. x in N(u)\{v}, y in N(x), y!=v.
        #    Endpoints: x and v. Need deg(x) != deg(v). deg(v) is now deg[v]+1.
        
        # So checks:
        # A. For all s in N(u)\{v}, t in N(v)\{u}, s!=t: deg[s] != deg[t]
        # B. For all x in N(v)\{u}, y in N(x), y!=u: (deg[u]+1) != deg[y]
        # C. For all x in N(u)\{v}, y in N(x), y!=v: deg[x] != (deg[v]+1)
        
        # But note: in case B, y could be in N(u) or not. And y!=u is required.
        # Also, we need to make sure we're not double-counting or missing constraints.
        
        # This is still O(deg(u)*deg(v) + deg(v)*avg_deg + deg(u)*avg_deg) per edge check.
        # For deg ~ 8, this is manageable.
        
        nu = adj[u] - {v}
        nv = adj[v] - {u}
        
        # Check A
        for s in nu:
            for t in nv:
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return False
        
        # Check B: u-v-x-y, endpoints u,y. deg[u]+1 vs deg[y]
        new_deg_u = deg[u] + 1
        for x in nv:
            for y in adj[x]:
                if y == u:
                    continue
                if new_deg_u == deg[y]:
                    return False
        
        # Check C: x-y-u-v, endpoints x,v. deg[x] vs deg[v]+1
        new_deg_v = deg[v] + 1
        for x in nu:
            for y in adj[x]:
                if y == v:
                    continue
                if deg[x] == new_deg_v:
                    return False
        
        return True
    
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
            
    missing_edges = list(all_edges - set((min(u,v), max(u,v)) for u,v in current_edges))
    # Normalize edge representation
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    missing_edges = [e for e in all_edges if e not in edge_set]
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for e in missing_edges:
            if e in edge_set:
                continue
            u, v = e
            if check_edge_add(u, v):
                # Add the edge
                current_edges.append(e)
                edge_set.add(e)
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                improved = True
                # Remove from missing
                missing_edges.remove(e)
    
    final_edges = [[int(u), int(v)] for u, v in current_edges]
    return final_edges
