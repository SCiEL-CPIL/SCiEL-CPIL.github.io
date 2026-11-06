import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    q = 5
    # Generate vertices of PG(2, q)
    vertices = []
    for i in range(q):
        for j in range(q):
            vertices.append((1, i, j))
    for k in range(q):
        vertices.append((0, 1, k))
    vertices.append((0, 0, 1))
    
    n = len(vertices)  # 31
    
    # Build adjacency list for the polarity graph
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    # We have 90 edges. Find the non-edge that adds the fewest C4s.
    best_cost = float('inf')
    best_edge = None
    
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            A = adj[u]
            B = adj[v]
            cost = 0
            # Count paths of length 3 between u and v
            # u-x-y-v
            for x in A:
                if x == v: continue
                for y in B:
                    if y == u or y == x: continue
                    if x in adj[y]:
                        cost += 1
            # Each C4 is counted twice? 
            # u-x-y-v and u-y-x-v?
            # No, x and y are distinct vertices in the path.
            # A C4 is u-x-y-v-u.
            # The path u-x-y-v is unique for the pair (x,y).
            # So cost is the number of C4s.
            if cost < best_cost:
                best_cost = cost
                best_edge = [u, v]
    
    edges.append(best_edge)
    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])
    
    # Helper to count C4s containing a specific edge (a,b) in the current graph
    def count_c4s_with_edge(a, b):
        count = 0
        for x in adj[a]:
            if x == b: continue
            for y in adj[b]:
                if y == a or y == x: continue
                if x in adj[y]:
                    count += 1
        return count

    # Helper to count C4s that would be created by adding edge (c,d) 
    # assuming (a,b) is removed.
    # This is counting paths c-x-y-d in G \ {(a,b)}
    def count_c4s_with_edge_if_removed(c, d, a, b):
        count = 0
        for x in adj[c]:
            if x == d: continue
            # x cannot be a or b if that affects connectivity?
            # The path is in G \ {(a,b)}.
            # So edges (x,y), (c,x), (y,d) must exist in G \ {(a,b)}.
            # (c,x) exists in G. If (c,x) is (a,b) or (b,a), it's removed.
            # But c,d are non-adjacent, so (c,x) is not (a,b) unless c=a and x=b?
            # We ensure c,d are not a,b in the swap selection.
            for y in adj[d]:
                if y == c or y == x: continue
                # Check if x-y edge exists in G \ {(a,b)}
                if x == a and y == b: continue
                if x == b and y == a: continue
                if x in adj[y]:
                    # Check if (c,x) or (y,d) is (a,b)?
                    # c,x != a,b and c,x != b,a?
                    # We assumed c,d distinct from a,b.
                    # But x could be a?
                    # If x=a, then (c,a) must exist.
                    # If (c,a) is (a,b)? No, c!=b.
                    # So (c,a) is fine.
                    # If (y,d) is (a,b)? No, d!=a.
                    # So we just need to ensure (x,y) is not (a,b).
                    count += 1
        return count

    edge_set = set(tuple(sorted(e)) for e in edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_swap = None
        
        # List of current edges
        current_edges = list(edges)
        
        # Precompute non-edges
        non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
        
        for ei, (a, b) in enumerate(current_edges):
            # Calculate C4s lost by removing (a,b)
            lost = count_c4s_with_edge(a, b)
            
            for c, d in non_edges:
                # Ensure (c,d) is not (a,b) (it's a non-edge, so it isn't)
                # Calculate C4s gained by adding (c,d) given (a,b) is removed
                gained = count_c4s_with_edge_if_removed(c, d, a, b)
                
                delta = gained - lost
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (ei, c, d)
        
        if best_swap is not None and best_delta < 0:
            ei, c, d = best_swap
            a, b = edges[ei]
            
            # Perform swap
            edges[ei] = [c, d]
            
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].add(d)
            adj[d].add(c)
            
            edge_set.discard((a, b) if a < b else (b, a))
            edge_set.add((c, d) if c < d else (d, c))
            
            improved = True

    # Final verification
    assert len(edges) == 91
    assert len(set(tuple(sorted(e)) for e in edges)) == 91
    
    return edges
