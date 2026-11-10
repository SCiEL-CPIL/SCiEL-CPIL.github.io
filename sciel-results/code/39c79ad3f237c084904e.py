import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    q = 5
    # Build PG(2, q) points: 31 points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in adj[i]:
            A[i, j] = 1
            A[j, i] = 1
    
    # Exact C4 count using common-neighbor pairs
    # A C4 is u-v-w-x-u where u~v, v~w, w~x, x~u, and no diagonals
    # Count: for each pair of non-adjacent vertices (u,w), count common neighbors
    # Each C4 is counted exactly twice (once for each diagonal pair)
    
    def count_c4s(adj_sets, n):
        """Count C4s exactly. Each C4 counted once per diagonal pair, total / 2."""
        total = 0
        for u in range(n):
            for w in range(u+1, n):
                if w not in adj_sets[u]:  # u and w not adjacent (diagonal of C4)
                    common = len(adj_sets[u] & adj_sets[w])
                    total += common
        return total // 2
    
    # Verify initial count
    adj_sets = [set(adj[i]) for i in range(n)]
    initial_c4 = count_c4s(adj_sets, n)
    # Should be 210 for PG(2,5) polarity graph
    
    # For 1-swap: remove edge (a,b), add edge (c,d) where (c,d) is non-edge
    # Delta in C4 count:
    # Removing (a,b) destroys C4s that use edge (a,b)
    # Adding (c,d) creates C4s that use edge (c,d)
    # But there's interaction: C4s that use both (a,b) and (c,d)
    
    def c4s_using_edge(a, b, adj_sets, n):
        """Count C4s that use edge (a,b)."""
        # A C4 using edge (a,b): a-b-x-y-a where b~x, x~y, y~a, and a!=x, b!=y, x!=y
        # Equivalently: x in N(b)\{a}, y in N(a)\{b}, x~y, x!=y
        count = 0
        for x in adj_sets[b]:
            if x == a:
                continue
            for y in adj_sets[a]:
                if y == b or y == x:
                    continue
                if x in adj_sets[y]:
                    count += 1
        return count
    
    # Precompute C4s using each edge
    edge_c4 = {}
    for e in edges:
        a, b = e
        edge_c4[(a, b)] = c4s_using_edge(a, b, adj_sets, n)
        edge_c4[(b, a)] = edge_c4[(a, b)]
    
    # For non-edges, compute how many C4s would be created by adding them
    # This is the same formula but with current adjacency
    non_edge_c4 = {}
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj_sets[u]:
                # If we add (u,v), C4s created = number of paths u-x-y-v with x~u, y~v, x~y, x!=y
                count = 0
                for x in adj_sets[u]:
                    for y in adj_sets[v]:
                        if x == y:
                            continue
                        if x in adj_sets[y]:
                            count += 1
                non_edge_c4[(u, v)] = count
                non_edge_c4[(v, u)] = count
    
    # Now try 1-swap: remove edge (a,b), add non-edge (c,d)
    # Net change ≈ non_edge_c4[(c,d)] - edge_c4[(a,b)] + interaction
    # Interaction: C4s that use both (a,b) and (c,d)
    # A C4 using both edges: must be a-c-b-d-a or a-d-b-c-a etc.
    # This requires specific adjacency patterns
    
    def interaction_c4s(a, b, c, d, adj_sets):
        """Count C4s that would use both edge (a,b) and edge (c,d)."""
        # Possible C4s with edges (a,b) and (c,d):
        # a-c-b-d-a: need a~c, c~b, b~d, d~a
        # a-d-b-c-a: need a~d, d~b, b~c, c~a
        count = 0
        # Check a-c-b-d-a
        if c in adj_sets[a] and b in adj_sets[c] and d in adj_sets[b] and a in adj_sets[d]:
            # Also need a!=c, b!=d, c!=b, d!=a (all distinct)
            if len({a, b, c, d}) == 4:
                count += 1
        # Check a-d-b-c-a
        if d in adj_sets[a] and b in adj_sets[d] and c in adj_sets[b] and a in adj_sets[c]:
            if len({a, b, c, d}) == 4:
                count += 1
        return count
    
    best_swap = None
    best_delta = float('inf')
    
    # Sort edges by C4 count (highest first - most valuable to remove)
    sorted_edges = sorted(edges, key=lambda e: edge_c4[(e[0], e[1])], reverse=True)
    # Sort non-edges by C4 count (lowest first - cheapest to add)
    sorted_non_edges = sorted(non_edge_c4.keys(), key=lambda x: non_edge_c4[x])
    # Only consider (u,v) where u<v to avoid duplicates
    sorted_non_edges = [ne for ne in sorted_non_edges if ne[0] < ne[1]]
    
    # Try top candidates
    max_edge_candidates = 50
    max_non_edge_candidates = 50
    
    for a, b in sorted_edges[:max_edge_candidates]:
        for c, d in sorted_non_edges[:max_non_edge_candidates]:
            # Skip if the non-edge shares a vertex with the edge in a way that creates issues
            # Actually, they can share vertices, but let's check
            if (a, b) == (c, d) or (a, b) == (d, c):
                continue
            
            # Compute exact delta
            # Remove (a,b), add (c,d)
            # New C4 count = old - C4s_lost + C4s_gained
            # C4s_lost = C4s using (a,b) in current graph
            # C4s_gained = C4s using (c,d) in graph with (a,b) removed
            # But also: some C4s might be both lost and gained (interaction)
            
            # Simpler: just compute the delta approximately, then verify with exact count
            approx_delta = non_edge_c4[(c, d)] - edge_c4[(a, b)]
            
            if approx_delta < best_delta:
                # Verify with exact computation
                # Create modified adjacency
                new_adj = [set(s) for s in adj_sets]
                new_adj[a].discard(b)
                new_adj[b].discard(a)
                new_adj[c].add(d)
                new_adj[d].add(c)
                
                new_c4 = count_c4s(new_adj, n)
                exact_delta = new_c4 - initial_c4
                
                if exact_delta < best_delta:
                    best_delta = exact_delta
                    best_swap = (a, b, c, d)
    
    # Apply best swap if it improves
    if best_swap is not None and best_delta < 0:
        a, b, c, d = best_swap
        # Remove edge (a,b)
        edges = [e for e in edges if not (set(e) == {a, b})]
        # Add edge (c,d)
        edges.append([c, d])
    
    # Ensure exactly 91 edges
    if len(edges) != 91:
        existing = set()
        for e in edges:
            existing.add(tuple(sorted(e)))
        for i in range(n):
            for j in range(i+1, n):
                if len(edges) >= 91:
                    break
                if (i, j) not in existing:
                    edges.append([i, j])
                    existing.add((i, j))
            if len(edges) >= 91:
                break
    
    return edges[:91]
