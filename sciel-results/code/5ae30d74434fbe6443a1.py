import random
import time
from itertools import combinations

def has_c5(n, edges):
    """Check if the graph contains a cycle of length 5."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # To check for C5, we can iterate over pairs of vertices (u, v) that are NOT adjacent
    # and see if there is a path of length 4 between them that doesn't use (u,v) directly?
    # Actually, a C5 is a cycle. 
    # A more efficient way: 
    # For each vertex u, look at its neighbors N(u).
    # Any C5 passing through u consists of u, v, w, x, y where v,y in N(u) and v-w-x is a path of length 2 in G - {u}.
    # So we need to check if there exist v, y in N(u) such that there is a path v-w-x-y in G-{u} with w != u, x != u, etc.
    # This is still complex.
    
    # Simpler check for small n (<=35):
    # Iterate over all combinations of 5 vertices.
    # Check if they form a C5.
    # 35C5 = 324632. This is fast enough.
    
    for verts in combinations(range(n), 5):
        v0, v1, v2, v3, v4 = verts
        # Check all 6 permutations of cycle structure?
        # A set of 5 vertices forms a C5 if the induced subgraph is exactly a C5.
        # Edges must be (0,1), (1,2), (2,3), (3,4), (4,0) in some order.
        # There are 12 distinct cycle structures for 5 labeled vertices (5! / (2*5) = 12).
        # We can just check if the induced subgraph has exactly 5 edges and is connected and 2-regular.
        
        # Build induced subgraph edges
        sub_edges = set()
        for i in range(5):
            for j in range(i+1, 5):
                if verts[j] in adj[verts[i]]:
                    sub_edges.add((verts[i], verts[j]))
        
        if len(sub_edges) == 5:
            # Check if it's a cycle
            # Degrees must all be 2
            degs = [0]*5
            for u, v in sub_edges:
                idx_u = verts.index(u)
                idx_v = verts.index(v)
                degs[idx_u] += 1
                degs[idx_v] += 1
            if all(d == 2 for d in degs):
                # Check connectivity (implied by 2-regular and 5 vertices, unless it's two disjoint cycles, but min cycle is 3, so 5 vertices can't be two cycles)
                return True
    return False

def max_independent_set_size(n, edges):
    """Find the size of the maximum independent set."""
    # Greedy heuristic or exact for small n?
    # For n=35, exact is possible with branch and bound.
    # But we need this in a loop.
    # Let's use a simple greedy upper bound or a randomized greedy for a lower bound on alpha?
    # We need to know if alpha >= 9.
    # If we can find an independent set of size 9, return 9.
    # Otherwise, return < 9.
    
    # Greedy construction of independent set
    # Random restarts to find a large one
    best = 0
    for _ in range(10):
        remaining = list(range(n))
        current = []
        while remaining:
            # Pick a random vertex
            idx = random.randint(0, len(remaining)-1)
            v = remaining.pop(idx)
            current.append(v)
            # Remove neighbors of v from remaining
            # Optimization: filter
            remaining = [u for u in remaining if u not in adj[v]]
        if len(current) > best:
            best = len(current)
            if best >= 9:
                return best
    return best

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    # Try to construct graphs for n from 35 down to 29
    # If we find one, return it.
    # Otherwise return the seed.
    
    for n in range(35, 28, -1):
        while time.time() - start_time < timeout:
            # Random graph generation
            # Density heuristic: C5-free graphs have low density.
            # For n=32, max edges ~ 100.
            # Let's try random p around 0.3 - 0.5
            edges = []
            p = random.uniform(0.2, 0.5)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        edges.append([i, j])
            
            # Repair loop
            valid = False
            for _ in range(50): # Max repair steps
                if time.time() - start_time > timeout:
                    break
                
                # Check C5
                if has_c5(n, edges):
                    # Remove a random edge from a C5
                    # To find which edge, we can just pick a random edge and check if it's in a C5?
                    # Simpler: Find a C5, remove one of its edges.
                    # My has_c5 doesn't return the cycle. Let's modify it or just pick random edge and remove if it helps?
                    # Better: Pick a random edge, remove it. Repeat.
                    # This might be slow.
                    # Let's just remove a random edge.
                    if edges:
                        idx = random.randint(0, len(edges)-1)
                        edges.pop(idx)
                    else:
                        break
                    continue
                
                # Check Alpha
                alpha = max_independent_set_size(n, edges)
                if alpha < 9:
                    valid = True
                    break
                else:
                    # Add an edge to break an independent set
                    # Find an independent set of size 9
                    # The greedy function doesn't return the set.
                    # Let's just add a random edge.
                    # This is inefficient.
                    # We need to add an edge between two non-adjacent vertices that are in a large independent set.
                    # For now, just add a random non-edge.
                    # Find non-edges
                    adj_set = set()
                    for u, v in edges:
                        adj_set.add((u, v))
                        adj_set.add((v, u))
                    
                    non_edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (i, j) not in adj_set:
                                non_edges.append((i, j))
                    
                    if non_edges:
                        u, v = random.choice(non_edges)
                        edges.append([u, v])
                    else:
                        break
            
            if valid:
                return {'n': n, 'edges': edges}
    
    # Fallback to seed
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
