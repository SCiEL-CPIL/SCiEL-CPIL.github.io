import time
import random
from random import Random

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    # We will process each instance.
    # To be safe, allocate time proportionally or just round-robin.
    # Given small sizes, we can just run a loop for each instance with a sub-deadline.
    
    for idx, (n, d, seed) in enumerate(specs):
        # Generate graph
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        
        edges = sorted(edges)
        m = len(edges)
        
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflicts
        # conflicts[i] is a list of edge indices j such that edge i and edge j conflict
        conflicts = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            # Conflict if share vertex
            for w in adj[u]:
                if w != v:
                    # Edge (u, w) conflicts with (u, v)
                    # Find index of (u, w)
                    e2 = (min(u, w), max(u, w))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for w in adj[v]:
                if w != u:
                    e2 = (min(v, w), max(v, w))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            
            # Conflict if joined by an edge
            # Edge (u, v) is joined by (u, x) or (v, x) or (u, y) or (v, y) where (x,y) is an edge?
            # Definition: "no two edges share a vertex or are joined by an edge"
            # Joined by an edge means there is an edge in the graph connecting an endpoint of e to an endpoint of f.
            # So if e=(u,v) and f=(x,y), they conflict if:
            # 1. {u,v} & {x,y} != empty
            # 2. (u,x) in edges OR (u,y) in edges OR (v,x) in edges OR (v,y) in edges
            
            # My loop above covers:
            # - w in adj[u] means (u,w) is an edge. If w is in f, then (u,w) connects u (in e) to w (in f).
            #   But my loop adds conflict if (u,w) is an edge.
            #   Wait, if (u,w) is an edge, and w is an endpoint of f, then e and f are joined by edge (u,w).
            #   So yes, if w is an endpoint of f, they conflict.
            #   My current loop adds ALL edges incident to u (except v) as conflicts.
            #   This is correct for "joined by an edge" IF the other edge shares a vertex with f.
            #   But I am adding ALL edges incident to u.
            #   Let's re-verify.
            #   If e=(u,v) and g=(u,w), they share u. Conflict.
            #   If e=(u,v) and g=(x,y) where (u,x) is an edge.
            #   Then x is in adj[u]. So I add index of (u,x) to conflicts.
            #   But g is (x,y). I need to check if g conflicts with e.
            #   The condition is: e and g conflict if they share a vertex OR are joined by an edge.
            #   If (u,x) is an edge, then e and g are joined by (u,x).
            #   So e conflicts with g.
            #   So for a fixed e=(u,v), any edge g=(x,y) such that x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v] conflicts with e?
            #   No.
            #   If (u,x) is an edge, then e=(u,v) and g=(x,y) are joined by (u,x). So they conflict.
            #   So YES, any edge g that has an endpoint in adj[u] (excluding v) or adj[v] (excluding u) conflicts with e?
            #   Wait. If g=(x,y) and (u,x) is an edge, then they are joined.
            #   So for e=(u,v), the conflicting edges are:
            #   1. Edges sharing u or v.
            #   2. Edges (x,y) where x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v].
            #   
            #   Let's check the loop I wrote:
            #   for w in adj[u]:
            #       if w != v:
            #           e2 = (u, w)
            #           conflicts[i].add(index(e2))
            #   This adds edges (u,w). These share u. Correct.
            #   But it does NOT add edges (x,y) where (u,x) is an edge but y is not u or v.
            #   Example: e=(0,1). adj[0]={1,2,3}.
            #   Edge g=(2,4). (0,2) is an edge.
            #   e and g are joined by (0,2). So they conflict.
            #   My loop adds (0,2). It does NOT add (2,4).
            #   So my conflict set is INCOMPLETE.
            
        # Correct Conflict Precomputation:
        conflicts = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            # 1. Share vertex
            for w in adj[u]:
                if w != v:
                    e2 = (min(u, w), max(u, w))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for w in adj[v]:
                if w != u:
                    e2 = (min(v, w), max(v, w))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            
            # 2. Joined by an edge
            # If there is an edge (u, x), then any edge (x, y) conflicts with e?
            # Yes, because (u, x) connects u (in e) to x (in g).
            # So for each neighbor w of u (w != v), any edge incident to w conflicts with e?
            # No, only if the edge is (w, z).
            # So for each w in adj[u] (w != v), all edges incident to w conflict with e?
            # Let's check: e=(u,v), g=(w,z). Edge (u,w) exists.
            # u is in e, w is in g. (u,w) connects them.
            # So yes, e and g conflict.
            # So for each w in adj[u] (w != v), add all edges incident to w.
            # But we already added (u,w) in step 1.
            # We need to add (w,z) for all z in adj[w].
            
            for w in adj[u]:
                if w != v:
                    for z in adj[w]:
                        if z != u: # (u,w) already handled, but (w,z) might be same as (u,w) if z=u? No, z != u.
                            e2 = (min(w, z), max(w, z))
                            if e2 in edge_to_idx:
                                conflicts[i].add(edge_to_idx[e2])
            
            for w in adj[v]:
                if w != u:
                    for z in adj[w]:
                        if z != v:
                            e2 = (min(w, z), max(w, z))
                            if e2 in edge_to_idx:
                                conflicts[i].add(edge_to_idx[e2])
                            
        # Remove self
        for i in range(m):
            conflicts[i].discard(i)

        # Greedy + Local Search
        best_coloring = None
        best_score = float('inf')
        
        # Sub-deadline for this instance
        # If 3 instances, ~45s each.
        # Let's use a global deadline check inside the loop.
        
        while time.time() < deadline - 5.0:
            # Random Order
            order = list(range(m))
            random.shuffle(order)
            
            # Greedy
            color = [-1] * m
            for e in order:
                taken = set()
                for f in conflicts[e]:
                    if color[f] != -1:
                        taken.add(color[f])
                c = 0
                while c in taken:
                    c += 1
                color[e] = c
            
            # Local Search
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                K = max(color) + 1
                if K <= 1: break
                
                # Try to reduce K
                # Sort edges by color, highest first
                edges_by_color = [[] for _ in range(K)]
                for i in range(m):
                    edges_by_color[color[i]].append(i)
                
                # Try to eliminate color K-1
                # We iterate through edges of color K-1 and try to move them to lower colors
                # If we can move ALL, K decreases.
                # But moving one might block another.
                # Simple approach:
                # For each pass:
                for i in range(m):
                    if color[i] == -1: continue
                    c = color[i]
                    if c == 0: continue
                    # Try to find smallest c' < c such that i can be colored c'
                    taken = set()
                    for f in conflicts[i]:
                        if color[f] != -1:
                            taken.add(color[f])
                    
                    c_prime = 0
                    while c_prime in taken:
                        c_prime += 1
                    
                    if c_prime < c:
                        color[i] = c_prime
                        improved = True
                        # Re-compress? Or just let it be.
                        # If we lower a color, we might create a gap.
                        # We should re-compress to keep K tight.
                        # But re-compressing every step is expensive.
                        # We can re-compress at the end of the pass.
            
                # Re-compress colors
                used_colors = set(color)
                if len(used_colors) < K:
                    mapping = {c: i for i, c in enumerate(sorted(used_colors))}
                    for i in range(m):
                        color[i] = mapping[color[i]]
                    improved = True # We made progress in compression
        
            # Evaluate
            score = max(color) + 1
            if score < best_score:
                best_score = score
                best_coloring = color[:]
                
            # If we hit the theoretical bound, we can stop early for this instance
            # Bound is 5*d*d/4
            bound = int(5 * d * d / 4)
            if best_score <= bound:
                break
                
        results.append(best_coloring)
        
    return results
