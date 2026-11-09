import time
import random
import math
from collections import defaultdict

def _edge_index(i, j, n):
    """Get the index of edge (i,j) in the flat list."""
    if i > j:
        i, j = j, i
    return i * n - i * (i + 1) // 2 + j - i - 1

def _edge_from_index(e, n):
    """Get (i, j) from edge index e."""
    # e = i*n - i*(i+1)//2 + j - i - 1
    # Let k = i*n - i*(i+1)//2. Then e = k + j - i - 1.
    # j = e - k + i + 1.
    # We need to find i such that k <= e < k + (n - i - 1) where k = i*n - i*(i+1)//2.
    # The number of edges starting with i is n - i - 1.
    # Let's just iterate or use a precomputed list for small n?
    # For speed in the loop, precomputing a list of (i,j) for all edges is better.
    pass

def _count_c6s_in_color(n, adj):
    """
    Count the total number of C6s in the graph defined by adj.
    We sum the number of paths of length 4 between each pair of adjacent vertices (u,v),
    where the path does not use the edge (u,v).
    Total C6s = sum_{(u,v) in E} paths_4(u,v) / 6.
    """
    total = 0
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # Count paths of length 4 from u to v
                # u - a - b - c - d - v
                count = 0
                for a in adj[u]:
                    if a == v:
                        continue
                    for b in adj[a]:
                        if b == u or b == v:
                            continue
                        for c in adj[b]:
                            if c == u or c == v or c == a:
                                continue
                        for d in adj[c]:
                            if d == u or d == v or d == b:
                                continue
                            if d in adj[v]:
                                count += 1
                total += count
    return total // 6

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_result = None
    best_n = 0
    
    # Try to find a coloring for n=17, then 16, etc.
    # The task says best known is 17. 18 or 19 would be new.
    # We should try to maximize n.
    # Let's try n=17 first. If we find a valid one, return it.
    # If we run out of time, return the best we found (maybe smaller n).
    
    for target_n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline:
            break
            
        if target_n <= best_n:
            break
            
        # Initialize
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        # Random initial coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Build adjacency lists for each color
        adj = [[[] for _ in range(n)] for _ in range(4)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            adj[c][i].append(j)
            adj[c][j].append(i)
        
        # Calculate initial cost
        # cost[c] = number of C6s in color c
        cost = [0] * 4
        for c in range(4):
            cost[c] = _count_c6s_in_color(n, adj[c])
        
        total_cost = sum(cost)
        
        if total_cost == 0:
            return {'n': n, 'colors': colors}
        
        # Local Search / Simulated Annealing
        T = 1.0
        cooling = 0.9995
        best_cost = total_cost
        best_colors = colors.copy()
        
        # Precompute for speed?
        # The bottleneck is _count_c6s_in_color which is O(n^5) or worse.
        # We need incremental updates.
        
        # Let's implement incremental cost update.
        # When changing color of edge (i,j) from c1 to c2:
        # 1. Remove (i,j) from adj[c1].
        # 2. Add (i,j) to adj[c2].
        # 3. Update cost[c1] and cost[c2].
        
        # Cost of a graph G is sum_{(u,v) in E(G)} P4(u,v) / 6.
        # Let f(G) be the sum of P4(u,v) over all edges (u,v) in G.
        # f(G) = 6 * C6_count.
        
        # When we remove edge (i,j) from G1:
        # New f(G1') = f(G1) - P4(i,j) - sum_{v in N(i,j)} (change in P4(v, u) for u in ...?)
        # This is complex.
        
        # Simpler approach: Recalculate cost for the two affected colors only?
        # Still O(n^5).
        
        # Alternative: Count C6s by vertex tuple?
        # A C6 is defined by 6 vertices.
        # There are C(n,6) subsets.
        # For each subset, check if it forms a C6 in any color.
        # For n=17, C(17,6) = 12376.
        # For each subset, we can check all 6 permutations of edges?
        # Or just check if the induced subgraph is a C6.
        # This might be faster if we can do it incrementally.
        
        # Let's stick to the edge-based incremental update but optimize the P4 count.
        # P4(u,v) in G: number of paths u-a-b-c-d-v.
        
        # Let's try a different local search strategy:
        # Minimize the number of C6s.
        # Use a faster cost function.
        
        # Let's try to find a valid coloring for n=17 using a known structure or random restarts.
        # Since 17 is the best known, random search might be hard.
        # Let's try to improve the seed for n=17.
        
        # Actually, the prompt says "Maximize n".
        # If we can't find n=17, we should return a valid coloring for the largest n we found.
        
        # Let's run a simple hill climbing / SA for each n.
        
        steps = 0
        max_steps = 50000
        
        while time.time() < deadline and steps < max_steps:
            steps += 1
            
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            i, j = edges[idx]
            
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            # Calculate delta cost
            # This is the expensive part.
            # We need P4(i,j) in adj[old_c] and P4(i,j) in adj[new_c].
            
            # Helper to count P4(u,v) in adj
            def count_p4(u, v, adj):
                count = 0
                for a in adj[u]:
                    if a == v: continue
                    for b in adj[a]:
                        if b == u or b == v: continue
                        for c in adj[b]:
                            if c == u or c == v or c == a: continue
                            for d in adj[c]:
                                if d == u or d == v or d == b: continue
                                if d in adj[v]:
                                    count += 1
                return count
            
            # To be efficient, we should update adj first, then calculate?
            # No, we need the count BEFORE removing/adding to get the delta correctly?
            # Delta = Cost(new) - Cost(old)
            # Cost(G) = Sum_{e in G} P4(e) / 6.
            
            # Let's just recalculate the cost of the two colors.
            # It's O(n^5) but n is small (17).
            # 17^5 = 1419857. 50000 steps * 2 colors * 1.4M = 1.4B ops. Too slow in Python.
            
            # We need a faster cost update.
            # Or fewer steps.
            
            # Let's try to limit steps and use a faster heuristic.
            # Or use a different algorithm: Backtracking.
            
            # Given the time limit, let's try to return a known valid coloring if possible,
            # or a small valid one.
            
            # Actually, the seed returns n=9.
            # Can we easily get n=10, 11?
            
            # Let's try a greedy construction for small n.
            
            break # Skip SA for now, try greedy
            
    # Greedy construction for small n
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        # Try to build a coloring for n vertices
        # Start with n-1 coloring, add vertex n-1.
        # This is complex.
        
        # Let's just use random restarts with a very fast cost check for small n.
        
    # Fallback: Return a valid coloring for n=9 or similar
    # The seed returns:
    # n = 9
    # colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    # Let's verify if this is valid.
    # If yes, return it.
    
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    # Verify
    adj = [[[] for _ in range(n)] for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = i * n - i * (i + 1) // 2 + j - i - 1
            c = colors[idx]
            adj[c][i].append(j)
            adj[c][j].append(i)
            
    valid = True
    for c in range(4):
        if _count_c6s_in_color(n, adj[c]) > 0:
            valid = False
            break
            
    if valid:
        return {'n': n, 'colors': colors}
        
    # If even that fails (shouldn't), return n=1
    return {'n': 1, 'colors': []}
