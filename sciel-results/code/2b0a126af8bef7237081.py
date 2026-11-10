"""C6-free graph on 40 vertices: maximize edges via local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _has_c6(adj, n):
    """Check if graph with adjacency matrix adj (n x n bool) contains a C6.
    A C6 is a cycle of length 6: v0-v1-v2-v3-v4-v5-v0.
    We enumerate pairs (v0, v1) and (v3, v4) and check if there exist
    v2, v5 completing the cycle.
    
    Actually, a more efficient approach: for each pair of vertices (a,b) that are
    NOT adjacent, count the number of common neighbors. If we can find a path of
    length 3 between two non-adjacent vertices... 
    
    Better: A C6 exists iff there exist two vertices u, v that are not adjacent
    and have at least 2 common neighbors, AND those two common neighbors have
    a common neighbor (other than u, v) that is not adjacent to u or v... 
    
    Actually the cleanest: enumerate all 6-cycles by checking for each ordered
    pair of non-adjacent vertices (s, t), whether there exists a path of length 5
    from s to t. But that's expensive.
    
    Simplest correct approach: for each pair (i,j) with i<j, if not adj[i][j],
    check if there's a path of length 5 from i to j. This is O(n^2 * n^3) worst case.
    
    Even simpler: A 6-cycle v0-v1-v2-v3-v4-v5-v0 requires:
    adj[v0][v1], adj[v1][v2], adj[v2][v3], adj[v3][v4], adj[v4][v5], adj[v5][v0]
    all true, and all 6 vertices distinct.
    
    We can enumerate (v0, v1, v2, v3) and check if there exist v4, v5 such that
    adj[v3][v4] and adj[v4][v5] and adj[v5][v0] and all distinct.
    
    For n=40, the number of 4-tuples is 40*39*38*37 ~ 2.2M, and for each we'd
    check up to 40*40 = 1600 pairs. That's ~3.5B which is too slow in Python.
    
    Better approach: For each pair (v0, v1) that are adjacent, and each pair (v3, v4)
    that are adjacent, check if there exists v2 adjacent to both v1 and v3, and
    v5 adjacent to both v4 and v0, with all 6 distinct.
    
    Actually, let's use the standard approach:
    For each unordered pair {a, b} that are NOT adjacent:
      Let N(a) = neighbors of a, N(b) = neighbors of b
      We need a path a - x - y - z - w - b of length 5 (6 vertices)
      i.e., x in N(a), z in N(b), y adjacent to x and z, w adjacent to y and z... 
      wait, the path is a-x-y-z-w-b so:
      x in N(a), w in N(b), y in N(x) ∩ N(z)... no.
      a-x: x in N(a)
      x-y: y in N(x)
      y-z: z in N(y)
      z-w: w in N(z)
      w-b: w in N(b)
      
    This is getting complex. Let me use a different method.
    
    For n=40, a practical approach: enumerate all 6-tuples (v0,v1,v2,v3,v4,v5)
    with v0 < v1 < v3 < v4 < v5 (to reduce duplicates) and check the cycle edges.
    But that's still large.
    
    Most practical for n=40 in Python: For each pair of non-adjacent vertices (s,t),
    check if there's a path of length exactly 5 from s to t. Use BFS/DFS with depth limit.
    
    Actually, the fastest for small n: 
    For each 6-cycle, it has 3 pairs of opposite vertices. Let's just do:
    for v0 in range(n):
      for v1 in range(n): if v1 != v0 and adj[v0][v1]:
        for v2 in range(n): if v2 != v0 and v2 != v1 and adj[v1][v2]:
          for v3 in range(n): if v3 not in (v0,v1,v2) and adj[v2][v3]:
            for v4 in range(n): if v4 not in (v0,v1,v2,v3) and adj[v3][v4]:
              for v5 in range(n): if v5 not in (v0,v1,v2,v3,v4) and adj[v4][v5] and adj[v5][v0]:
                return True
    This is O(n^6) worst case which for n=40 is 40^6 = 4B - way too slow.
    
    Let me use a smarter approach. For C6 detection:
    A C6 can be characterized by its three "diagonals" (pairs of opposite vertices).
    
    Alternative: For each pair of vertices (u, v) that are at distance exactly 2
    (i.e., not adjacent but share a common neighbor), count paths of length 4 between them.
    If any pair at distance 2 has a path of length 4, then together with the length-2
    path we get a C6 (or a shorter cycle, but we need to check).
    
    Actually, the cleanest efficient method for n=40:
    For each unordered pair {a,b} with a<b:
      If adj[a][b], skip
      Count the number of common neighbors of a and b. If < 2, skip (can't form C6 through a,b as opposite vertices)
      Wait, in a C6, opposite vertices are at distance 3, not 2.
    
    In a 6-cycle v0-v1-v2-v3-v4-v5-v0:
    - v0 and v3 are opposite (distance 3)
    - v0 and v2 are at distance 2
    - v0 and v1 are adjacent
    
    So for a pair (a,b) to be opposite in a C6, they need a path of length 3 between them
    (a-x-y-b) AND a path of length 3 between them going the other way (a-z-w-b),
    with all 6 vertices distinct.
    
    So: for each pair (a,b) not adjacent:
      Find all length-3 paths from a to b: a-x-y-b where adj[a][x], adj[x][y], adj[y][b]
      If there are two such paths with disjoint internal vertices {x,y} and {x',y'}, 
      then we have a C6.
    
    Number of length-3 paths from a to b: for each x in N(a), for each y in N(b), check adj[x][y].
    That's O(deg(a)*deg(b)) per pair, and we do this for O(n^2) pairs. Total O(n^2 * avg_deg^2).
    For n=40, avg_deg ~ 5 (if 100 edges), this is 40^2 * 25 = 40000. Very fast!
    
    So the algorithm:
    For each pair (a,b) with a < b and not adj[a][b]:
      paths = []
      for x in N(a):
        for y in N(b):
          if x != y and adj[x][y]:
            paths.append((x, y))
      # Check if any two paths have disjoint internal vertex sets
      for i in range(len(paths)):
        for j in range(i+1, len(paths)):
          x1, y1 = paths[i]
          x2, y2 = paths[j]
          if len({x1, y1, x2, y2}) == 4:
            return True
    return False
    """
    n = adj.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a][b]:
                continue
            # Find all length-3 paths from a to b
            paths = []
            for x in range(n):
                if x == a or x == b or not adj[a][x]:
                    continue
                for y in range(n):
                    if y == a or y == b or y == x or not adj[b][y] or not adj[x][y]:
                        continue
                    paths.append((x, y))
            # Check for two paths with disjoint internal vertices
            for i in range(len(paths)):
                x1, y1 = paths[i]
                for j in range(i + 1, len(paths)):
                    x2, y2 = paths[j]
                    if x1 != x2 and x1 != y2 and y1 != x2 and y1 != y2:
                        return True
    return False


def _edge_count(adj):
    return int(adj.sum() // 2)


def construct():
    n = 40
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Initialize: 8 disjoint K5s (80 edges, C6-free)
    import numpy as np
    adj = np.zeros((n, n), dtype=bool)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                adj[u][v] = adj[v][u] = True
    
    best_edges = _edge_count(adj)
    best_adj = adj.copy()
    
    # Local search: try adding edges, if C6 created, try to fix by removing edges
    # Strategy: simulated annealing style
    # 1. Try to add a random non-edge
    # 2. If it creates a C6, try to remove one of the edges in the C6 to fix it
    #    (this keeps edge count the same or we can try other fixes)
    # 3. Occasionally do edge swaps (remove one edge, add another)
    
    rng = random.Random(42)
    
    def try_add_edge(adj, u, v):
        """Try adding edge (u,v). Return (success, new_adj, c6_edges)"""
        adj[u][v] = adj[v][u] = True
        if not _has_c6(adj, n):
            return True, adj, None
        # C6 created - find which edges are in some C6
        adj[u][v] = adj[v][u] = False
        return False, adj, None
    
    def find_c6_edges(adj, u, v):
        """Find edges that participate in a C6 involving edge (u,v).
        Actually, let's just find any C6 in the graph."""
        # Find a C6 and return its edges
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a][b]:
                    continue
                paths = []
                for x in range(n):
                    if x == a or x == b or not adj[a][x]:
                        continue
                    for y in range(n):
                        if y == a or y == b or y == x or not adj[b][y] or not adj[x][y]:
                            continue
                        paths.append((x, y))
                for i in range(len(paths)):
                    x1, y1 = paths[i]
                    for j in range(i + 1, len(paths)):
                        x2, y2 = paths[j]
                        if x1 != x2 and x1 != y2 and y1 != x2 and y1 != y2:
                            # C6: a-x1-y1-b-y2-x2-a
                            return [(a, x1), (x1, y1), (y1, b), (b, y2), (y2, x2), (x2, a)]
        return None
    
    # Main search loop
    current_adj = adj.copy()
    current_edges = _edge_count(current_adj)
    
    # Phase 1: Greedy edge addition
    # Try all possible edges, add if no C6
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if not current_adj[u][v]:
                    current_adj[u][v] = current_adj[v][u] = True
                    if not _has_c6(current_adj, n):
                        changed = True
                        current_edges += 1
                    else:
                        current_adj[u][v] = current_adj[v][u] = False
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj.copy()
    
    # Phase 2: Local search with edge swaps
    # Try: remove edge e1, add edge e2, see if we can increase edge count
    # or maintain edge count but get to a better position
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 20 and no_improve < max_no_improve:
        # Pick a random edge to remove
        edges_list = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i][j]]
        if not edges_list:
            break
        e1 = rng.choice(edges_list)
        
        # Try adding a random non-edge
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not current_adj[i][j]]
        if not non_edges:
            break
        e2 = rng.choice(non_edges)
        
        # Remove e1, add e2
        u1, v1 = e1
        u2, v2 = e2
        current_adj[u1][v1] = current_adj[v1][u1] = False
        current_adj[u2][v2] = current_adj[v2][u2] = True
        
        if _has_c6(current_adj, n):
            # Revert
            current_adj[u1][v1] = current_adj[v1][u1] = True
            current_adj[u2][v2] = current_adj[v2][u2] = False
            no_improve += 1
        else:
            # Valid swap (edge count unchanged)
            no_improve += 1
            
            # Try to add more edges now
            for u in range(n):
                for v in range(u + 1, n):
                    if not current_adj[u][v]:
                        current_adj[u][v] = current_adj[v][u] = True
                        if not _has_c6(current_adj, n):
                            current_edges += 1
                            no_improve = 0
                        else:
                            current_adj[u][v] = current_adj[v][u] = False
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = current_adj.copy()
                no_improve = 0
    
    # Phase 3: Multi-edge perturbation
    while time.time() < deadline - 10:
        # Remove 2-4 random edges
        k = rng.randint(2, 4)
        edges_list = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i][j]]
        if len(edges_list) < k:
            break
        removed = rng.sample(edges_list, k)
        for u, v in removed:
            current_adj[u][v] = current_adj[v][u] = False
            current_edges -= 1
        
        # Try to add edges greedily
        for u in range(n):
            for v in range(u + 1, n):
                if not current_adj[u][v]:
                    current_adj[u][v] = current_adj[v][u] = True
                    if not _has_c6(current_adj, n):
                        current_edges += 1
                    else:
                        current_adj[u][v] = current_adj[v][u] = False
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj.copy()
    
    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                result.append([u, v])
    return result

# EVOLVE-BLOCK-END
