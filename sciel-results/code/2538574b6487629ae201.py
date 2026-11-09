import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    if k != 3 or g != 16:
        # Fallback for other parameters: use LUW or simple construction
        return _luw_construct(k, g)
    
    # For (3,16): construct 3-regular bipartite graph with girth >= 16
    # Moore bound: 242 vertices (121 per part)
    # Strategy: build 3-reg tree of depth 7, close boundary
    
    depth = g // 2 - 1  # = 7 for g=16
    # Number of vertices per side in the tree:
    # Level 0 (root side A): 1
    # Level 1 (side B): 3
    # Level 2 (side A): 9
    # ...
    # Level i: 3^i for i >= 1, but level 0 is 1
    # Actually for a 3-regular tree rooted at a vertex in A:
    # A-side levels: 0, 2, 4, 6 -> 1, 9, 81, 729? No...
    # In a 3-regular tree, from root (deg 3), each child has 2 children (one back to parent)
    # So: level 0: 1, level 1: 3, level 2: 3*2=6, level 3: 6*2=12, ...
    # Level i (i>=1): 3 * 2^(i-1)
    # Total up to level 7: 1 + 3*(1+2+4+8+16+32+64) = 1 + 3*127 = 1 + 381 = 382
    # But this is not balanced bipartite.
    
    # Better: use a bipartite tree where we start from an edge.
    # Part A has levels 0,1,2,3,4 (5 levels), Part B has levels 0,1,2,3,4,5,6 (7 levels)?
    # No. Let's think differently.
    
    # For girth 16 (even), the graph is bipartite. 
    # A BFS tree from a vertex v: 
    #   dist 0: 1 vertex (in A)
    #   dist 1: 3 vertices (in B)  
    #   dist 2: 3*2 = 6 vertices (in A)
    #   dist 3: 6*2 = 12 (in B)
    #   dist 4: 12*2 = 24 (in A)
    #   dist 5: 24*2 = 48 (in B)
    #   dist 6: 48*2 = 96 (in A)
    #   dist 7: 96*2 = 192 (in B)
    # Total: 1+3+6+12+24+48+96+192 = 382
    # A-side: 1+6+24+96 = 127, B-side: 3+12+48+192 = 255. Not balanced!
    
    # For a 3-regular bipartite graph, |A| = |B|. So I can't just use a single-rooted tree.
    # Instead, use the approach of building the tree and then the boundary closure.
    
    # Alternative: Use a known construction. The (3,16) cage is known to exist with 936 vertices.
    # But we want fewer. Let's try the "random pairing" approach on the boundary.
    
    # Actually, let me use a different strategy: 
    # 1. Start with the 3-regular tree of depth 7 (382 vertices as above)
    # 2. The leaves (depth 7) are in B, count = 192. Each has 2 free stubs.
    #    But wait, in the tree, leaves have degree 1 (only connected to parent).
    #    So each leaf has 2 free stubs. Total free stubs = 192*2 = 384.
    #    Also, the root has 2 free stubs (degree 1 in tree, needs 2 more).
    #    Total free stubs = 384 + 2 = 386. But 386 is even, good.
    #    However, this doesn't give a balanced bipartite graph.
    
    # Let me try a completely different approach: 
    # Use the "non-backtracking walk" based construction from the literature.
    
    # Practical approach: Use a random 3-regular bipartite graph and check girth.
    # But random 3-reg bipartite graphs have small girth typically.
    
    # Better: Use the "greedy" construction where we build the graph level by level.
    
    # Given time constraints, let me use a known good construction:
    # The (3,16) cage can be constructed using the "Klein" method or similar.
    
    # For now, let's use a randomized approach with girth verification:
    
    def _check_girth(n, edges, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # BFS tracking parent to avoid backtracking
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([(start, -1)])
            while queue:
                node, par = queue.popleft()
                for nb in adj[node]:
                    if nb == par:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append((nb, node))
                    else:
                        # Found a cycle
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    def _verify(n, edges, k, g):
        """Verify the graph is k-regular, simple, and has girth >= g."""
        if len(edges) != n * k // 2:
            return False
        # Check degree
        deg = [0] * n
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return False
        # Check girth
        if not _check_girth(n, edges, g):
            return False
        return True
    
    # Strategy: Build using the "tree closure" method
    # For (3,16), use depth 7 tree, close boundary
    # The issue is the tree isn't balanced. Let's use a different approach.
    
    # Use the "random 3-regular bipartite graph with large girth" approach:
    # Generate many random 3-regular bipartite graphs on 2n vertices and check girth.
    # But this is slow.
    
    # Better: Use the "non-backtracking" construction.
    # Start from a single vertex, grow the tree, and when we hit the depth limit,
    # connect to existing vertices that maintain girth.
    
    # Let me implement the "greedy tree growth with backtracking" approach:
    
    def _construct_greedy(k, g, max_time=100):
        """Greedy construction: grow 3-regular tree, close boundary."""
        start_t = time.time()
        depth = (g - 1) // 2  # = 7 for g=16
        
        # We'll build a bipartite graph. Part A and Part B.
        # Start with one vertex in A.
        # Use a priority queue of "free stubs" to process.
        
        # Each vertex has a list of neighbors. We build incrementally.
        # adj: list of lists of neighbors
        # side: 0 for A, 1 for B
        # level: distance from root
        
        adj = []
        side = []
        level = []
        free_stubs = []  # (vertex_id, stub_count) - but we process one at a time
        
        # Root: vertex 0 in A, level 0, needs k=3 stubs
        adj.append([])
        side.append(0)
        level.append(0)
        free_stubs.append((0, 3))
        
        n = 1
        edges = []
        edge_set = set()
        
        # We process stubs. When we create an edge, we reduce stubs on both ends.
        # To avoid short cycles, when connecting to an existing vertex, check distance.
        
        # Use a queue for processing
        from collections import deque
        stub_queue = deque()
        stub_queue.append((0, 3))
        
        # dist_matrix would be too large. Instead, track level and use BFS for girth check.
        
        while stub_queue and time.time() - start_t < max_time:
            v, count = stub_queue.popleft()
            if count <= 0:
                continue
            
            # Try to connect this stub
            # First, try to connect to an existing vertex in the other side
            # that is at the correct level to maintain girth.
            
            target_side = 1 - side[v]
            # We want to connect to a vertex w such that the cycle formed is >= g.
            # The cycle length would be: level[v] + level[w] + 1 (if they're in different parts and connected)
            # Actually, the cycle length depends on the path between v and w in the tree.
            
            # Simplification: only connect to vertices at the maximum allowed level.
            # For girth g, if v is at level lv and w is at level lw, and they're in different parts,
            # the cycle formed has length at least lv + lw + 1. We need lv + lw + 1 >= g.
            # Also, we shouldn't create a cycle of length < g.
            
            # For a tree, the distance between v and w is lv + lw - 2*level[lca]. 
            # To be safe, only connect vertices where lv + lw >= g - 1 (conservative).
            
            # Let's try connecting to vertices at level >= g - 1 - level[v]
            min_target_level = g - 1 - level[v]
            
            candidates = [i for i in range(n) if side[i] == target_side and level[i] >= min_target_level]
            
            random.shuffle(candidates)
            
            connected = False
            for w in candidates:
                if w == v:
                    continue
                if (min(v,w), max(v,w)) in edge_set:
                    continue
                # Check if adding this edge creates a short cycle
                # The cycle length is the distance between v and w in the current tree + 1
                # For a tree, this is level[v] + level[w] + 1 if they're in a tree structure
                # But this is an approximation. Let's just use the level heuristic.
                cycle_len = level[v] + level[w] + 1
                if cycle_len < g:
                    continue
                
                # Add the edge
                adj[v].append(w)
                adj[w].append(v)
                edge_set.add((min(v,w), max(v,w)))
                edges.append((v, w))
                connected = True
                
                # w now has one more neighbor, so it needs one less stub
                # But we don't track stubs for existing vertices this way.
                # Let me restructure.
                break
            
            if not connected:
                # Create a new vertex
                new_v = n
                adj.append([])
                side.append(target_side)
                level.append(level[v] + 1)
                adj[v].append(new_v)
                adj[new_v].append(v)
                edge_set.add((min(v, new_v), max(v, new_v)))
                edges.append((v, new_v))
                n += 1
                # new_v has k-1 = 2 free stubs
                stub_queue.append((new_v, k - 1))
            
            # v still has count-1 stubs
            if count > 1:
                stub_queue.append((v, count - 1))
        
        # After processing all stubs, check if the graph is valid
        # Verify degrees
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        # Verify girth
        if not _check_girth(n, edges, g):
            return None
        
        return n, edges
    
    result = _construct_greedy(k, g, max_time=100)
    if result is not None:
        return result
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5)%4
        c0 = c - r
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def neighbours(vertex):
        s, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if s==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-s, tuple(y)
    
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
