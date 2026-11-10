import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth g on as few vertices as possible.
    Uses specific known constructions for small cases and a heuristic search for (4,10).
    Falls back to Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave margin
    
    # Handle specific known cages with hardcoded or fast constructions
    if k == 3 and g == 6:
        # Petersen Graph
        n = 10
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                 (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
                 (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        return n, edges

    if k == 3 and g == 8:
        # Tutte-Coxeter Graph (n=30)
        # This is a bipartite graph.
        # We can construct it using a known edge list or permutation.
        # For simplicity and speed, we return a known valid graph if we can.
        # The Tutte-Coxeter graph is the bipartite double cover of the 15-vertex (3,6) cage?
        # Actually, let's just use the LUW fallback for k=3, g=8 if not hardcoded, 
        # but LUW for k=3, g=8: q=3, d=3? Girth d+5=8. n=2*3^4 = 162? 
        # Best is 30. So LUW is bad here too.
        # I will implement a small search for k=3, g=8 as well if time permits, 
        # but for now, focus on k=4, g=10.
        pass

    if k == 4 and g == 10:
        # Target: 320 vertices (160 per side)
        # Heuristic Search
        n_target = 320
        n_side = 160
        search_deadline = start_time + 10.0  # Give 10s to search
        
        # Initialize random 4-regular bipartite graph
        # Permutations P0..P3
        perms = [list(range(n_side)) for _ in range(4)]
        for p in perms:
            random.shuffle(p)
        
        best_n = None
        best_edges = None
        
        def get_edges(perms):
            edges = []
            for i in range(n_side):
                for p in perms:
                    j = p[i]
                    edges.append((i, n_side + j))
            return edges

        def check_girth(perms, min_girth):
            # BFS from each node in one partition to find shortest cycle
            # Since it's bipartite, cycles are even.
            # We look for cycles of length < min_girth.
            # If min_girth=10, we check for 4, 6, 8.
            # Build adjacency list
            adj = [[] for _ in range(2 * n_side)]
            for i in range(n_side):
                for p in perms:
                    j = p[i]
                    u = i
                    v = n_side + j
                    adj[u].append(v)
                    adj[v].append(u)
            
            # BFS from each node in U (0..n_side-1)
            # To speed up, we can stop if we find a cycle < min_girth
            for start in range(n_side):
                visited = [-1] * (2 * n_side)
                parent = [-1] * (2 * n_side)
                dist = [0] * (2 * n_side)
                queue = [start]
                visited[start] = start
                
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if visited[v] == -1:
                            visited[v] = u
                            parent[v] = u
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]: # Avoid backtracking
                            # Found a cycle
                            # Length is dist[u] + dist[v] + 1? No.
                            # In BFS tree, if we see a visited node v that is not parent, 
                            # the cycle length is dist[u] + dist[v] + 1?
                            # Actually, standard BFS girth check:
                            # If v is visited and v != parent[u], cycle length is dist[u] + dist[v] + 1.
                            # Wait, this is for general graphs.
                            # For bipartite, we can just check distance.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
                            # If we found a cycle, we don't need to continue this BFS
                            # if we already found a short one, but we want the MINIMUM.
                            # Actually, BFS finds shortest paths.
                            # The first time we see a cross edge, it's the shortest cycle involving that edge.
                            # We need the global minimum.
                            pass
                # Optimization: If we complete BFS from start without finding cycle < min_girth,
                # it doesn't mean the whole graph is girth >= min_girth.
                # We need to check all starts.
            return True

        # The check_girth above is O(N^2) per start, so O(N^3) total. 160^3 is ~4M. Fast.
        # But we need to find the cycle to break it.
        
        def find_short_cycle(perms, min_girth):
            adj = [[] for _ in range(2 * n_side)]
            for i in range(n_side):
                for p in perms:
                    j = p[i]
                    u = i
                    v = n_side + j
                    adj[u].append(v)
                    adj[v].append(u)
            
            for start in range(n_side):
                visited = [-1] * (2 * n_side)
                parent = [-1] * (2 * n_side)
                dist = [0] * (2 * n_side)
                queue = [start]
                visited[start] = start
                
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if visited[v] == -1:
                            visited[v] = u
                            parent[v] = u
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                # Reconstruct cycle
                                path1 = []
                                curr = u
                                while curr != -1:
                                    path1.append(curr)
                                    curr = parent[curr]
                                path1.reverse()
                                path2 = []
                                curr = v
                                while curr != -1:
                                    path2.append(curr)
                                    curr = parent[curr]
                                path2.reverse()
                                # Find LCA? No, they meet at start?
                                # Actually, the cycle is path1 + path2 reversed?
                                # Simplest: just return the edges involved.
                                # For breaking, we can just pick one edge from the cycle.
                                # Let's return the edge (u, v) which is a cross edge.
                                return (u, v), cycle_len
            return None, min_girth

        # Search Loop
        iterations = 0
        while time.time() < search_deadline:
            iterations += 1
            # Check girth
            edge, cycle_len = find_short_cycle(perms, g)
            if edge is None:
                # Girth is at least g
                best_n = n_target
                best_edges = get_edges(perms)
                break
            
            # Break the cycle by swapping edges
            # Pick a random edge in the graph and swap with the problematic edge?
            # Simple heuristic: Pick the edge (u, v) from the cycle.
            # u is in U, v is in V.
            # Find the permutation index p_idx such that perms[p_idx][u] == v - n_side.
            u, v = edge
            v_idx = v - n_side
            p_idx = -1
            for i in range(4):
                if perms[i][u] == v_idx:
                    p_idx = i
                    break
            
            if p_idx != -1:
                # Swap this edge with a random other edge
                # Pick a random node u2 in U and random perm p2
                u2 = random.randint(0, n_side - 1)
                p2 = random.randint(0, 3)
                if u2 != u and p2 != p_idx:
                    # Swap perms[p_idx][u] and perms[p2][u2]
                    # This maintains 4-regularity
                    perms[p_idx][u], perms[p2][u2] = perms[p2][u2], perms[p_idx][u]
        
        if best_n is not None:
            return best_n, best_edges

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q)
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
