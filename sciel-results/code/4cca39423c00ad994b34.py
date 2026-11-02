"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
    
    n_luw, edges_luw = len(order), edges
    best_n, best_edges = n_luw, edges_luw

    # Attempt Greedy Random Search to find a smaller graph.
    # Strategy: Grow a tree-like structure, keeping track of "dangling" half-edges.
    # When a dangling half-edge is to be connected to another, we check the distance.
    # If distance >= g - 1, we can connect them (creating a cycle of length >= g).
    # Otherwise, we must extend the tree with a new node.
    
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    def try_greedy():
        nonlocal best_n, best_edges
        if time.time() > deadline:
            return
        
        # State
        adj = []  # adj[i] = list of neighbors
        dangling = []  # list of (node, slot_count) ? No, just list of nodes with free slots.
        # Actually, we need to know how many free slots each node has.
        free_slots = []
        
        # Initialize with a single node 0 with k free slots.
        # But a single node with k=3 is a star center? No, it's just a node.
        # Let's start with a path of length 2 to avoid small cycles immediately?
        # Or just start with one node.
        
        # Let's start with one node.
        nodes = 1
        adj.append([])
        free_slots.append(k)
        dangling = [0]
        
        # To avoid creating triangles immediately, we should probably not connect a node to its parent?
        # But we don't have a tree structure explicitly, just a graph.
        # The "dangling" list contains nodes that still have free degree.
        # When we pick a node u from dangling, we want to connect its free slot to something.
        # Option 1: Connect to a new node v.
        # Option 2: Connect to an existing node w in dangling (or not in dangling?)
        # If we connect to w, we form a cycle. The length of the cycle is dist(u, w) + 1.
        # We need dist(u, w) + 1 >= g.
        
        # We will use a strategy:
        # 1. Pick a random node u from dangling.
        # 2. Try to connect to a random node w from dangling (w != u).
        # 3. If dist(u, w) >= g - 1, connect.
        # 4. Else, create a new node v, connect u-v.
        
        # To make this efficient, we need fast distance queries.
        # For small graphs, BFS is fine.
        
        def dist(u, w):
            if u == w:
                return 0
            visited = {u: 0}
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                if visited[curr] >= g - 1: # Prune if we've gone deep enough
                    continue
                for nb in adj[curr]:
                    if nb not in visited:
                        visited[nb] = visited[curr] + 1
                        if nb == w:
                            return visited[nb]
                        queue.append(nb)
            return -1 # Disconnected (should not happen in this construction)
        
        max_nodes = n_luw // 2 # Don't try to build something larger than LUW
        max_steps = 5000 # Hard cap on steps to prevent infinite loops in greedy
        
        for _ in range(max_steps):
            if time.time() > deadline:
                return
            if len(dangling) == 0:
                # Graph is complete (all degrees satisfied)
                # Check if it's better
                if len(adj) < best_n:
                    # Verify girth and degree
                    if verify_graph(adj, k, g):
                        best_n = len(adj)
                        # Convert adj to edges
                        edges = []
                        for u in range(len(adj)):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_edges = edges
                return # Success, stop searching for this attempt? Or try more?
                # Since we found a valid smaller graph, we can stop this greedy run.
                # But maybe we can find even smaller?
                # For now, let's just return to the main loop which might restart greedy.
            
            if len(adj) > max_nodes:
                return # Failed to close within budget
            
            # Pick a random dangling node
            u_idx = random.randint(0, len(dangling) - 1)
            u = dangling[u_idx]
            
            # Try to connect to another dangling node
            # To increase chance of success, maybe try a few random candidates
            connected = False
            candidates = list(dangling)
            random.shuffle(candidates)
            
            # Try up to 5 random candidates from dangling
            for w in candidates[:5]:
                if w == u:
                    continue
                d = dist(u, w)
                if d >= g - 1:
                    # Connect u and w
                    adj[u].append(w)
                    adj[w].append(u)
                    free_slots[u] -= 1
                    free_slots[w] -= 1
                    
                    if free_slots[u] == 0:
                        dangling.remove(u)
                    if free_slots[w] == 0:
                        dangling.remove(w)
                    
                    connected = True
                    break
            
            if not connected:
                # Create a new node
                v = len(adj)
                adj.append([])
                free_slots.append(k)
                adj[u].append(v)
                adj[v].append(u)
                free_slots[u] -= 1
                free_slots[v] -= 1
                
                if free_slots[u] == 0:
                    dangling.remove(u)
                
                dangling.append(v)
            
        # If we didn't close, it's a failure (incomplete graph)
        return

    def verify_graph(adj, k, g):
        # Check degree
        for i in range(len(adj)):
            if len(adj[i]) != k:
                return False
            if len(set(adj[i])) != k:
                return False # Multi-edge or loop? Loops are i in adj[i].
            if i in adj[i]:
                return False
        
        # Check girth using non-backtracking walks?
        # Simpler: BFS from each node to find shortest cycle containing that node?
        # Or just BFS from each node to all others and check distances?
        # A graph has girth < g iff there is a cycle of length < g.
        # We can check for cycles of length < g by doing BFS from each node.
        # For each node u, do BFS. If we find a node v at distance d < g/2, and there is another path?
        # Actually, standard way:
        # For each edge (u, v), remove it and check if u and v are connected by a path of length < g - 1.
        # This is O(E * (V+E)) which might be slow for V=500. 500 * 1000 = 500,000. OK.
        
        n = len(adj)
        # Build adjacency sets for faster lookup?
        adj_set = [set(adj[i]) for i in range(n)]
        
        for u in range(n):
            for v in adj[u]:
                # Remove edge (u, v) temporarily
                adj_set[u].remove(v)
                adj_set[v].remove(u)
                
                # BFS from u to v
                visited = {u: 0}
                queue = deque([u])
                found = False
                while queue:
                    curr = queue.popleft()
                    if visited[curr] >= g - 1:
                        continue
                    for nb in adj_set[curr]:
                        if nb not in visited:
                            visited[nb] = visited[curr] + 1
                            if nb == v:
                                found = True
                                break
                            queue.append(nb)
                    if found:
                        break
                
                # Restore edge
                adj_set[u].add(v)
                adj_set[v].add(u)
                
                if found:
                    # Cycle length = dist(u, v) + 1 < (g - 1) + 1 = g
                    return False
        
        return True

    # Run greedy a few times
    attempts = 10
    for i in range(attempts):
        if time.time() > deadline:
            break
        try_greedy()
        if best_n < n_luw:
            # We found an improvement. Maybe try more to improve further?
            # But time is precious.
            # Let's just break if we found something significantly better?
            # Or keep trying if time permits.
            pass

    return best_n, best_edges
# EVOLVE-BLOCK-END
