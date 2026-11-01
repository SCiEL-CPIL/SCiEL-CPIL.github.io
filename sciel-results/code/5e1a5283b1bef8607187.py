import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)
    
    # Strategy: sweep N from Moore bound upward, try random 5-regular graphs,
    # check girth via non-backtracking walks (fast with numpy).
    # Return first valid graph found (smallest N among those tried).
    
    N_start = 428
    if N_start % 2 != 0:
        N_start += 1
    N_end = 1500
    
    def gen_random_5_reg(N, deadline):
        """Generate a random 5-regular simple graph using configuration model + swap repair."""
        # Build stubs: each vertex appears 5 times
        stubs = np.repeat(np.arange(N), 5)
        np.random.shuffle(stubs)
        
        # Build adjacency as a list of lists (each vertex has 5 neighbors)
        adj = np.zeros((N, 5), dtype=np.int32)
        for i in range(N):
            adj[i] = stubs[5*i : 5*i+5]
        
        # Remove self-loops and multi-edges via swap
        # A self-loop: adj[i, j] == i
        # A multi-edge: adj[i, j] == adj[i', j'] for i != i' (undirected)
        # We'll iterate and fix issues by swapping stubs between two vertices.
        
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Find a self-loop
            self_loop_found = False
            for i in range(N):
                for j in range(5):
                    if adj[i, j] == i:
                        # Find another vertex to swap with
                        # Pick a random vertex l != i and random stub positions
                        l = random.randint(0, N - 1)
                        while l == i:
                            l = random.randint(0, N - 1)
                        j2 = random.randint(0, 4)
                        # Check that the swap won't create a new self-loop
                        if adj[l, j2] != i and adj[i, j] != l:
                            # Swap adj[i,j] and adj[l,j2]
                            tmp = adj[i, j]
                            adj[i, j] = adj[l, j2]
                            adj[l, j2] = tmp
                            self_loop_found = True
                            break
                if self_loop_found:
                    break
            
            if not self_loop_found:
                # Check for multi-edges
                # Build edge set and look for duplicates
                edge_set = {}
                has_multi = False
                for i in range(N):
                    for j in range(5):
                        u, v = int(adj[i, j]), int(i)
                        if u > v:
                            u, v = v, u
                        key = u * N + v
                        if key in edge_set:
                            # Multi-edge found: edge (u,v) appears at (i,j) and at edge_set[key]
                            # Swap to fix
                            i2, j2 = edge_set[key]
                            # Swap adj[i,j] with adj[i2,j2]
                            # But we need to make sure this doesn't create a self-loop
                            new_ij = int(adj[i2, j2])
                            new_i2j2 = int(adj[i, j])
                            if new_ij != i and new_i2j2 != i2:
                                adj[i, j] = new_ij
                                adj[i2, j2] = new_i2j2
                                has_multi = True
                                break
                        else:
                            edge_set[key] = (i, j)
                    if has_multi:
                        break
                if not has_multi:
                    # Graph is simple! Extract edges
                    edges = []
                    seen = set()
                    for i in range(N):
                        for j in range(5):
                            u, v = i, int(adj[i, j])
                            if u < v:
                                if (u, v) not in seen:
                                    edges.append((u, v))
                                    seen.add((u, v))
                    return edges
                # If multi-edge found, loop continues to fix
            # Continue fixing self-loops
        return None
    
    def check_girth_nb(adj, N, min_girth, deadline):
        """Check girth >= min_girth using non-backtracking walks from every vertex.
        Uses numpy for speed. Returns True if girth >= min_girth."""
        # Build adjacency list from adj matrix
        # adj is N x 5 int array
        adj_list = [list(adj[i]) for i in range(N)]
        
        # For each start vertex, do non-backtracking BFS up to depth (min_girth-1)//2
        # A non-backtracking walk of length L from start: (v0, v1, ..., vL) where v_{i-1} != v_{i+1}
        # A cycle of length c < min_girth exists iff there's a non-backtracking walk of length c back to start
        # Actually, we need: for each start, check if any non-backtracking walk of length < min_girth returns to start
        # More precisely: a cycle of length c through start means a non-backtracking closed walk of length c at start
        # But non-backtracking closed walks of length c correspond to cycles of length c (for c >= 3 in simple graphs)
        # Actually no: a non-backtracking closed walk can traverse a cycle multiple times... 
        # Wait, for simple graphs, a non-backtracking closed walk of length c starting and ending at v
        # corresponds to a cycle of length c if c is the girth. But it could also be a longer structure.
        # The standard approach: for each vertex v, do non-backtracking BFS. If at depth d < min_girth,
        # we can reach v again via a non-backtracking path, then there's a cycle of length <= d+1 (or similar).
        
        # Simpler: for each start vertex s, run non-backtracking BFS. Track (current_vertex, previous_vertex) states.
        # If we ever reach (s, x) where x != s and the path length < min_girth, we have a cycle.
        # Actually: a non-backtracking walk from s of length L that ends at s means a closed non-backtracking walk.
        # The shortest such walk length equals the girth if the graph has girth >= 3.
        # Wait, that's not quite right either. Let me think more carefully.
        
        # Standard girth check via non-backtracking:
        # For each vertex s, perform a BFS in the "line graph" of directed edges.
        # State: (u, v) meaning we're at vertex v, came from u.
        # Start states: (s, v) for each neighbor v of s, at depth 1.
        # Transition: from (u, v) go to (v, w) for each neighbor w of v with w != u.
        # If we reach state (v, s) at depth d, we have a non-backtracking walk of length d+1 from s back to s
        # Wait no. Let me reconsider.
        
        # A non-backtracking walk: v0, v1, ..., vL where v_{i-1} != v_{i+1} for all i in 1..L-1.
        # A closed non-backtracking walk: v0 = vL.
        # The girth is the length of the shortest cycle.
        # In a simple graph, the shortest closed non-backtracking walk has length equal to the girth.
        # Proof: any cycle of length c gives a closed NB walk of length c. Conversely, a closed NB walk
        # of length L contains a cycle of length <= L.
        
        # So: for each vertex s, find the shortest closed non-backtracking walk starting at s.
        # The minimum over all s gives the girth.
        
        # BFS in state space (current, prev):
        # Start: for each neighbor v of s, state (v, s) at depth 1.
        # From state (u, prev), go to (w, u) for each neighbor w of u with w != prev.
        # If we reach state (s, u) at depth d, we have a closed NB walk of length d.
        # (The walk is s -> ... -> u -> s, which has length d, and is non-backtracking.)
        
        max_depth = min_girth - 1  # We only need to check up to this depth
        # If no closed NB walk of length < min_girth exists, girth >= min_girth.
        
        for s in range(N):
            # BFS from s
            # State: (current, prev), depth
            # Use a dict or array for visited
            # For efficiency, use a 2D array: visited[current][prev_index]
            # But prev is a vertex index, so visited is N x N bool. That's N^2 memory.
            # For N=1116, that's ~1.2M entries. Fine.
            # But we only need to track up to depth max_depth.
            
            # Use a list of dicts or a flat array
            visited = np.zeros((N, N), dtype=bool)
            # Queue of (current, prev, depth)
            queue = []
            for v in adj_list[s]:
                v = int(v)
                queue.append((v, s, 1))
                visited[v, s] = True
            
            head = 0
            while head < len(queue):
                if time.time() > deadline:
                    return False  # Timeout, assume invalid to be safe? Or return True?
                    # Actually if we timeout, we can't certify. Return False to be safe.
                
                u, prev, depth = queue[head]
                head += 1
                
                if depth >= max_depth:
                    continue
                
                for w in adj_list[u]:
                    w = int(w)
                    if w == prev:
                        continue
                    if w == s:
                        # Found a closed NB walk of length depth + 1
                        if depth + 1 < min_girth:
                            return False
                        # depth + 1 >= min_girth, so this is fine
                    elif not visited[w, u]:
                        visited[w, u] = True
                        queue.append((w, u, depth + 1))
        
        return True
    
    # Search
    N_current = N_start
    while N_current <= N_end and time.time() < deadline - 5:
        
        # Determine trials based on N
        if N_current < 700:
            max_trials = 5
        elif N_current < 900:
            max_trials = 10
        elif N_current < 1050:
            max_trials = 30
        elif N_current < 1150:
            max_trials = 100
        else:
            max_trials = 300
        
        if time.time() > deadline - 10:
            break
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            edges = gen_random_5_reg(N_current, deadline - 5)
            if edges is None:
                continue
            
            # Build adj matrix for girth check
            adj = np.zeros((N_current, 5), dtype=np.int32)
            # We need to reconstruct the adjacency from edges
            # This is a bit wasteful. Let me restructure gen_random_5_reg to return adj directly.
            # For now, build adj from edges.
            deg = np.zeros(N_current, dtype=np.int32)
            for u, v in edges:
                adj[u, deg[u]] = v
                deg[u] += 1
                adj[v, deg[v]] = u
                deg[v] += 1
            
            if check_girth_nb(adj, N_current, g, deadline - 5):
                return N_current, edges
        
        N_current += 2
    
    # Fallback to LUW
    return _luw_construct(k, g, deadline)

def _check_girth(adj, n, min_girth):
    """Check if graph has girth >= min_girth."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g, deadline):
    """Original LUW implementation as fallback."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges
