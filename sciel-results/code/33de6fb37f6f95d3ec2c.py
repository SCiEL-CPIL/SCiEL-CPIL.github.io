import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
        
    best_n = None
    best_edges = None
    
    # Helper to check girth >= g using non-backtracking walks
    def has_girth_at_least_g(adj, n, g):
        visit_id = np.zeros(n * n, dtype=np.int32)
        current_id = 1
        
        for start in range(n):
            queue = []
            for nb in adj[start]:
                idx = start * n + nb
                if visit_id[idx] != current_id:
                    visit_id[idx] = current_id
                    queue.append((start, nb, 1))
            
            while queue:
                new_queue = []
                for prev, curr, depth in queue:
                    if depth >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start:
                            if depth + 1 < g:
                                return False
                        else:
                            idx = curr * n + nxt
                            if visit_id[idx] != current_id:
                                visit_id[idx] = current_id
                                new_queue.append((curr, nxt, depth + 1))
                queue = new_queue
            current_id += 1
            if current_id > 2**31:
                visit_id[:] = 0
                current_id = 1
        return True

    # Returns (min_cycle_length, list_of_edges_in_shortest_cycle)
    def find_shortest_cycle(adj, n, max_girth):
        visit_id = np.zeros(n * n, dtype=np.int32)
        current_id = 1
        min_cycle = max_girth + 1
        best_cycle_edges = None
        
        for start in range(n):
            queue = []
            for nb in adj[start]:
                idx = start * n + nb
                if visit_id[idx] != current_id:
                    visit_id[idx] = current_id
                    queue.append((start, nb, 1, [(start, nb)]))
            
            while queue:
                new_queue = []
                for prev, curr, depth, path in queue:
                    if depth >= max_girth - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start:
                            cycle_len = depth + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                                cycle_edges = list(path) + [(curr, start)]
                                best_cycle_edges = cycle_edges
                        else:
                            idx = curr * n + nxt
                            if visit_id[idx] != current_id:
                                visit_id[idx] = current_id
                                new_path = path + [(curr, nxt)]
                                new_queue.append((curr, nxt, depth + 1, new_path))
                queue = new_queue
            current_id += 1
            if current_id > 2**31:
                visit_id[:] = 0
                current_id = 1
            if min_cycle == 3:
                break
        return min_cycle, best_cycle_edges

    def random_regular_graph(n, k, rng):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        rng.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        seen = [set() for _ in range(n)]
        
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if v in seen[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
            seen[u].add(v)
            seen[v].add(u)
            
        return adj

    def get_all_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    def try_swap(adj, n, k, rng, prefer_edges=None):
        edges = get_all_edges(adj, n)
        if len(edges) < 2:
            return False
            
        if prefer_edges and len(prefer_edges) >= 2:
            e1 = rng.choice(prefer_edges)
            e2 = rng.choice(edges)
            if e1 == e2:
                e2 = rng.choice(edges)
                if e1 == e2:
                    return False
        else:
            e1_idx = rng.randint(0, len(edges) - 1)
            e2_idx = rng.randint(0, len(edges) - 2)
            if e2_idx >= e1_idx:
                e2_idx += 1
            e1 = edges[e1_idx]
            e2 = edges[e2_idx]
            
        u1, v1 = e1
        u2, v2 = e2
        
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return False
            
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        if u2 not in adj[u1] and v2 not in adj[v1]:
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            return True
        
        if v2 not in adj[u1] and u2 not in adj[v1]:
            adj[u1].append(v2)
            adj[v2].append(u1)
            adj[v1].append(u2)
            adj[u2].append(v1)
            return True
            
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        
        return False

    # BFS-Tree Pruning construction
    def bfs_tree_construction(k, g, rng, max_n=2000):
        """
        Build graph by generating non-backtracking walks from root up to depth floor(g/2),
        then merge leaf vertices to satisfy k-regularity.
        """
        half = g // 2 if g % 2 == 0 else (g - 1) // 2
        
        # Generate tree structure: root + k children + (k-1)^(i) nodes at each level
        # Node representation: (depth, parent, child_index)
        # We'll build the tree level by level
        
        # Level 0: root (vertex 0)
        # Level 1: k children of root
        # Level i (i>=2): each node at level i-1 has (k-1) children (excluding parent)
        
        # Count nodes at each level
        level_sizes = [1]
        for i in range(1, half + 1):
            if i == 1:
                level_sizes.append(k)
            else:
                level_sizes.append(level_sizes[-1] * (k - 1))
        
        total_nodes = sum(level_sizes)
        
        if total_nodes > max_n:
            return None
        
        # Assign vertex IDs: level 0 gets 0, level 1 gets 1..k, etc.
        vertex_id = {}
        current_id = 0
        for level in range(half + 1):
            for i in range(level_sizes[level]):
                vertex_id[(level, i)] = current_id
                current_id += 1
        
        n = total_nodes
        adj = [[] for _ in range(n)]
        
        # Build tree edges
        # Level 0 to Level 1
        for i in range(k):
            u = vertex_id[(0, 0)]
            v = vertex_id[(1, i)]
            adj[u].append(v)
            adj[v].append(u)
        
        # Level i to Level i+1 for i >= 1
        for level in range(1, half):
            for i in range(level_sizes[level]):
                parent_id = vertex_id[(level, i)]
                for j in range(k - 1):
                    child_idx = i * (k - 1) + j
                    if child_idx < level_sizes[level + 1]:
                        child_id = vertex_id[(level + 1, child_idx)]
                        adj[parent_id].append(child_id)
                        adj[child_id].append(parent_id)
        
        # Now we have a tree. Vertices at level < half have degree < k (except root which has degree k).
        # Vertices at level == half are leaves with degree 1.
        # We need to add edges to make it k-regular.
        # Leaves at level half each need (k-1) more edges.
        # Internal nodes at level < half-1 have degree k (root) or k (others: 1 parent + (k-1) children = k).
        # Wait, let me recount:
        # Root: degree k (k children) - OK
        # Level 1: degree 1 (parent) + (k-1) (children) = k - OK
        # Level i (1<=i<half): degree 1 + (k-1) = k - OK
        # Level half: degree 1 (parent) - need (k-1) more edges
        
        # So we need to add (k-1) edges per leaf, connecting leaves to each other or to internal nodes.
        # But we can't connect to internal nodes without creating short cycles!
        # Two leaves at level half connected: path between them goes through their LCA.
        # If LCA is at level L, path length = 2*(half - L) + 1... wait no.
        # Leaf at level half, LCA at level L: distance from leaf to LCA is half-L, so two leaves 
        # with LCA at level L have distance 2*(half-L). If we add an edge between them, cycle length = 2*(half-L)+1.
        # For girth >= g, we need 2*(half-L)+1 >= g.
        # If g is even: half = g/2, need 2*(half-L)+1 >= g => 2*(g/2-L)+1 >= g => g-2L+1 >= g => L <= 0.5, so L=0.
        # So leaves can only be connected if their LCA is the root (level 0).
        # If g is odd: half = (g-1)/2, need 2*(half-L)+1 >= g => 2*((g-1)/2-L)+1 >= g => g-1-2L+1 >= g => -2L >= 0 => L <= 0.
        # So again, only leaves whose LCA is root can be connected.
        
        # Leaves under different root children have LCA = root.
        # Leaves under the same root child have LCA at level >= 1.
        
        # Number of leaves: level_sizes[half]
        # Leaves under root child i: level_sizes[half-1] * ... wait
        # Level 1 has k nodes. Each level 1 node has (k-1) children at level 2, etc.
        # Leaves under root child i: (k-1)^(half-1)
        
        leaves_per_root_child = (k - 1) ** (half - 1) if half >= 1 else k
        
        # We can only connect leaves from different root subtrees.
        # Total leaves = k * leaves_per_root_child
        # Each leaf needs (k-1) additional edges.
        # Total additional edges needed = k * leaves_per_root_child * (k-1) / 2
        
        # But we can only connect leaves from different subtrees.
        # This is like a complete k-partite graph problem.
        
        # Actually, this simple tree approach may not work well for large girth because
        # we can't connect leaves within the same subtree.
        
        # Alternative: use the random approach with local search, but start from the tree.
        # Let's just return the tree structure and let the random search improve it.
        return adj, n

    # Main search loop
    n_start = moore
    if (n_start * k) % 2 != 0:
        n_start += 1
        
    max_n_to_try = 500
    
    for n in range(n_start, max_n_to_try + 1, 2):
        if time.time() > deadline - 10:
            break
            
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and time.time() < deadline - 5:
            adj = random_regular_graph(n, k, rng)
            if adj is None:
                attempts += 1
                continue
                
            if has_girth_at_least_g(adj, n, g):
                edges = get_all_edges(adj, n)
                return n, edges
            
            # Local search: repeatedly find shortest cycle, try to break it
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                
                min_cycle, cycle_edges = find_shortest_cycle(adj, n, g)
                
                if min_cycle >= g:
                    edges = get_all_edges(adj, n)
                    return n, edges
                
                # Try to break the shortest cycle by swapping one of its edges
                # with a random edge
                if cycle_edges:
                    canon_cycle = []
                    for a, b in cycle_edges:
                        canon_cycle.append((min(a,b), max(a,b)))
                    
                    for _attempt in range(10):
                        if time.time() > deadline - 5:
                            break
                        if try_swap(adj, n, k, rng, prefer_edges=canon_cycle):
                            break
                else:
                    try_swap(adj, n, k, rng)
                
                if _ % 20 == 0:
                    if has_girth_at_least_g(adj, n, g):
                        edges = get_all_edges(adj, n)
                        return n, edges
            
            attempts += 1
            
    # Fallback: LUW construction
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
        if time.time() > deadline - 2:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
