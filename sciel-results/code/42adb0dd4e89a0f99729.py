import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback: Lazebnik-Ustimenko-Woldar (LUW) construction
    def seed_construct(k, g):
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

    # Fast girth check using non-backtracking walks
    def check_girth_fast(adj, n, g):
        """
        Check if graph has girth >= g using non-backtracking BFS from each vertex.
        Returns True if girth >= g, False otherwise.
        """
        # Pre-compute neighbor indices for fast lookup
        # adj[i] is a list of 3 neighbors
        for start in range(n):
            # State: (node, prev_node, dist)
            # We want to detect any non-backtracking walk of length < g that starts and ends at start
            # Actually, a cycle of length L < g corresponds to a non-backtracking walk of length L from start to start.
            # We can do a BFS in the non-backtracking graph.
            # State is (u, v) meaning we are at v, coming from u.
            # Start states: (start, nb) for nb in adj[start]. Dist 1.
            # Target: any state (u, start) where dist < g.
            # Wait, if we reach (u, start), the path is start -> ... -> u -> start.
            # Length is dist. If dist < g, we have a cycle of length dist.
            
            # Visited set of (u, v)
            visited = set()
            # Queue of (u, v, dist)
            queue = []
            for nb in adj[start]:
                visited.add((start, nb))
                queue.append((start, nb, 1))
            
            while queue:
                u, v, dist = queue.pop(0)
                if dist >= g:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == start:
                        # Found a cycle of length dist + 1
                        if dist + 1 < g:
                            return False
                    else:
                        if (v, w) not in visited:
                            visited.add((v, w))
                            queue.append((v, w, dist + 1))
        return True

    # Heuristic: Count short cycles to guide optimization
    def count_short_cycles(adj, n, g):
        """
        Estimate number of cycles of length < g.
        Uses non-backtracking walks of length L < g.
        """
        count = 0
        for start in range(n):
            visited = set()
            queue = []
            for nb in adj[start]:
                visited.add((start, nb))
                queue.append((start, nb, 1))
            while queue:
                u, v, dist = queue.pop(0)
                if dist >= g:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == start:
                        if dist + 1 < g:
                            count += 1
                    else:
                        if (v, w) not in visited:
                            visited.add((v, w))
                            queue.append((v, w, dist + 1))
        return count

    def build_petersen_lift(m, rng):
        """
        Build a random m-lift of the Petersen graph.
        Returns adjacency list or None if invalid.
        """
        # Petersen graph edges (15 edges)
        petersen_edges = []
        # Outer pentagon: 0-1-2-3-4-0
        for i in range(5):
            petersen_edges.append((i, (i + 1) % 5))
        # Inner star: 5-7-9-6-8-5 (pentagram)
        inner = [5, 6, 7, 8, 9]
        for i in range(5):
            petersen_edges.append((inner[i], inner[(i + 2) % 5]))
        # Spokes: i to i+5
        for i in range(5):
            petersen_edges.append((i, i + 5))

        # Assign random permutations to each Petersen edge
        edge_perms = {}
        for (p, q) in petersen_edges:
            perm = list(range(m))
            rng.shuffle(perm)
            edge_perms[(p, q)] = perm
            inv_perm = [0] * m
            for i in range(m):
                inv_perm[perm[i]] = i
            edge_perms[(q, p)] = inv_perm

        n = 10 * m
        adj = [[] for _ in range(n)]
        
        for (p, q) in petersen_edges:
            perm = edge_perms[(p, q)]
            for i in range(m):
                u = p * m + i
                v = q * m + perm[i]
                adj[u].append(v)
                adj[v].append(u)
        
        # Verify 3-regular and simple
        for i in range(n):
            if len(adj[i]) != 3:
                return None
            if len(set(adj[i])) != 3:
                return None
            for nb in adj[i]:
                if nb < 0 or nb >= n:
                    return None
                if i not in adj[nb]:
                    return None
        
        return adj

    def improve_lift(m, rng, max_iters=500):
        """
        Local search to improve a random lift of Petersen.
        Objective: minimize number of short cycles.
        """
        # Initial random lift
        adj = build_petersen_lift(m, rng)
        if adj is None:
            return None
        
        # Check if already valid
        if check_girth_fast(adj, 10*m, g):
            return adj
        
        # Get current score
        current_score = count_short_cycles(adj, 10*m, g)
        best_adj = [list(neighs) for neighs in adj]
        best_score = current_score
        
        # Pre-compute Petersen structure for fast moves
        petersen_edges = []
        for i in range(5):
            petersen_edges.append((i, (i + 1) % 5))
        inner = [5, 6, 7, 8, 9]
        for i in range(5):
            petersen_edges.append((inner[i], inner[(i + 2) % 5]))
        for i in range(5):
            petersen_edges.append((i, i + 5))
        
        edge_list = petersen_edges
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge in Petersen
            edge_idx = rng.randint(0, len(edge_list) - 1)
            p, q = edge_list[edge_idx]
            
            # Pick a random vertex in the fiber
            i = rng.randint(0, m - 1)
            
            # Current permutation for this edge
            # We store permutations in a dict for easy access
            # Reconstructing perms from adj is slow, so let's maintain perms separately
            
            # Actually, let's maintain the edge_perms dict alongside adj
            # This function is getting complex. Let's simplify.
            # We will just restart with a new random lift if local search is too complex to implement robustly in time.
            # Or, we can do a simple "swap" in the adjacency list directly.
            
            # Move: Swap two connections in the fiber of edge (p,q)
            # Current: p_i is connected to q_{perm[i]}
            # Let's pick two indices i, j in the fiber
            # Swap the targets of p_i and p_j
            
            j = rng.randint(0, m - 1)
            if i == j:
                continue
            
            u1 = p * m + i
            u2 = p * m + j
            
            # Current neighbors of u1 and u2 in the q-fiber
            # Each has exactly one neighbor in the q-fiber
            v1 = None
            v2 = None
            for nb in adj[u1]:
                if nb // m == q:
                    v1 = nb
            for nb in adj[u2]:
                if nb // m == q:
                    v2 = nb
            
            if v1 is None or v2 is None:
                continue
            
            # Try swapping: u1 connects to v2, u2 connects to v1
            # Remove old edges
            adj[u1].remove(v1)
            adj[u2].remove(v2)
            adj[v1].remove(u1)
            adj[v2].remove(u2)
            
            # Add new edges
            adj[u1].append(v2)
            adj[u2].append(v1)
            adj[v1].append(u2)
            adj[v2].append(u1)
            
            # Check validity (simple graph)
            valid = True
            if v2 in adj[u1][:adj[u1].index(v2)] or v2 in adj[u1][adj[u1].index(v2)+1:]: # Duplicate check
                valid = False
            if v1 in adj[u2][:adj[u2].index(v1)] or v1 in adj[u2][adj[u2].index(v1)+1:]:
                valid = False
            if v1 == u2 or v2 == u1: # Loop
                valid = False
            
            if not valid:
                # Revert
                adj[u1].remove(v2)
                adj[u2].remove(v1)
                adj[v1].remove(u2)
                adj[v2].remove(u1)
                adj[u1].append(v1)
                adj[u2].append(v2)
                adj[v1].append(u1)
                adj[v2].append(u2)
                continue
            
            # Evaluate new score
            new_score = count_short_cycles(adj, 10*m, g)
            
            if new_score < best_score:
                best_score = new_score
                best_adj = [list(neighs) for neighs in adj]
                if best_score == 0:
                    break
            else:
                # Revert
                adj[u1].remove(v2)
                adj[u2].remove(v1)
                adj[v1].remove(u2)
                adj[v2].remove(u1)
                adj[u1].append(v1)
                adj[u2].append(v2)
                adj[v1].append(u1)
                adj[v2].append(u2)
                
                # Random restart of this local search phase?
                # Or just continue.
        
        return best_adj

    # Main search
    rng = random.Random(42)
    
    # Try different m values
    # Start small, go up.
    # For g=16, m=50 (n=500) might be too small (Moore bound 512).
    # m=60 (n=600) is a good start.
    m_values = [60, 70, 80, 90, 100, 120, 150, 200, 300, 500]
    
    best_result = None
    
    for m in m_values:
        if time.time() > deadline - 10:
            break
        
        n = 10 * m
        
        # Try a few random lifts
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            adj = build_petersen_lift(m, rng)
            if adj is None:
                continue
            
            # Check girth
            if check_girth_fast(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
        
        # Try local search improvement
        if time.time() < deadline - 20:
            adj = improve_lift(m, rng, max_iters=100)
            if adj is not None:
                if check_girth_fast(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
    
    # Fallback to seed
    return seed_construct(k, g)
