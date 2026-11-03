import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Fallback: The original LUW construction
    def luw_construct(k, g):
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

    # Heuristic Search for smaller graph
    # We target a 3-regular bipartite graph
    # Girth g means no cycles of length < g.
    # For bipartite, we check even lengths 4, 6, ..., g-2.
    
    def generate_random_bipartite(n):
        # n must be even
        half = n // 2
        # Vertices 0..half-1 are U, half..n-1 are V
        # Each u in U connects to 3 in V
        # Each v in V connects to 3 in U
        # We generate a random 3-regular bipartite graph using the configuration model
        # Stubs for U: 3 * half
        # Stubs for V: 3 * half
        u_stubs = [i for i in range(half) for _ in range(3)]
        v_stubs = [i + half for i in range(half) for _ in range(3)]
        
        random.shuffle(v_stubs)
        
        adj = {i: [] for i in range(n)}
        edges = []
        
        # Pair u_stubs with v_stubs
        # To ensure simplicity (no multiple edges), we might need to retry or use a smarter pairing
        # For now, simple random pairing. If multi-edge, we can swap later or regenerate.
        # Given n is large, probability of multi-edge is low but not zero.
        
        for i in range(len(u_stubs)):
            u = u_stubs[i]
            v = v_stubs[i]
            if v not in adj[u]: # Simple check
                adj[u].append(v)
                adj[v].append(u)
                edges.append((u, v))
            else:
                # Multi-edge found, this is a problem for "simple graph"
                # We'll just flag this and try to fix or regenerate
                pass
        
        # Check if simple
        if len(edges) != 3 * half:
            return None, None
            
        return adj, edges

    def count_short_cycles(adj, n, max_len):
        # Counts cycles of length 4, 6, ..., max_len
        # Since it's bipartite, only even lengths.
        # 4-cycles:
        count = 0
        # For each u, for each pair of neighbors v1, v2, check if they share another neighbor
        # This is O(n * k^2)
        for u in range(n // 2):
            neighbors = adj[u]
            # k=3, so 3 pairs
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    v1 = neighbors[i]
                    v2 = neighbors[j]
                    # Check if v1 and v2 have a common neighbor other than u
                    # v1 is in V, v2 is in V. Their neighbors are in U.
                    # Common neighbor w in U such that w != u
                    common = set(adj[v1]) & set(adj[v2])
                    if len(common) > 1: # u is in common, so if >1, there is another
                        count += len(common) - 1
        # 6-cycles:
        # u1-v1-u2-v2-u3-v3-u1
        # This is more expensive.
        # We can sample or just rely on breaking 4-cycles first.
        # If 4-cycles are gone, 6-cycles might remain.
        # For simplicity in this heuristic, we'll just minimize 4-cycles and hope the rest follow,
        # or do a BFS check at the end.
        return count

    def is_valid(adj, n, g):
        # Check girth >= g using BFS
        # For bipartite, check even cycles < g
        # BFS from each vertex up to depth g/2 - 1
        # If we meet a node from a different path, cycle found.
        
        # Optimization: Only need to check from U vertices?
        # No, cycles can start anywhere.
        # But in bipartite, a cycle of length 2L corresponds to a path of length L in the "square" or similar?
        # Just standard BFS.
        
        for start in range(n):
            # BFS
            visited = {start: (None, 0)} # node: (parent, dist)
            queue = [(start, 0)]
            found_cycle = False
            while queue:
                node, dist = queue.pop(0)
                if dist >= g // 2:
                    continue
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited[neighbor] = (node, dist + 1)
                        queue.append((neighbor, dist + 1))
                    else:
                        parent, p_dist = visited[neighbor]
                        if neighbor != parent and dist + 1 + p_dist < g:
                            # Cycle length is dist + 1 + p_dist
                            # But this might count the same cycle multiple times or paths not cycles
                            # If we meet a visited node that is not parent, we have a cycle.
                            # The length is dist + 1 + p_dist.
                            # We need to ensure this is a simple cycle.
                            # In BFS, the first time we meet a visited node, the cycle is simple.
                            return False
            # If BFS completes without finding a cycle < g, continue
        return True

    # Main Search Logic
    # Try increasing n
    target_ns = [936, 1000, 1200, 1500, 2000, 3000, 5000]
    
    for n in target_ns:
        if n % 2 != 0:
            n += 1
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds for this n
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            random.seed(seed)
            adj, edges = generate_random_bipartite(n)
            if adj is None:
                continue
                
            # Local Search to remove short cycles
            # We'll do a simple greedy: while there are 4-cycles, break one.
            # Limit iterations
            max_iters = 5000
            for _ in range(max_iters):
                if time.time() > deadline - 2:
                    break
                # Check for 4-cycles
                # If none, check full girth
                # For speed, just check 4-cycles first
                has_4 = False
                for u in range(n // 2):
                    neighbors = adj[u]
                    for i in range(len(neighbors)):
                        for j in range(i+1, len(neighbors)):
                            v1 = neighbors[i]
                            v2 = neighbors[j]
                            common = set(adj[v1]) & set(adj[v2])
                            if len(common) > 1:
                                has_4 = True
                                # Found a 4-cycle involving u, v1, w, v2
                                # w is the other common neighbor
                                w = list(common - {u})[0]
                                # Break the cycle by swapping an edge
                                # Swap (u, v1) with (w, v2)? No, (u, v1) and (w, v2) are edges.
                                # We can swap (u, v1) and (w, v2) to (u, v2) and (w, v1)?
                                # This might create new 4-cycles.
                                # Just pick a random edge in the cycle and swap with a random non-edge.
                                # Edge (u, v1) is in cycle.
                                # Find a non-edge (u, v_new) where v_new is not neighbor of u.
                                # And (w, v1) is in cycle.
                                # Let's just swap (u, v1) with (some u2, v1) where u2 is not neighbor of v1.
                                # This is getting complex.
                                # Simple heuristic: Just regenerate if 4-cycles persist?
                                # Or use a "2-opt" move:
                                # Pick edge (u, v1).
                                # Pick random non-neighbor u2 of v1.
                                # Swap (u, v1) and (u2, v1) -> (u, v1) becomes (u, v1_new)?
                                # Let's try: Remove (u, v1). Add (u, v_rand) where v_rand is not neighbor.
                                # But we must maintain 3-regularity.
                                # So we must swap two edges.
                                # Edge e1 = (u, v1).
                                # Find another edge e2 = (u2, v2) such that swapping gives (u, v2) and (u2, v1) and both are non-edges.
                                # This is a standard 2-opt.
                                
                                # Find u2, v2
                                # v2 is a neighbor of u2.
                                # We want (u, v2) not in E and (u2, v1) not in E.
                                # Try a few random candidates.
                                swapped = False
                                for _ in range(20):
                                    u2 = random.randint(0, n//2 - 1)
                                    if u2 == u: continue
                                    v2 = random.choice(adj[u2])
                                    if (u, v2) in adj[u] or (u2, v1) in adj[u2]:
                                        continue
                                    # Check if swap creates multi-edge or self loop (bipartite so no self loop)
                                    # Perform swap
                                    adj[u].remove(v1)
                                    adj[v1].remove(u)
                                    adj[u2].remove(v2)
                                    adj[v2].remove(u2)
                                    
                                    adj[u].append(v2)
                                    adj[v2].append(u)
                                    adj[u2].append(v1)
                                    adj[v1].append(u2)
                                    
                                    edges.remove((u, v1))
                                    edges.remove((u2, v2))
                                    edges.append((u, v2))
                                    edges.append((u2, v1))
                                    
                                    swapped = True
                                    break
                                if swapped:
                                    break
                if not has_4:
                    # Check full girth
                    if is_valid(adj, n, g):
                        return n, edges
                    # If 4-cycles are gone but longer cycles exist, we can continue searching or accept if g is small?
                    # No, must be valid.
                    # We can try more swaps to break longer cycles.
                    # But for now, if 4-cycles are gone, we might be close.
                    # Let's just break and try next seed/n.
                    break

    # Fallback
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
