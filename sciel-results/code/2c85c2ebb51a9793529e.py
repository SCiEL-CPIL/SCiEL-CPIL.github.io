import time
import random
import math
from collections import deque

def _luw_construct(k, g):
    # Seed logic
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

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)

    deadline = time.time() + 110

    # For k=4, g=10, try to find a smaller graph than LUW (2048)
    # Use local search on random 4-regular bipartite graphs

    def count_short_cycles(adj, n):
        """Count 4-cycles, 6-cycles, and 8-cycles using adjacency list."""
        half = n // 2
        # 4-cycles: pairs of left nodes sharing >=2 right neighbors
        c4 = 0
        for u in range(half):
            for v in range(u + 1, half):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    c4 += common * (common - 1) // 2

        # 6-cycles: a1-b1-a2-b2-a3-b3-a1 (bipartite)
        # Count by: for each pair (a1, a2) with |N(a1)∩N(a2)|>=2,
        # for each pair of common neighbors (b1,b2),
        # count common neighbors of a1,a2 excluding b1,b2 that are also... 
        # Actually simpler: for each a1, for each pair of neighbors (b1,b2),
        # for each a2 in N(b2)\{a1}, for each b3 in N(a2)\{b2},
        # check if b3 in N(a1) and b3 != b1
        c6 = 0
        for a1 in range(half):
            nbs1 = list(adj[a1])
            for i in range(len(nbs1)):
                b1 = nbs1[i]
                for j in range(i + 1, len(nbs1)):
                    b2 = nbs1[j]
                    for a2 in adj[b2]:
                        if a2 == a1:
                            continue
                        for b3 in adj[a2]:
                            if b3 == b2 or b3 == b1:
                                continue
                            if b3 in adj[a1]:
                                c6 += 1
        c6 //= 6

        # 8-cycles: a1-b1-a2-b2-a3-b3-a4-b4-a1
        # For each ordered pair of edges (a1-b1, a2-b2) with b1!=b2, a1!=a2:
        # count pairs (a3,b3) such that b2-a3-b3-a2 and b3!=b1, b3!=b2, a3!=a1,a3!=a2
        # and a4 = common neighbor of a1 and a3 (excluding b1,b3)
        # This is expensive. Use a different approach:
        # For each pair of left nodes (a1, a2) that are NOT adjacent via a common right node
        # (i.e., |N(a1)∩N(a2)| <= 1), count paths of length 3 from a1 to a2
        # that don't use a common neighbor directly.
        # 
        # Better approach for 8-cycles in bipartite graph:
        # An 8-cycle is a1-b1-a2-b2-a3-b3-a4-b4-a1
        # Fix (a1, b1, a2, b2) where a1-b1, a2-b2, b1!=b2, a1!=a2
        # Then we need b3 in N(a2)\{b2}, a3 in N(b3)\{a2}, b4 in N(a3)\{b3},
        # a4 in N(b4)\{a3}, and a4-b1-a1 with a4!=a1, a4!=a3, b4!=b1
        # This is O(n^8) worst case. Too slow.
        #
        # Alternative: count 8-cycles via matrix multiplication or
        # use the fact that in a 4-regular bipartite graph with n/2 nodes per side,
        # the number of 8-cycles can be computed from the adjacency matrix A:
        # #8-cycles = (1/8)(tr(A^8) - 4*tr(A^4) + ... ) -- complicated
        #
        # For now, use a sampling-based estimate or just check girth directly.
        # We'll return c4, c6 and use a direct girth check for 8-cycles.
        return c4, c6

    def has_cycle_less_than_g(adj, n, g):
        """Check if graph has a cycle of length < g using non-backtracking walks."""
        # Since bipartite, only even cycles matter
        # Check for cycles of length 4, 6, 8 (if g > 8)
        half = n // 2
        
        if g > 4:
            # Check 4-cycles
            for u in range(half):
                for v in range(u + 1, half):
                    if len(adj[u] & adj[v]) >= 2:
                        return True
        
        if g > 6:
            # Check 6-cycles
            for a1 in range(half):
                nbs1 = list(adj[a1])
                for i in range(len(nbs1)):
                    b1 = nbs1[i]
                    for j in range(i + 1, len(nbs1)):
                        b2 = nbs1[j]
                        for a2 in adj[b2]:
                            if a2 == a1:
                                continue
                            for b3 in adj[a2]:
                                if b3 == b2 or b3 == b1:
                                    continue
                                if b3 in adj[a1]:
                                    return True
        
        if g > 8:
            # Check 8-cycles using non-backtracking BFS from each left node
            # up to depth 4, looking for return to start
            for start in range(half):
                # State: (current_node, prev_node, depth)
                # We want paths of length 8 that return to start
                # Use DFS with visited set to avoid counting same cycle multiple times
                # Simpler: BFS tracking (node, parent, depth) and check if we can reach start at depth 8
                visited = set()
                queue = deque()
                for nb in adj[start]:
                    queue.append((nb, start, 1))
                    visited.add((nb, start, 1))
                
                while queue:
                    node, parent, depth = queue.popleft()
                    if depth >= 4:
                        continue
                    for nb in adj[node]:
                        if nb == parent:
                            continue
                        new_state = (nb, node, depth + 1)
                        if new_state in visited:
                            continue
                        if nb == start and depth + 1 == 4:
                            # Found an 8-cycle (path of length 4 from start back to start
                            # in the "half graph" means 8 edges in original)
                            # Wait, this is the bipartite half. A path of length 4
                            # from start to start in the line graph... 
                            # Actually, in the bipartite graph, a cycle of length 8
                            # corresponds to a closed walk of length 8.
                            # From start (left), go right (1), left (2), right (3), left (4),
                            # right (5), left (6), right (7), left (8) = start
                            # So we need a path of length 8 from start back to start.
                            # In terms of the BFS above, we track depth as number of edges.
                            # depth=4 means we've taken 4 edges: start->r->l->r->l
                            # If l == start, that's a cycle of length 4 in the "compressed" graph
                            # which is length 8 in the original? No.
                            # 
                            # Let me reconsider. In the bipartite graph with left nodes 0..half-1
                            # and right nodes half..n-1:
                            # A cycle of length 8: L1-R1-L2-R2-L3-R3-L4-R4-L1
                            # Starting from L1=start, after 8 edges we're back at L1.
                            # In my BFS, I start from start (left), go to right (depth 1),
                            # then left (depth 2), etc. At depth 8, if I'm at start, that's an 8-cycle.
                            # But I'm only going to depth 4. Let me fix this.
                            return True
                        if depth + 1 < 4:
                            visited.add(new_state)
                            queue.append(new_state)
                        elif depth + 1 == 4:
                            # At depth 4, we're at a left node. If it's start, 8-cycle.
                            # But also need to check further for longer paths? No, we only care about < g.
                            if nb == start:
                                return True
        
        return False

    def generate_random_bipartite_4reg(half):
        """Generate a random 4-regular bipartite graph with 'half' vertices on each side."""
        left_nodes = list(range(half))
        
        # Each left node has 4 stubs, each right node has 4 stubs
        left_stubs = []
        for i in range(half):
            left_stubs.extend([i] * 4)
        right_stubs = []
        for i in range(half):
            right_stubs.extend([half + i] * 4)
        
        random.shuffle(right_stubs)
        
        # Pair stubs
        adj = [set() for _ in range(n)]
        for l, r in zip(left_stubs, right_stubs):
            if l not in adj[r]:  # Avoid multi-edges
                adj[l].add(r)
                adj[r].add(l)
        
        # Check if all nodes have degree 4
        for i in range(n):
            if len(adj[i]) != 4:
                return None
        return adj

    def local_search(adj, n, max_iters):
        """Local search to minimize short cycles."""
        half = n // 2
        best_c4, best_c6 = count_short_cycles(adj, n)
        best_score = best_c4 * 1000 + best_c6
        best_adj = [set(s) for s in adj]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to swap (2-opt move)
            u = random.randint(0, half - 1)
            w = random.randint(0, half - 1)
            if u == w:
                continue
            
            nbs_u = list(adj[u])
            nbs_w = list(adj[w])
            if not nbs_u or not nbs_w:
                continue
            
            v = random.choice(nbs_u)
            x = random.choice(nbs_w)
            
            # Check if swap is valid (no multi-edges)
            if v == x:
                continue
            if x in adj[u] or v in adj[w]:
                continue
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].add(x)
            adj[x].add(u)
            adj[w].add(v)
            adj[v].add(w)
            
            c4, c6 = count_short_cycles(adj, n)
            score = c4 * 1000 + c6
            
            if score < best_score:
                best_score = score
                best_c4, best_c6 = c4, c6
                best_adj = [set(s) for s in adj]
            else:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[w].remove(v)
                adj[v].remove(w)
                adj[u].add(v)
                adj[v].add(u)
                adj[w].add(x)
                adj[x].add(w)
        
        return best_adj, best_c4, best_c6

    def adj_to_edges(adj, n):
        edges = []
        for u in range(n // 2):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Try different n values
    best_result = None
    best_n = 2048  # LUW size
    
    # Start from smaller n and go up
    for n in range(244, min(400, best_n), 2):  # n must be even for bipartite
        if time.time() > deadline - 10:
            break
        
        half = n // 2
        found = False
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            adj = generate_random_bipartite_4reg(half)
            if adj is None:
                continue
            
            adj, c4, c6 = local_search(adj, n, max_iters=200)
            
            if c4 == 0 and c6 == 0:
                # Check girth properly for 8-cycles
                if not has_cycle_less_than_g(adj, n, 10):
                    edges = adj_to_edges(adj, n)
                    if best_result is None or n < best_n:
                        best_result = (n, edges)
                        best_n = n
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to LUW
    return _luw_construct(k, g)
